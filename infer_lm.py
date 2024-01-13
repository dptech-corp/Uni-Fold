#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
from pathlib import Path
import sys
import gc
import os
import time
import re
from unicore import options
from functools import lru_cache
from unicore.data import Dictionary
import numpy as np
import torch
from unifold.musse.plm.model.bert import BertModel
import time

import pickle
import gzip

import torch.utils.data.distributed
from tqdm import tqdm

HHBLITS_AA_TO_ID = {
    "[CLS]": 0,
    "[PAD]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 24,
    "B": 25,
    "U": 26,
    "Z": 27,
    "O": 28,
    ".": 29,
    "-": 30,
    "<null_1>": 31,
    "[MASK]": 32,
}
AA_TO_ESM = HHBLITS_AA_TO_ID


def parse_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


class MultimerDataset:
    def __init__(self, inputfn):
        self.key = Path(inputfn).stem
        fasta_str = open(inputfn).read()
        input_seqs, _ = parse_fasta(fasta_str)
        self.seqs = input_seqs

    def __len__(self):
        return 1

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        assert idx == 0
        sequences = self.seqs
        all_chain_features = self.get_chain_features(sequences)
        is_same_entity = self.is_same_entity_func(all_chain_features)
        has_same_sequence = self.has_same_sequence_func(all_chain_features)
        tokenized_data = [AA_TO_ESM[tok] for sequence in sequences for tok in sequence]
        tokenized_data = [0] + tokenized_data + [2]
        sequence = torch.from_numpy(np.array(tokenized_data)).long()
        return self.key, sequence, is_same_entity, has_same_sequence

    def get_chain_features(self, sequences):
        chain_features = {}
        all_chain_features = []
        seq_to_entity_id = {}
        chain_id = 1
        for seq in sequences:
            chain_features = {}
            if str(seq) not in seq_to_entity_id:
                seq_to_entity_id[str(seq)] = len(seq_to_entity_id) + 1
            chain_features["seq_length"] = len(seq)
            chain_features["seq"] = seq
            chain_features["asym_id"] = chain_id * np.ones(chain_features["seq_length"])
            chain_features["entity_id"] = seq_to_entity_id[str(seq)] * np.ones(
                chain_features["seq_length"]
            )

            all_chain_features.append(chain_features)
            chain_id += 1
        return all_chain_features

    def is_same_entity_func(self, all_chain_features):
        entity_id_list = []
        for chain_id in range(1, len(all_chain_features) + 1):
            if all_chain_features[chain_id - 1] is None:
                continue
            else:
                entity_id_list.append(all_chain_features[chain_id - 1]["entity_id"])

        entity_id = np.concatenate(entity_id_list)
        seq_length = len(entity_id)
        mask = entity_id.reshape(seq_length, 1) == entity_id.reshape(1, seq_length)
        mask = torch.tensor(mask)
        is_same_entity = torch.zeros(seq_length, seq_length).long()
        is_same_entity = is_same_entity.masked_fill_(mask, 1)

        is_same_entity_bos_eos = torch.zeros(
            is_same_entity.shape[0] + 2, is_same_entity.shape[0] + 2
        ).long()

        is_same_entity_bos_eos[1:-1, 1:-1] = is_same_entity
        return is_same_entity_bos_eos

    def has_same_sequence_func(self, all_chain_features):
        asym_id_list = []
        for chain_id in range(1, len(all_chain_features) + 1):
            if all_chain_features[chain_id - 1] is None:
                continue
            else:
                asym_id_list.append(all_chain_features[chain_id - 1]["asym_id"])

        asym_id = np.concatenate(asym_id_list)
        seq_length = len(asym_id)
        mask = asym_id.reshape(seq_length, 1) == asym_id.reshape(1, seq_length)
        mask = torch.tensor(mask)
        has_same_sequence = torch.zeros(seq_length, seq_length).long()
        has_same_sequence = has_same_sequence.masked_fill_(mask, 1)

        has_same_sequence_bos_eos = torch.zeros(
            has_same_sequence.shape[0] + 2, has_same_sequence.shape[0] + 2
        ).long()

        has_same_sequence_bos_eos[1:-1, 1:-1] = has_same_sequence
        return has_same_sequence_bos_eos.long()


def load_model_ensemble_and_task(
    filenames,
):

    from unicore import tasks

    filename = filenames[0]
    state = torch.load(filename, map_location=torch.device("cpu"))
    args = state["args"]
    dictionary = Dictionary.load(os.path.join("unifold/musse/plm", "dict_esm.txt"))
    model = BertModel(args, dictionary)

    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder."]  # ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {
            pattern.sub("", name): param for name, param in state_dict.items()
        }

        prefixes = [
            "sentence_encoder.embed_tokens"
        ]  # ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {
            pattern.sub("embed_tokens", name): param
            for name, param in state_dict.items()
        }
        return state_dict

    state["model"] = upgrade_state_dict(state["model"])
    model.load_state_dict(state["model"], strict=True, model_args=args)
    return model


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument("--input", type=str, default="")
    parser.add_argument(
        "--path",  # model_location
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="where to use bf16",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="where to use fp16",
    )
    # parser.add_argument(
    #     "fasta_file",
    #     type=pathlib.Path,
    #     help="FASTA file on which to extract representations",
    # )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument(
        "--toks_per_batch", type=int, default=4096, help="maximum batch size"
    )
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts", "attentions"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )
    parser.add_argument(
        "--user-dir",
        default=None,
        help="path to a python module containing custom extensions (tasks and/or architectures)",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="not used")

    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    return parser


def main(args):
    if args.bf16:
        raise "not support bf16"
   
    this_rank = 0
   
    if this_rank == 0:
        print(f"model_path: {args.path}")
    model = load_model_ensemble_and_task(
        [args.path],
    )

    model = model.half()

  
    gc.collect()

    if this_rank == 0:
        print("loaded model successfully")
    
    model.eval()

    dataset = MultimerDataset(args.input)
   
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    if this_rank == 0:
        print(f"{len(dataset)} sequences")
    torch.cuda.set_device(this_rank)

    model = model.cuda(this_rank)
    gc.collect()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # return_contacts = "contacts" in args.include
    # need_head_weights = "attentions" in args.include

    num_layers = 36
    # args.repr_layers = [i+1 for i in range(args.num_layers)]

    assert all(-(num_layers + 1) <= i <= num_layers for i in args.repr_layers)
    repr_layers = [(i + num_layers + 1) % (num_layers + 1) for i in args.repr_layers]

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (labels, sequence, is_same_entity, has_same_sequence) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
           
            assert sequence.shape[0] == 1
            sequence = sequence.cuda(this_rank, non_blocking=True)
            is_same_entity = is_same_entity.cuda(this_rank, non_blocking=True)
            has_same_sequence = has_same_sequence.cuda(this_rank, non_blocking=True)

            
            out = model(
                sequence,
                is_same_entity=is_same_entity,
                has_same_sequence=has_same_sequence,
                features_only=True,
            )

           
            representations = torch.stack(
                [out[1].to(device="cpu"), out[36].to(device="cpu")], dim=2
            )

            for i, label in enumerate(labels):
                args.output_file = (
                    args.output_dir
                    / label
                    / f"{label}.esm2_multimer_finetune_emb.pkl.gz"
                )
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {}  # {"label": label}
               
                result["token"] = representations[i, 1:-1].clone().numpy()
                assert result["token"].shape[1] == 2
                assert result["token"].shape[2] == 2560

               
                pickle.dump(result, gzip.GzipFile(args.output_file, "wb"), protocol=4)
    t1 = time.time()
    if this_rank == 0:
        print(f"total inference time for {len(dataset)} samples: {t1-t0}s")


if __name__ == "__main__":
    parser = create_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
