import os

import glob
from Bio.PDB import protein_letters_3to1
import json
from tqdm import tqdm
from multiprocessing import Pool
from unifold.msa.mmcif import parse
import argparse
import gzip
import numpy as np
from unifold.data.residue_constants import restype_order_with_x
from unifold.msa.templates import _get_atom_positions as get_atom_positions
import pickle


def mmcif_object_to_fasta(mmcif_object, auth_chain_id: str) -> str:
    residues = mmcif_object.seqres_to_structure[auth_chain_id]
    residue_names = [residues[t].name for t in range(len(residues))]
    residue_letters = [
        protein_letters_3to1[n] if n in protein_letters_3to1.keys() else "X"
        for n in residue_names
    ]
    # take care of cases where residue letters are of length 3
    # simply by replacing them as 'X' ('UNK')
    filter_out_triple_letters = lambda x: x if len(x) == 1 else "X"
    fasta_string = "".join([filter_out_triple_letters(n) for n in residue_letters])
    return fasta_string


def get_label(input_args):
    mmcif_file, label_dir = input_args
    pdb_id = os.path.basename(mmcif_file).split(".")[0]
    with gzip.open(mmcif_file, "rb") as fn:
        cif_string = fn.read().decode("utf8")
    parsing_result = parse(file_id=pdb_id, mmcif_string=cif_string)
    mmcif_obj = parsing_result.mmcif_object

    information = []
    if mmcif_obj is not None:
        for chain_id in mmcif_obj.chain_to_seqres:
            label_name = f"{pdb_id}_{chain_id}"
            label_path = os.path.join(label_dir, f"{label_name}.label.pkl.gz")
            try:
                all_atom_positions, all_atom_mask = get_atom_positions(
                    mmcif_obj, chain_id, max_ca_ca_distance=float("inf")
                )
                sequence = mmcif_object_to_fasta(mmcif_obj, chain_id)
                aatype_idx = np.array(
                    [
                        restype_order_with_x[rn]
                        if rn in restype_order_with_x
                        else restype_order_with_x["X"]
                        for rn in sequence
                    ]
                )
                resolution = np.array([mmcif_obj.header["resolution"]])
                seq_len = aatype_idx.shape[0]
                _, counts = np.unique(aatype_idx, return_counts=True)
                freqs = counts.astype(np.float32) / seq_len
                max_freq = np.max(freqs)
                if resolution > 9 or max_freq > 0.8:
                    continue

                date = mmcif_obj.header["release_date"]
                release_date = np.array([date])
                label = {
                    "aatype_index": aatype_idx.astype(np.int8),  # [NR,]
                    "all_atom_positions": all_atom_positions.astype(
                        np.float32
                    ),  # [NR, 37, 3]
                    "all_atom_mask": all_atom_mask.astype(np.int8),  # [NR, 37]
                    "resolution": resolution.astype(np.float32),  # [1,]
                    "release_date": release_date,
                }
                pickle.dump(label, gzip.GzipFile(label_path, "wb"), protocol=4)

            except Exception as e:
                information.append("{} {} error".format(label_name, str(e)))
    else:
        print(pdb_id, "Parse mmcif error")
        return pdb_id, "Parse mmcif error"

    if len(information) > 0:
        print(pdb_id, "\t".join(information))
        return pdb_id, "\t".join(information)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmcif-dir", type=str, default="")
    parser.add_argument("--label-dir", type=str, default="")
    parser.add_argument("--output-fn", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    print(args)

    os.makedirs(args.label_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_fn), exist_ok=True)

    mmcif_files = glob.glob(os.path.join(args.mmcif_dir, "*.cif.gz"))
    file_cnt = len(mmcif_files)
    print(f"len(mmcif_files): {len(mmcif_files)}")

    def input_files():
        for fn in mmcif_files:
            yield fn, args.label_dir

    meta_dict = {}
    with Pool(1 if args.debug else 64) as pool:
        for ret in tqdm(
            pool.imap(get_label, input_files(), chunksize=10), total=file_cnt
        ):
            if ret is not None:
                meta_dict[ret[0]] = ret[1]

    json.dump(meta_dict, open(args.output_fn, "w"), indent=2)


if __name__ == "__main__":
    main()
