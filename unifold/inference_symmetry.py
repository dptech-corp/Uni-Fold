import argparse
import gzip
import logging
import numpy as np
import os
import pathlib
import time
import torch
import pickle
from unifold.data import residue_constants, protein
from unifold.dataset import UnifoldDataset
from unicore.utils import (
    tensor_tree_map,
)

from unifold.symmetry import (
    UFSymmetry,
    load_and_process_symmetry,
    uf_symmetry_config,
    assembly_from_prediction,
)

from unifold.inference import (
    automatic_chunk_size,
)


def load_feature_for_one_target(
    config, data_folder, symmetry, seed=0, is_multimer=False, use_uniprot=False
):
    if not is_multimer:
        uniprot_msa_dir = None
        sequence_ids = ["A"]
        if use_uniprot:
            uniprot_msa_dir = data_folder
    else:
        uniprot_msa_dir = data_folder
        sequence_ids = open(os.path.join(data_folder, "chains.txt")).readline().split()
    batch, _ = load_and_process_symmetry(
        config=config.data,
        mode="predict",
        seed=seed,
        batch_idx=None,
        data_idx=0,
        is_distillation=False,
        symmetry=symmetry,
        sequence_ids=sequence_ids,
        monomer_feature_dir=data_folder,
        uniprot_msa_dir=uniprot_msa_dir,
        is_monomer=(not is_multimer),
    )
    batch = UnifoldDataset.collater([batch])
    return batch


def main(args):
    config = uf_symmetry_config()
    config.data.common.max_recycling_iters = args.max_recycling_iters
    config.globals.max_recycling_iters = args.max_recycling_iters
    config.data.predict.num_ensembles = args.num_ensembles
    is_multimer = config.model.is_multimer
    if args.sample_templates:
        # enable template samples for diversity
        config.data.predict.subsample_templates = True
    # faster prediction with large chunk
    config.globals.chunk_size = 128
    
    model = UFSymmetry(config)

    print("start to load params {}".format(args.param_path))
    state_dict = torch.load(args.param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(args.model_device)
    model.eval()
    model.inference_mode()
    if args.bf16:
        model.bfloat16()

    # data path is based on target_name
    data_dir = os.path.join(args.data_dir, args.target_name)
    output_dir = os.path.join(args.output_dir, args.target_name)
    os.system("mkdir -p {}".format(output_dir))
    param_name = pathlib.Path(args.param_path).stem
    name_suffix = ""
    if args.sample_templates:
        name_suffix += "_st"
    if not is_multimer and args.use_uniprot:
        name_suffix += "_uni"
    if args.max_recycling_iters != 3:
        name_suffix += "_r" + str(args.max_recycling_iters)
    if args.num_ensembles != 2:
        name_suffix += "_e" + str(args.num_ensembles)
    
    symmetry = args.symmetry
    if symmetry[0] != 'C':
        raise NotImplementedError(f"symmetry {symmetry} is not supported currently.")

    print("start to predict {}".format(args.target_name))
    for seed in range(args.times):
        cur_seed = hash((args.data_random_seed, seed)) % 100000
        batch = load_feature_for_one_target(
            config,
            data_dir,
            args.symmetry,
            cur_seed,
            is_multimer=is_multimer,
            use_uniprot=args.use_uniprot,
        )
        seq_len = batch["aatype"].shape[-1]
        
        # faster prediction with large chunk/block size
        chunk_size, block_size = automatic_chunk_size(
                                    seq_len,
                                    args.model_device,
                                    args.bf16
                                )
        model.globals.chunk_size = chunk_size
        model.globals.block_size = block_size


        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=args.model_device)
                for k, v in batch.items()
            }
            shapes = {k: v.shape for k, v in batch.items()}
            print(shapes)
            t = time.perf_counter()
            raw_out = model(batch, expand=True)     # when expand, output assembly.
            print(f"Inference time: {time.perf_counter() - t}")

        def to_float(x):
            if x.dtype == torch.bfloat16 or x.dtype == torch.half:
                return x.float()
            else:
                return x

        out = raw_out
        
        # Toss out the recycling dimensions --- we don't need them anymore
        batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
        batch = tensor_tree_map(to_float, batch)
        out = tensor_tree_map(lambda t: t[0, ...], out)
        out = tensor_tree_map(to_float, out)
        batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        plddt = out["plddt"]
        plddt_b_factors = np.repeat(
            plddt[..., None], residue_constants.atom_type_num, axis=-1
        )
        plddt_b_factors_assembly = np.repeat(
            plddt_b_factors, batch["symmetry_opers"].shape[0], axis=-2)
        
        cur_assembly = assembly_from_prediction(
            result=out, b_factors=plddt_b_factors_assembly
        )
        cur_save_name = (
            f"ufsymm_{param_name}_{cur_seed}{name_suffix}"
        )
        with open(os.path.join(output_dir, cur_save_name + '.pdb'), "w") as f:
            f.write(protein.to_pdb(cur_assembly))
        if args.save_raw_output:
            out = {
                k: v for k, v in out.items()
                if k.startswith("final_") or k.startswith("expand_final_") or k == "plddt"
            }
            with gzip.open(os.path.join(output_dir, cur_save_name + '_outputs.pkl.gz'), 'wb') as f:
                pickle.dump(out, f)
        del out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_device",
        type=str,
        default="cuda:0",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")""",
    )
    # Fixed model name as `uf_symmetry` so no need for --model_name
    parser.add_argument(
        "--param_path", type=str, default=None, help="Path to model parameters."
    )
    parser.add_argument(
        "--data_random_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        default="C1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max_recycling_iters",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_ensembles",
        type=int,
        default=2,
    )
    parser.add_argument("--sample_templates", action="store_true")
    parser.add_argument("--use_uniprot", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_raw_output", action="store_true")

    args = parser.parse_args()

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying
            --model_device for better performance"""
        )

    main(args)
