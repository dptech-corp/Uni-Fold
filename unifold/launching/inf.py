from typing import *
import torch
import time
import numpy as np
import json
import os
import os.path as osp
from unicore.utils import (
    tensor_tree_map,
)

from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold
from unifold.data import protein, residue_constants
from unifold.colab.data import load_feature_for_one_target
from unifold.symmetry import (
    UFSymmetry,
    uf_symmetry_config,
    assembly_from_prediction,
)
from unifold.inference import automatic_chunk_size
import hashlib
import os
from typing import *


from unifold.data import residue_constants, protein
from unifold.dataset import load_and_process, UnifoldDataset
from unifold.symmetry import load_and_process_symmetry
import pathlib

from .data import unique_job_id_gen

def load_feature_for_one_target(
    seqids,
    config,
    feat_dir,
    symmetry_group,
    seed,
):
    if symmetry_group is None:
        batch, _ = load_and_process(
            config=config.data,
            mode="predict",
            seed=seed,
            batch_idx=None,
            data_idx=0,
            is_distillation=False,
            sequence_ids=seqids,
            monomer_feature_dir=feat_dir,
            uniprot_msa_dir=None,
        )
    else:
        batch, _ = load_and_process_symmetry(
            config=config.data,
            mode="predict",
            seed=seed,
            batch_idx=None,
            data_idx=0,
            is_distillation=False,
            symmetry=symmetry_group,
            sequence_ids=seqids,
            monomer_feature_dir=feat_dir,
            uniprot_msa_dir=None,
        )
    batch = UnifoldDataset.collater([batch])
    return batch


def config_and_model(
    model_name, param_path, is_symmetry, device="cuda",
):
    config = model_config(model_name)
    model = AlphaFold(config) if not is_symmetry else UFSymmetry(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.inference_mode()
    model = model.to(device)
    return config, model


def get_results(batch, out, symmetry_group, use_multimer):
    def to_float(x):
        if x.dtype == torch.bfloat16 or x.dtype == torch.half:
            return x.float()
        else:
            return x
    # Toss out the recycling dimensions --- we don't need them anymore
    batch = tensor_tree_map(lambda t: t[-1, 0, ...], batch)
    batch = tensor_tree_map(to_float, batch)
    out = tensor_tree_map(lambda t: t[0, ...], out)
    out = tensor_tree_map(to_float, out)
    batch = tensor_tree_map(lambda x: np.array(x.cpu()), batch)
    out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

    plddt = out["plddt"]
    mean_plddt = np.mean(plddt)
    plddt_b_factors = np.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )
    # TODO: , may need to reorder chains, based on entity_ids
    if symmetry_group is None:
        cur_protein = protein.from_prediction(
            features=batch, result=out, b_factors=plddt_b_factors
        )
    else:
        plddt_b_factors_assembly = np.concatenate(
            [plddt_b_factors for _ in range(batch["symmetry_opers"].shape[0])])
        cur_protein = assembly_from_prediction(
            result=out, b_factors=plddt_b_factors_assembly,
        )
    if use_multimer and symmetry_group is None:
        mean_ptm = np.mean(out["iptm+ptm"])
        return cur_protein, mean_plddt, mean_ptm
    else:
        return cur_protein, mean_plddt, None


def launching_inference(
    all_targets: List[List[str]],
    feat_dir: str,
    param_dir: str,
    output_dir: str,
    use_multimer: bool,
    symmetry_group: str,
    max_recycling_iters: int,
    num_ensembles: int,
    times: int,
    manual_seed: int,
    device: str = "cuda:0",
):
    if symmetry_group is not None:
        model_name = "uf_symmetry"
        param_path = os.path.join(param_dir, "uf_symmetry.pt")
    elif use_multimer:
        model_name = "multimer_ft"
        param_path = os.path.join(param_dir, "multimer.unifold.pt")
    else:
        model_name = "model_2_ft"
        param_path = os.path.join(param_dir, "monomer.unifold.pt")

    if symmetry_group is None:
        config = model_config(model_name)
    else:
        config = uf_symmetry_config()

    config.data.common.max_recycling_iters = max_recycling_iters
    config.globals.max_recycling_iters = max_recycling_iters
    config.data.predict.num_ensembles = num_ensembles

    # faster prediction with large chunk
    config.globals.chunk_size = 128
    model = AlphaFold(config) if symmetry_group is None else UFSymmetry(config)
    print("start to load params {}".format(param_path))
    state_dict = torch.load(param_path)["ema"]["params"]
    state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.inference_mode()

    jidgen = unique_job_id_gen()

    for seqids in all_targets:
        target_id = next(jidgen)
        # data path is based on target_name
        cur_output_dir = osp.join(output_dir, target_id)

        print("start to predict {}".format(target_id))
        plddts = {}
        ptms = {}
        best_result = None
        best_score = 0

        for seed in range(times):
            cur_seed = hash((manual_seed, seed)) % 100000
            batch = load_feature_for_one_target(
                seqids, config, feat_dir, symmetry_group, cur_seed
            )
            seq_len = batch["aatype"].shape[-1]
            chunk_size, block_size = automatic_chunk_size(
                                        seq_len,
                                        device,
                                        is_bf16=False,
                                    )
            model.globals.chunk_size = chunk_size
            model.globals.block_size = block_size

            with torch.no_grad():
                batch = {
                    k: torch.as_tensor(v, device=device)
                    for k, v in batch.items()
                }
                shapes = {k: v.shape for k, v in batch.items()}
                print(shapes)
                t = time.perf_counter()
                out = model(batch)
                print(f"Inference time: {time.perf_counter() - t}")

            cur_protein, mean_plddt, mean_ptm = get_results(
                batch, out, symmetry_group, use_multimer
            )
            cur_save_name = (
                f"{target_id}_{model_name}_{cur_seed:06d}"
            )
            plddts[cur_save_name] = str(mean_plddt)
            if mean_ptm is not None:
                ptms[cur_save_name] = str(mean_ptm)

            with open(os.path.join(cur_output_dir, cur_save_name + '.pdb'), "w") as f:
                f.write(protein.to_pdb(cur_protein))

            cur_score = mean_ptm if mean_ptm is not None else mean_plddt
            if cur_score > best_score:
                best_score = cur_score
                best_result = {
                    "protein": cur_protein,
                    "plddt": mean_plddt,
                    "ptm": mean_ptm,
                }

        jobname = f"{target_id}_{model_name}"
        print("plddts", plddts)
        json.dump(plddts, open(os.path.join(cur_output_dir, f"{jobname}_plddt.json"), "w"), indent=1)
        best_save_name = jobname + f"_best_plddt={best_result['plddt'][:6]}"
        if ptms:
            print("ptms", ptms)
            json.dump(ptms, open(os.path.join(output_dir, f"{jobname}_ptm.json"), "w"), indent=1)
            best_save_name += f"_ptm={best_result['ptm'][:6]}"

        with open(os.path.join(cur_output_dir, f'{best_save_name}.pdb'), "w") as f:
            f.write(protein.to_pdb(best_result["protein"]))
