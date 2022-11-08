from typing import *
import torch
import time
import numpy as np
import json
import os
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


def colab_inference(
    target_id: str,
    data_dir: str,
    param_dir: str,
    output_dir: str,
    symmetry_group: Optional[str],
    is_multimer: bool,
    max_recycling_iters: int,
    num_ensembles: int,
    times: int,
    manual_seed: int,
    device: str = "cuda:0",
):

    if symmetry_group is not None:
        model_name = "uf_symmetry"
        param_path = os.path.join(param_dir, "uf_symmetry.pt")
    elif is_multimer:
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

    # data path is based on target_name
    cur_param_path_postfix = os.path.split(param_path)[-1]

    print("start to predict {}".format(target_id))
    plddts = {}
    ptms = {}
    best_result = None
    best_score = 0

    for seed in range(times):
        cur_seed = hash((manual_seed, seed)) % 100000
        batch = load_feature_for_one_target(
            config,
            data_dir,
            cur_seed,
            is_multimer=is_multimer,
            use_uniprot=is_multimer,
            symmetry_group=symmetry_group,
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
        cur_save_name = (
            f"{cur_param_path_postfix}_{cur_seed}"
        )
        plddts[cur_save_name] = str(mean_plddt)
        if is_multimer and symmetry_group is None:
            ptms[cur_save_name] = str(np.mean(out["iptm+ptm"]))
        with open(os.path.join(output_dir, cur_save_name + '.pdb'), "w") as f:
            f.write(protein.to_pdb(cur_protein))

        if is_multimer and symmetry_group is None:
            mean_ptm = np.mean(out["iptm+ptm"])
            if mean_ptm>best_score:
                best_result = {
                    "protein": cur_protein,
                    "plddt": out["plddt"],
                    "pae": out["predicted_aligned_error"]
                }
        else:
            if mean_plddt>best_score:
                best_result = {
                    "protein": cur_protein,
                    "plddt": out["plddt"],
                    "pae": None
                }

    print("plddts", plddts)
    score_name = f"{model_name}_{cur_param_path_postfix}"
    plddt_fname = score_name + "_plddt.json"
    json.dump(plddts, open(os.path.join(output_dir, plddt_fname), "w"), indent=4)
    if ptms:
        print("ptms", ptms)
        ptm_fname = score_name + "_ptm.json"
        json.dump(ptms, open(os.path.join(output_dir, ptm_fname), "w"), indent=4)
    
    return best_result
