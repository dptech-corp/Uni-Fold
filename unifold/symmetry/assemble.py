import torch
import numpy as np

from unifold.data.protein import Protein
from ..modules.featurization import atom14_to_atom37
from ..modules.frame import Frame


def expand_frames(frames: Frame, ops: Frame) -> torch.Tensor:
    """
    Args:
        frames: Rigid of shape [*, NR]
        ops: Rigid of shape [NG]
    Returns:
        Tensor of shape [*, NGxNR, 4, 4]
    """
    batch_shape = frames.shape[:-1]
    ret = ops[..., None].compose(frames[..., None, :]).to_tensor_4x4()
    ret = ret.reshape(*batch_shape, -1, 4, 4)
    return ret


def expand_sc_frames(sc_frames: Frame, ops: Frame) -> torch.Tensor:
    """
    Args:
        frames: Rigid of shape [*, NR]
        ops: Rigid of shape [NG]
    Returns:
        Tensor of shape [*, NGxNR, 4, 4]
    """
    batch_shape = sc_frames.shape[:-2]
    ret = ops[..., None, None].compose(sc_frames[..., None, :, :]).to_tensor_4x4()
    ret = ret.reshape(*batch_shape, -1, sc_frames.shape[-1], 4, 4)
    return ret


def expand_atom_positions(positions: torch.Tensor, ops: Frame) -> torch.Tensor:
    """
    Args:
        positions: Tensor of shape [*, NR, 37, 3]
        ops: Rigid of shape [NG]
    Returns:
        Tensor of shape [*, NG * NR]
    """
    batch_shape = positions.shape[:-3]
    position_shape = positions.shape[-2:]
    ret = ops[..., None, None].apply(positions[..., None, :, :, :])
    ret = ret.reshape(*batch_shape, -1, *position_shape)
    return ret


def expand_symmetry(sm_out, batch):
    ops = Frame.from_tensor_4x4(batch["symmetry_opers"][-1, 0, ...].float())    # reduce recycle and batch dims.
    num_expand = ops.shape[0]
    frames = Frame.from_tensor_4x4(sm_out["frames"].float())
    sidechain_frames = Frame.from_tensor_4x4(sm_out["sidechain_frames"].float())
    positions = sm_out["positions"].float()
    
    def repeat_fn(tensor, repeats, dim):
        shape = [1 for _ in tensor.shape]
        shape[dim] = repeats
        return tensor.repeat(shape)

    symm_out = {
        "frames": expand_frames(frames, ops),
        "sidechain_frames": expand_sc_frames(sidechain_frames, ops),
        "unnormalized_angles": repeat_fn(sm_out["unnormalized_angles"], num_expand, dim=-3),
        "angles": repeat_fn(sm_out["angles"], num_expand, dim=-3),
        "single": repeat_fn(sm_out["single"], num_expand, dim=-2),
        "positions": expand_atom_positions(positions, ops),
    }
    
    feats_expand_dims = {
        "residx_atom37_to_atom14": -2,
        "entity_id": -1,
        "num_sym": -1,
        "aatype": -1,
        "residue_index": -1,
        "atom37_atom_exists": -2,
        "seq_mask": -1,
    }
    symm_feats = {
        k: repeat_fn(batch[k], num_expand, dim=v)[-1] for k, v in feats_expand_dims.items() if k in batch
    }

    asym_id = batch["asym_id"]
    def asym_fn(asym_id, i, num_asym):
        ret = asym_id + num_asym * i
        ret[asym_id == 0] = 0
        return ret
    asym_ids = torch.cat(
        [asym_fn(asym_id, i, batch["num_asym"]) for i in range(num_expand)], dim=-1
    ).long()
    symm_feats["asym_id"] = asym_ids[-1]
    symm_feats["num_sym"] = symm_feats["num_sym"] * num_expand
    symm_feats["num_asym"] = batch["num_asym"][-1] * num_expand

    if "all_atom_positions" in batch:
        symm_feats["all_atom_positions"] = expand_atom_positions(batch["all_atom_positions"], ops)[-1]
        symm_feats["all_atom_mask"] = repeat_fn(batch["all_atom_mask"], num_expand, -2)[-1]
    
    symm_out["expand_final_atom_positions"] = atom14_to_atom37(symm_out["positions"], symm_feats)
    symm_out["expand_final_atom_mask"] = symm_feats["atom37_atom_exists"]
    
    return symm_feats, symm_out


def assembly_from_prediction(
    result,
    b_factors=None) -> Protein:

    chain_index = result["expand_batch"]["asym_id"]
    aatype = result["expand_batch"]["aatype"]
    residue_index = result["expand_batch"]["residue_index"]
    atom_positions = result["expand_final_atom_positions"]
    atom_mask = result["expand_final_atom_mask"]
    if b_factors is None:
        b_factors = np.zeros_like(atom_mask)
    return Protein(
        aatype=aatype,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        residue_index=residue_index + 1,
        chain_index=chain_index - 1,
        b_factors=b_factors
    )
