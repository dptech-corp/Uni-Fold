import torch as th
from unifold.data import residue_constants as rc
from unifold.modules.frame import Frame
from .geometry import kabsch_rmsd, get_optimal_transform, compute_rmsd
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional


def compute_x_fape(
    pred_frames: Frame,
    target_frames: Frame,
    pred_points: th.Tensor,
    target_points: th.Tensor,
    frames_mask: Optional[th.Tensor] = None,
    points_mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """ FAPE matrix from all frames to the cross-matrix of points,
    used to find a permutation which gives the optimal loss for a symmetric structure.

    Notation for use with n chains of length p, under f=k frames:
    - n: number of protein chains
    - p: number of points (length of chain)
    - d: dimension of points = 3
    - f=k: arbitrary number of frames
    (..., k, 1, 1) @ (..., 1, n, p, d) -> (..., k, n, p, d)
    (..., k, ni, p, d) - (..., k, nj, p, d) -> (..., ni, nj)

    Iff the frames from the same chains are used, then k=(n f) and f=p

    Args:
        pred_frames: (..., k)
        target_frames: (..., k)
        pred_points: (..., n, p, d)
        target_points: (..., n, p, d)
        frames_mask: (..., k) float tensor
        points_mask: (..., n, p) float tensor

    Returns:
        (..., n, n) th.Tensor
    """
    # define masks for reduction
    mask = 1.
    if frames_mask is not None:
        mask = mask * frames_mask[..., None, None, None]
    if points_mask is not None:
        mask = mask * points_mask[..., None, :, None, :, :] * points_mask[..., None, None, :, :, :]

    # (..., k, 1, 1) @ (..., 1, n, p, d) -> (..., k, n, p, d)
    local_pred_pos = pred_frames[..., None, None].invert().apply(
        pred_points[..., None, :, :, :].float(),
    )
    local_target_pos = target_frames[..., None, None].invert().apply(
        target_points[..., None, :, :, :].float(),
    )
    # (..., k, ni, p, d) - (..., k, nj, p, d) -> (..., k, ni, nj, p)
    d_pt2 = (local_pred_pos.unsqueeze(dim=-3) - local_target_pos.unsqueeze(dim=-4)).square().sum(-1)
    d_pt = d_pt2.add_(1e-5).sqrt()

    # (..., k, ni, nj, p) -> (..., ni, nj)
    if frames_mask is not None or points_mask is not None:
        x_fape = (d_pt * mask).sum(dim=(-1, -4)) / mask.sum(dim=(-1, -4))
    else:
        x_fape = d_pt.mean(dim=(-1, -4))

    return x_fape


def multi_chain_perm_align(out: Dict, batch: Dict, labels: List[Dict]) -> Dict:
    """ Permutes labels so that a structural loss wrt preds is minimized.
    Framed as a linear assignment problem, loss is sum of "individual" losses
    and the permutation is found by the Hungarian algorithm on the cross matrix.

    WARNING! All keys in `out` have no batch size
    """
    assert isinstance(labels, list)
    # get all unique chains
    unique_asym_ids = th.unique(batch["asym_id"])
    best_global_curr = th.clone(unique_asym_ids)
    best_global_perm = th.clone(unique_asym_ids)
    best_global_perm_list = best_global_perm.tolist()

    # all indices associated with a chain (asymmetric unit)
    per_asym_residue_index = {}
    for cur_asym_id in unique_asym_ids:
        asym_mask = (batch["asym_id"] == cur_asym_id).bool()
        per_asym_residue_index[int(cur_asym_id)] = batch["residue_index"][asym_mask]

    # Get values to compute cross-matrix, use reference frames
    true_frames, true_frames_mask = [], []
    for l, cur_asym_id in zip(labels, unique_asym_ids):
        asym_res_idx = per_asym_residue_index[int(cur_asym_id)]
        true_frames.append(Frame.from_tensor_4x4(l["true_frame_tensor"][asym_res_idx]))
        true_frames_mask.append(l["frame_mask"][asym_res_idx])

    # [bsz, nres, d=3]
    true_frames = Frame.cat(true_frames, dim=0)
    # [bsz, nres]
    true_frames_mask = th.cat(true_frames_mask, dim=0)
    # [bsz, nres, d=3]
    pred_frames = Frame.from_tensor_4x4(out["pred_frame_tensor"])
    # [bsz, nres]
    pred_frames_mask = batch["final_atom_mask"][..., [0, 1, 2]].float().prod(dim=-1)

    # will rename only for every unique structure with symmetric counterparts (works for abab, abb, abbcc, ...)
    unique_ent_ids, unique_ent_counts = th.unique(batch["entity_id"], return_counts=True)

    # use all frames from non-symmetric chains, already-renamed symmetric chains and current entity
    ref_frames_mask = th.zeros_like(pred_frames_mask)
    for ent_id in unique_ent_ids:
        ent_mask = batch["entity_id"] == ent_id
        asym_ids = th.unique(batch["asym_id"][ent_mask])
        if len(asym_ids) == 1:
            asym_mask = batch["asym_id"] == asym_ids[0]
            ref_frames_mask[asym_mask] = pred_frames_mask[asym_mask]

    # rename symmetric chains
    for ent_id, ent_count in zip(unique_ent_ids, unique_ent_counts):
        # see how many chains for the entity, if just 1, continue
        ent_mask = batch["entity_id"] == ent_id
        asym_ids = th.unique(batch["asym_id"][ent_mask])
        if len(asym_ids) == 1:
            continue
        # create placeholders for points and corresponding masks of shape (n, l, d) and (n, l)
        local_perm_idxs = []
        ent_mask = batch["entity_id"] == ent_id
        ent_res_idx = batch["residue_index"][ent_mask]
        min_res, max_res = ent_res_idx.amin().item(), ent_res_idx.amax().item()
        n = len(asym_ids)
        l = max_res - min_res + 1
        ph_pred_ca_pos = pred_frames._t.new_zeros((n, l, 3))
        ph_true_ca_pos = pred_frames._t.new_zeros((n, l, 3))
        points_mask = pred_frames._t.new_zeros((n, l,))
        # fill placeholders with points and masks
        for i, ni in enumerate(asym_ids):
            local_perm_idxs.append(best_global_perm_list[ni.item()])
            asym_mask = batch["asym_id"] == ni
            asym_res_idx = batch["residue_index"][asym_mask]
            ph_pred_ca_pos[i, asym_res_idx - min_res] = pred_frames._t[i, asym_mask].clone()
            ph_true_ca_pos[i, asym_res_idx - min_res] = true_frames._t[i, asym_mask].clone()
            points_mask[i, asym_mask] = 1.

        # include all frames of non-symmetric, already assigned symmetric, and this entity
        frames_mask_ = (pred_frames_mask * true_frames_mask) * (ref_frames_mask + ent_mask.float()).bool().float()

        # cross-matrix and hungarian algorithm finds the best permutation
        # (n=N f=L), (n=N f=L), (n=N, p=L), (n=N, p=L) -> (ni=N, nj=N)
        x_mat = compute_x_fape(
            pred_frames=pred_frames,
            target_frames=true_frames,
            pred_points=ph_pred_ca_pos,
            target_points=ph_true_ca_pos,
            frames_mask=frames_mask_,
            points_mask=points_mask,
        ).cpu().transpose(-1, -2).numpy()
        rows, cols = linear_sum_assignment(x_mat)

        # update frames_mask to include already assigned frames for the next iteration
        ref_frames_mask[frames_mask_] = 1.

        # remap labels like: labels["ent_mask"] = ph_true_ca_pos[cols][ph_true_ca_mask[cols]]
        global_rows = local_perm_idxs
        global_cols = [local_perm_idxs[c] for c in cols]
        best_global_perm[global_rows] = best_global_perm[global_cols]

    # (N,) -> (2, N)
    ij_label_align = th.stack((best_global_curr, best_global_perm), dim=0).tolist()
    best_labels = merge_labels(
        batch=batch,
        per_asym_residue_index=per_asym_residue_index,
        labels=labels,
        align=ij_label_align
    )

    return best_labels


def merge_labels(batch: Dict, per_asym_residue_index: Dict[List[int]], labels: Dict, align: List[Tuple[int, int]]) -> Dict:
    """ Reorders the labels

    Args:
        batch: dict of tensors
        per_asym_residue_index: dict mapping every asym_id to a list of its residue indices
        labels: list of label dicts, each with shape [nk, *]
        align: list of int tuples (i,j) such that label j will soon be label i. ex. [(1, 2), (2, 1)]

    Returns:
        merged_labels: dict of tensors with reordered labels
    """
    num_res = batch["msa_mask"].shape[-1]
    outs = {}
    for k, v in labels[0].items():
        if k in [
            "resolution",
        ]:
            continue
        cur_out = {}
        for i, j in align:
            label = labels[j][k]
            # to 1-based
            cur_residue_index = per_asym_residue_index[i + 1]
            cur_out[i] = label[cur_residue_index]
        cur_out = [x[1] for x in sorted(cur_out.items())]
        new_v = th.cat(cur_out, dim=0)
        merged_nres = new_v.shape[0]
        assert (
            merged_nres <= num_res
        ), f"bad merged num res: {merged_nres} > {num_res}. something is wrong."
        if merged_nres < num_res:  # must pad
            pad_dim = new_v.shape[1:]
            pad_v = new_v.new_zeros((num_res - merged_nres, *pad_dim))
            new_v = th.cat((new_v, pad_v), dim=0)
        outs[k] = new_v
    return outs
