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
    Notation:
    - n: number of protein chains
    - f: number of frames (length of chain)
    - p: number of points (length of chain)
    - d: dimension of points = 3
    (..., (n f), 1, 1) @ (..., 1, n, p, d) -> (..., (n f), n, p, d)
    (..., (n f), ni, p, d) - (..., (n f), nj, p, d) -> (..., ni, nj)

    Args:
        pred_frames: (..., (n f))
        target_frames: (..., (n f))
        pred_points: (..., n, p=f)
        target_points: (..., n, p=f)
        frames_mask: (..., (n f)) float tensor
        points_mask: (..., n, p=f) float tensor

    Returns:
        (..., n, n) th.Tensor
    """
    # define masks for reduction
    mask = 1.
    if frames_mask is not None:
        mask = mask * frames_mask[..., None, None, None]
    if points_mask is not None:
        mask = mask * points_mask[..., None, :, None, :, :] * points_mask[..., None, None, :, :, :]

    # (..., (n f), 1, 1) @ (..., 1, n, p, d) -> (..., (n f), n, p, d)
    local_pred_pos = pred_frames[..., None, None].invert().apply(
        pred_points[..., None, :, :, :].float(),
    )
    local_target_pos = target_frames[..., None, None].invert().apply(
        target_points[..., None, :, :, :].float(),
    )
    # (..., (n f), ni, p, d) - (..., (n f), nj, p, d) -> (..., (n f), ni, nj, p)
    d_pt2 = (local_pred_pos.unsqueeze(dim=-3) - local_target_pos.unsqueeze(dim=-4)).square().sum(-1)
    d_pt = d_pt2.add_(1e-5).sqrt()

    # (..., (n f), ni, nj, p) -> (..., ni, nj)
    if frames_mask is not None or points_mask is not None:
        x_fape = (d_pt * mask).sum(dim=(-1, -4)) / mask.sum(dim=(-1, -4))
    else:
        x_fape = d_pt.mean(dim=(-1, -4))

    return x_fape


def multi_chain_perm_align(out: Dict, batch: Dict, labels: List[Dict], shuffle_times: int = 2) -> Dict:
    """ Permutes labels so that a structural loss wrt preds is minimized.
    Framed as a linear assignment problem, loss is sum of "individual" losses
    and the permutation is found by the Hungarian algorithm on the cross matrix.
    """
    assert isinstance(labels, list)
    ca_idx = rc.atom_order["CA"]
    f_idxs = [ca_idx-1, ca_idx, ca_idx+1]

    # [bsz, nres, f=3, d=3]
    pred_ca_pos = out["final_atom_positions"][..., f_idxs, :].float()
    # [bsz, nres]
    pred_ca_mask = out["final_atom_mask"][..., f_idxs].float().prod(dim=-1)
    # list([nres, f=3, d=3])
    true_ca_poses = [l["all_atom_positions"][..., f_idxs, :].float() for l in labels]
    # list([nres,])
    true_ca_masks = [l["all_atom_mask"][..., f_idxs].float().prod(dim=-1) for l in labels]

    unique_asym_ids = th.unique(batch["asym_id"])
    best_global_curr = th.clone(unique_asym_ids)
    best_global_perm = th.clone(unique_asym_ids)
    best_global_perm_list = best_global_perm.tolist()

    per_asym_residue_index = {}
    for cur_asym_id in unique_asym_ids:
        asym_mask = (batch["asym_id"] == cur_asym_id).bool()
        per_asym_residue_index[int(cur_asym_id)] = batch["residue_index"][asym_mask]

    # for every unique structure with symmetric counterparts (works for 2a-2b structs, etc)
    unique_ent_ids, unique_ent_counts = th.unique(batch["entity_id"], return_counts=True)
    for ent_id, ent_count in zip(unique_ent_ids, unique_ent_counts):
        # see how many chains for the entity, if just 1, continue
        ent_mask = batch["entity_id"] == ent_id
        asym_ids = th.unique(batch["asym_id"][ent_mask])
        if len(asym_ids) == 1:
            continue
        # create placeholders for values and corresponding masks
        local_perm_idxs = []
        ent_mask = batch["entity_id"] == ent_id
        ent_res_idx = batch["residue_index"][ent_mask]
        min_res, max_res = ent_res_idx.amin().item(), ent_res_idx.amax().item()
        N = len(asym_ids)
        L = max_res - min_res + 1
        ph_pred_ca_pos = pred_ca_pos.new_zeros((N, L, 3, 3))
        ph_true_ca_pos = pred_ca_pos.new_zeros((N, L, 3, 3))
        ph_pred_ca_mask = pred_ca_mask.new_zeros((L,))
        ph_true_ca_mask = pred_ca_mask.new_zeros((L,))
        for i, ni in enumerate(asym_ids):
            local_perm_idxs.append(best_global_perm_list[ni.item()])
            ni_mask = batch["asym_id"] == ni
            ni_res_idx = batch["residue_index"][ni_mask]
            ni_res_idx = ni_res_idx - min_res
            ph_pred_ca_pos[ni_res_idx, ...] = pred_ca_pos[ni_mask, ...]
            # TODO: check indexing for true items
            ph_true_ca_pos[ni_res_idx, ...] = true_ca_poses[ni][ni_mask, ...]
            ph_pred_ca_mask[ni_res_idx] = pred_ca_mask[ni_mask]
            ph_true_ca_mask[ni_res_idx] = true_ca_masks[ni][ni_mask]

        # TODO: include all frames, not just the ones for the symmetric entity
        # (N, L, 3, 3) -> ((n=N f=L),)
        frames_pred = Frame.from_3_points(ph_pred_ca_pos.reshape(-1, 3, 3).unbind(dim=-2))
        frames_true = Frame.from_3_points(ph_true_ca_pos.reshape(-1, 3, 3).unbind(dim=-2))
        # (N, L) -> ((n=N f=L),)
        frames_mask = (ph_pred_ca_mask * ph_true_ca_mask).reshape(-1)
        # (N, L) -> ((n=N, p=L))
        points_mask = ph_pred_ca_mask * ph_true_ca_mask

        # cross-matrix and hungarian algorithm finds the best permutation
        # (n=N f=L), (n=N f=L), (n=N, p=L), (n=N, p=L) -> (ni=N, nj=N)
        x_mat = compute_x_fape(
            pred_frames=frames_pred,
            target_frames=frames_true,
            pred_points=ph_pred_ca_pos,
            target_points=ph_true_ca_pos,
            frames_mask=frames_mask,
            points_mask=points_mask,
        ).cpu().transpose(-1, -2).numpy()
        rows, cols = linear_sum_assignment(x_mat)

        # remap labels like: labels["ent_mask"] = ph_true_ca_pos[cols][ph_true_ca_mask[cols]]
        global_rows = local_perm_idxs
        global_cols = [local_perm_idxs[c] for c in cols]
        best_global_perm[global_rows] = best_global_perm[global_cols]

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
