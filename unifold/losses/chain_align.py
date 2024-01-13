import torch as th
from unifold.modules.frame import Frame
from scipy.optimize import linear_sum_assignment
from .geometry import kabsch_rmsd, get_optimal_transform, compute_rmsd
from typing import List, Tuple, Dict, Optional


def compute_xx_fape(
    pred_frames: Frame,
    target_frames: Frame,
    pred_points: th.Tensor,
    target_points: th.Tensor,
    frames_mask: Optional[th.Tensor] = None,
    points_mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    """ FAPE cross-matrix from frames to the cross-matrix of points,
    used to find a permutation which gives the optimal loss for a symmetric structure.

    Notation for use with n chains of length p, under f=k frames:
    - n: number of protein chains
    - p: number of points (length of chain)
    - d: dimension of points = 3
    - f: arbitrary number of frames
    - ': frames dimension
    (..., n', f, 1, 1) @ (..., 1, 1, n, p, d) -> (..., n', f, n, p, d)
    (..., ni', f, ni, p, d) - (..., nj', f, nj, p, d) -> (..., ni', nj', ni, nj)

    Args:
        pred_frames: (..., n(i'), f)
        target_frames: (..., n(j'), f)
        pred_points: (..., n(i), p, d)
        target_points: (..., n(j), p, d)
        frames_mask: (..., n', f) float tensor
        points_mask: (..., n, p) float tensor

    Returns:
        (..., n(i'), n(j'), n(i), n(j)) th.Tensor
    """
    # define masks for reduction, mask is (ni', nj', f, ni, nj, p)
    mask = 1.
    if frames_mask is not None:
        mask = mask * (
            frames_mask[..., :, None, :, None, None, None] + frames_mask[..., None, :, :, None, None, None]
        ).bool().float()
    if points_mask is not None:
        mask = mask * (
            points_mask[..., None, None, None, :, None, :] + points_mask[..., None, None, None, None, :, :]
        ).bool().float()

    # (..., n', f) · (..., n, p, d) -> (..., n', f, n, p, d)
    local_pred_pos = pred_frames[..., None, None].invert().apply(
        pred_points[..., None, None, :, :, :].float(),
    )
    # (..., n', f) · (..., n, p) -> (..., n', f, n, p)
    local_target_pos = target_frames[..., None, None].invert().apply(
        target_points[..., None, None, :, :, :].float(),
    )
    # chunk in ni, nj to avoid memory errors
    n_, n = local_pred_pos.shape[-5], local_pred_pos.shape[-3]
    xx_fape = local_pred_pos.new_zeros(*local_pred_pos.shape[:-5], n_, n_, n, n)
    for i_ in range(n_):
        for j_ in range(n_):
            # (..., ni, f, ni, p, d) - (..., nj, f, nj, p, d) -> (..., ni, nj, f, ni', nj', p)
            d_pt2 = (
                local_pred_pos[..., i_:i_ + 1, None, :, :, None, :, :] -
                local_target_pos[..., None, j_:j_ + 1, :, None, :, :, :]
            ).square().sum(-1)
            d_pt = d_pt2.add_(1e-5).sqrt()
            # (..., ni, nj, f, ni', nj', p) -> (..., ni, nj, ni', nj')
            if frames_mask is not None or points_mask is not None:
                mask_ = mask[..., i_:i_+1, j_:j_+1, :, :, :, :]
                x_fape_ij = (d_pt * mask_).sum(dim=(-1, -4)) / mask_.sum(dim=(-1, -4))
            else:
                x_fape_ij = d_pt.mean(dim=(-1, -4))
            xx_fape[..., i_, j_, :, :] = x_fape_ij
            # save memory
            del d_pt2, d_pt, x_fape_ij

    return xx_fape


def multi_chain_perm_align(out: Dict, batch: Dict, labels: List[Dict]) -> Dict:
    """ Permutes labels so that a structural loss wrt preds is minimized.
    Framed as a linear assignment problem, loss is sum of "individual" losses
    and the permutation is found by the Hungarian algorithm on the cross matrix.

    WARNING! All keys in `out` have no batch size
    """
    assert isinstance(labels, list)
    # get all unique chains - remove padding tokens with no labels
    unique_asym_ids = th.unique(batch["asym_id"])
    if len(unique_asym_ids) == len(labels) + 1:
        unique_asym_ids = th.tensor(list((set(unique_asym_ids.tolist()) - {0}))).to(batch["asym_id"])
    assert len(unique_asym_ids) == len(labels)
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
    pred_frames_mask = batch["atom14_gt_exists"][..., [0, 1, 2]].float().prod(dim=-1)

    # rename symmetric chains, (works for abab, abb, abbcc, ...)
    unique_ent_ids = th.unique(batch["entity_id"])
    for ent_id in unique_ent_ids:
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
        ph_frames_pred = Frame.identity((n, l), device=pred_frames.device, dtype=pred_frames.dtype)
        ph_frames_true = Frame.identity((n, l), device=pred_frames.device, dtype=pred_frames.dtype)
        frames_mask = points_mask = true_frames_mask.new_zeros((n, l,))
        # fill placeholders with points and masks
        for i, ni in enumerate(asym_ids):
            local_perm_idxs.append(best_global_perm_list[ni.item()])
            asym_mask = batch["asym_id"] == ni
            asym_res_idx = batch["residue_index"][asym_mask]
            ph_frames_pred[i, asym_res_idx - min_res] = pred_frames[asym_mask].clone()
            ph_frames_true[i, asym_res_idx - min_res] = true_frames[asym_mask].clone()
            points_mask[i, asym_res_idx - min_res] = pred_frames_mask[asym_mask] * true_frames_mask[asym_mask]
            frames_mask[i, asym_res_idx - min_res] = pred_frames_mask[asym_mask] * true_frames_mask[asym_mask]

        # cross-matrix and hungarian algorithm finds the best permutation
        # (n=N, f=L), (n=N, f=L), (n=N, p=L), (n=N, p=L) -> (ni'=N, nj'=N, ni=N, nj=N)
        x_mat = compute_xx_fape(
            pred_frames=ph_frames_pred,
            target_frames=ph_frames_true,
            pred_points=ph_frames_pred._t,
            target_points=ph_frames_true._t,
            frames_mask=frames_mask,
            points_mask=points_mask,
        ).detach().cpu()
        # (ni'=N, nj'=N, ni=N, nj=N) -> (N, N)
        x_mat_frames = x_mat.sum(dim=(-1, -2)) / (x_mat.shape[-1] * x_mat.shape[-2])
        x_mat_points = x_mat.sum(dim=(-3, -4)) / (x_mat.shape[-3] * x_mat.shape[-4])
        rows, cols = linear_sum_assignment((x_mat_frames + x_mat_points).numpy())

        # remap labels like: labels["ent_mask"] = ph_true_ca_pos[cols][ph_true_ca_mask[cols]]
        global_rows = local_perm_idxs
        global_cols = [local_perm_idxs[c] for c in cols]
        best_global_perm[global_rows] = best_global_perm[global_cols]

    # (N,) -> (2, N) and match indices of labels list
    ij_label_align = th.stack((best_global_curr, best_global_perm), dim=0).long().T
    ij_label_align = (ij_label_align - ij_label_align.amin()).tolist()
    best_labels = merge_labels(
        batch=batch,
        per_asym_residue_index=per_asym_residue_index,
        labels=labels,
        align=ij_label_align
    )
    return best_labels


def multi_chain_perm_align_outdated(out, batch, labels, shuffle_times=2):
    assert isinstance(labels, list)
    ca_idx = rc.atom_order["CA"]
    pred_ca_pos = out["final_atom_positions"][..., ca_idx, :].float()  # [bsz, nres, 3]
    pred_ca_mask = out["final_atom_mask"][..., ca_idx].float()  # [bsz, nres]
    true_ca_poses = [
        l["all_atom_positions"][..., ca_idx, :].float() for l in labels
    ]  # list([nres, 3])
    true_ca_masks = [
        l["all_atom_mask"][..., ca_idx].float() for l in labels
    ]  # list([nres,])

    unique_asym_ids = th.unique(batch["asym_id"])

    per_asym_residue_index = {}
    for cur_asym_id in unique_asym_ids:
        asym_mask = (batch["asym_id"] == cur_asym_id).bool()
        per_asym_residue_index[int(cur_asym_id)] = batch["residue_index"][asym_mask]

    anchor_gt_asym, anchor_pred_asym = get_anchor_candidates(
        batch, per_asym_residue_index, true_ca_masks
    )
    anchor_gt_idx = int(anchor_gt_asym) - 1

    best_rmsd = 1e9
    best_labels = None

    unique_entity_ids = th.unique(batch["entity_id"])
    entity_2_asym_list = {}
    for cur_ent_id in unique_entity_ids:
        ent_mask = batch["entity_id"] == cur_ent_id
        cur_asym_id = th.unique(batch["asym_id"][ent_mask])
        entity_2_asym_list[int(cur_ent_id)] = cur_asym_id

    for cur_asym_id in anchor_pred_asym:
        asym_mask = (batch["asym_id"] == cur_asym_id).bool()
        anchor_residue_idx = per_asym_residue_index[int(cur_asym_id)]
        anchor_true_pos = true_ca_poses[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_pos = pred_ca_pos[asym_mask]
        anchor_true_mask = true_ca_masks[anchor_gt_idx][anchor_residue_idx]
        anchor_pred_mask = pred_ca_mask[asym_mask]

        r, x = get_optimal_transform(
            anchor_true_pos,
            anchor_pred_pos,
            (anchor_true_mask * anchor_pred_mask).bool(),
        )

        aligned_true_ca_poses = [ca @ r + x for ca in true_ca_poses]  # apply transforms
        for _ in range(shuffle_times):
            shuffle_idx = th.randperm(
                unique_asym_ids.shape[0], device=unique_asym_ids.device
            )
            shuffled_asym_ids = unique_asym_ids[shuffle_idx]
            align = greedy_align(
                batch,
                per_asym_residue_index,
                shuffled_asym_ids,
                entity_2_asym_list,
                pred_ca_pos,
                pred_ca_mask,
                aligned_true_ca_poses,
                true_ca_masks,
            )

            merged_labels = merge_labels(
                batch,
                per_asym_residue_index,
                labels,
                align,
            )

            rmsd = kabsch_rmsd(
                merged_labels["all_atom_positions"][..., ca_idx, :] @ r + x,
                pred_ca_pos,
                (pred_ca_mask * merged_labels["all_atom_mask"][..., ca_idx]).bool(),
            )

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_labels = merged_labels
    return best_labels


def get_anchor_candidates(batch, per_asym_residue_index, true_masks):
    def find_by_num_sym(min_num_sym):
        best_len = -1
        best_gt_asym = None
        asym_ids = th.unique(batch["asym_id"][batch["num_sym"] == min_num_sym])
        for cur_asym_id in asym_ids:
            assert cur_asym_id > 0
            cur_residue_index = per_asym_residue_index[int(cur_asym_id)]
            j = int(cur_asym_id - 1)
            cur_true_mask = true_masks[j][cur_residue_index]
            cur_len = cur_true_mask.sum()
            if cur_len > best_len:
                best_len = cur_len
                best_gt_asym = cur_asym_id
        return best_gt_asym, best_len

    sorted_num_sym = batch["num_sym"][batch["num_sym"] > 0].sort()[0]
    best_gt_asym = None
    best_len = -1
    for cur_num_sym in sorted_num_sym:
        if cur_num_sym <= 0:
            continue
        cur_gt_sym, cur_len = find_by_num_sym(cur_num_sym)
        if cur_len > best_len:
            best_len = cur_len
            best_gt_asym = cur_gt_sym
        if best_len >= 3:
            break
    best_entity = batch["entity_id"][batch["asym_id"] == best_gt_asym][0]
    best_pred_asym = th.unique(batch["asym_id"][batch["entity_id"] == best_entity])
    return best_gt_asym, best_pred_asym


def greedy_align(
    batch,
    per_asym_residue_index,
    unique_asym_ids,
    entity_2_asym_list,
    pred_ca_pos,
    pred_ca_mask,
    true_ca_poses,
    true_ca_masks,
):
    used = [False for _ in range(len(true_ca_poses))]
    align = []
    for cur_asym_id in unique_asym_ids:
        # skip padding
        if cur_asym_id == 0:
            continue
        i = int(cur_asym_id - 1)
        asym_mask = batch["asym_id"] == cur_asym_id
        num_sym = batch["num_sym"][asym_mask][0]
        # don't need to align
        if (num_sym) == 1:
            align.append((i, i))
            assert used[i] == False
            used[i] = True
            continue
        cur_entity_ids = batch["entity_id"][asym_mask][0]
        best_rmsd = 1e10
        best_idx = None
        cur_asym_list = entity_2_asym_list[int(cur_entity_ids)]
        cur_residue_index = per_asym_residue_index[int(cur_asym_id)]
        cur_pred_pos = pred_ca_pos[asym_mask]
        cur_pred_mask = pred_ca_mask[asym_mask]
        for next_asym_id in cur_asym_list:
            if next_asym_id == 0:
                continue
            j = int(next_asym_id - 1)
            if not used[j]:  # posesible candidate
                cropped_pos = true_ca_poses[j][cur_residue_index]
                mask = true_ca_masks[j][cur_residue_index]
                rmsd = compute_rmsd(
                    cropped_pos, cur_pred_pos, (cur_pred_mask * mask).bool()
                )
                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_idx = j

        assert best_idx is not None
        used[best_idx] = True
        align.append((i, best_idx))

    return align

def merge_labels(
    batch: Dict,
    per_asym_residue_index: Dict,
    labels: List[Dict],
    align: List[Tuple]
) -> Dict:
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
