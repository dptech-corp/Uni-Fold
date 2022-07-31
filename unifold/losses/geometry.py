import torch
from unifold.data import residue_constants as rc
from unifold.modules.frame import Frame
from typing import Dict, Tuple
from unicore.utils import (
    permute_final_dims,
    set_jit_fusion_options,
)

set_jit_fusion_options()


def compute_lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))

    return score


def compute_fape(
    pred_frames: Frame,
    target_frames: Frame,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    pair_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :].float(),
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :].float(),
    )

    frames_mask = frames_mask.float()
    positions_mask = positions_mask.float()
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error *= frames_mask[..., None]
    normed_error *= positions_mask[..., None, :]
    if pair_mask is not None:
        normed_error *= pair_mask

    if pair_mask is not None:
        mask = frames_mask.unsqueeze(-1) * positions_mask.unsqueeze(-2)
        mask *= pair_mask
        norm_factor = mask.sum(dim=(-1, -2))
    else:
        norm_factor = torch.sum(frames_mask, dim=-1) * torch.sum(positions_mask, dim=-1)

    normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)

    return normed_error


def compute_distogram(
    positions,
    mask,
    min_bin=2.3125,
    max_bin=21.6875,
    num_bins=64,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        num_bins - 1,
        device=positions.device,
    )
    boundaries = boundaries**2
    positions = positions.float()

    dists = torch.sum(
        (positions[..., None, :] - positions[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    ).detach()

    mask = mask.float()
    pair_mask = mask[..., None] * mask[..., None, :]

    return torch.sum(dists > boundaries, dim=-1), pair_mask


def compute_aligned_error(
    pred_affine_tensor: torch.Tensor,
    true_affine_tensor: torch.Tensor,
    affine_mask: torch.Tensor,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-10,
):
    pred_affine = Frame.from_tensor_4x4(pred_affine_tensor.float())
    true_affine = Frame.from_tensor_4x4(true_affine_tensor.float())

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum(
        (_points(pred_affine) - _points(true_affine)) ** 2, dim=-1
    ).detach()

    boundaries = torch.linspace(
        0, max_bin, steps=(num_bins - 1), device=pred_affine_tensor.device
    )
    boundaries = boundaries**2

    affine_mask = affine_mask.float()
    pair_mask = affine_mask[..., None] * affine_mask[..., None, :]

    return (
        torch.sqrt(sq_diff + eps),
        torch.sum(sq_diff[..., None] > boundaries, dim=-1),
        pair_mask,
    )


def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    atom14_pred_positions = atom14_pred_positions.float()
    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"].float()
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"].float()
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"].float()
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"].float()
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = atom14_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * atom14_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
        "atom14_alt_gt_exists"
    ].float()

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


@torch.jit.script
def compute_rmsd(
    true_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: torch.Tensor = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    # shape check
    sq_diff = torch.square(true_atom_pos - pred_atom_pos).sum(dim=-1, keepdim=False)
    if atom_mask is not None:
        sq_diff = sq_diff[atom_mask]
    msd = torch.mean(sq_diff)
    msd = torch.nan_to_num(msd, nan=1e8)
    return torch.sqrt(msd + eps)


@torch.jit.script
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = P.transpose(-1, -2) @ Q

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, _, W = torch.linalg.svd(C)
    d = (torch.linalg.det(V) * torch.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = V @ W
    return U


@torch.jit.script
def get_optimal_transform(
    src_atoms: torch.Tensor,
    tgt_atoms: torch.Tensor,
    mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert src_atoms.shape == tgt_atoms.shape, (src_atoms.shape, tgt_atoms.shape)
    assert src_atoms.shape[-1] == 3
    if mask is not None:
        assert mask.dtype == torch.bool
        assert mask.shape[-1] == src_atoms.shape[-2]
        if mask.sum() == 0:
            src_atoms = torch.zeros((1, 3), device=src_atoms.device).float()
            tgt_atoms = src_atoms
        else:
            src_atoms = src_atoms[mask, :]
            tgt_atoms = tgt_atoms[mask, :]
    src_center = src_atoms.mean(-2, keepdim=True)
    tgt_center = tgt_atoms.mean(-2, keepdim=True)
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


@torch.jit.script
def kabsch_rmsd(
    true_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    atom_mask: torch.Tensor,
):
    r, x = get_optimal_transform(
        true_atom_pos,
        pred_atom_pos,
        atom_mask,
    )
    aligned_true_atom_pos = true_atom_pos @ r + x
    return compute_rmsd(aligned_true_atom_pos, pred_atom_pos, atom_mask)


def get_optimal_transform_v2(
    p: torch.Tensor,
    q: torch.Tensor,
    m: torch.Tensor,
    num_dim: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    calculate u such that p @ u ~ q.
    p, q has shape  [*, *dim, 3]
    mask has shape  [*, *dim]
    ret has shape     [*, *dim, 3, 3]
    """
    rd = p.shape[-1]
    batch_shape = p.shape[: -(num_dim + 1)]
    m = m.reshape(*batch_shape, -1, 1)

    def process_input(p):
        p = p.reshape(*batch_shape, -1, rd)
        p = p * m
        cp = p.sum(dim=-2, keepdim=True) / (
            eps + m.sum(dim=-2, keepdim=True)
        )  # [*, 1, 3]
        p = p - cp
        p = p * m
        return p, cp

    p_rc, cp = process_input(p)  # rc for remove center
    q_rc, cq = process_input(q)
    c = p_rc.transpose(-1, -2) @ q_rc  # [*, 3, 3]
    v, _, w = torch.linalg.svd(c)  # [*, 3, 3]
    d = (torch.linalg.det(v) * torch.linalg.det(w) >= 0.0).type(
        v.dtype
    ) * 2.0 - 1.0  # [*]
    v[..., -1] = v[..., -1] * d[..., None]  # [*, 3]
    u = v @ w  # [*, 3, 3]
    u = u.reshape(*batch_shape, *((1,) * num_dim), rd, rd)
    cp = cp.reshape(*batch_shape, *((1,) * num_dim), rd)
    cq = cq.reshape(*batch_shape, *((1,) * num_dim), rd)
    x = cq[..., None, :] - cp[..., None, :] @ u
    return u, x.squeeze(-2)


def apply_optimal_transform_v2(x, r, t):
    return (x.unsqueeze(-2) @ r + t.unsqueeze(-2)).squeeze(-2)


def compute_rmsd_v2(p1, p2, mask, dim=-1, eps=1e-8):
    sd = torch.square(p1 - p2).sum(dim=-1, keepdim=False)
    msd = torch.sum(sd * mask, dim=dim) / (eps + torch.sum(mask, dim=dim))
    return torch.sqrt(msd + eps)


def kabsch_rmsd_v2(
    true_atom_pos: torch.Tensor,
    pred_atom_pos: torch.Tensor,
    true_atom_mask: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    num_dim: int = 1,
):
    r, t = get_optimal_transform_v2(
        true_atom_pos, pred_atom_pos, true_atom_mask * pred_atom_mask, num_dim
    )
    aligned_true_atom_pos = apply_optimal_transform_v2(true_atom_pos, r, t)
    reduce_dim = tuple(-k - 1 for k in range(num_dim))
    return compute_rmsd_v2(
        aligned_true_atom_pos,
        pred_atom_pos,
        true_atom_mask * pred_atom_mask,
        dim=reduce_dim,
    )


def compute_metric(features, out, eps=1e-6):
    ca_idx = rc.atom_order["CA"]
    true_ca: torch.Tensor = features["all_atom_positions"][..., ca_idx, :]
    pred_ca = out["final_atom_positions"][..., ca_idx, :]
    mask: torch.Tensor = features["all_atom_mask"] * out["final_atom_mask"]
    mask = mask[..., ca_idx]
    r, t = get_optimal_transform_v2(pred_ca, true_ca, mask, num_dim=1)
    aln_pred_ca: torch.Tensor = apply_optimal_transform_v2(pred_ca, r, t)
    sd = (aln_pred_ca - true_ca).square().sum(dim=-1)  # [*, n]

    nres = mask.sum(dim=-1, keepdim=True)  # [*, 1]
    d0 = 1.24 * torch.clamp(nres, min=15) ** (1.0 / 3.0) - 1.8
    tm_term = 1.0 / (1.0 + (sd / d0) ** 2)

    msd = torch.sum(sd * mask, dim=-1) / (eps + torch.sum(mask, dim=-1))
    rmsd = torch.sqrt(msd + eps)
    tm = torch.sum(tm_term * mask, dim=-1) / (eps + torch.sum(mask, dim=-1))

    return {
        "rmsd": rmsd.data,
        "tm_score": tm.data,
    }
