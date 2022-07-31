import torch
from unicore.utils import one_hot
from unifold.data import residue_constants as rc
from .utils import (
    sigmoid_cross_entropy,
    softmax_cross_entropy,
    masked_mean,
)
from .geometry import (
    compute_aligned_error,
    compute_distogram,
    compute_lddt,
)


def experimentally_resolved_loss(
    logits: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
    loss_dict: dict = None,
    **kwargs,
) -> torch.Tensor:
    atom37_atom_exists = atom37_atom_exists.float()
    all_atom_mask = all_atom_mask.float()
    errors = sigmoid_cross_entropy(logits.float(), all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    dnorm = torch.sum(atom37_atom_exists, dim=(-1, -2)).unsqueeze(-1)
    
    loss = loss / (eps + dnorm)
    loss = torch.sum(loss, dim=-1)
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))
    
    loss_dict["experimentally_resolved"] = loss.data
    
    return loss


def plddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    num_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    loss_dict: dict = None,
    **kwargs,
) -> torch.Tensor:
    # TODO: bin utils

    ca_pos = rc.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :].float()
    all_atom_positions = all_atom_positions[..., ca_pos, :].float()
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)].float()  # keep dim

    lddt = compute_lddt(
        all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps
    ).detach()

    bin_index = torch.floor(lddt * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_ca_one_hot = one_hot(bin_index, num_classes=num_bins)

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)

    loss = masked_mean(all_atom_mask, errors, dim=-1, eps=eps)
    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    ca_lddt = masked_mean(all_atom_mask, lddt, dim=-1, eps=eps)

    loss_dict["ca_lddt_score"] = ca_lddt.data
    loss_dict["plddt_loss"] = loss.data
    return loss


def supervised_chi_loss(
    pred_angles_sin_cos: torch.Tensor,
    pred_unnormed_angles_sin_cos: torch.Tensor,
    true_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    loss_dict=None,
    **kwargs,
) -> torch.Tensor:
    # TODO: refactor this.
    pred_angles_sin_cos = pred_angles_sin_cos.float()
    pred_unnormed_angles_sin_cos = pred_unnormed_angles_sin_cos.float()
    true_angles_sin_cos = true_angles_sin_cos.unsqueeze(0).float()
    seq_mask = seq_mask.float()
    chi_mask = chi_mask.float()

    pred_angles = pred_angles_sin_cos[..., 3:, :]
    residue_type_one_hot = one_hot(
        aatype,
        rc.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "ijk, kl->ijl",
        residue_type_one_hot.type(pred_angles_sin_cos.dtype),
        pred_angles_sin_cos.new_tensor(rc.chi_pi_periodic),
    )
    true_chi = true_angles_sin_cos
    shifted_mask = (1.0 - 2.0 * chi_pi_periodic)[None, ..., None]
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum((true_chi_shifted - pred_angles) ** 2, dim=-1)
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    # permute nblock and batch dim
    sq_chi_error = sq_chi_error.transpose(0, 1)
    mask = chi_mask.unsqueeze(1)
    sq_chi_loss = masked_mean(mask, sq_chi_error, dim=(-1, -2, -3))
    loss_dict["chi_loss"] = sq_chi_loss.data
    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(torch.sum(pred_unnormed_angles_sin_cos**2, dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.transpose(0, 1)
    mask = seq_mask[..., None, :, None]
    angle_norm_loss = masked_mean(mask, norm_error, dim=(-1, -2, -3))

    loss_dict["angle_norm_loss"] = angle_norm_loss.data
    loss = loss + angle_norm_weight * angle_norm_loss

    return loss


def repr_norm_loss(
    msa_norm: torch.Tensor,
    pair_norm: torch.Tensor,
    msa_mask: torch.Tensor,
    pseudo_beta_mask: torch.Tensor,
    loss_dict=None,
    eps=1e-5,
    tolerance=0.0,
    **kwargs,
) -> torch.Tensor:
    def norm_loss(x):
        max_norm = x.shape[-1] ** 0.5
        norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
        error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
        return error

    pair_norm_error = norm_loss(pair_norm.float())
    msa_norm_error = norm_loss(msa_norm.float())
    pair_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
    
    pair_norm_loss = masked_mean(pair_mask.float(), pair_norm_error, dim=(-1, -2))
    msa_norm_loss = masked_mean(msa_mask.float(), msa_norm_error, dim=(-1, -2))
    
    loss = pair_norm_loss + msa_norm_loss

    loss_dict["pair_norm"] = pair_norm_loss.data
    loss_dict["msa_norm"] = msa_norm_loss.data
    
    return loss


def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    num_bins=64,
    eps=1e-6,
    loss_dict=None,
    **kwargs,
):
    distogram, mask = compute_distogram(
        pseudo_beta, pseudo_beta_mask, min_bin, max_bin, num_bins)

    errors = softmax_cross_entropy(logits, one_hot(distogram, num_bins))

    loss = masked_mean(mask, errors, dim=(-1, -2), eps=eps)
    
    loss_dict["distogram"] = loss.data
    
    return loss


def pae_loss(
    logits,
    pred_frame_tensor,
    true_frame_tensor,
    frame_mask,
    resolution,
    max_bin=31,
    num_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    loss_dict=None,
    **kwargs,
):
    aligned_error_val, aligned_error_bin, mask = compute_aligned_error(
        pred_frame_tensor,
        true_frame_tensor,
        frame_mask,
        max_bin,
        num_bins,
    )

    errors = softmax_cross_entropy(logits.float(), one_hot(aligned_error_bin, num_bins))

    loss = masked_mean(mask, errors, dim=(-1, -2), eps=eps)

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    loss_dict["pae_loss"] = loss.data
    loss_dict["aligned_error"] = aligned_error_val.data
    
    return loss


def masked_msa_loss(logits, true_msa, bert_mask, eps=1e-8, loss_dict=None, **kwargs):
    bert_mask = bert_mask.float()
    errors = softmax_cross_entropy(
        logits.float(), one_hot(true_msa.long(), num_classes=logits.shape[-1])
    )

    loss = masked_mean(bert_mask, errors, dim=(-1, -2), eps=eps)
    loss_dict["masked_msa"] = loss.data
    return loss


def get_asym_mask(asym_id):
    """get the mask for each asym_id. [*, NR] -> [*, NC, NR]"""
    # this func presumes that valid asym_id ranges [1, NC] and is dense.
    asym_type = torch.arange(1, torch.amax(asym_id) + 1, device=asym_id.device)  # [NC]
    return (asym_id[..., None, :] == asym_type[:, None]).float()


def chain_centre_mass_loss(
    pred_atom_positions: torch.Tensor,
    true_atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    eps: float = 1e-10,
    loss_dict=None,
    **kwargs,
) -> torch.Tensor:

    ca_pos = rc.atom_order["CA"]
    pred_atom_positions = pred_atom_positions[..., ca_pos, :].float()  # [B, NR, 3]
    true_atom_positions = true_atom_positions[..., ca_pos, :].float()  # [B, NR, 3]
    atom_mask = atom_mask[..., ca_pos].bool()  # [B, NR]
    assert len(pred_atom_positions.shape) == 3

    asym_mask = get_asym_mask(asym_id) * atom_mask[..., None, :]  # [B, NC, NR]
    asym_exists = torch.any(asym_mask, dim=-1).float()  # [B, NC]

    def get_asym_centres(pos):
        pos = pos[..., None, :, :] * asym_mask[..., :, :, None]  # [B, NC, NR, 3]
        return torch.sum(pos, dim=-2) / (torch.sum(asym_mask, dim=-1)[..., None] + eps)

    pred_centres = get_asym_centres(pred_atom_positions)  # [B, NC, 3]
    true_centres = get_asym_centres(true_atom_positions)  # [B, NC, 3]

    def get_dist(p1: torch.Tensor, p2: torch.Tensor):
        return torch.sqrt(
            (p1[..., :, None, :] - p2[..., None, :, :]).square().sum(-1) + eps
        )

    pred_centres2 = pred_centres
    true_centres2 = true_centres
    pred_dists = get_dist(pred_centres, pred_centres2)  # [B, NC, NC]
    true_dists = get_dist(true_centres, true_centres2)  # [B, NC, NC]
    losses = (pred_dists - true_dists + 4).clamp(max=0).square() * 0.0025
    loss_mask = asym_exists[..., :, None] * asym_exists[..., None, :]  # [B, NC, NC]

    loss = masked_mean(loss_mask, losses, dim=(-1, -2))
    loss_dict["chain_centre_loss"] = loss.data

    return loss
