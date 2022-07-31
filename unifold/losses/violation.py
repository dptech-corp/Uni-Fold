import torch
from typing import Dict

from unicore.utils import one_hot
from .utils import masked_mean
from unifold.data import residue_constants as rc


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    residue_index: torch.Tensor,
    aatype: torch.Tensor,
    asym_id: torch.Tensor,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    pred_atom_positions = pred_atom_positions.float()
    pred_atom_mask = pred_atom_mask.float()
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    # mask gap between different chains
    if asym_id is not None:
        has_no_gap_mask &= asym_id[..., :-1] == asym_id[..., 1:]
    has_no_gap_mask = has_no_gap_mask.float()

    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )
    next_is_proline = (aatype[..., 1:] == rc.resname_to_idx["PRO"]).float()
    gt_length = (1.0 - next_is_proline) * rc.between_res_bond_length_c_n[
        0
    ] + next_is_proline * rc.between_res_bond_length_c_n[1]
    gt_stddev = (1.0 - next_is_proline) * rc.between_res_bond_length_stddev_c_n[
        0
    ] + next_is_proline * rc.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = (
        mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)).float()
    )

    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_ca_c_n[0]
    gt_stddev = rc.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_c_n_ca[0]
    gt_stddev = rc.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    asym_id: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
) -> Dict[str, torch.Tensor]:
    atom14_pred_positions = atom14_pred_positions.float()
    fp_type = atom14_pred_positions.dtype

    dists = torch.sqrt(
        1e-10
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)
    dists_mask = (
        dists_mask
        * (
            residue_index[..., :, None, None, None]
            <= residue_index[..., None, :, None, None]
        ).float()
    )
    diagonal = (
        residue_index[..., :, None, None, None]
        == residue_index[..., None, :, None, None]
    )
    if asym_id is not None:
        in_one_chain = (
            asym_id[..., :, None, None, None] == asym_id[..., None, :, None, None]
        )
        diagonal = diagonal & in_one_chain
    dists_mask = dists_mask * (1.0 - (diagonal).float())
    c_one_hot = one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residue_index[..., :, None] + 1) == residue_index[..., None, :]
    if asym_id is not None:
        neighbour_mask &= asym_id[..., :, None] == asym_id[..., None, :]
    neighbour_mask = neighbour_mask[..., None, None].float()
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(
        -1
    )
    cys_sg_one_hot = one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None].float()
        + atom14_atom_radius[..., None, :, None, :].float()
    )
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )
    clash_mask = (
        dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard)).float()
    )

    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )
    per_atom_clash_count = torch.sum(clash_mask, dim=(-4, -2)) + torch.sum(
        clash_mask, dim=(-3, -1)
    )

    return {
        "mean_loss": mean_loss,
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_clash_mask": per_atom_clash_mask,
        "per_atom_clash_count": per_atom_clash_count,
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
) -> Dict[str, torch.Tensor]:
    atom14_pred_positions = atom14_pred_positions.float()
    atom14_atom_exists = atom14_atom_exists.float()
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    dists = torch.sqrt(
        1e-10
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    violations = (
        dists_masks
        * (
            (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
        ).float()
    )

    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, dim=-1)[0]
    )

    per_atom_clash_count = torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1)
    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
        "per_atom_clash_count": per_atom_clash_count,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    atom14_pred_positions = atom14_pred_positions.float()
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        asym_id=batch["asym_id"] if "asym_id" in batch else None,
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    atomtype_radius = [rc.van_der_waals_radius[name[0]] for name in rc.atom_types]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        batch["atom14_atom_exists"] * atomtype_radius[batch["residx_atom14_to_atom37"]]
    )
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        asym_id=batch["asym_id"] if "asym_id" in batch else None,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    restype_atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(between_residue_clashes["per_atom_clash_mask"], dim=-1)[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],
            "angles_ca_c_n_loss_mean": connection_violations["ca_c_n_loss_mean"],
            "angles_c_n_ca_loss_mean": connection_violations["c_n_ca_loss_mean"],
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],
            "clashes_mean_loss": between_residue_clashes["mean_loss"],
            "clashes_per_atom_loss_sum": between_residue_clashes["per_atom_loss_sum"],
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],
            "clashes_per_atom_clash_count": between_residue_clashes[
                "per_atom_clash_count"
            ],
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations["per_atom_loss_sum"],
            "per_atom_violations": residue_violations["per_atom_violations"],
            "per_atom_clash_count": residue_violations["per_atom_clash_count"],
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,
    }


def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    residue_index: torch.Tensor,
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    pred_atom_positions = pred_atom_positions.float()

    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (
        (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ).float()
    ca_ca_distance = torch.sqrt(
        eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1)
    )
    violations = (ca_ca_distance - rc.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    atom14_pred_positions = atom14_pred_positions.float()
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"]["connections_per_residue_violation_mask"],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(violations["within_residues"]["per_atom_violations"], dim=-1)[
            0
        ],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def violation_loss(
    violations: Dict[str, torch.Tensor],
    eps=1e-6,
    loss_dict=None,
    bond_angle_loss_weight: float = 0.3,
    **kwargs,
) -> torch.Tensor:
    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"],
        dim=(-1, -2),
    )
    cnt_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_clash_count"]
        + violations["within_residues"]["per_atom_clash_count"],
        dim=(-1, -2),
    )
    l_clash = l_clash / (eps + cnt_clash)
    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + bond_angle_loss_weight
        * violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + bond_angle_loss_weight
        * violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )
    loss_dict["violation"] = loss.data
    return loss
