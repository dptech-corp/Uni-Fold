import ml_collections
import torch
from typing import Dict

from .geometry import compute_fape
from unifold.modules.frame import Frame


def backbone_loss(
    true_frame_tensor: torch.Tensor,
    frame_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: torch.Tensor,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    clamp_distance_between_chains: float = 30.0,
    loss_unit_distance_between_chains: float = 20.0,
    intra_chain_mask: torch.Tensor = None,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_aff = Frame.from_tensor_4x4(traj)
    gt_aff = Frame.from_tensor_4x4(true_frame_tensor)

    use_clamped_fape = int(use_clamped_fape) == 1
    if intra_chain_mask is None:
        return compute_fape(
            pred_aff,
            gt_aff[None],
            frame_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            frame_mask[None],
            pair_mask=None,
            l1_clamp_distance=clamp_distance if use_clamped_fape else None,
            length_scale=loss_unit_distance,
            eps=eps,
        )
    else:
        intra_chain_mask = intra_chain_mask.float().unsqueeze(0)
        intra_chain_bb_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            frame_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            frame_mask[None],
            pair_mask=intra_chain_mask,
            l1_clamp_distance=clamp_distance if use_clamped_fape else None,
            length_scale=loss_unit_distance,
            eps=eps,
        )
        interface_fape = compute_fape(
            pred_aff,
            gt_aff[None],
            frame_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            frame_mask[None],
            pair_mask=1.0 - intra_chain_mask,
            l1_clamp_distance=clamp_distance_between_chains
            if use_clamped_fape
            else None,
            length_scale=loss_unit_distance_between_chains,
            eps=eps,
        )
        return intra_chain_bb_loss, interface_fape


def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames

    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Frame.from_tensor_4x4(sidechain_frames)

    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Frame.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(*batch_dims, -1, 3)
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        pair_mask=None,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )
    return fape


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
    loss_dict: dict,
) -> torch.Tensor:
    for key in out["sm"]:
        out["sm"][key] = out["sm"][key].float()
    if "asym_id" in batch:
        intra_chain_mask = (
            batch["asym_id"][..., :, None] == batch["asym_id"][..., None, :]
        )
        bb_loss, interface_loss = backbone_loss(
            traj=out["sm"]["frames"],
            **{**batch, **config.backbone},
            intra_chain_mask=intra_chain_mask,
        )
        # only show the loss on last layer
        loss_dict["fape"] = bb_loss[-1].data
        loss_dict["interface_fape"] = interface_loss[-1].data
        bb_loss = torch.mean(bb_loss, dim=0) + torch.mean(interface_loss, dim=0)
    else:
        bb_loss = backbone_loss(
            traj=out["sm"]["frames"],
            **{**batch, **config.backbone},
            intra_chain_mask=None,
        )
        # only show the loss on last layer
        loss_dict["fape"] = bb_loss[-1].data
        bb_loss = torch.mean(bb_loss, dim=0)

    sc_loss = sidechain_loss(
        out["sm"]["sidechain_frames"],
        out["sm"]["positions"],
        **{**batch, **config.sidechain},
    )
    loss_dict["sc_fape"] = sc_loss.data
    loss = config.backbone.weight * bb_loss + config.sidechain.weight * sc_loss

    return loss
