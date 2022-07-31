import torch
from unifold.data import residue_constants as rc
from .geometry import kabsch_rmsd, get_optimal_transform, compute_rmsd


def multi_chain_perm_align(out, batch, labels, shuffle_times=2):
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

    unique_asym_ids = torch.unique(batch["asym_id"])

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

    unique_entity_ids = torch.unique(batch["entity_id"])
    entity_2_asym_list = {}
    for cur_ent_id in unique_entity_ids:
        ent_mask = batch["entity_id"] == cur_ent_id
        cur_asym_id = torch.unique(batch["asym_id"][ent_mask])
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
            shuffle_idx = torch.randperm(
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
        asym_ids = torch.unique(batch["asym_id"][batch["num_sym"] == min_num_sym])
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
    best_pred_asym = torch.unique(batch["asym_id"][batch["entity_id"] == best_entity])
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


def merge_labels(batch, per_asym_residue_index, labels, align):
    """
    batch:
    labels: list of label dicts, each with shape [nk, *]
    align: list of int, such as [2, None, 0, 1], each entry specify the corresponding label of the asym.
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
        new_v = torch.concat(cur_out, dim=0)
        merged_nres = new_v.shape[0]
        assert (
            merged_nres <= num_res
        ), f"bad merged num res: {merged_nres} > {num_res}. something is wrong."
        if merged_nres < num_res:  # must pad
            pad_dim = new_v.shape[1:]
            pad_v = new_v.new_zeros((num_res - merged_nres, *pad_dim))
            new_v = torch.concat((new_v, pad_v), dim=0)
        outs[k] = new_v
    return outs
