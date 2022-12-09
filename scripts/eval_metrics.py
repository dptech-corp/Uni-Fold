from Bio.PDB.PDBParser import PDBParser
import numpy as np
from itertools import permutations


def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x


def get_pdb(filename):
    parser = PDBParser()
    data = parser.get_structure("tmp", filename)
    chains = []

    for chain in data.get_chains():
        residues = {}
        for res in chain.get_residues():
            res_id = res.get_id()[1]
            d = {}
            d["resname"] = res.resname
            atoms = {}
            for a in res.get_atoms():
                atoms[a.name] = a.get_coord()
            d["atoms"] = atoms
            residues[res_id] = d
        chains.append(residues)
    return chains


def recursive_perm(intervals, cur_idx=0):
    if cur_idx >= len(intervals):
        return [()]
    ret = []
    for cur_perm in permutations(intervals[cur_idx]):
        for right in recursive_perm(intervals, cur_idx + 1):
            ret.append(cur_perm + right)
    return ret


def generate_perm(entity):
    intervals = []
    pre_eid = -1
    for i, eid in enumerate(entity):
        if eid != pre_eid:
            intervals.append([])
        intervals[-1].append(i)
        pre_eid = eid
    return recursive_perm(intervals)


def get_coords(gt, pred):
    gt_coords = []
    pred_coords = []
    for i in range(len(gt)):
        for r in gt[i]:
            if gt[i][r]["resname"] == "UNK":
                continue
            assert r in pred[i]
            if "CA" in gt[i][r]["atoms"] and "CA" in pred[i][r]["atoms"]:
                gt_coords.append(gt[i][r]["atoms"]["CA"])
                pred_coords.append(pred[i][r]["atoms"]["CA"])
    if gt_coords and pred_coords:
        gt_coords = np.stack(gt_coords)
        pred_coords = np.stack(pred_coords)
        return gt_coords, pred_coords
    else:
        return [], []


def compute_rmsd(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    msd = np.mean(sd)
    return np.sqrt(msd + eps)


def compute_tm(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    num_res = true_atom_pos.shape[0]
    d0 = 1.24 * (num_res - 15) ** (1.0 / 3) - 1.8
    nsd = 1.0 / (1.0 + (sd) / (d0**2.0))
    return nsd.mean()


def compute_gdt(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    d = np.sqrt(np.square(true_atom_pos - pred_atom_pos).sum(axis=-1))

    def p(d, k):
        return (d <= k).astype(np.float32).sum() / d.size

    p0_5 = p(d, 0.5)
    p1 = p(d, 1)
    p2 = p(d, 2)
    p4 = p(d, 4)
    p8 = p(d, 8)
    return 0.25 * (p1 + p2 + p4 + p8), 0.25 * (p0_5 + p1 + p2 + p4)


def compute_lddt(
    true_atom_pos,
    pred_atom_pos,
    cutoff: float = 15.0,
    eps: float = 1e-10,
):
    n = true_atom_pos.shape[-2]
    dmat_true = np.sqrt(
        eps
        + np.sum(
            (true_atom_pos[..., None, :] - true_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )

    dmat_pred = np.sqrt(
        eps
        + np.sum(
            (pred_atom_pos[..., None, :] - pred_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )
    dists_to_score = (dmat_true < cutoff).astype(np.float32) * (1.0 - np.eye(n))

    dist_l1 = np.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).astype(np.float32)
        + (dist_l1 < 1.0).astype(np.float32)
        + (dist_l1 < 2.0).astype(np.float32)
        + (dist_l1 < 4.0).astype(np.float32)
    )
    score = score * 0.25

    norm = 1.0 / (eps + np.sum(dists_to_score, axis=-1))
    score = norm * (eps + np.sum(dists_to_score * score, axis=-1))
    return score.mean()


def compute_monomer(gt_pdb, pred_pdb):
    """
    Compute monomer metrics
    : param gt_pdb: ground truth pdb file
    : param pred_pdb: predicted pdb file
    """
    gt = get_pdb(gt_pdb)
    pred = get_pdb(pred_pdb)
    gt_coords, pred_coords = get_coords(gt, pred)
    r, x = get_optimal_transform(pred_coords, gt_coords)
    pred_coords = pred_coords @ r + x
    best_rmsd = compute_rmsd(gt_coords, pred_coords)
    best_tm = compute_tm(gt_coords, pred_coords)
    best_lddt = compute_lddt(gt_coords, pred_coords)
    best_gdt_ts, best_gdt_ha = compute_gdt(gt_coords, pred_coords)
    return {
        "rmsd": float(best_rmsd),
        "tm": float(best_tm),
        "lddt": float(best_lddt),
        "gdt_ts": float(best_gdt_ts),
        "gdt_ha": float(best_gdt_ha),
    }


def compute_multimer(gt_pdb, pred_pdb, entity, max_permutations=120):
    """
    Compute multimer metrics
    : param gt_pdb: ground truth pdb file
    : param pred_pdb: predicted pdb file
    : param entity: entity names for the chains in the multimer, e.g. for a 2-chain multimer A2, entity = ["A", "A"],
                    Permutaions is based on the entity names
    : param max_permutations: maximum number of permutations to try
    """
    gt = get_pdb(gt_pdb)
    pred = get_pdb(pred_pdb)
    best_rmsd = 1e10
    best_tm = 0
    best_lddt = 0
    best_gdt_ts = 0
    best_gdt_ha = 0
    perms = generate_perm(entity)
    if len(perms) > max_permutations:
        assert False, f"Too many permutations for {name}"
    for indices in perms:
        cur_pred = []
        for i in indices:
            cur_pred.append(pred[i])
        gt_coords, pred_coords = get_coords(gt, cur_pred)
        r, x = get_optimal_transform(pred_coords, gt_coords)
        pred_coords = pred_coords @ r + x
        cur_rmsd = compute_rmsd(gt_coords, pred_coords)
        cur_tm = compute_tm(gt_coords, pred_coords)
        cur_lddt = compute_lddt(gt_coords, pred_coords)
        cur_gdt_ts, cur_gdt_ha = compute_gdt(gt_coords, pred_coords)
        # use tm-score to select the best permutation
        if best_tm < cur_tm:
            best_tm = cur_tm
            best_lddt = cur_lddt
            best_rmsd = cur_rmsd
            best_gdt_ts = cur_gdt_ts
            best_gdt_ha = cur_gdt_ha
    return {
        "rmsd": float(best_rmsd),
        "tm": float(best_tm),
        "lddt": float(best_lddt),
        "gdt_ts": float(best_gdt_ts),
        "gdt_ha": float(best_gdt_ha),
    }
