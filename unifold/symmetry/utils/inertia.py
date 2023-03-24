import numpy as np
from typing import List, Tuple


def smallest_gap(floats: List[float]) -> float:
    s = 1e8
    # s = floats[0]
    for i in range(len(floats) - 1):
        t = floats[i + 1] - floats[i]
        s = min(s, t)
    return s


def inertia_gap_by_ref_vec(inertia: np.ndarray, ref_vec: np.ndarray) -> Tuple[np.ndarray, float]:
    inertia_ = np.copy(inertia)
    inner_products = [np.dot(inertia_[:, i], ref_vec) for i in range(3)]
    for i in range(3):
        if inner_products[i] < 0:
            inertia_[:, i] = -inertia_[:, i]
            inner_products[i] = -inner_products[i]

    ret_inertia = inertia_[:, np.argsort(inner_products)]
    ref_inner_products = [float(np.dot(ret_inertia[:, i], ref_vec)) for i in range(3)]
    gap = smallest_gap(ref_inner_products) / np.linalg.norm(ref_vec)
    if np.dot(np.cross(ret_inertia[:, 2], ret_inertia[:, 1]), ret_inertia[:, 0]) < 0:
        ret_inertia[:, 0] = -ret_inertia[:, 0]
    return ret_inertia, gap


def inertia_gap(inertia: np.ndarray, list_ref_vec: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
    list_ref_inertia_gap = [inertia_gap_by_ref_vec(inertia, ref_vec) for ref_vec in list_ref_vec]
    list_inertia, list_gap = [], []
    for inert, gap in list_ref_inertia_gap:
        list_inertia.append(np.copy(inert))
        list_gap.append(gap)
    return list_inertia, list_gap


def best_inertia_gap_id(list_inertia_gaps: List[Tuple[List[np.ndarray], List[float]]]) -> int:
    lg_mat = np.array([inertia_gaps[1] for inertia_gaps in list_inertia_gaps])
    min_per_gap = np.min(lg_mat, axis=0)
    max_min_gap_id = int(np.argmax(min_per_gap))
    return max_min_gap_id


def get_ref_idx(n_a) -> List[Tuple[int, int]]:
    assert n_a >= 3
    if n_a == 3:
        return [(0, 1), (1, 2)]
    if n_a == 4:
        return [(0, 1), (1, 2), (2, 3)]
    ref_idx = [
        (0, int(n_a * 0.25)),
        (int(n_a * 0.25), int(n_a * 0.50)),
        (int(n_a * 0.50), int(n_a * 0.75)),
        (int(n_a * 0.75), -1)
    ]
    return ref_idx
