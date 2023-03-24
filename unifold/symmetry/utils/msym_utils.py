from functools import reduce
import string
from libmsym import libmsym as m
import numpy as np
from typing import *
from .geometry_utils import remainder, normalize_axis, ANGLE_EPS, get_standard_syms_axes, \
    get_rotation_from_axis_theta, expand_groups, get_num_AU, PairingFailError, LabelType


def arr_to_elems(arr: np.ndarray, name: str = 'C') -> List[m.Element]:
    assert len(arr.shape) == 2 and arr.shape[-1] == 3, f"invalid points shape {arr.shape}"
    return [m.Element(name=name, coordinates=r) for r in arr]


def get_rotate_theta_axis(pts: np.ndarray, group: Optional[str] = None, eps: Optional[float] = None, debug=False
                          ) -> List[Tuple[float, np.ndarray]]:
    """
        Get the axes and their rotation orders (converted to rotation angles) by libmsym toolkit.


        Parameters
        ----------
        pts : array-like
            The point set i.e. the centers of units.
        group : str, optional
            Symmetry type.
        eps : float, optional
            Thresholds in libmsym.
        debug : bool
            If to print debug info.
        Returns
        -------
        list_theta_axis: list of Tuple[float, np.ndarray]
            Fetched axes and their rotation angles.

        See Also
        --------
        calc_format_pair_theta_axis in unifold.fold.data.geometry_units.py
    """
    ctx = m.Context(elements=arr_to_elems(pts))
    if eps is not None:
        ctx.set_thresholds(zero=eps, equivalence=eps, permutation=eps)
    try:
        group_ = ctx.find_symmetry()
    except Exception as e:
        raise PairingFailError(str(e))
    if group is not None:
        if group != 'C1' and group != group_:
            raise PairingFailError(f"unexpected group extracted: {group} vs {group_}")
    ops = ctx.symmetry_operations
    list_theta_axis = []
    for op in ops:
        if op.order > 1 and op.power == 1:
            theta = np.pi * 2 / op.order
            list_theta_axis.append((theta, np.array(op.vector)))
    return list_theta_axis


def collect_axis_with_fac(list_theta_axis: List[Tuple[float, np.ndarray]], angle_fac, debug=False
                          ) -> List[np.ndarray]:
    """
        Collect the axes which can rotate a unit to another with angle angle_fac i.e. 2 * pi / N.
        They are potential N-fold axis.
        For efficiency of later processing, we normalize the axes and merge similar ones.

        See Also
        --------
        collect_axis_with_fac in unifold.fold.data.geometry_units.py
    """
    list_axis = []
    for theta, axis in list_theta_axis:
        rem = remainder(np.abs(theta), angle_fac)
        if rem > ANGLE_EPS:
            continue
        axis = normalize_axis(axis)
        list_axis.append(axis)

    return list_axis


def combine_axes(list_list_axis: List[List[np.ndarray]],
                 standard_axes: np.ndarray, symmetry: str, accept_delta=1e-2, debug=False
                 ) -> List[np.ndarray]:
    """
        Find the axis combination to match the standard orientation 'standard_axes'.

        See Also
        --------
        combine_succeed_axis_group in unifold.fold.data.geometry_units.py
    """
    t_len = len(list_list_axis)
    axis_pair_angle_cos = (standard_axes @ standard_axes.T).clip(min=-1, max=1)

    best_list_ret_axis = []
    best_delta = 1e8
    search_cnt = 0

    def search(state: Tuple, reverse: Tuple):
        nonlocal best_list_ret_axis, best_delta, search_cnt
        search_cnt += 1
        if best_delta < accept_delta:
            return
        depth = len(state)
        assert len(reverse) == depth
        list_ret_axis = [list_list_axis[i][s] * r for i, (s, r) in enumerate(zip(state, reverse))]
        if depth == t_len:
            ret_axes = np.stack(list_ret_axis)
            ret_axis_pair_angle_cos = ret_axes @ ret_axes.T
            delta = np.linalg.norm(ret_axis_pair_angle_cos - axis_pair_angle_cos)
            if delta < best_delta:
                best_delta = delta
                best_list_ret_axis = list_ret_axis
            return

        for i, p_axis in enumerate(list_list_axis[depth]):
            if best_delta < accept_delta:
                return
            iterations = [(1, p_axis), (-1, -p_axis)] if depth > 0 else [(1, p_axis)]
            for rev, axis in iterations:
                flag = True
                for j in range(depth):
                    cos_j = np.dot(list_ret_axis[j], axis).clip(min=-1, max=1)
                    if np.abs(np.arccos(cos_j) - np.arccos(axis_pair_angle_cos[j, depth])) > ANGLE_EPS * 5:
                        flag = False
                if not flag:
                    continue
                if symmetry in 'TOI' and depth == 2:
                    ref_direction = np.sign(np.dot(np.cross(list_ret_axis[0], list_ret_axis[1]), axis))
                    tgt_direction = np.sign(np.dot(np.cross(standard_axes[0], standard_axes[1]), standard_axes[depth]))
                    if ref_direction != tgt_direction:
                        continue
                search((*state, i), (*reverse, rev))

    search((), ())
    return best_list_ret_axis


def recover_groups_by_axis(centers: np.ndarray, entity_ids: List, num_sym: int, axis: np.ndarray,
                           debug=False, strict_mod=False
                           ) -> Tuple[bool, List[List[int]], float]:
    """
        Check if the units can be matched when rotating around axis num_sym times with angle 2 * pi / num_sym.
        The cyclically matched units will be put into the same group.

        See Also
        --------
        recover_groups_by_axis in unifold.fold.data.geometry_units.py
    """
    n_f = centers.shape[0]
    theta = 2 * np.pi / num_sym
    groups = []
    max_delta = -1
    left_set = set(range(n_f))
    rot_mat_0 = get_rotation_from_axis_theta(axis, theta)
    temp_rot_mat = np.eye(3, dtype=np.float64)
    rot_mats = []
    for i in range(num_sym - 1):
        temp_rot_mat = temp_rot_mat @ rot_mat_0
        rot_mats.append(temp_rot_mat)
    while len(left_set):
        i = left_set.pop()
        group = [i]
        for rm_idx, rm in enumerate(rot_mats):
            target = rm @ centers[i, :]
            best_delta = 1e6
            best_j = -1
            for j in left_set:
                if entity_ids[i] != entity_ids[j]:
                    continue
                delta_dis = np.linalg.norm(centers[j, :] - target)
                delta = delta_dis
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            max_delta = max(max_delta, best_delta)
            if best_delta < 50:
                left_set.remove(best_j)
                group.append(best_j)
            else:
                if strict_mod:
                    return False, [[i] for i in range(n_f)], 1e8
                continue
        groups.append(group)
    return n_f == num_sym * len(groups), groups, max_delta


def get_au_with_groups(labels: List[LabelType], groups: List[List[int]], first_axis: np.ndarray, centers: np.ndarray
                       ) -> List[LabelType]:
    """
        Select a unit from each group to construct asymmetric units.

        See Also
        --------
        get_au_with_groups in unifold.fold.data.geometry_units.py
    """
    centroid = first_axis
    au_labels = []
    for group in groups:
        chosen_label = None
        best_cos = -1
        for idx in group:
            direction = centers[idx, :]
            cos = np.dot(direction / np.linalg.norm(direction), centroid)
            if cos > best_cos:
                best_cos = cos
                chosen_label = labels[idx]
        assert chosen_label is not None
        au_labels.append(chosen_label)
    return au_labels


def au_with_axes_msym(labels: List[LabelType], symmetry: str, centers: List[np.ndarray], debug=False
                      ) -> Tuple[List[LabelType], List[np.ndarray]]:
    """
        Find the asymmetric units and the spin axes (use libmsym toolkit).

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        symmetry : str
            Symmetry type.
        centers : sequence of array-like
            Unit centers.
        debug : bool
            If to print debug info.

        Returns
        -------
        au_labels : sequence of LabelType
            The asymmetric units.
        list_ret_axis : sequence of np.ndarray
            Spin axes, representing the orientation of units.
    """
    list_num_sym, standard_axes = get_standard_syms_axes(symmetry)
    entity_ids = [label['entity_id'][0] for label in labels]
    centers = np.stack(centers)
    list_theta_axis = get_rotate_theta_axis(centers, symmetry, debug=debug)
    list_list_axis = [
        collect_axis_with_fac(list_theta_axis, 2 * np.pi / n_sym, debug=debug) for n_sym in list_num_sym]
    list_ret_axis = combine_axes(list_list_axis, standard_axes, symmetry, debug=debug)
    list_groups = []
    for num_sym, axis in zip(list_num_sym, list_ret_axis):
        succeed, groups, delta = recover_groups_by_axis(centers, entity_ids, num_sym, axis,
                                                        debug=False, strict_mod=False)
        list_groups.append(groups)
    groups = expand_groups(list_groups, len(labels))
    au_labels = get_au_with_groups(labels, groups, list_ret_axis[0], centers)
    if len(au_labels) != len(labels) / get_num_AU(symmetry):
        raise PairingFailError(f'au length {len(au_labels)} rather than {len(labels) / get_num_AU(symmetry)}')
    return au_labels, list_ret_axis
