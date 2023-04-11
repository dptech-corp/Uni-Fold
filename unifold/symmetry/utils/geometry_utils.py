import numpy as np
from typing import List, Tuple, Dict, Sequence, Optional, MutableMapping, Any

ANGLE_EPS = 0.05 * np.pi

LabelType = MutableMapping[str, Any]
FormatType = np.ndarray

class MaskNotEnoughError(Exception):
    def __init__(self, *args, **kwargs):
        super(MaskNotEnoughError, self).__init__(*args, **kwargs)


class PairingFailError(Exception):
    def __init__(self, *args, **kwargs):
        super(PairingFailError, self).__init__(*args, **kwargs)


class RotatingFailError(Exception):
    def __init__(self, *args, **kwargs):
        super(RotatingFailError, self).__init__(*args, **kwargs)


def kabsch_rot_mat(p: np.ndarray, q: np.ndarray):
    """
        Using the Kabsch algorithm with two sets of paired point P and Q, centered
        around the centroid. Each vector set is represented as an NxD
        matrix, where D is the dimension of the space.
        The algorithm works in three steps:
        - a centroid translation of P and Q (assumed done before this function
          call)
        - the computation of a covariance matrix C
        - computation of the optimal rotation matrix U
        For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

        Parameters
        ----------
        p : array
            (N,D) matrix, where N is points and D is dimension.
        q : array
            (N,D) matrix, where N is points and D is dimension.

        Returns
        -------
        u : matrix
            Rotation matrix (D,D)
    """
    assert p.shape[0] == q.shape[0]
    assert p.shape[1] == q.shape[1] == 3

    # Computation of the covariance matrix
    c = np.dot(np.transpose(p), q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    c = np.array(c, dtype=np.float32)
    v, s, w = np.linalg.svd(c)
    d = (np.linalg.det(v) * np.linalg.det(w)) < 0.0
    # d = np.linalg.det(c) < 0.0

    if d:
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]

    # Create Rotation matrix U
    u = np.dot(v, w)

    return u


def get_rotation_from_axis_theta(axis: Sequence[float], theta: float) -> np.ndarray:
    """
        Calculates a rotation matrix given an axis and angle.

        Parameters
        ----------
        axis : sequence of float
            The rotation axis.
        theta : float
            The rotation angle.

        Returns
        -------
        rot_mat : np.ndarray
            The rotation matrix.
    """
    assert len(axis) == 3
    k_x, k_y, k_z = axis
    c, s = np.cos(theta), np.sin(theta)
    r_00 = c + (k_x ** 2) * (1 - c)
    r_11 = c + (k_y ** 2) * (1 - c)
    r_22 = c + (k_z ** 2) * (1 - c)
    r_01 = -s * k_z + (1 - c) * k_x * k_y
    r_10 = s * k_z + (1 - c) * k_x * k_y
    r_20 = -s * k_y + (1 - c) * k_x * k_z
    r_02 = s * k_y + (1 - c) * k_x * k_z
    r_12 = -s * k_x + (1 - c) * k_y * k_z
    r_21 = s * k_x + (1 - c) * k_y * k_z
    return np.array([
        [r_00, r_01, r_02],
        [r_10, r_11, r_12],
        [r_20, r_21, r_22],
    ], dtype=np.float64)


def check_rotation_with_axis(f1: FormatType, f2: FormatType, eps=1e-5) -> List[Tuple[float, np.ndarray]]:
    """
        Check if f1 can be rotated to f2 with ONLY ONE ROTATION-AXIS.
        If not, return []; otherwise, return all the possible rotation-angles thetas.
        Reference: https://blog.csdn.net/xhtchina/article/details/122767359 Section 2

        Parameters
        ----------
        f1 : FormatType
            Format 1.
        f2 : FormatType
            Format 2.
        eps : float
            Threshold.

        Returns
        -------
        list_theta_axis : List[Tuple[float, np.ndarray]
            The possible (angle, axis) tuples to rotate 'f1' to 'f2'.
    """

    def _calc(x, y, z):
        # Specially handle ZeroDivision
        if np.abs(z) < eps:
            if np.abs(x) < eps:
                ret = y
            else:
                ret = x
        else:
            ret = x * y / z
        if ret < 0:
            ret = -ret
        return ret

    list_theta_axis = []
    v1, v2 = f1[:, 0], f2[:, 0]
    r1, r2 = f1[:, 1:], f2[:, 1:]
    rot_mat = r2 @ r1.T
    if np.linalg.norm(v2 - rot_mat @ v1) > 20:
        return list_theta_axis
    tr = np.trace(rot_mat)
    if tr < -1 - eps or tr > 3 + eps:
        return list_theta_axis
    c = np.clip((tr - 1) / 2, a_min=-1, a_max=3)
    theta = np.arccos(c)
    t_xy = rot_mat[0, 1] + rot_mat[1, 0]
    t_xz = rot_mat[0, 2] + rot_mat[2, 0]
    t_yz = rot_mat[1, 2] + rot_mat[2, 1]
    q_xy = rot_mat[1, 0] - rot_mat[0, 1]
    q_xz = rot_mat[0, 2] - rot_mat[2, 0]
    q_yz = rot_mat[2, 1] - rot_mat[1, 2]
    if np.abs(q_xy) + np.abs(q_xz) + np.abs(q_yz) < 3 * eps:
        # Case 1: theta = 0 or pi
        if tr > 0:
            # theta = 0
            return []
        else:
            # theta = pi
            pk_x = np.sqrt((rot_mat[0, 0] + 1).clip(min=0) / 2)
            pk_y = np.sqrt((rot_mat[1, 1] + 1).clip(min=0) / 2)
            pk_z = np.sqrt((rot_mat[2, 2] + 1).clip(min=0) / 2)
    elif np.abs(t_xy * t_xz * t_yz - rot_mat[0, 0] * rot_mat[1, 1] * rot_mat[2, 2]) < eps:
        # Case 2: theta = pi / 2
        pk_x = np.sqrt(np.abs(rot_mat[0, 0]))
        pk_y = np.sqrt(np.abs(rot_mat[1, 1]))
        pk_z = np.sqrt(np.abs(rot_mat[2, 2]))
    elif np.abs(t_xy) + np.abs(t_xz) + np.abs(t_yz) < 3 * eps:
        # Case 3: (k_x, k_y, k_z) parallel to xyz
        ps = np.sqrt(1 - c ** 2)
        pk_x = np.abs(q_yz / (2 * ps))
        pk_y = np.abs(q_xz / (2 * ps))
        pk_z = np.abs(q_xy / (2 * ps))
    else:
        # Case 4: others
        pk_x = np.sqrt(_calc(t_xy, t_xz, t_yz)) / np.sqrt((3 - tr).clip(min=0))
        pk_y = np.sqrt(_calc(t_xy, t_yz, t_xz)) / np.sqrt((3 - tr).clip(min=0))
        pk_z = np.sqrt(_calc(t_xz, t_yz, t_xy)) / np.sqrt((3 - tr).clip(min=0))
    for k_x in [pk_x]:
        for k_y in [pk_y, -pk_y]:
            for k_z in [pk_z, -pk_z]:
                pseudo_rot_mat = get_rotation_from_axis_theta([k_x, k_y, k_z], theta)
                recovered = np.all(np.abs(rot_mat - pseudo_rot_mat) < 5e-1)
                if recovered:
                    # can recover the rotation matrix by single-axis rotation
                    axis = np.array([k_x, k_y, k_z], dtype=np.float64)
                    axis /= np.linalg.norm(axis)
                    list_theta_axis.append((theta, axis))
    return list_theta_axis


def remainder(a, b):
    c = a / b
    ic = np.round(c)
    return np.abs(c - ic)


def normalize_axis(axis: np.ndarray) -> np.ndarray:
    """
        Unifying polar axes, which are actually same axes but their representing vectors
            point to opposite directions.

        Parameters
        ----------
        axis : array-like
            The axis.

        Returns
        -------
        axis : np.ndarray
            Normalized axis.
    """
    sign = np.sign(np.sum(axis))
    if np.abs(sign) > 0.5:
        axis = axis * sign
    return axis


def calc_format_pair_theta_axis(formats: List[FormatType], entity_ids: List[int], debug=False
                                ) -> Dict[str, List[Tuple[float, np.ndarray]]]:
    """
        Find the format-pairs that can be rotationally transformed with single axis, and calculate the angle.

        Parameters
        ----------
        formats : sequence of FormatType
            Units' formats.
        entity_ids : sequence of int
            Units' entities.
        debug : bool
            If to print debug info

        Returns
        -------
        format_pair_theta_axis : Dict[str, List[Tuple[float, np.ndarray]]]
            Store all the axes and their angles to rotate unit i to unit j.
    """
    n_f = len(formats)
    assert len(entity_ids) == n_f
    format_pair_theta_axis = {}

    for i in range(n_f):
        for j in range(n_f):
            if i >= j or entity_ids[i] != entity_ids[j]:
                continue
            f1, f2 = formats[i], formats[j]
            list_theta_axis = check_rotation_with_axis(f1, f2, eps=1e-4) + check_rotation_with_axis(f1, f2, eps=5e-2)
            if len(list_theta_axis) == 0:
                continue
            format_pair_theta_axis[f'{i}-{j}'] = list_theta_axis

    return format_pair_theta_axis


def collect_axis_with_fac(format_pair_theta_axis: Dict[str, List[Tuple[float, np.ndarray]]], angle_fac
                          ) -> List[np.ndarray]:
    """
        Collect the axes which can rotate a unit to another with angle angle_fac i.e. 2 * pi / N.
        They are potential N-fold axis.
        For efficiency of later processing, we normalize the axes and merge similar ones.

        Parameters
        ----------
        format_pair_theta_axis : Dict[str, List[Tuple[float, np.ndarray]]]
            Store all the axes and their angles which can rotate a unit to another.
        angle_fac : float
            The angle.

        Returns
        -------
        list_axis : List of np.ndarray
            The potential N-fold axis.
    """
    list_axis = []
    list_axis_sim_set = []
    for ij, list_theta_axis in format_pair_theta_axis.items():
        list_angle_rem = [remainder(np.abs(a), angle_fac) for a, _ in list_theta_axis]
        for a_rem, (_, axis) in zip(list_angle_rem, list_theta_axis):
            axis = normalize_axis(axis)
            if a_rem > ANGLE_EPS:
                continue
            idx = -1
            for t, t_axis in enumerate(list_axis):
                if np.dot(axis, t_axis) > 0.9:
                    idx = t
                    break
            if idx == -1:
                list_axis.append(axis)
                list_axis_sim_set.append([axis])
            else:
                list_axis_sim_set[idx].append(axis)
                if len(list_axis_sim_set[idx]) < 10:
                    list_axis[idx] = sum(list_axis_sim_set[idx]) / len(list_axis_sim_set[idx])
                    list_axis[idx] /= np.sqrt(np.dot(list_axis[idx], list_axis[idx]))

    for i in range(len(list_axis)):
        list_axis[i] = sum(list_axis_sim_set[i]) / len(list_axis_sim_set[i])
        list_axis[i] /= np.sqrt(np.dot(list_axis[i], list_axis[i]))

    return list_axis


def delta_neg_translation(f1: FormatType, f2: FormatType) -> Tuple[float, float]:
    """
        Calculate the translation and rotation discrepancy between two formats.

        Parameters
        ----------
        f1 : FormatType
            Format 1.
        f2 : FormatType
            Format 1.

        Returns
        -------
        delta_dis : float
            Translation discrepancy.
        delta_rot : float
            Rotation discrepancy.
    """
    delta = f1 - f2
    delta_dis = np.linalg.norm(delta[:, 0])
    delta_rot = np.linalg.norm(delta[:, 1:])
    if delta_rot > 1:
        # chirality exceptions
        delta_rot = np.abs(2.0 - delta_rot)
    return delta_dis, delta_rot


def recover_groups_by_axis(formats: List[FormatType], entity_ids: List[str], num_sym: int, axis: np.ndarray,
                           debug=False, strict_mod=False
                           ) -> Tuple[bool, List[List[int]], float]:
    """
        Check if the units can be matched when rotating around axis num_sym times with angle 2 * pi / num_sym.
        The cyclically matched units will be put into the same group.

        Parameters
        ----------
        formats : sequence of FormatType
            Units' formats.
        entity_ids : sequence of str
            Units' entities
        num_sym : int
            Cyclic number.
        axis : array-like
            The axis to rotate.
        debug : bool
            If to print debug info.
        strict_mod : bool
            Recommended. If True, an axis will be dropped once a unit can't be cyclically matched;
                otherwise, group as many units as we can.

        Returns
        -------
        succeed : bool
            If recovery succeeds.
        groups : List[List[int]]
            Groups recovered.
        max_delta : float
            The max discrepancy in recovery. Lower max_delta means better match.
    """
    n_f = len(formats)
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
            target = rm @ formats[i]
            best_delta = 1e6
            best_j = -1
            for j in left_set:
                if entity_ids[i] != entity_ids[j]:
                    continue
                delta_dis, delta_rot = delta_neg_translation(formats[j], target)
                delta = delta_dis + delta_rot * 20
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
    return len(formats) == num_sym * len(groups), groups, max_delta


def combine_succeed_axis_group(list_list_axis_group: List[List[Tuple[np.ndarray, List[List[int]]]]],
                               standard_axes: np.ndarray, symmetry: str, accept_delta=1e-2, debug=False
                               ) -> Tuple[List[List[List[int]]], List[np.ndarray]]:
    """
        Find the axis combination to match the standard orientation 'standard_axes'.
        The function is designed to be called recursively, which starts from an empty list and ends
            when finishes an axis combination within accept_delta compared to standard_axes.
        In each recursive step, it tries to add another axis to the current combination.
        A new axis will be added to the current combination if it does not violate the angle constraint
            between two existing axes.

        Parameters
        ----------
        list_list_axis_group : List[List[Tuple[np.ndarray, List[List[int]]]]]
            Store the axis and the list of group indices.
        standard_axes : array-like
            Representing standard orientation.
        symmetry : str
            Symmetry type.
        accept_delta : float
            Threshold to accept the axis combination.
        debug : bool
            If to print debug info.

        Returns
        -------
        best_list_groups : List[List[List[int]]
            List of groups of unit indices, corresponding to the best axis combination
        best_list_ret_axis : List of np.ndarray
            The best axis combination.
    """
    t_len = len(list_list_axis_group)
    axis_pair_angle_cos = (standard_axes @ standard_axes.T).clip(min=-1, max=1)

    best_list_groups = []
    best_list_ret_axis = []
    best_delta = 1e8
    search_cnt = 0

    def search(state: Tuple, reverse: Tuple):
        nonlocal best_list_groups, best_list_ret_axis, best_delta, search_cnt
        search_cnt += 1
        if best_delta < accept_delta:
            return
        depth = len(state)
        assert len(reverse) == depth
        list_ret_axis = [list_list_axis_group[i][s][0] * r for i, (s, r) in enumerate(zip(state, reverse))]
        if depth == t_len:
            ret_axes = np.stack(list_ret_axis)
            ret_axis_pair_angle_cos = ret_axes @ ret_axes.T
            delta = np.linalg.norm(ret_axis_pair_angle_cos - axis_pair_angle_cos)
            if delta < best_delta:
                best_delta = delta
                best_list_groups = [list_list_axis_group[i][s][1] for i, s in enumerate(state)]
                best_list_ret_axis = list_ret_axis
            return

        for i, (p_axis, _) in enumerate(list_list_axis_group[depth]):
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
                # check chirality
                if symmetry in 'TOI' and depth == 2:
                    ref_direction = np.sign(np.dot(np.cross(list_ret_axis[0], list_ret_axis[1]), axis))
                    tgt_direction = np.sign(np.dot(np.cross(standard_axes[0], standard_axes[1]), standard_axes[depth]))
                    if ref_direction != tgt_direction:
                        continue
                search((*state, i), (*reverse, rev))

    search((), ())

    if len(best_list_groups) == 0 or len(best_list_ret_axis) == 0:
        raise PairingFailError("can't find axis combination.")
    return best_list_groups, best_list_ret_axis


def expand_groups(list_groups: List[List[List[int]]], num: int) -> List[List[int]]:
    """
        Expand groups with union-find sets.

        Parameters
        ----------
        list_groups : List[List[List[int]]]
            The cyclic groups under different spin axes.
        num : int
            Number of units

        Returns
        -------
        extracted_groups : List[List[int]]
            The extracted groups.
    """
    father = [i for i in range(num)]

    def find(i):
        if father[i] == i:
            return i
        f = find(father[i])
        father[i] = f
        return f

    def union(i, j):
        fi, fj = find(i), find(j)
        if fi != fj:
            fi, fj = sorted([fi, fj])
            father[fj] = fi

    def extract_groups() -> List[List[int]]:
        offsprings_dict = {}
        for i in range(num):
            fi = find(i)
            offsprings_dict.setdefault(fi, []).append(i)
        return list(offsprings_dict.values())

    for groups in list_groups:
        for group in groups:
            for idx in group[1:]:
                union(group[0], idx)
    return extract_groups()


def get_au_with_groups(labels: List[LabelType], groups: List[List[int]], first_axis: np.ndarray
                       ) -> List[LabelType]:
    """
        Select a unit from each group to construct asymmetric units.
        We hope the selected units are around the direction of first_axis.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        groups : List[List[int]]
            The cyclic groups, containing indices of units.
        first_axis: array-like
            The reference axis (N-fold in C/D, 3-fold in IOT).

        Returns
        -------
        au_labels : List of LabelType
            The asymmetric units.
    """
    centroid = first_axis
    au_labels = []
    for group in groups:
        chosen_label = None
        best_cos = -1
        for idx in group:
            direction = labels[idx]['format'][:, 0]
            cos = np.dot(direction / np.linalg.norm(direction), centroid)
            if cos > best_cos:
                best_cos = cos
                chosen_label = labels[idx]
        assert chosen_label is not None
        au_labels.append(chosen_label)
    return au_labels


STANDARD_AXES_C = np.array([
    [0, 0, 1],
], dtype=np.float64)

STANDARD_AXES_D = np.array([
    [0, 0, 1],
    [0, 1, 0],
], dtype=np.float64)

NUM_SYM_T = [3, 2, 2]
STANDARD_AXES_T = np.array([
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    [0, 1, 0],
    [1, 0, 0],
], dtype=np.float64)

NUM_SYM_O = [3, 4, 2]
STANDARD_AXES_O = np.array([
    [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
    [0, 0, 1],
    [1, 0, 0],
], dtype=np.float64)

NUM_SYM_I = [3, 5, 2, 2]
STANDARD_AXES_I = np.array([
    [0, -1 / np.sqrt(3), np.sqrt(2) / np.sqrt(3)],
    [0, 0, 1],
    [1 / 3, -2 * np.sqrt(2) / 3, 0],
    [0, -np.sqrt(2) / np.sqrt(3), 1 / np.sqrt(3)],
], dtype=np.float64)


def get_standard_syms_axes(symmetry: str) -> Tuple[List[int], np.ndarray]:
    """
        Get the information of spin axes of a symmetry type, including the axis vectors and their cyclic numbers.

        Parameters
        ----------
        symmetry : str
            Symmetry type.

        Returns
        -------
        list_num_sym : List of int
            The axes' cyclic numbers.
        standard_axes: np.ndarray
            The spin axes (normalized).
    """
    if symmetry.startswith('C'):
        list_num_sym = [int(symmetry[1:])]
        standard_axes = STANDARD_AXES_C
    elif symmetry.startswith('D'):
        list_num_sym = [int(symmetry[1:]), 2]
        standard_axes = STANDARD_AXES_D
    elif symmetry == 'T':
        list_num_sym = NUM_SYM_T
        standard_axes = STANDARD_AXES_T
    elif symmetry == 'O':
        list_num_sym = NUM_SYM_O
        standard_axes = STANDARD_AXES_O
    elif symmetry == 'I':
        list_num_sym = NUM_SYM_I
        standard_axes = STANDARD_AXES_I
    else:
        assert False, f'{symmetry}'
    return list_num_sym, standard_axes


def get_num_AU(symmetry: Optional[str]):
    """
        The get_num_AU function takes a string as input and returns the number of
            asymmetric units in that symmetry group.

        Parameters
        ----------
        symmetry : str, optional
            Symmetry type.

        Returns
        -------
        num_AU : int
            Number of asymmetric units.
    """
    if symmetry is None:
        return 1
    elif symmetry.startswith('C'):
        return int(symmetry[1:])
    elif symmetry.startswith('D'):
        return int(symmetry[1:]) * 2
    elif symmetry.startswith('T'):
        return 12
    elif symmetry.startswith('O'):
        return 24
    elif symmetry.startswith('I'):
        return 60
    else:
        raise NotImplementedError(f"do not support symmetry {symmetry} currently.")


def rotation_z(theta):
    ca = np.cos(theta)
    sa = np.sin(theta)
    ret = np.array([[ ca, -sa,  0.,  0.],
                    [ sa,  ca,  0.,  0.],
                    [ 0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  1.]])
    ret[np.abs(ret) < 1e-10] = 0
    return ret


def get_transform_C(grpnum):
    interval = 2 * np.pi / grpnum
    ret = np.stack([rotation_z(theta).astype(float) for theta in np.arange(0, 2 * np.pi, step=interval)])
    return ret


def get_transform_D(grpnum):
    assert grpnum % 2 == 0
    c_transform = get_transform_C(grpnum // 2)
    rot_y = np.array([[-1.,  0.,  0.,  0.],
                      [ 0.,  1.,  0.,  0.],
                      [ 0.,  0., -1.,  0.],
                      [ 0.,  0.,  0.,  1.]])
    ret = np.concatenate([c_transform, c_transform @ rot_y], axis=0)
    return ret


def combine_rigid_groups(rigid_groups: List[List[np.ndarray]]) -> np.ndarray:
    list_rigid = [np.eye(4, dtype=np.float64)]
    for rigid_group in rigid_groups:
        temp_list_rigid = []
        for r1 in list_rigid:
            for r2 in rigid_group:
                temp_list_rigid.append(r2 @ r1)
        list_rigid = temp_list_rigid
    return np.stack(list_rigid)


def get_transform_TOI(symmetry: str) -> np.ndarray:
    list_num_sym, standard_axes = get_standard_syms_axes(symmetry)
    n = len(list_num_sym)
    rigid_groups = []
    for i in range(n):
        num_sym, axis = list_num_sym[i], standard_axes[i, :]
        angles = [j * 2 * np.pi / num_sym for j in range(num_sym)]
        rigid_group = []
        for angle in angles:
            rigid = np.eye(4, dtype=np.float64)
            rigid[:3, :3] = get_rotation_from_axis_theta(axis, angle)
            rigid_group.append(rigid)
        rigid_groups.append(rigid_group)
    if symmetry == 'I':
        xs = [0, 3, 2, 1]
        rigid_groups = [rigid_groups[i] for i in xs]
    return combine_rigid_groups(rigid_groups)


TRANSFORM_T = get_transform_TOI('T')
TRANSFORM_O = get_transform_TOI('O')
TRANSFORM_I = get_transform_TOI('I')
assert TRANSFORM_T.shape[0] == 12
assert TRANSFORM_O.shape[0] == 24
assert TRANSFORM_I.shape[0] == 60


def get_transform(symmetry: str) -> np.ndarray:
    """
        Get symmetry operators of the given symmetry.

        Parameters
        ----------
        symmetry : str
            Symmetry type.

        Returns
        -------
        sym_opers: np.ndarray
            (N * 4 * 4) Symmetry operators.
    """
    if symmetry is None:
        ret = get_transform_C(1)
    elif symmetry.startswith('C'):
        ret = get_transform_C(get_num_AU(symmetry))
    elif symmetry.startswith('D'):
        ret = get_transform_D(get_num_AU(symmetry))
    elif symmetry.startswith('T'):
        ret = TRANSFORM_T
    elif symmetry.startswith('O'):
        ret = TRANSFORM_O
    elif symmetry.startswith('I'):
        ret = TRANSFORM_I
    else:
        raise NotImplementedError(f"do not support symmetry {symmetry} currently.")
    return ret


def au_with_axes(labels: List[LabelType], symmetry: str, debug=False, strict_mod=True
                 ) -> Tuple[List[LabelType], List[np.ndarray]]:
    """
        Find the asymmetric units and the spin axes

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        symmetry : str
            Symmetry type.
        debug : bool
            If to print debug info.
        strict_mod : bool
            Recommended. If True, an axis will be dropped once a unit can't be cyclically matched.

        Returns
        -------
        au_labels : List of LabelType
            The asymmetric units.
        list_ret_axis : sequence of np.ndarray
            Spin axes, representing the orientation of units.
    """
    list_num_sym, standard_axes = get_standard_syms_axes(symmetry)
    list_fac = [2 * np.pi / num_sym for num_sym in list_num_sym]
    formats = [label['format'] for label in labels]

    entity_ids = [label['entity_id'][0] for label in labels]
    format_pair_theta_axis = calc_format_pair_theta_axis(formats, entity_ids, debug=False)
    list_list_axis = [collect_axis_with_fac(format_pair_theta_axis, fac) for fac in list_fac]

    list_list_axis_group: List[List[Tuple[np.ndarray, List[List[int]]]]] = []
    for i, (num_sym, list_axis) in enumerate(zip(list_num_sym, list_list_axis)):
        if strict_mod:
            list_delta_axis_group = []
            for axis in list_axis:
                succeed, groups, delta = recover_groups_by_axis(formats, entity_ids, num_sym, axis, debug=False,
                                                                strict_mod=True)
                if succeed:
                    list_delta_axis_group.append((delta, axis, groups))
            list_delta_axis_group.sort(key=lambda x: x[0])
            list_axis_group = [(axis, groups) for _, axis, groups in list_delta_axis_group]
        else:
            list_axis_group = []
            for axis in list_axis:
                succeed, groups, _ = recover_groups_by_axis(formats, entity_ids, num_sym, axis, debug=False,
                                                            strict_mod=False)
                list_axis_group.append((axis, groups))
        list_list_axis_group.append(list_axis_group)
    list_groups, list_ret_axis = combine_succeed_axis_group(list_list_axis_group, standard_axes, symmetry,
                                                            debug=debug)

    groups = expand_groups(list_groups, len(labels))
    au_labels = get_au_with_groups(labels, groups, list_ret_axis[0])
    if len(au_labels) != len(labels) / get_num_AU(symmetry):
        raise PairingFailError(f'au length {len(au_labels)} rather than {len(labels) / get_num_AU(symmetry)}')
    return au_labels, list_ret_axis


def calc_distance_map(vectors: np.ndarray) -> np.ndarray:
    """
        Calculate the distances among the vectors.

        Parameters
        ----------
        vectors : array-like
            The vectors.

        Returns
        -------
        distance_matrix : np.ndarray
            The distance matrix.
    """
    return np.linalg.norm(np.expand_dims(vectors, axis=0) - np.expand_dims(vectors, axis=1), axis=2)
