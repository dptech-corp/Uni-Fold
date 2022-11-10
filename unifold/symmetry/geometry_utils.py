import numpy as np
from typing import List, Tuple, Sequence, Optional


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
    elif symmetry == 'T':
        return 12
    elif symmetry == 'O':
        return 24
    elif symmetry == 'I':
        return 60
    elif symmetry == 'H':
        raise NotImplementedError("helical structures not supported currently.")
    else:
        raise ValueError(f"unknown symmetry type {symmetry}")


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
    elif symmetry == 'T':
        ret = TRANSFORM_T
    elif symmetry == 'O':
        ret = TRANSFORM_O
    elif symmetry == 'I':
        ret = TRANSFORM_I
    elif symmetry == 'H':
        raise NotImplementedError("helical structures not supported currently.")
    else:
        raise ValueError(f"unknown symmetry type {symmetry}")
    return ret

