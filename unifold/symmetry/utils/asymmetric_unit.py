from typing import Dict, List, Tuple, Any, Union, Sequence, Optional
import numpy as np
from .geometry_utils import MaskNotEnoughError, PairingFailError, RotatingFailError, LabelType
from .geometry_utils import get_standard_syms_axes, get_transform, au_with_axes
from .quaternion import quaternion_rotate
from .au_span import span_au, get_pair_distance_set_dict
from .msym_utils import au_with_axes_msym
from .inertia import inertia_gap, best_inertia_gap_id, get_ref_idx
from itertools import groupby
from functools import reduce


def masked_mean_pos(
        pos: np.ndarray,
        mask: np.ndarray,
        strict: bool = True,
        axis: Union[int, Tuple] = -2):
    '''
    pos: [*, 3]
    mask: [*]
    strict: if true, raise when all is masked.
    axis: the axis to be avg (on pos)
    '''
    mask = np.expand_dims(mask, -1)
    if np.any(mask):
        return np.sum(pos * mask, axis=axis) / np.sum(mask, axis=axis)
    elif strict:
        raise ValueError("inputs are empty")
    else:  # return 0
        return np.sum(pos * mask, axis=axis)


def norm_vec(xs):
    """
    get the norm vec n [3] of points xs [N, 3], presumed that 0 is on n.
    see weijie's proof of norm vecs of points.
    """
    _, vec = np.linalg.eig(xs.T @ xs)
    n = vec[:, 0]
    if n[-1] < 0:
        n *= -1  # n points to z+
    return n


def rodrigues(a, b, eps=1e-6, normalized: bool = False):
    """
    a, b shape [3]
    return r 3x3 s.t. r @ a == b
    """
    if not normalized:
        a /= np.linalg.norm(a, 2)
        b /= np.linalg.norm(b, 2)
    v = np.cross(a, b)
    s = np.linalg.norm(v, 2)
    c = np.dot(a, b)
    if np.abs(1. + c) < eps:  # cosa = -1, i.e. a, b opposite
        return -np.eye(3)
    vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )
    r = np.eye(3) + vx + (1. / (1. + c)) * vx @ vx
    # assert np.all(r @ a - b < eps)
    return r


def get_centers(labels: List[LabelType], entity_all_atom_mask=None, removal=True
                ) -> List[np.ndarray]:
    """
        Get unit centers.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        entity_all_atom_mask : dict of str / ndarray
            Intersected 'all_atom_mask' for each 'entity_id'.
        removal : bool
            If to remove 'temp_all_atom_mask' key after processing.

        Returns
        -------
        label_centers : List of np.ndarray
            Unit centers.
    """
    if entity_all_atom_mask is None:
        assert False
        # entity_all_atom_mask = get_entity_all_atom_mask(labels)
    for label in labels:
        label["temp_all_atom_mask"] = entity_all_atom_mask[label['entity_id'][0]]

    ca_idx = 1 # rc.atom_order["CA"]
    label_centers = []
    for label in labels:
        ca_pos = label['all_atom_positions'][..., ca_idx, :]
        ca_mask = (label['temp_all_atom_mask'][..., ca_idx])
        label_centers.append(masked_mean_pos(ca_pos, ca_mask, strict=True))
        if removal:
            label.pop("temp_all_atom_mask")
    return label_centers


def get_entity_all_atom_mask(labels: List[LabelType]) -> Dict[str, Any]:
    """
        Get intersected 'all_atom_mask' for all units with the same 'entity_id'.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.

        Returns
        -------
        entity_all_atom_mask: Dict of str / np.ndarray
            Intersected 'all_atom_mask' for each 'entity_id'.
    """
    entity_group_labels = groupby(labels, key=lambda x: x['entity_id'][0])
    entity_all_atom_mask = {}
    for entity_id, ls in entity_group_labels:
        entity_all_atom_mask[entity_id] = reduce(lambda x, y: x * y, map(lambda x: x["all_atom_mask"], ls))
    return entity_all_atom_mask


def normalize_update_labels(labels: List[LabelType], use_inertia=False, debug=False):
    """
        1) Unifying atom masks of units.
        2) Decentralize units.
        3) Get the formats of units, including their position (for translation) and orientation (for rotation).

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        use_inertia : bool
            If to use unit's inertia as format, not recommended.
        debug : bool
            If to print debug info.
    """
    ca_idx = 1 # rc.atom_order["CA"]
    # step 1: find the intersection of labels' atom masks
    entity_all_atom_mask = get_entity_all_atom_mask(labels)
    for label in labels:
        label["temp_all_atom_mask"] = entity_all_atom_mask[label['entity_id'][0]]
    entity_aai = {k: np.argwhere(aam[:, ca_idx] > 0.5)[:, 0] for k, aam in entity_all_atom_mask.items()}

    # step 2: decentralize
    centers = []
    for i, label in enumerate(labels):
        xs = label["all_atom_positions"]
        aai = entity_aai[label['entity_id'][0]]
        if len(aai) < 3:
            raise MaskNotEnoughError("all_atom_mask less than 2")
        xs = np.stack([xs[i, ca_idx, :] for i in aai])
        center = np.mean(xs, axis=0)
        centers.append(center)

    all_center = sum(centers) / len(centers)
    for label in labels:
        label["all_atom_positions"] -= all_center

    # step 3: get the format of each unit
    if use_inertia:
        # use inertia
        list_inertia_gaps = []
        for i, label in enumerate(labels):
            xs = label["all_atom_positions"]
            aai = np.argwhere(label['temp_all_atom_mask'] > 0.5)
            if aai.shape[0] < 5:
                raise PairingFailError
            list_ref_vec = [
                xs[aai[-1][0], aai[-1][1]] - xs[aai[0][0], aai[0][1]],
                xs[aai[-int(aai.shape[0] / 2)][0], aai[-int(aai.shape[0] / 2)][1]] - xs[aai[0][0], aai[0][1]],
                xs[aai[-4][0], aai[-4][1]] - xs[aai[0][0], aai[0][1]],
            ]
            xs = np.stack([xs[i, j, :] for i, j in aai])
            center = np.mean(xs, axis=0)
            xs -= center
            _, inertia = np.linalg.eig(xs.T @ xs)
            list_inertia_gaps.append(inertia_gap(inertia, list_ref_vec))
            label['format'] = np.zeros(shape=[3, 4], dtype=np.float64)
            label['format'][:, 0] = center

        inertia_id = best_inertia_gap_id(list_inertia_gaps)

        for i, label in enumerate(labels):
            label['format'][:, 1:] = list_inertia_gaps[i][0][inertia_id]
    else:
        # use reference vectors
        entity_ref_cos = {}
        main_ref_vecs, list_ref_vecs = [], []
        for i, label in enumerate(labels):
            xs = label["all_atom_positions"]
            aai = entity_aai[label['entity_id'][0]]
            main_ref_vec = xs[aai[-1], ca_idx, :] - xs[aai[0], ca_idx, :]
            list_ref_vec = [xs[aai[v], ca_idx, :] - xs[aai[u], ca_idx, :] for u, v in get_ref_idx(len(aai))]
            main_ref_vecs.append(main_ref_vec)
            list_ref_vecs.append(list_ref_vec)
            entity_ref_cos.setdefault(label['entity_id'][0], []).append([
                np.abs(np.dot(main_ref_vec, ref_vec)) / (np.linalg.norm(ref_vec) * np.linalg.norm(main_ref_vec))
                for ref_vec in list_ref_vec])
            xs = np.stack([xs[i, ca_idx, :] for i in aai])
            center = np.mean(xs, axis=0)
            xs -= center
            label['format'] = np.zeros(shape=[3, 4], dtype=np.float64)
            label['format'][:, 0] = center  # position

        # find the reference vector with the biggest intersection angle with main_ref_vec
        entity_ref_id = {}
        for k, ref_cos in entity_ref_cos.items():
            ref_cos = np.array(ref_cos)
            entity_ref_id[k] = np.argmin(np.max(ref_cos, axis=0))

        for i, label in enumerate(labels):
            main_ref_vec, ref_vec = main_ref_vecs[i], list_ref_vecs[i][entity_ref_id[label['entity_id'][0]]]
            f1 = main_ref_vec / np.linalg.norm(main_ref_vec)
            f2_ = np.cross(main_ref_vec, ref_vec)  # f2 ⊥ f1
            f2 = f2_ / np.linalg.norm(f2_)
            f3 = np.cross(f1, f2)  # f3 ⊥ f1 and f3 ⊥ f2
            label['format'][:, 1], label['format'][:, 2], label['format'][:, 3] = f1, f2, f3  # orientation


def restore_labels(labels: List[LabelType]):
    """
        Remove useless keys.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
    """
    for label in labels:
        if "temp_all_atom_mask" in label.keys():
            label.pop("temp_all_atom_mask")
        if "format" in label.keys():
            label.pop("format")


def get_au_axes(labels: List[LabelType], symmetry: str, use_msym=False, debug=False
                ) -> Tuple[List[LabelType], Optional[np.ndarray]]:
    """
        Get the asymmetric units and the orientation with given symmetry type.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        symmetry : str
            Symmetry type.
        use_msym : bool
            If to use libmsym toolkit.
        debug : bool
            If to print debug info.

        Returns
        -------
        au_labels : List of LabelType
            The asymmetric units.
        axes : np.ndarray, optional
            Spin axes, representing the orientation of units, None for C1.
    """
    if symmetry == 'C1':
        return labels, None

    if use_msym:
        au_labels, axes = au_with_axes_msym(
            labels, symmetry, get_centers(labels, get_entity_all_atom_mask(labels), removal=True), debug=debug)
    else:
        au_labels, axes = au_with_axes(labels, symmetry, debug=debug)

    axes = np.stack(axes)
    return au_labels, axes


def translate_rotate_labels(labels: List[LabelType], axes: Optional[np.ndarray], symmetry: str, debug=False):
    """
        Rotate the units to match the standard orientation.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        axes : array-like, optional
            Spin axes, representing the original orientation of units, None for C1.
        symmetry : str
            Symmetry type.
        debug : bool
            If to print debug info.
    """
    if symmetry.startswith('C'):
        if axes is not None:
            axis = axes[0, :]
            rot_mat = rodrigues(axis / np.linalg.norm(axis), np.array([0., 0., 1.]))
            for label in labels:
                label["all_atom_positions"] = label["all_atom_positions"] @ rot_mat.T
        return

    assert axes is not None
    _, standard_axes = get_standard_syms_axes(symmetry)
    rot_mat = quaternion_rotate(axes, standard_axes)
    for label in labels:
        label["all_atom_positions"] = label["all_atom_positions"] @ rot_mat


def score_au_symmetry_match(all_labels: List[LabelType], au_labels: List[LabelType], symm_op: np.ndarray,
                            thresholds: Sequence[int] = (3, 5, 8, 13, 21)) -> Tuple[float, List[float]]:
    """
        Try to recover the structure by spanning asymmetric units with symmetric operators, and calculate a score.
        Higher score indicates lower discrepancy between spanned units and original structure units.

        Parameters
        ----------
        all_labels : sequence of LabelType
            Structure units.
        au_labels : sequence of LabelType
            The asymmetric units.
        symm_op : array-like
            Symmetric operators.
        thresholds : sequence of float
            Thresholds of distance discrepancy.

        Returns
        -------
        score : float
            Rotation score.
        scores : List of float
            Rotation scores with different thresholds.
    """
    if len(all_labels) == 1:
        return 1., [1. for _ in thresholds]
    entity_all_atom_mask = get_entity_all_atom_mask(all_labels)
    spanned_labels = span_au(au_labels, symm_op)
    spanned_centers = get_centers(spanned_labels, entity_all_atom_mask, removal=False)
    all_centers = get_centers(all_labels, entity_all_atom_mask, removal=False)
    src_dis = get_pair_distance_set_dict(spanned_labels, spanned_centers)
    tgt_dis = get_pair_distance_set_dict(all_labels, all_centers)
    delta_dis = np.concatenate([np.abs(np.array(src_dis[k]) - np.array(tgt_dis[k])) for k in src_dis.keys()])
    hierarchy_passed_points = [delta_dis < i for i in thresholds]
    scores = [np.sum(passed_points) / len(passed_points) for passed_points in hierarchy_passed_points]
    score = sum(scores) / len(scores)
    return score, scores


def get_normalized_au(labels: List[LabelType], symmetry: Optional[str],
                      accept_score=0.6, rotate_fail_tolerate=True, debug=False
                      ) -> Tuple[List[LabelType], Dict[str, Any]]:
    """
        Get the asymmetric units, which are translated and rotated to reference positions for later recovery.
        Recommended setting:
            - Set 'accept_score' to a relatively large number (0.6~0.8) to instantly accept AUs with good span recovery.
            - Set 'rotate_fail_tolerate=True' to allow to return AU with unsatisfying span recovery,
                and let users decide whether to use AU or to regard them as 'C1' symmetry with 'score' in 'ret_dict'.

        Parameters
        ----------
        labels : sequence of LabelType
            Structure units.
        symmetry : str, optional
            Symmetry type.
        accept_score : float
            Accept rotation score.
        rotate_fail_tolerate : bool
            If Ture, suppress RotatingFailError.
        debug : bool
            If to print debug info.

        Returns
        -------
        au_labels : list of unit
            The asymmetric units
        ret_dict : dict
            The return info, including following keys:
                'handle_symmetry': (str) The symmetry type finally decided to use to handle the units. C1 if failed.
                'error_string': (str, optional key) Error info.
                'score': (float) The rotation score of output asymmetric units. We have score < 0 if failed.
    """
    if symmetry is None:
        symmetry = "C1"
    # initialize temporary all_atom_mask
    random_state = np.random.get_state()
    np.random.seed(1)

    # group labels
    au_labels, handle_symmetry, score = [lb for lb in labels], symmetry, -1.0

    def switch_symmetry(use_msym=False):
        nonlocal au_labels, handle_symmetry
        au_labels, axes = get_au_axes(labels, symmetry, use_msym, debug)
        handle_symmetry = symmetry
        translate_rotate_labels(labels, axes, symmetry, debug)

    def fail_process():
        # decide to use C1 instead
        nonlocal au_labels, handle_symmetry, score
        au_labels, axes_ = get_au_axes(labels, 'C1', debug)
        handle_symmetry = 'C1'
        translate_rotate_labels(labels, axes_, 'C1', debug)
        ret_dict['error_string'] = str(ex)
        score = -1.0

    ret_dict = {}
    try:
        normalize_update_labels(labels, debug=debug)
        switch_symmetry()
        score, _ = score_au_symmetry_match(labels, au_labels, symm_op=get_transform(symmetry))
        if score < accept_score:
            raise RotatingFailError(f'Too low rotation score: {score} < {accept_score}')
    except MaskNotEnoughError as ex:
        fail_process()
    except PairingFailError:
        try:
            switch_symmetry(use_msym=True)
            score, _ = score_au_symmetry_match(labels, au_labels, symm_op=get_transform(symmetry))
            if score < accept_score:
                raise RotatingFailError(f'Too low rotation score: {score} < {accept_score}')
        except PairingFailError as ex:
            fail_process()
        except RotatingFailError as ex:
            if rotate_fail_tolerate:
                pass
            else:
                fail_process()
    except RotatingFailError as ex:
        try:
            former_au_labels, former_handle_symmetry, former_score = au_labels, handle_symmetry, score
            switch_symmetry(use_msym=True)
            score, _ = score_au_symmetry_match(labels, au_labels, symm_op=get_transform(symmetry))
            if score < former_score:
                au_labels, handle_symmetry, score = former_au_labels, former_handle_symmetry, former_score
            if score < accept_score:
                raise RotatingFailError(f'Too low rotation score: {score} < {accept_score}')
        except PairingFailError as ex:
            if rotate_fail_tolerate:
                pass
            else:
                fail_process()
        except RotatingFailError as ex:
            if rotate_fail_tolerate:
                pass
            else:
                fail_process()
    ret_dict['handle_symmetry'] = handle_symmetry
    ret_dict['score'] = score

    restore_labels(labels)
    np.random.set_state(random_state)
    return au_labels, ret_dict


def get_factors(i):
    # slow yet effective
    return tuple(f for f in range(1, i // 2 + 1) if i % f == 0)


def get_subgroups(symmetry):
    if symmetry == "I":
        return ("D5", "D3", "D2", "C5", "C3", "C2", "C1")
    elif symmetry == "O":
        return ("D4", "D3", "D2", "C4", "C3", "C2", "C1")
    elif symmetry == "T":
        return ("C3", "C2", "C1")
    elif symmetry == "C1":
        return ()
    elif symmetry[0] == "C":
        return tuple(f"C{t}" for t in get_factors(int(symmetry[1:])))
    elif symmetry[0] == "D":
        return tuple(f"D{t}" for t in get_factors(int(symmetry[1:]))[1:]) + tuple(
            f"C{t}" for t in get_factors(int(symmetry[1:])))
    else:
        raise NotImplementedError(f"{symmetry} unsupported.")
