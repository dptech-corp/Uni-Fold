import numpy as np
from copy import deepcopy
from typing import List, Dict, Sequence, Tuple

from .geometry_utils import calc_distance_map, LabelType


def span_au(au_labels: List[LabelType], symmetry_opers: np.ndarray):
    """
        span asymmetric units with symmetry operators
    """
    spanned_labels = []
    num_asym = len(au_labels)
    for i, op in enumerate(symmetry_opers):
        this_au_labels = deepcopy(au_labels)
        for l in this_au_labels:
            l["all_atom_positions"] = l["all_atom_positions"] @ op[:3, :3].T
            l["asym_id"] = (l["asym_id"][0] + num_asym * i),
        spanned_labels.extend(this_au_labels)
    return spanned_labels


def get_pair_distance_set_dict(labels: List[LabelType], centers: List[np.ndarray]) -> Dict[str, Sequence]:
    """
        unit-pair distance set is SE(3)-invariant and index-insensitive
    """
    n = len(labels)
    assert n > 1
    labels = sorted(labels, key=lambda x: x['entity_id'][0])
    vectors = np.stack(centers)
    center_distance_matrix = calc_distance_map(vectors)
    pair_distance_set_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{labels[i]['entity_id'][0]}-@-{labels[j]['entity_id'][0]}"
            pair_distance_set_dict.setdefault(key, []).append(center_distance_matrix[i, j])
    for k, v in pair_distance_set_dict.items():
        pair_distance_set_dict[k] = sorted(v)
    return pair_distance_set_dict
