import numpy as np
import ml_collections as mlc
from typing import *

from .utils import get_transform
from ..dataset import load_and_process

import torch

def get_pseudo_residue_feat(symmetry: str):
    circ = 2. * np.pi
    symmetry = "C1" if symmetry is None else symmetry
    if symmetry == 'C1':
        ret = np.array([1., 0., 0., 0., 0., 0., 1., 0.], dtype=float)
    elif symmetry[0] == 'C':
        theta = circ / float(symmetry[1:])
        ret = np.array([0., 1., 0., 0., 0., 0., np.cos(theta), np.sin(theta)], dtype=float)
    elif symmetry[0] == 'D':
        theta = circ / float(symmetry[1:])
        ret = np.array([0., 0., 1., 0., 0., 0., np.cos(theta), np.sin(theta)], dtype=float)
    elif symmetry == 'I':
        ret = np.array([0., 0., 0., 1., 0., 0., 1., 0.], dtype=float)
    elif symmetry == 'O':
        ret = np.array([0., 0., 0., 0., 1., 0., 1., 0.], dtype=float)
    elif symmetry == 'T':
        ret = np.array([1., 0., 0., 0., 0., 1., 1., 0.], dtype=float)
    elif symmetry == 'H':
        raise NotImplementedError("helical structures not supported currently.")
    else:
        raise ValueError(f"unknown symmetry type {symmetry}")
    return ret


def load_and_process_symmetry(
    config: mlc.ConfigDict,
    mode: str,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
    symmetry: str = 'C1',
    **load_kwargs,
):
    if mode == "train":
        raise NotImplementedError("training UF-Symmetry not implemented.")
    if not symmetry.startswith('C'):
        raise NotImplementedError(f"symmetry group {symmetry} not supported currently.")
    feats, _ = load_and_process(config, mode, seed, batch_idx, data_idx, is_distillation, **load_kwargs)
    feats["symmetry_opers"] = torch.tensor(get_transform(symmetry), dtype=float)[None, :]
    feats["pseudo_residue_feat"] = torch.tensor(get_pseudo_residue_feat(symmetry), dtype=float)[None, :]
    feats["num_asym"] = torch.max(feats["asym_id"])[None]

    return feats, None
