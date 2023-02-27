# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

import numpy as np
import torch

from unicore.data import BaseWrapperDataset

logger = logging.getLogger(__name__)

def collate_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    row_size = max(v.size(0) for v in values)
    size = max(v.size(1) for v in values)
    assert row_size == size
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        # logger.info("dst:", dst.numel())
        # logger.info("src:", src.numel())
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][:v.size(0),size - v.size(1):] if left_pad else res[i][:v.size(0),:v.size(1)])
    return res

class PositionPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return collate_tokens(samples, self.pad_idx, left_pad=self.left_pad, pad_to_multiple=8)


class PositionLeftPadDataset(PositionPadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class PositionRightPadDataset(PositionPadDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)
    
