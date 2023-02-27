
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import dataset
import lmdb
import os
import pickle
import torch
import numpy as np
import collections
from functools import lru_cache
from unicore.data import data_utils, BaseWrapperDataset
import logging
logger = logging.getLogger(__name__)

HHBLITS_AA_TO_ID = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    '-': 21,
}

ID_TO_AA = {value:key for key, value in HHBLITS_AA_TO_ID.items()}

AA_TO_ESM = {
    "[CLS]":0,
    "[PAD]":1,
    "[SEP]":2,
    "[UNK]":3,
    "L":4,
    "A":5,
    "G":6,
    "V":7,
    "S":8,
    "E":9,
    "R":10,
    "T":11,
    "I":12,
    "D":13,
    "P":14,
    "K":15,
    "Q":16,
    "N":17,
    "F":18,
    "Y":19,
    "M":20,
    "H":21,
    "W":22,
    "C":23,
    "X":24,
    "B":25,
    "U":26,
    "Z":27,
    "O":28,
    ".":29,
    "-":30,
    "<null_1>":31,
    "[MASK]":32,
}

    
class MultimerDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        self._sizes = None
        

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        sequences = data['sequences']
        tokenized_sequences = [np.array([AA_TO_ESM[tok] for tok in sequence]) for sequence in sequences]
        data['sequences'] = tokenized_sequences
        return data # metaname, sequences
    
    @property
    def sample_weight(self):
        return None

class PPIDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))
        self._sizes = None
        

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        sequences = [data['sequence_a'], data['sequence_b']]
        tokenized_sequences = [np.array([AA_TO_ESM[tok] for tok in sequence]) for sequence in sequences]
        # data['sequences'] = tokenized_sequences
        return {"sequences": tokenized_sequences}
    
    @property
    def sample_weight(self):
        return None
    
class SingleDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(
            self.db_path
        )
        env = self.connect_db(self.db_path)
        # with env.begin() as txn:
        #     self._keys = list(txn.cursor().iternext(values=False))
        with open(os.path.join(os.path.dirname(db_path), "5000_sequence_train.keys.pkl"), "rb") as f:
            self._keys = pickle.load(f)
        self._sizes = None
        

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(self._keys[idx])
        data = pickle.loads(datapoint_pickled)
        sequences = [data]
        tokenized_sequences = [np.array([AA_TO_ESM[ID_TO_AA[tok]] for tok in sequence]) for sequence in sequences]
        # data['sequences'] = tokenized_sequences
        return {'sequences': tokenized_sequences}
    
    @property
    def sample_weight(self):
        return None 

    
class MSAConcatDataset:
    def __init__(self, dataset_dict):
        self._dataset_dict = dataset_dict
        self._dataset_name = list(dataset_dict.keys())
        self._sizes_list = [len(dataset_dict[x]) for x in self._dataset_name]
        self._dataset_sum = np.cumsum(self._sizes_list)
        self._dataset_start = np.cumsum([0] + self._sizes_list[:-1])
        self._sample_weight = [dataset_dict[x].sample_weight for x in self._dataset_name]
        
    def __len__(self):
        return sum(self._sizes_list)
    
    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        dataidx = int(np.argmax(self._dataset_sum>idx))
        idx = idx - self._dataset_start[dataidx] 
        dataset_name = self._dataset_name[dataidx]
        return self._dataset_dict[dataset_name][idx]
    
    @property
    def sample_weight(self):
        return self._sample_weight
        
        