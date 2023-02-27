# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import os
import pickle
import gzip
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)

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

    @lru_cache(maxsize=16)
    def get_by_key(self, key):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        key = key.encode("ascii")
        datapoint_pickled = self.env.begin().get(key)
        if datapoint_pickled is None:
            raise ValueError(f"cannot find key {key} in {self.db_path}")
        return pickle.loads(gzip.decompress(datapoint_pickled))
