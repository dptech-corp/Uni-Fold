from functools import lru_cache

import numpy as np
import torch
import collections
# from tokenizers import BertWordPieceTokenizer
from unicore.data import Dictionary, data_utils

from unicore.data import BaseWrapperDataset, LRUCacheDataset


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
# AA

class BertTruncateDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary,
        max_seq_len: int=512,
        seed: int=1,
        is_tgt: bool=False,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.is_tgt = is_tgt
    
    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, epoch, index):
            all_chain_features = self.dataset[index]['all_chain_features']
            total_length = sum([chain_features['seq_length'] for chain_features in all_chain_features])
            sample_len = self.max_seq_len - 2
            if total_length > sample_len:
                asym_id_all = np.concatenate([chain_features['asym_id'] for chain_features in all_chain_features])
                selected_asym_id = np.random.choice(asym_id_all, sample_len, replace=False)
                cnt_asym_id = collections.Counter(selected_asym_id)
                for chain_id in range(1, len(all_chain_features)+1):
                    if chain_id not in cnt_asym_id:
                        all_chain_features[chain_id-1] = None
                    else:
                        cnt = cnt_asym_id[chain_id]
                        chain_features = all_chain_features[chain_id-1]
                        chain_features["asym_id"] = chain_id * np.ones(cnt)
                        chain_features["entity_id"] = chain_features["entity_id"][0] * np.ones(cnt)
                        if chain_features['seq_length'] - cnt > 0:
                            start = np.random.randint(0, chain_features['seq_length'] - cnt)
                            chain_features["seq"] = chain_features["seq"][start:start+cnt]
                        all_chain_features[chain_id-1] = chain_features
                    
            ret = torch.cat([chain_features['seq'] for chain_features in all_chain_features if chain_features is not None], dim=-1)
            ret = torch.LongTensor(ret)
            ret_bos_eos = torch.zeros(len(ret)+2).long()
            if self.is_tgt:
                ret_bos_eos[0] = self.dictionary.pad()
                ret_bos_eos[-1] = self.dictionary.pad()
                ret_bos_eos[1:-1] = ret
            else:
                ret_bos_eos[0] = self.dictionary.bos()
                ret_bos_eos[-1] = self.dictionary.eos()
                ret_bos_eos[1:-1] = ret
            return {"multimer_concat_seqence": ret_bos_eos, "all_chain_features": all_chain_features}
        
        
class BertDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dictionary,
        max_seq_len: int=512,
        seed: int=1,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
        self.seed = seed
    
    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, epoch, index):
            return self.dataset[index]['multimer_concat_seqence']


