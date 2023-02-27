from audioop import mul
from functools import lru_cache

import numpy as np
import torch
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

class HasSameSequenceDataset(BaseWrapperDataset):
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

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes
    
    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, epoch, index):
            all_chain_features = self.dataset[index]['all_chain_features']
            asym_id_list = []
            for chain_id in range(1, len(all_chain_features)+1):
                if all_chain_features[chain_id-1] is None:
                    continue
                else:
                    asym_id_list.append(all_chain_features[chain_id-1]["asym_id"])
            
            asym_id = np.concatenate(asym_id_list)
            seq_length = len(asym_id)
            mask = asym_id.reshape(seq_length, 1) == asym_id.reshape(1, seq_length)
            mask = torch.tensor(mask)
            has_same_sequence = torch.zeros(seq_length, seq_length).long()
            has_same_sequence = has_same_sequence.masked_fill_(mask, 1)
                
            has_same_sequence_bos_eos = torch.ones(has_same_sequence.shape[0]+2, has_same_sequence.shape[0]+2).long()

            has_same_sequence_bos_eos[1:-1, 1:-1] = has_same_sequence
            return has_same_sequence_bos_eos.long()
    
class IsSameEntityDataset(BaseWrapperDataset):
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

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes
    
    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        with data_utils.numpy_seed(self.seed, epoch, index):
            all_chain_features = self.dataset[index]['all_chain_features']
            entity_id_list = []
            for chain_id in range(1, len(all_chain_features)+1):
                if all_chain_features[chain_id-1] is None:
                    continue
                else:
                    entity_id_list.append(all_chain_features[chain_id-1]["entity_id"])
                    
                    
            entity_id = np.concatenate(entity_id_list)
            seq_length = len(entity_id)
            mask = entity_id.reshape(seq_length, 1) == entity_id.reshape(1, seq_length)
            mask = torch.tensor(mask)
            is_same_entity = torch.zeros(seq_length, seq_length).long()
            is_same_entity = is_same_entity.masked_fill_(mask, 1)
            
            is_same_entity_bos_eos = torch.ones(is_same_entity.shape[0]+2, is_same_entity.shape[0]+2).long()

            is_same_entity_bos_eos[1:-1, 1:-1] = is_same_entity
            return is_same_entity_bos_eos