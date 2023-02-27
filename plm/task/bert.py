import logging
import os
import torch 
import contextlib
from typing import Optional
import math 
import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
    UnicoreDataset,
    BaseWrapperDataset
)
from unicore.data.iterators import CountingIterator, ShardedIterator, BufferedIterator, EpochBatchIterator
from unicore.tasks import UnicoreTask, register_task

from plm.data import BertTruncateDataset, MaskTokensDataset, PositionRightPadDataset, IsSameEntityDataset, HasSameSequenceDataset, BertDataset
from plm.data.msa_dataset import MultimerDataset, PPIDataset, SingleDataset, MSAConcatDataset


logger = logging.getLogger(__name__)


@register_task("bert")
class BertTask(UnicoreTask):
    """Task for training masked language models (e.g., BERT)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        
        parser.add_argument(
            "--max-msa-len",
            default=512,
            type=int,
            help="max msa len",
        )
        parser.add_argument(
            "--epoch-size",
            default=512000,
            type=int, 
        )
        parser.add_argument(
            "--multimer-perc",
            default=0.1,
            type=float, 
        )
        parser.add_argument(
            "--ppi-perc",
            default=0.1,
            type=float, 
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.epoch_size = args.epoch_size
        self.multimer_perc = args.multimer_perc
        self.ppi_perc = args.ppi_perc
        assert self.multimer_perc + self.ppi_perc <= 1.0
        

    @classmethod
    def setup_task(cls, args, **kwargs):
        try:
            dictionary = Dictionary.load(os.path.join(args.data, "dict_esm.txt"))
        except:
            dictionary = Dictionary.load(os.path.join("/mnt/vepfs/projects/msa_pretrain/multimer_data", "dict_esm.txt"))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)
    
    def load_sizes(self, path, split):
        if split=='train':
           
            sizes = np.load(os.path.join(path, 'train_sizes.npy'))
        else:
            sizes = np.load(os.path.join(path, 'valid_sizes.npy'))
        return sizes

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        if split == 'train' and self.multimer_perc != 1.0:
            dataset = MSAConcatDataset(
                {
                    "multimer": MultimerDataset(os.path.join(self.args.data, "idx_to_sequences_train.lmdb")),
                    "ppi": PPIDataset(os.path.join(self.args.data, "ppi_train.lmdb")),
                    "single": SingleDataset(os.path.join(self.args.data, "5000_sequence_train.lmdb")),
                }
            )
            sizes = []
            file_names = [
                os.path.join(self.args.data, "idx_to_sequences_train.size.npy"),
                os.path.join(self.args.data, "ppi_train.size.npy"),
                os.path.join(self.args.data, "5000_sequence_train.size.npy")
            ]
            for fn in file_names:
                sizes.append(np.load(fn))
            sizes = np.concatenate(sizes, axis=0)
           
        elif split == 'train' and self.multimer_perc == 1.0:
            dataset  = MultimerDataset(os.path.join(self.args.data, "idx_to_sequences_train.lmdb"))
            sizes = np.load(os.path.join(self.args.data, "idx_to_sequences_train.size.npy"))
            
        else:
            dataset = MultimerDataset(os.path.join(self.args.data, "idx_to_sequences_valid_clean.lmdb"))
            
        
        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.args.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
        )
        src_dataset = BertTruncateDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len, seed=self.args.seed)
        tgt_dataset = BertTruncateDataset(tgt_dataset, self.dictionary, max_seq_len=self.args.max_seq_len, seed=self.args.seed, is_tgt=True)
        
        is_same_entity_dataset = IsSameEntityDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len, seed=self.args.seed)
        has_same_sequence_dataset = HasSameSequenceDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len, seed=self.args.seed)
        
        src_dataset = BertDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len, seed=self.args.seed)
        tgt_dataset = BertDataset(tgt_dataset, self.dictionary, max_seq_len=self.args.max_seq_len, seed=self.args.seed)


        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_dataset))

        if split == 'train' and self.multimer_perc != 1.0:
            nestd_dataset = NestedDictionaryDataset(
                                {
                                    "net_input": {
                                        "src_tokens": RightPadDataset(
                                            src_dataset,
                                            pad_idx=self.dictionary.pad(),
                                        ),
                                        "is_same_entity": PositionRightPadDataset(
                                            is_same_entity_dataset,
                                            pad_idx=0,
                                        ),
                                        "has_same_sequence": PositionRightPadDataset(
                                            has_same_sequence_dataset,
                                            pad_idx=0,
                                        )
                                    },
                                    "target": RightPadDataset(
                                        tgt_dataset,
                                        pad_idx=self.dictionary.pad(),
                                    ),
                                },
                            )
            self.datasets[split] = NoiseOrderedSampleDataset(nestd_dataset,
                                        sort_order=[shuffle, sizes],
                                        seed=self.args.seed,
                                        order_noise=5,
                                        epoch_size=self.epoch_size,
                                        names=dataset._dataset_name,
                                        start=dataset._dataset_start, 
                                        end=dataset._dataset_sum,
                                        sample_weight=dataset.sample_weight,
                                        multimer_perc=self.multimer_perc,
                                        ppi_perc=self.ppi_perc
                                        )
        elif split == 'train' and self.multimer_perc == 1.0:
            nestd_dataset = NestedDictionaryDataset(
                                {
                                    "net_input": {
                                        "src_tokens": RightPadDataset(
                                            src_dataset,
                                            pad_idx=self.dictionary.pad(),
                                        ),
                                        "is_same_entity": PositionRightPadDataset(
                                            is_same_entity_dataset,
                                            pad_idx=0,
                                        ),
                                        "has_same_sequence": PositionRightPadDataset(
                                            has_same_sequence_dataset,
                                            pad_idx=0,
                                        )
                                    },
                                    "target": RightPadDataset(
                                        tgt_dataset,
                                        pad_idx=self.dictionary.pad(),
                                    ),
                                },
                            )
            self.datasets[split] = NoiseOrderedDataset(nestd_dataset,
                                        sort_order=[shuffle],
                                        seed=self.args.seed,
                                        order_noise=5)
        
        else:
            self.datasets[split] = SortDataset(
                NestedDictionaryDataset(
                    {
                        "net_input": {
                            "src_tokens": RightPadDataset(
                                src_dataset,
                                pad_idx=self.dictionary.pad(),
                            ),
                            "is_same_entity": PositionRightPadDataset(
                                is_same_entity_dataset,
                                pad_idx=0,
                            ),
                            "has_same_sequence": PositionRightPadDataset(
                                has_same_sequence_dataset,
                                pad_idx=0,
                            )
                        },
                        "target": RightPadDataset(
                            tgt_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                    },
                ),
                sort_order=[
                    shuffle,
                ],
            )

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model

    def get_batch_iterator(
        self,
        dataset,
        batch_size=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~unicore.data.UnicoreDataset): dataset to batch
            batch_size (int, optional): max number of samples in each
                batch (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `UnicoreTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~unicore.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.info("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]
        else:
            logger.info("get EpochBatchIterator for epoch {}".format(epoch))

        assert isinstance(dataset, UnicoreDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            batch_size=batch_size,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = CustomEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            disable_shuffling=self.disable_shuffling(),
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter


class CustomEpochBatchIterator(EpochBatchIterator):

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            total_batchs = math.ceil(len(batches) / self.num_shards)
            idxs = np.arange(total_batchs)
            with data_utils.numpy_seed(seed):
                np.random.shuffle(idxs)
            batches_new = []
            for i in range(total_batchs):
                for j in range(self.num_shards):
                    idx = idxs[i] * self.num_shards + j 
                    if idx < len(batches):
                        batches_new.append(batches[idx])
            return batches_new

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)
        return itr
    
    
class NoiseOrderedDataset(BaseWrapperDataset):
    def __init__(self, dataset, sort_order, seed, order_noise):
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order
        self.seed = seed
        self.order_noise = order_noise

        assert all(len(so) == len(dataset) for so in sort_order)
        self._epoch = 0

    def ordered_indices(self):
        sort_order = []
        with data_utils.numpy_seed(self.seed + self._epoch):
            for so in self.sort_order:
                sort_order.append(
                    so +
                    np.random.randint(low=-self.order_noise, high=self.order_noise, size=so.shape))
            return np.lexsort(sort_order)

    def set_epoch(self, epoch):
        self._epoch = epoch
        super().set_epoch(epoch)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    
class NoiseOrderedSampleDataset(BaseWrapperDataset):
    def __init__(self, dataset, sort_order, seed, order_noise, epoch_size, names, start, end, sample_weight, multimer_perc, ppi_perc):
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order 
        self.seed = seed 
        self.order_noise = order_noise
        assert all(len(so) == len(dataset) for so in sort_order)
        self._epoch = 0
        
        self.epoch_size = epoch_size 
        self._dataset_size = []
        assert multimer_perc + ppi_perc < 1.0
        for n in names[:-1]:
            if n == "single":
                self._dataset_size.append(int(epoch_size * (1 - multimer_perc - ppi_perc)))
            elif n == "ppi":
                self._dataset_size.append(int(epoch_size * ppi_perc))
            else:
                self._dataset_size.append(int(epoch_size * multimer_perc))
        self._dataset_size.append(epoch_size - sum(self._dataset_size))
        self._start = start
        self._end = end 
        self._sample_weight = sample_weight
    
    def __len__(self):
        return self.epoch_size
        
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
        
    def set_epoch(self, epoch):
        self._epoch = epoch
        super().set_epoch(epoch)
        
    def ordered_indices(self):
        with data_utils.numpy_seed(self.seed + self._epoch):
            selected_indices = []
            for s, e, l, weight in zip(self._start, self._end, self._dataset_size, self._sample_weight):
                selected_indices.append(np.random.choice(np.arange(s, e), size=l, replace=True, p=weight))
            selected_indices = np.concatenate(selected_indices, axis=0)
            
            sort_order = []
        
            for so in self.sort_order:
                sort_order.append(
                    so[selected_indices] +
                    np.random.randint(low=-self.order_noise, high=self.order_noise, size=self.epoch_size))
            sorted_order = np.lexsort(sort_order)
            return selected_indices[sorted_order]
            
    
