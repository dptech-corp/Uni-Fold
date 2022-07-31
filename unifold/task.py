import logging
import os

import contextlib
from typing import Optional

import numpy as np

from unifold.dataset import UnifoldDataset, UnifoldMultimerDataset
from unicore.data import data_utils
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


@register_task("af2")
class AlphafoldTask(UnicoreTask):
    """Task for training masked language models (e.g., BERT)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
        )
        parser.add_argument("--disable-sd", action="store_true")
        parser.add_argument(
            "--json-prefix",
            type=str,
            default="",
        )
        parser.add_argument(
            "--max-chains",
            type=int,
            default=18,
        )
        parser.add_argument(
            "--sd-prob",
            type=float,
            default=0.75,
        )

    def __init__(self, args):
        super().__init__(args)
        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.config.model.is_multimer:
            data_class = UnifoldMultimerDataset
        else:
            data_class = UnifoldDataset
        if split == "train":
            dataset = data_class(
                self.args,
                self.args.seed + 81,
                self.config,
                self.args.data,
                mode="train",
                max_step=self.args.max_update,
                disable_sd=self.args.disable_sd,
                json_prefix=self.args.json_prefix,
            )
        else:
            dataset = data_class(
                self.args,
                self.args.seed + 81,
                self.config,
                self.args.data,
                mode="eval",
                max_step=None,
                json_prefix=self.args.json_prefix,
            )

        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        self.config = model.config
        return model

    def disable_shuffling(self) -> bool:
        return True
