import logging
from typing import Any

from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unifold.modules.alphafold import AlphaFold
from unifold.config import model_config


logger = logging.getLogger(__name__)


@register_model("af2")
class AlphafoldModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--model-name",
            help="choose the model config",
        )

    def __init__(self, args):
        super().__init__()
        base_architecture(args)
        self.args = args
        config = model_config(
            self.args.model_name,
            train=True,
        )
        self.model = AlphaFold(config)
        self.config = config

    def half(self):
        self.model = self.model.half()
        return self

    def bfloat16(self):
        self.model = self.model.bfloat16()
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def forward(self, batch, **kwargs):
        outputs = self.model.forward(batch)
        return outputs, self.config.loss


@register_model_architecture("af2", "af2")
def base_architecture(args):
    args.model_name = getattr(args, "model_name", "model_2")
