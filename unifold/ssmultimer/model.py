from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unifold.config import model_config
from .modules.alphafold import AlphaFoldSingle


@register_model("af2_single")
class AlphafoldSingleModel(BaseUnicoreModel):
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
        self.model = AlphaFoldSingle(config)
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


@register_model_architecture("af2_single", "af2_single")
def base_architecture(args):
    args.model_name = getattr(args, "model_name", "single_multimer")
