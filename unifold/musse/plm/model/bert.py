import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.data import Dictionary
from unicore.modules import LayerNorm # , init_bert_params
from .transformer_encoder import TransformerEncoder


logger = logging.getLogger(__name__)

def init_position_params(module):
    if not getattr(module, 'can_global_init', True):
        return
    def normal_(data):
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.zero_()


@register_model("bert")
class BertModel(BaseUnicoreModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout", type=float, metavar="D", help="dropout probability for embeddings"
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument("--ignore-inter-rotary", type=bool, default="")
        parser.add_argument("--share-pos-emb", type=bool, default="")


    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.mask_idx = dictionary.index('[MASK]')
        self.embed_tokens = nn.Embedding(len(dictionary), args.encoder_embed_dim, self.padding_idx)
        # self.embed_positions = nn.Embedding(args.max_seq_len, args.encoder_embed_dim)
        self.sentence_encoder = TransformerEncoder(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            rel_pos=True,
            rel_pos_bins=32,
            max_rel_pos=128,
            post_ln=args.post_ln,
            ignore_inter_rotary=args.ignore_inter_rotary,
            share_pos_emb=args.share_pos_emb,
        )

        self.lm_head = RobertaLMHead(embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.embed_tokens.weight,
        )
        # self.freq_head = FreqHead(embed_dim=args.encoder_embed_dim,
        #     output_dim=22,
        #     activation_fn=args.activation_fn,
        #     # weight=self.embed_tokens.weight,
        # )
        self.classification_heads = nn.ModuleDict()
        self.apply(init_position_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)
    
    # def __make_freq_head_float__(self):
    #     self.freq_head = self.freq_head.float()
    
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_args = None,
    ):
        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder."]# ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
            
            prefixes = ["sentence_encoder.embed_tokens"]# ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("embed_tokens", name): param for name, param in state_dict.items()}
            return state_dict
        
        state_dict = upgrade_state_dict(state_dict)
        
        return super().load_state_dict(state_dict, strict)

    def half(self):
        super().half()
        # if (not getattr(self, "inference", False)):
        # self.__make_freq_head_float__()
        self.dtype = torch.half
        return self

    def forward(
        self,
        src_tokens,
        is_same_entity,
        has_same_sequence,
        masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        return_attn=False,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True
        # print("src_tokens:", src_tokens, src_tokens.shape)
        # print("is_same_entity:", is_same_entity, is_same_entity.shape)
        # print("has_same_sequence:", has_same_sequence[:, -20:, :], has_same_sequence.shape)
        # None.shape
        padding_mask = src_tokens.eq(self.padding_idx)
        x = self.embed_tokens(src_tokens)
        # x += self.embed_positions.weight[:src_tokens.size(1), :]
        x.masked_fill_((src_tokens == self.mask_idx).unsqueeze(-1), 0.0)
        # x: B x T x C
        if not self.training:
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (src_tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]    
            
        if not padding_mask.any():
            padding_mask = None

        x = self.sentence_encoder(x, padding_mask=padding_mask, return_attn=return_attn, is_same_entity=is_same_entity, has_same_sequence=has_same_sequence, features_only=features_only)

        if return_attn:
            _, attn = x
            return attn
        if not features_only:
            x = self.lm_head(x, masked_tokens)
            # freq_x = self.freq_head(x)
        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        return x


    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BertClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        features = features.type(self.dense.weight.dtype)
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
    
class FreqHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        features = features.type(self.dense.weight.dtype)
        x = self.layer_norm(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model_architecture("bert", "bert")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 33)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 5120)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.0)
    args.emb_dropout = getattr(args, "emb_dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 1024)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.ignore_inter_rotary = getattr(args, "ignore_inter_rotary", False)


@register_model_architecture("bert", "bert_base")
def bert_base_architecture(args):
    base_architecture(args)


@register_model_architecture("bert", "bert_large")
def bert_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)


@register_model_architecture("bert", "xlm")
def xlm_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)