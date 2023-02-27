# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore.modules import LayerNorm
from functools import partial
from torch.utils.checkpoint import checkpoint
from .transformer_encoder_layer import TransformerLayer

logger = logging.getLogger(__name__)

def checkpoint_sequential(
    functions,
    input,
):
    def wrap_tuple(a):
        return (a,) if type(a) is not tuple else a

    def exec(func, a):
        return wrap_tuple(func(*a))

    def get_wrap_exec(func):
        def wrap_exec(*a):
            return exec(func, a)

        return wrap_exec

    input = wrap_tuple(input)

    is_grad_enabled = torch.is_grad_enabled()

    if is_grad_enabled:
        for func in functions:
            input = checkpoint(get_wrap_exec(func), *input)
    else:
        for func in functions:
            input = exec(func, input)
    return input

def init_bert_params(module):
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
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    sign = torch.sign(relative_position)
    num_buckets //= 2
    n = torch.abs(relative_position)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact
    max_bucket_val = num_buckets - 1 - max_exact
    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + torch.ceil(
        torch.log(n.float() / max_exact) / math.log((max_distance - 1) / max_exact) * (max_bucket_val)
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret = torch.where(is_small, n, val_if_large) * sign
    return ret


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        rel_pos: bool = True,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        post_ln: bool = False,
        ignore_inter_rotary = False,
        share_pos_emb = False,
    ) -> None:

        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        # self.emb_layer_norm = LayerNorm(self.embed_dim)
        if not post_ln:
            self.emb_layer_norm_after = LayerNorm(self.embed_dim)
        else:
            self.emb_layer_norm_after = None
        
        self.share_pos_emb = share_pos_emb

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                    ignore_inter_rotary=ignore_inter_rotary,
                    share_pos_emb=share_pos_emb,
                )
                for _ in range(encoder_layers)
            ]
        )
        if share_pos_emb:
            self.is_same_entity_emb = nn.Embedding(2, attention_heads)
            self.has_same_sequence_emb = nn.Embedding(2, attention_heads)

        self.rel_pos = rel_pos

   
    def forward(
        self,
        emb: torch.Tensor,
        return_attn=False,
        features_only=False,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        is_same_entity: Optional[torch.Tensor] = None,
        has_same_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        seq_len = emb.size(1)       
        bsz = emb.size(0) 
        # x = self.emb_layer_norm(emb)
        x = F.dropout(emb, p=self.emb_dropout, training=self.training)
        hidden_representations = {}
        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        
        if self.share_pos_emb:
            multimer_pos_emb = self.is_same_entity_emb(is_same_entity) + self.has_same_sequence_emb(has_same_sequence)
            

        if attn_mask is not None and padding_mask is not None:
            # merge key_padding_mask and attn_mask
            attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
            attn_mask.masked_fill_(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
            padding_mask = None
            
        attn_probs_list = []
        if not self.training and not return_attn:
            if self.share_pos_emb:
                multimer_pos_emb = multimer_pos_emb.permute(0, 3, 1, 2).contiguous() \
                                                .view(bsz * self.attention_heads, seq_len, seq_len)
            for layer_id, layer in enumerate(self.layers):
                x = layer(x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=return_attn, is_same_entity=is_same_entity, has_same_sequence=has_same_sequence, multimer_pos_emb=multimer_pos_emb)
                if self.share_pos_emb:
                    x, multimer_pos_emb = x
                if features_only and layer_id == 0:
                    hidden_representations[layer_id + 1] = x
            
        elif return_attn:
            if self.share_pos_emb:
                multimer_pos_emb = multimer_pos_emb.permute(0, 3, 1, 2).contiguous() \
                                                .view(bsz * self.attention_heads, seq_len, seq_len)
            for layer in self.layers:
                x, attn_weights, attn_probs = layer(x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=return_attn, is_same_entity=is_same_entity, has_same_sequence=has_same_sequence, multimer_pos_emb=multimer_pos_emb)
                if attn_probs.dim() == 3:
                    attn_probs = attn_probs.unsqueeze(0)
                attn_probs_list.append(attn_probs) # B*H, L, L
        

        else:
            blocks = [
                partial(
                    b,
                    padding_mask=padding_mask,
                    is_same_entity=is_same_entity, 
                    has_same_sequence=has_same_sequence,
                    # multimer_pos_emb=multimer_pos_emb,
                    # attn_bias=attn_mask,
                )
                for b in self.layers
            ]
            if self.share_pos_emb:
                multimer_pos_emb = multimer_pos_emb.permute(0, 3, 1, 2).contiguous() \
                                                .view(bsz * self.attention_heads, seq_len, seq_len)
                x = checkpoint_sequential(
                    blocks,
                    input=(x, multimer_pos_emb),
                )[0]
            else:
                x = checkpoint_sequential(
                    blocks,
                    input=x,
                )[0]  
        
        if self.emb_layer_norm_after is not None:
            x = self.emb_layer_norm_after(x)
        
        if features_only:
            hidden_representations[layer_id + 1] = x
            assert 36 in hidden_representations
            assert 1 in hidden_representations
        if return_attn:
            return x, torch.cat(attn_probs_list, dim=0) # num_layer, B*H, L, L
        elif features_only:
            return hidden_representations
        else:
            return x