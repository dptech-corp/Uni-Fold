import torch
import torch.nn as nn
from typing import Tuple, Optional
from functools import partial

from .common import (
    Linear,
    Transition,
    OuterProductMean,
    SimpleModuleList,
    residual,
    bias_dropout_residual,
    tri_mul_residual,
)
from .attentions import (
    MSARowAttentionWithPairBias,
    MSAColumnAttention,
    MSAColumnGlobalAttention,
    TriangleAttentionStarting,
    TriangleAttentionEnding,
)
from .triangle_multiplication import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from unicore.utils import checkpoint_sequential


import torch.distributed as dist
from unicore.distributed.comm_group import scg
from unicore.distributed import bp


class EvoformerIteration(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hid_msa_att: int,
        d_hid_opm: int,
        d_hid_mul: int,
        d_hid_pair_att: int,
        num_heads_msa: int,
        num_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        outer_product_mean_pos: bool,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
    ):
        super(EvoformerIteration, self).__init__()

        self._is_extra_msa_stack = _is_extra_msa_stack
        self.outer_product_mean_pos = outer_product_mean_pos

        self.msa_att_row = MSARowAttentionWithPairBias(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hid=d_hid_msa_att,
            num_heads=num_heads_msa,
        )

        if _is_extra_msa_stack:
            self.msa_att_col = MSAColumnGlobalAttention(
                d_in=d_msa,
                d_hid=d_hid_msa_att,
                num_heads=num_heads_msa,
                inf=inf,
                eps=eps,
            )
        else:
            self.msa_att_col = MSAColumnAttention(
                d_msa,
                d_hid_msa_att,
                num_heads_msa,
            )

        self.msa_transition = Transition(
            d_in=d_msa,
            n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            d_msa,
            d_pair,
            d_hid_opm,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            d_pair,
            d_hid_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            d_pair,
            d_hid_mul,
        )

        self.tri_att_start = TriangleAttentionStarting(
            d_pair,
            d_hid_pair_att,
            num_heads_pair,
        )
        self.tri_att_end = TriangleAttentionEnding(
            d_pair,
            d_hid_pair_att,
            num_heads_pair,
        )

        self.pair_transition = Transition(
            d_in=d_pair,
            n=transition_n,
        )

        self.row_dropout_share_dim = -3
        self.col_dropout_share_dim = -2
        self.msa_dropout = msa_dropout
        self.pair_dropout = pair_dropout

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        msa_row_attn_mask: torch.Tensor,
        msa_col_attn_mask: torch.Tensor,
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if scg.get_bp_world_size() > 1:


            assert self.outer_product_mean_pos == 'end', "Branch Parallellism only support outer_product_mean_pos == 'end'"

            # Note(GuoxiaWang): add zeros trigger the status of stop_gradient=False within recompute context.
            z = z + torch.zeros_like(z)
            m = m + torch.zeros_like(m)

            # # Note(GuoxiaWang): reduce the pair_act's gradient from msa branch and pair branch
            if z.requires_grad:
                z.register_hook(bp.all_reduce)

            if scg.get_bp_rank_in_group() == 0:
                m = bias_dropout_residual(
                    self.msa_att_row,
                    m,
                    self.msa_att_row(
                        m, z=z, attn_mask=msa_row_attn_mask, chunk_size=chunk_size
                    ),
                    self.row_dropout_share_dim,
                    self.msa_dropout,
                    self.training,
                )
                if self._is_extra_msa_stack:
                    m = residual(
                        m, self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size),
                        self.training
                    )
                else:
                    m = bias_dropout_residual(
                        self.msa_att_col,
                        m,
                        self.msa_att_col(m, attn_mask=msa_col_attn_mask, chunk_size=chunk_size),
                        self.col_dropout_share_dim,
                        self.msa_dropout,
                        self.training,
                    )
                m = residual(
                    m, self.msa_transition(m, chunk_size=chunk_size),
                    self.training,
                )
                outer = self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size)

            if scg.get_bp_rank_in_group() == 1:

                z = tri_mul_residual(
                    self.tri_mul_out,
                    z,
                    self.tri_mul_out(z, mask=pair_mask, block_size=block_size),
                    self.row_dropout_share_dim,
                    self.pair_dropout,
                    self.training,
                    block_size=block_size,
                )

                z = tri_mul_residual(
                    self.tri_mul_in,
                    z,
                    self.tri_mul_in(z, mask=pair_mask, block_size=block_size),
                    self.row_dropout_share_dim,
                    self.pair_dropout,
                    self.training,
                    block_size=block_size,
                )

                z = bias_dropout_residual(
                    self.tri_att_start,
                    z,
                    self.tri_att_start(z, attn_mask=tri_start_attn_mask, chunk_size=chunk_size),
                    self.row_dropout_share_dim,
                    self.pair_dropout,
                    self.training,
                )

                z = bias_dropout_residual(
                    self.tri_att_end,
                    z,
                    self.tri_att_end(z, attn_mask=tri_end_attn_mask, chunk_size=chunk_size),
                    self.col_dropout_share_dim,
                    self.pair_dropout,
                    self.training,
                )
                z = residual(
                    z, self.pair_transition(z, chunk_size=chunk_size),
                    self.training,
                )
                outer = torch.zeros_like(z)
                outer.requires_grad = z.requires_grad

            # Note(GuoxiaWang): z = residual(z, outer) in sync_evoformer_results
            m, z = bp.sync_evoformer_results(outer, m, z, self.training)
            z = z.clone()
            m = m.clone()

        else:

            if self.outer_product_mean_pos == 'first':
                z = residual(
                    z, self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size),
                    self.training
                )

            m = bias_dropout_residual(
                self.msa_att_row,
                m,
                self.msa_att_row(
                    m, z=z, attn_mask=msa_row_attn_mask, chunk_size=chunk_size
                ),
                self.row_dropout_share_dim,
                self.msa_dropout,
                self.training,
            )
            if self._is_extra_msa_stack:
                m = residual(
                    m, self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size),
                    self.training
                )
            else:
                m = bias_dropout_residual(
                    self.msa_att_col,
                    m,
                    self.msa_att_col(m, attn_mask=msa_col_attn_mask, chunk_size=chunk_size),
                    self.col_dropout_share_dim,
                    self.msa_dropout,
                    self.training,
                )
            m = residual(
                m, self.msa_transition(m, chunk_size=chunk_size),
                self.training
            )
            if self.outer_product_mean_pos == 'middle' or self.outer_product_mean_pos == 'end':
                outer = self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size)

            if self.outer_product_mean_pos == 'middle':
                z = residual(z, outer, self.training)

            z = tri_mul_residual(
                self.tri_mul_out,
                z,
                self.tri_mul_out(z, mask=pair_mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.pair_dropout,
                self.training,
                block_size=block_size,
            )

            z = tri_mul_residual(
                self.tri_mul_in,
                z,
                self.tri_mul_in(z, mask=pair_mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.pair_dropout,
                self.training,
                block_size=block_size,
            )

            z = bias_dropout_residual(
                self.tri_att_start,
                z,
                self.tri_att_start(z, attn_mask=tri_start_attn_mask, chunk_size=chunk_size),
                self.row_dropout_share_dim,
                self.pair_dropout,
                self.training,
            )

            z = bias_dropout_residual(
                self.tri_att_end,
                z,
                self.tri_att_end(z, attn_mask=tri_end_attn_mask, chunk_size=chunk_size),
                self.col_dropout_share_dim,
                self.pair_dropout,
                self.training,
            )

            z = residual(
                z, self.pair_transition(z, chunk_size=chunk_size),
                self.training
            )

            if self.outer_product_mean_pos == 'end':
                z = residual(z, outer)

        return m, z


class EvoformerStack(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hid_msa_att: int,
        d_hid_opm: int,
        d_hid_mul: int,
        d_hid_pair_att: int,
        d_single: int,
        num_heads_msa: int,
        num_heads_pair: int,
        num_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        outer_product_mean_pos: bool,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
        **kwargs,
    ):
        super(EvoformerStack, self).__init__()

        self._is_extra_msa_stack = _is_extra_msa_stack

        self.blocks = SimpleModuleList()

        for _ in range(num_blocks):
            self.blocks.append(
                EvoformerIteration(
                    d_msa=d_msa,
                    d_pair=d_pair,
                    d_hid_msa_att=d_hid_msa_att,
                    d_hid_opm=d_hid_opm,
                    d_hid_mul=d_hid_mul,
                    d_hid_pair_att=d_hid_pair_att,
                    num_heads_msa=num_heads_msa,
                    num_heads_pair=num_heads_pair,
                    transition_n=transition_n,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    outer_product_mean_pos=outer_product_mean_pos,
                    inf=inf,
                    eps=eps,
                    _is_extra_msa_stack=_is_extra_msa_stack,
                )
            )
        if not self._is_extra_msa_stack:
            self.linear = Linear(d_msa, d_single)
        else:
            self.linear = None

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        msa_row_attn_mask: torch.Tensor,
        msa_col_attn_mask: torch.Tensor,
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        chunk_size: int,
        block_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                msa_row_attn_mask=msa_row_attn_mask,
                msa_col_attn_mask=msa_col_attn_mask,
                tri_start_attn_mask=tri_start_attn_mask,
                tri_end_attn_mask=tri_end_attn_mask,
                chunk_size=chunk_size,
                block_size=block_size
            )
            for b in self.blocks
        ]

        m, z = checkpoint_sequential(
            blocks,
            input=(m, z),
        )

        s = None
        if not self._is_extra_msa_stack:
            seq_dim = -3
            index = torch.tensor([0], device=m.device)
            s = self.linear(torch.index_select(m, dim=seq_dim, index=index))
            s = s.squeeze(seq_dim)

        return m, z, s


class ExtraMSAStack(EvoformerStack):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hid_msa_att: int,
        d_hid_opm: int,
        d_hid_mul: int,
        d_hid_pair_att: int,
        num_heads_msa: int,
        num_heads_pair: int,
        num_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        outer_product_mean_pos: bool,
        inf: float,
        eps: float,
        **kwargs,
    ):
        super(ExtraMSAStack, self).__init__(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hid_msa_att=d_hid_msa_att,
            d_hid_opm=d_hid_opm,
            d_hid_mul=d_hid_mul,
            d_hid_pair_att=d_hid_pair_att,
            d_single=None,
            num_heads_msa=num_heads_msa,
            num_heads_pair=num_heads_pair,
            num_blocks=num_blocks,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            outer_product_mean_pos=outer_product_mean_pos,
            inf=inf,
            eps=eps,
            _is_extra_msa_stack=True,
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        msa_row_attn_mask: torch.Tensor = None,
        msa_col_attn_mask: torch.Tensor = None,
        tri_start_attn_mask: torch.Tensor = None,
        tri_end_attn_mask: torch.Tensor = None,
        chunk_size: int = None,
        block_size: int = None,
    ) -> torch.Tensor:
        _, z, _ = super().forward(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            msa_row_attn_mask=msa_row_attn_mask,
            msa_col_attn_mask=msa_col_attn_mask,
            tri_start_attn_mask=tri_start_attn_mask,
            tri_end_attn_mask=tri_end_attn_mask,
            chunk_size=chunk_size,
            block_size=block_size
        )
        return z
