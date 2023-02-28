from unifold.modules.evoformer import *


class EvoformerIterationSingle(EvoformerIteration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_att_col = None

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        msa_row_attn_mask: torch.Tensor,
        msa_col_attn_mask: Optional[torch.Tensor],
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.outer_product_mean_first:
            z = residual(
                z,
                self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size),
                self.training,
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
        # if self._is_extra_msa_stack:
        #     m = residual(
        #         m, self.msa_att_col(m, mask=msa_mask, chunk_size=chunk_size),
        #         self.training
        #     )
        # else:
        #     m = bias_dropout_residual(
        #         self.msa_att_col,
        #         m,
        #         self.msa_att_col(m, attn_mask=msa_col_attn_mask, chunk_size=chunk_size),
        #         self.col_dropout_share_dim,
        #         self.msa_dropout,
        #         self.training,
        #     )
        m = residual(m, self.msa_transition(m, chunk_size=chunk_size), self.training)
        if not self.outer_product_mean_first:
            z = residual(
                z,
                self.outer_product_mean(m, mask=msa_mask, chunk_size=chunk_size),
                self.training,
            )

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
        z = residual(z, self.pair_transition(z, chunk_size=chunk_size), self.training)
        return m, z


class EvoformerStackSingle(EvoformerStack):
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
        outer_product_mean_first: bool,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
        **kwargs
    ):
        super().__init__(
            d_msa,
            d_pair,
            d_hid_msa_att,
            d_hid_opm,
            d_hid_mul,
            d_hid_pair_att,
            d_single,
            num_heads_msa,
            num_heads_pair,
            num_blocks,
            transition_n,
            msa_dropout,
            pair_dropout,
            outer_product_mean_first,
            inf,
            eps,
            _is_extra_msa_stack,
            **kwargs
        )
        self.blocks = SimpleModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                EvoformerIterationSingle(
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
                    outer_product_mean_first=outer_product_mean_first,
                    inf=inf,
                    eps=eps,
                    _is_extra_msa_stack=_is_extra_msa_stack,
                )
            )
