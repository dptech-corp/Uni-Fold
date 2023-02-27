import torch
import torch.nn as nn
from typing import Optional, Tuple

from unicore.utils import one_hot

from .common import Linear
from .common import SimpleModuleList
from unicore.modules import LayerNorm


class InputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        d_pair: int,
        d_msa: int,
        relpos_k: int,
        use_chain_relative: bool = False,
        max_relative_chain: Optional[int] = None,
        **kwargs,
    ):
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.d_pair = d_pair
        self.d_msa = d_msa

        self.linear_tf_z_i = Linear(tf_dim, d_pair)
        self.linear_tf_z_j = Linear(tf_dim, d_pair)
        self.linear_tf_m = Linear(tf_dim, d_msa)
        self.linear_msa_m = Linear(msa_dim, d_msa)

        # RPE stuff
        self.relpos_k = relpos_k
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if not self.use_chain_relative:
            self.num_bins = 2 * self.relpos_k + 1
        else:
            self.num_bins = 2 * self.relpos_k + 2
            self.num_bins += 1  # entity id
            self.num_bins += 2 * max_relative_chain + 2

        self.linear_relpos = Linear(self.num_bins, d_pair)

    def _relpos_indices(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ):

        max_rel_res = self.relpos_k
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res
        if not self.use_chain_relative:
            return rp
        else:
            asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
            rp[~asym_id_same] = 2 * max_rel_res + 1
            entity_id_same = entity_id[..., :, None] == entity_id[..., None, :]
            rp_entity_id = entity_id_same.type(rp.dtype)[..., None]

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain

            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
            )

            clipped_rel_chain[~entity_id_same] = 2 * max_rel_chain + 1
            return rp, rp_entity_id, clipped_rel_chain

    def relpos_emb(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
        num_sym: Optional[torch.Tensor] = None,
    ):

        dtype = self.linear_relpos.weight.dtype
        if not self.use_chain_relative:
            rp = self._relpos_indices(res_id=res_id)
            return self.linear_relpos(
                one_hot(rp, num_classes=self.num_bins, dtype=dtype)
            )
        else:
            rp, rp_entity_id, rp_rel_chain = self._relpos_indices(
                res_id=res_id, sym_id=sym_id, asym_id=asym_id, entity_id=entity_id
            )
            rp = one_hot(rp, num_classes=(2 * self.relpos_k + 2), dtype=dtype)
            rp_entity_id = rp_entity_id.type(dtype)
            rp_rel_chain = one_hot(
                rp_rel_chain, num_classes=(2 * self.max_relative_chain + 2), dtype=dtype
            )
            return self.linear_relpos(
                torch.cat([rp, rp_entity_id, rp_rel_chain], dim=-1)
            )

    def forward(
        self,
        tf: torch.Tensor,
        msa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [*, N_res, d_pair]
        if self.tf_dim == 21:
            # multimer use 21 target dim
            tf = tf[..., 1:]
        # convert type if necessary
        tf = tf.type(self.linear_tf_z_i.weight.dtype)
        msa = msa.type(self.linear_tf_z_i.weight.dtype)
        n_clust = msa.shape[-3]

        msa_emb = self.linear_msa_m(msa)
        # target_feat (aatype) into msa representation
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))  # expand -3 dim
        )
        msa_emb += tf_m

        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]

        return msa_emb, pair_emb


class RecyclingEmbedder(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        min_bin: float,
        max_bin: float,
        num_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        super(RecyclingEmbedder, self).__init__()

        self.d_msa = d_msa
        self.d_pair = d_pair
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins
        self.inf = inf

        self.squared_bins = None

        self.linear = Linear(self.num_bins, self.d_pair)
        self.layer_norm_m = LayerNorm(self.d_msa)
        self.layer_norm_z = LayerNorm(self.d_pair)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m_update = self.layer_norm_m(m)
        z_update = self.layer_norm_z(z)

        return m_update, z_update

    def recyle_pos(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.squared_bins is None:
            bins = torch.linspace(
                self.min_bin,
                self.max_bin,
                self.num_bins,
                dtype=torch.float,
                device=x.device,
                requires_grad=False,
            )
            self.squared_bins = bins**2
        upper = torch.cat(
            [self.squared_bins[1:], self.squared_bins.new_tensor([self.inf])], dim=-1
        )
        xf = x.float()
        d = torch.sum(
            (xf[..., None, :] - xf[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )
        d = ((d > self.squared_bins) * (d < upper)).type(self.linear.weight.dtype)
        d = self.linear(d)
        return d


class TemplateAngleEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        **kwargs,
    ):
        super(TemplateAngleEmbedder, self).__init__()

        self.d_out = d_out
        self.d_in = d_in

        self.linear_1 = Linear(self.d_in, self.d_out, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.d_out, self.d_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x.type(self.linear_1.weight.dtype))
        x = self.act(x)
        x = self.linear_2(x)
        return x


class TemplatePairEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        v2_d_in: list,
        d_out: int,
        d_pair: int,
        v2_feature: bool = False,
        **kwargs,
    ):
        super(TemplatePairEmbedder, self).__init__()

        self.d_out = d_out
        self.v2_feature = v2_feature
        if self.v2_feature:
            self.d_in = v2_d_in
            self.linear = SimpleModuleList()
            for d_in in self.d_in:
                self.linear.append(Linear(d_in, self.d_out, init="relu"))
            self.z_layer_norm = LayerNorm(d_pair)
            self.z_linear = Linear(d_pair, self.d_out, init="relu")
        else:
            self.d_in = d_in
            self.linear = Linear(self.d_in, self.d_out, init="relu")

    def forward(
        self,
        x,
        z,
    ) -> torch.Tensor:
        if not self.v2_feature:
            x = self.linear(x.type(self.linear.weight.dtype))
        else:
            t = 0
            for i, s in enumerate(x):
                dtype = self.z_linear.weight.dtype
                t = t + self.linear[i](s.type(dtype))
            t = t + self.z_linear(self.z_layer_norm(z))
            x = t
        return x


class ExtraMSAEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        **kwargs,
    ):
        super(ExtraMSAEmbedder, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.linear = Linear(self.d_in, self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.type(self.linear.weight.dtype))
