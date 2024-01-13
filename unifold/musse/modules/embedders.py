from unifold.modules.embedders import *


class InputEmbedderSingle(InputEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_msa_m = None

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
        # msa = msa.type(self.linear_tf_z_i.weight.dtype)
        # n_clust = msa.shape[-3]
        n_clust = 1

        # msa_emb = self.linear_msa_m(msa)
        # target_feat (aatype) into msa representation
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))  # expand -3 dim
        )
        msa_emb = tf_m

        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]

        return msa_emb, pair_emb


class Esm2Embedder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        d_msa: int,
        d_pair: int,
        dropout: float,
        **kwargs,
    ):
        super(Esm2Embedder, self).__init__()
        
            
        self.linear_token = Linear(token_dim, d_msa)
        # self.linear_pair = Linear(pair_dim, d_pair)
        self.combine = nn.Parameter(torch.tensor([0.0, 2.3]))
        self.dropout = dropout

    def forward(
        self,
        m,
        z,
        token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_shape = token.shape[:-2]
        token = token[..., None, :, :, :]
        token = token.type(self.linear_token.weight.dtype)
        token = torch.einsum(
            "...nh,n->...h",
            token,
            nn.functional.softmax(self.combine.float(), dim=0).type(
                self.linear_token.weight.dtype
            ),
        )
        # pair = pair.type(self.linear_pair.weight.dtype)

        with torch.no_grad():
            token_mask = (
                torch.rand(mask_shape, dtype=token.dtype, device=token.device)
                >= self.dropout
            ).type(token.dtype)

            # pair_mask = token_mask[..., None, :] * token_mask[..., None]
            token_mask = token_mask[..., None, :, None]  # / (1.0 - self.dropout)
            # pair_mask = pair_mask[..., None]  # / (1.0 - self.dropout)

        token = token * token_mask
        # pair = pair * pair_mask
        m = residual(m, self.linear_token(token), self.training)
        # z = residual(z, self.linear_pair(pair), self.training)
        return m, z
       
