import torch
import torch.nn as nn
import torch.nn.functional as F

from unifold.modules.structure_module import *
from ..modules.common import Linear
from ..modules.embedders import InputEmbedder

from typing import *

import torch
import torch.nn as nn


class PseudoResidueResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(PseudoResidueResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(self.c_hidden, self.c_hidden)
        self.act = nn.GELU()
        self.linear_2 = Linear(self.c_hidden, self.c_hidden)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_0 = x

        x = self.act(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return x + x_0


class PseudoResidueEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        num_blocks: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(PseudoResidueEmbedder, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.num_blocks = num_blocks

        self.linear_in = Linear(self.d_in, self.d_hidden)
        self.act = nn.GELU()

        self.layers = nn.ModuleList()
        for _ in range(self.num_blocks):
            layer = PseudoResidueResnetBlock(c_hidden=self.d_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.d_hidden, self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] pseudo residue feature
        Returns:
            [*, C_out] embedding
        """
        x = x.type(self.linear_in.weight.dtype)

        x = self.linear_in(x)
        x = self.act(x)
        for l in self.layers:
            x = l(x)
        x = self.linear_out(x)

        return x


class SymmInputEmbedder(InputEmbedder):
    def __init__(
        self,
        pr_dim: Optional[int] = None,
        **kwargs,
    ):
        super(SymmInputEmbedder, self).__init__(**kwargs)
        d_pair = kwargs.get("d_pair")
        d_msa = kwargs.get("d_msa")
        self.pr_dim = pr_dim
        self.linear_pr_z_i = Linear(pr_dim, d_pair)
        self.linear_pr_z_j = Linear(pr_dim, d_pair)
        self.linear_pr_m = Linear(pr_dim, d_msa)
        
    def forward(
        self,
        tf: torch.Tensor,
        msa: torch.Tensor,
        prf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [*, N_res, c_z]
        if self.tf_dim == 21:
            # multimer use 21 target dim
            tf = tf[...,1:]
        # convert type if necessary
        tf = tf.type(self.linear_tf_z_i.weight.dtype)
        msa = msa.type(self.linear_tf_z_i.weight.dtype)
        tf_emb_i = self.linear_tf_z_i(tf)   # [*, N_res, c_z]
        tf_emb_j = self.linear_tf_z_j(tf)

        pr_emb_i = self.linear_pr_z_i(prf)  # [*, c_z]
        pr_emb_j = self.linear_pr_z_j(prf)

        tf_emb_i = torch.cat([pr_emb_i[..., None, :], tf_emb_i], dim=-2)
        tf_emb_j = torch.cat([pr_emb_j[..., None, :], tf_emb_j], dim=-2)

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., :, None, :] + tf_emb_j[..., None, :, :]

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        pr_m = self.linear_pr_m(prf)[..., None, None, :]
        pr_m_expand = pr_m.expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
        msa_emb = torch.cat([pr_m_expand, msa_emb], dim=-2)
        return msa_emb, pair_emb, pr_m


class SymmStructureModule(StructureModule):
     def forward(
        self,
        s,
        z,
        aatype,
        mask=None,
    ):
        if mask is None:
            mask = s.new_ones(s.shape[:-1])
        mask = F.pad(mask, (1, 0), "constant", 1.)

        # generate square mask
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = gen_attn_mask(square_mask, -self.inf).unsqueeze(-3)
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        initial_s = s
        s = self.linear_in(s)

        quat_encoder = Quaternion.identity(
            s.shape[:-1],
            s.dtype,
            s.device,
            requires_grad=False,
        )
        backb_to_global = Frame(
            Rotation(
                mat=quat_encoder.get_rot_mats(),
            ),
            quat_encoder.get_trans(),
        )
        outputs = []
        for i in range(self.num_blocks):
            s = residual(s, self.ipa(s, z, backb_to_global, square_mask), self.training)
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # update quaternion encoder
            # use backb_to_global to avoid quat-to-rot conversion
            quat_encoder = quat_encoder.compose_update_vec(
                self.bb_update(s), pre_rot_mat=backb_to_global.get_rots()
            )

            # initial_s is always used to update the backbone
            unnormalized_angles, angles = self.angle_resnet(s[..., 1:, :], initial_s[..., 1:, :])

            # convert quaternion to rotation matrix
            backb_to_global = Frame(
                Rotation(
                    mat=quat_encoder.get_rot_mats(),
                ),
                quat_encoder.get_trans(),
            )

            global_frame = backb_to_global[..., 0:1]
            local_frames = backb_to_global[..., 1:]
            local_frames = global_frame.compose(local_frames)

            preds = {
                "frames": local_frames.scale_translation(
                    self.trans_scale_factor
                ).to_tensor_4x4(),  # no pr
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
            }

            outputs.append(preds)
            if i < (self.num_blocks - 1):
                # stop gradient in iteration
                quat_encoder = quat_encoder.stop_rot_gradient()
                backb_to_global = backb_to_global.stop_rot_gradient()
            else:
                all_frames_to_global = self.torsion_angles_to_frames(
                    local_frames.scale_translation(self.trans_scale_factor),
                    angles,
                    aatype,
                ) # no pr
                pred_positions = self.frames_and_literature_positions_to_atom14_pos(
                    all_frames_to_global,
                    aatype,
                ) # no pr
        
        outputs = dict_multimap(torch.stack, outputs)
        outputs["sidechain_frames"] = all_frames_to_global.to_tensor_4x4()
        outputs["positions"] = pred_positions
        outputs["single"] = s[..., 1:, :]
        outputs["global_center_position"] = global_frame.get_trans()

        return outputs