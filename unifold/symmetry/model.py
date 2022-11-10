import torch.nn.functional as F
from ..modules.alphafold import *
from .modules import SymmInputEmbedder, PseudoResidueEmbedder, SymmStructureModule
from .assemble import expand_symmetry


class UFSymmetry(AlphaFold):    # inherit the main model. alterations implemented here.
    def __init__(self, config):
        assert not config.globals.alphafold_original_mode
        super(UFSymmetry, self).__init__(config)
        # replace input embedder with symm input embedder
        self.input_embedder = SymmInputEmbedder(**config.model["input_embedder"], use_chain_relative=True)
        self.pseudo_residue_embedder = PseudoResidueEmbedder(**config.model["pseudo_residue_embedder"])
        self.structure_module = SymmStructureModule(
            **config.model["structure_module"],
        )

    def __make_input_float__(self):
        super().__make_input_float__()
        self.pseudo_residue_embedder = self.pseudo_residue_embedder.float()
    
    def iteration_evoformer(self, feats, m_1_prev, z_prev, x_prev):
        batch_dims = feats["target_feat"].shape[:-2]
        n = feats["target_feat"].shape[-2] + 1 # pr
        seq_mask = feats["seq_mask"]
        msa_mask = feats["msa_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

        pr_feat = feats["pseudo_residue_feat"]
        pr_feat = self.pseudo_residue_embedder(pr_feat)

        m, z, pr_m = self.input_embedder(
            feats["target_feat"],
            feats["msa_feat"],
            pr_feat
        )

        if m_1_prev is None:
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.d_msa),
                requires_grad=False,
            )
        if z_prev is None:
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.d_pair),
                requires_grad=False,
            )
        if x_prev is None:
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )
        
        x_prev_ = pseudo_beta_fn(feats["aatype"], x_prev[..., 1:, :, :], None)
        x_prev = torch.cat([x_prev[..., 0:1, 0, :], x_prev_], dim=-2)

        z += self.recycling_embedder.recyle_pos(x_prev)

        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
        )

        m[..., 0, :, :] += m_1_prev_emb

        z += z_prev_emb

        relpos = self.input_embedder.relpos_emb(
            feats["residue_index"].long(),
            feats.get("sym_id", None),
            feats.get("asym_id", None),
            feats.get("entity_id", None),
            feats.get("num_sym", None),
        )

        z[..., 1:, 1:, :] += relpos

        m = m.type(self.dtype)
        z = z.type(self.dtype)

        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(pair_mask, self.inf)

        # m_in = m[..., 1:, :]
        z_in = z[..., 1:, 1:, :]

        if self.config.template.enabled:
            template_mask = feats["template_mask"]
            if torch.any(template_mask):
                z_in = residual(
                    z_in,
                    self.embed_templates_pair(
                        feats,
                        z_in,
                        pair_mask,
                        tri_start_attn_mask,
                        tri_end_attn_mask,
                        templ_dim=-4,
                    ),
                    self.training,
                )

        if self.config.extra_msa.enabled:
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))
            extra_msa_row_mask = gen_msa_attn_mask(
                feats["extra_msa_mask"],
                inf=self.inf,
                gen_col_mask=False,
            )
            z_in = self.extra_msa_stack(
                a,
                z_in,
                msa_mask=feats["extra_msa_mask"],
                chunk_size=self.globals.chunk_size,
                block_size=self.globals.block_size,
                pair_mask=pair_mask,
                msa_row_attn_mask=extra_msa_row_mask,
                msa_col_attn_mask=None,
                tri_start_attn_mask=tri_start_attn_mask,
                tri_end_attn_mask=tri_end_attn_mask,
            )

        if self.config.template.embed_angles:
            template_1d_feat, template_1d_mask = self.embed_templates_angle(feats)
            expand_shape = list(pr_m.shape)
            expand_shape[-3] = template_1d_feat.shape[-3]
            template_1d_feat = torch.cat([pr_m.expand(expand_shape), template_1d_feat], dim=-2)
            msa_mask = torch.cat([feats["msa_mask"], template_1d_mask], dim=-2)

        # compose tensors back
        m = torch.cat([m, template_1d_feat], dim=-3)
        z_tmp = torch.cat([z[..., 1:, 0:1, :], z_in], dim=-2)
        z = torch.cat([z[..., 0:1, :, :], z_tmp], dim=-3)

        # pad pr mask
        pad_fn = lambda msk: F.pad(msk, (1, 0), "constant", 1.)
        seq_mask = pad_fn(seq_mask)
        msa_mask = pad_fn(msa_mask)
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]

        msa_row_mask, msa_col_mask = gen_msa_attn_mask(
            msa_mask,
            inf=self.inf,
        )
        tri_start_attn_mask, tri_end_attn_mask = gen_tri_attn_mask(pair_mask, self.inf)

        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            msa_row_attn_mask=msa_row_mask,
            msa_col_attn_mask=msa_col_mask,
            tri_start_attn_mask=tri_start_attn_mask,
            tri_end_attn_mask=tri_end_attn_mask,
            chunk_size=self.globals.chunk_size,
            block_size=self.globals.block_size,
        )
        return m, z, s, msa_mask, pair_mask, m_1_prev_emb, z_prev_emb

    def iteration_evoformer_structure_module(
        self, batch, m_1_prev, z_prev, x_prev, cycle_no, num_recycling, num_ensembles=1
    ):
        z, s = 0, 0
        n_seq = batch["msa_feat"].shape[-3]
        assert num_ensembles >= 1
        for ensemble_no in range(num_ensembles):
            idx = cycle_no * num_ensembles + ensemble_no
            fetch_cur_batch = lambda t: t[min(t.shape[0] - 1, idx), ...]
            feats = tensor_tree_map(fetch_cur_batch, batch)
            m, z0, s0, msa_mask, pair_mask, m_1_prev_emb, z_prev_emb = self.iteration_evoformer(
                feats, m_1_prev, z_prev, x_prev
            )
            z += z0
            s += s0
            del z0, s0
        if num_ensembles > 1:
            z /= float(num_ensembles)
            s /= float(num_ensembles)

        outputs = {}

        outputs["msa"] = m[..., :n_seq, 1:, :]
        outputs["pair"] = z[..., 1:, 1:, :]
        outputs["single"] = s[..., 1:, :]

        # norm loss
        if (not getattr(self, "inference", False)) and num_recycling == (cycle_no + 1):
            delta_msa = m
            delta_msa[..., 0, :, :] = delta_msa[..., 0, :, :] - m_1_prev_emb.detach()
            delta_pair = z - z_prev_emb.detach()
            outputs["delta_msa"] = delta_msa
            outputs["delta_pair"] = delta_pair
            outputs["msa_norm_mask"] = msa_mask
            outputs["pair_norm_mask"] = pair_mask

        outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"],
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["pred_frame_tensor"] = outputs["sm"]["frames"][-1]

        global_center = outputs['sm']['global_center_position'][..., None, :]
        global_center = global_center.repeat(*((1,) * len(global_center.shape[:-2])), 37, 1)
        x_prev = torch.cat([global_center, outputs["final_atom_positions"]], dim=-3)

        # use float32 for numerical stability
        if (not getattr(self, "inference", False)):
            m_1_prev = m[..., 0, :, :].float()
            z_prev = z.float()
            x_prev = x_prev.float()
        else:
            m_1_prev = m[..., 0, :, :]
            z_prev = z
            x_prev = x_prev

        return outputs, m_1_prev, z_prev, x_prev


    def forward(self, batch, expand=True):

        m_1_prev = batch.get("m_1_prev", None)
        z_prev = batch.get("z_prev", None)
        x_prev = batch.get("x_prev", None)

        is_grad_enabled = torch.is_grad_enabled()

        num_iters = int(batch["num_recycling_iters"]) + 1
        num_ensembles = int(batch["msa_mask"].shape[0]) // num_iters
        if self.training:
            # don't use ensemble during training
            assert num_ensembles == 1

        # convert dtypes in batch
        batch = self.__convert_input_dtype__(batch)
        for cycle_no in range(num_iters):
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                (
                    outputs,
                    m_1_prev,
                    z_prev,
                    x_prev,
                ) = self.iteration_evoformer_structure_module(
                    batch,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    cycle_no=cycle_no,
                    num_recycling=num_iters,
                    num_ensembles=num_ensembles,
                )
            if not is_final_iter:
                del outputs
        
        if expand:
            symmetry_feat, symmetry_output = expand_symmetry(outputs["sm"], batch)
            outputs["expand_batch"] = symmetry_feat
            outputs["expand_sm"] = symmetry_output
            outputs["expand_final_atom_positions"] = symmetry_output["expand_final_atom_positions"]
            outputs["expand_final_atom_mask"] = symmetry_output["expand_final_atom_mask"]
        
        if "asym_id" in batch:
            outputs["asym_id"] = batch["asym_id"][0, ...]
        outputs.update(self.aux_heads(outputs))
        return outputs
