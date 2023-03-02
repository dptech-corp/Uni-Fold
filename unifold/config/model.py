from chanfig import Config

from .variables import aux_distogram_bins, d_extra_msa, d_msa, d_pair, d_single, d_template, is_multimer, use_templates


class ModelConfig(Config):
    def __init__(self, *args, **kwargs):
        self.is_multimer = is_multimer
        self.input_embedder = Config(
            tf_dim=22,
            msa_dim=49,
            d_pair=d_pair,
            d_msa=d_msa,
            relpos_k=32,
            max_relative_chain=2,
        )
        self.recycling_embedder = Config(
            d_pair=d_pair,
            d_msa=d_msa,
            min_bin=3.25,
            max_bin=20.75,
            num_bins=15,
            inf=1e8,
        )
        self.template = TemplateConfig()
        self.extra_msa = ExtraMSAConfig()
        self.evoformer_stack = Config(
            d_msa=d_msa,
            d_pair=d_pair,
            d_hid_msa_att=32,
            d_hid_opm=32,
            d_hid_mul=128,
            d_hid_pair_att=32,
            d_single=d_single,
            num_heads_msa=8,
            num_heads_pair=4,
            num_blocks=48,
            transition_n=4,
            msa_dropout=0.15,
            pair_dropout=0.25,
            inf=1e9,
            eps=1e-10,
            outer_product_mean_first=False,
        )
        self.structure_module = Config(
            d_single=d_single,
            d_pair=d_pair,
            d_ipa=16,
            d_angle=128,
            num_heads_ipa=12,
            num_qk_points=4,
            num_v_points=8,
            dropout_rate=0.1,
            num_blocks=8,
            no_transition_layers=1,
            num_resnet_blocks=2,
            num_angles=7,
            trans_scale_factor=10,
            epsilon=1e-12,
            inf=1e5,
            separate_kv=False,
            ipa_bias=True,
        )
        self.heads = HeadsConfig()
        super().__init__(*args, **kwargs)


class TemplateConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distogram = Config(
            min_bin=3.25,
            max_bin=50.75,
            num_bins=39,
        )
        self.template_angle_embedder = Config(
            d_in=57,
            d_out=d_msa,
        )
        self.template_pair_embedder = Config(
            d_in=88,
            v2_d_in=[39, 1, 22, 22, 1, 1, 1, 1],
            d_pair=d_pair,
            d_out=d_template,
            v2_feature=False,
        )
        self.template_pair_stack = Config(
            d_template=d_template,
            d_hid_tri_att=16,
            d_hid_tri_mul=64,
            num_blocks=2,
            num_heads=4,
            pair_transition_n=2,
            dropout_rate=0.25,
            inf=1e9,
            tri_attn_first=True,
        )
        self.template_pointwise_attention = Config(
            enabled=True,
            d_template=d_template,
            d_pair=d_pair,
            d_hid=16,
            num_heads=4,
            inf=1e5,
        )
        self.inf = 1e5
        self.eps = 1e-6
        self.enabled = use_templates
        self.embed_angles = use_templates


class ExtraMSAConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_msa_embedder = Config(
            d_in=25,
            d_out=d_extra_msa,
        )
        self.extra_msa_stack = Config(
            d_msa=d_extra_msa,
            d_pair=d_pair,
            d_hid_msa_att=8,
            d_hid_opm=32,
            d_hid_mul=128,
            d_hid_pair_att=32,
            num_heads_msa=8,
            num_heads_pair=4,
            num_blocks=4,
            transition_n=4,
            msa_dropout=0.15,
            pair_dropout=0.25,
            inf=1e9,
            eps=1e-10,
            outer_product_mean_first=False,
        )
        self.enabled = True


class HeadsConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plddt = PLDDTHeadConfig()
        self.distogram = DistogramHeadConfig()
        self.pae = PAEHeadConfig()
        self.masked_msa = MaskedMSAtHeadConfig()
        self.experimentally_resolved = ExperimentallyResolvedHeadConfig()


class PLDDTHeadConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bins = 50
        self.d_in = d_single
        self.d_hid = 128


class DistogramHeadConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_pair = d_pair
        self.num_bins = aux_distogram_bins
        self.disable_enhance_head = False


class PAEHeadConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_pair = d_pair
        self.num_bins = aux_distogram_bins
        self.enabled = False
        self.iptm_weight = 0.8
        self.disable_enhance_head = False


class MaskedMSAtHeadConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_msa = d_msa
        self.d_out = 23
        self.disable_enhance_head = False


class ExperimentallyResolvedHeadConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_single = d_single
        self.d_out = 37
        self.enabled = False
        self.disable_enhance_head = False
