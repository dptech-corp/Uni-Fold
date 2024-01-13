from chanfig import Config

N_RES = "number of residues"
N_MSA = "number of MSA sequences"
N_EXTRA_MSA = "number of extra MSA sequences"
N_TPL = "number of templates"

from .variables import is_multimer, max_recycling_iters, use_templates


class DataConfig(Config):
    def __init__(self, *args, **kwargs):
        self.common = CommonDataConfig()
        self.supervised = SupervisedDataConfig()
        self.train = TrainDataConfig()
        self.eval = EvalDataConfig()
        self.predict = PredictDataConfig()
        super().__init__(*args, **kwargs)


class CommonDataConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = FeaturesConfig()
        self.masked_msa = Config(
            profile_prob=0.1,
            same_prob=0.1,
            uniform_prob=0.1,
        )
        self.block_delete_msa = Config(
            msa_fraction_per_block=0.3,
            randomize_num_blocks=False,
            num_blocks=5,
            min_num_msa=16,
        )
        self.random_delete_msa = Config(
            max_msa_entry=1 << 25,  # := 33554432
        )
        self.v2_feature = False
        self.gumbel_sample = False
        self.max_extra_msa = 1024
        self.msa_cluster_features = True
        self.reduce_msa_clusters_by_max_templates = True
        self.resample_msa_in_recycling = True
        self.template_features = [
            "template_all_atom_positions",
            "template_sum_probs",
            "template_aatype",
            "template_all_atom_mask",
        ]
        self.unsupervised_features = [
            "aatype",
            "residue_index",
            "msa",
            "msa_chains",
            "num_alignments",
            "seq_length",
            "between_segment_residues",
            "deletion_matrix",
            "num_recycling_iters",
            "crop_and_fix_size_seed",
        ]
        self.recycling_features = [
            "msa_chains",
            "msa_mask",
            "msa_row_mask",
            "bert_mask",
            "true_msa",
            "msa_feat",
            "extra_msa_deletion_value",
            "extra_msa_has_deletion",
            "extra_msa",
            "extra_msa_mask",
            "extra_msa_row_mask",
            "is_distillation",
        ]
        self.multimer_features = [
            "assembly_num_chains",
            "asym_id",
            "sym_id",
            "num_sym",
            "entity_id",
            "asym_len",
            "cluster_bias_mask",
        ]
        self.use_templates = use_templates
        self.is_multimer = is_multimer
        self.use_template_torsion_angles = use_templates
        self.max_recycling_iters = max_recycling_iters


class FeaturesConfig(Config):
    def __init__(self):
        self.aatype = [N_RES]
        self.all_atom_mask = [N_RES, None]
        self.all_atom_positions = [N_RES, None, None]
        self.alt_chi_angles = [N_RES, None]
        self.atom14_alt_gt_exists = [N_RES, None]
        self.atom14_alt_gt_positions = [N_RES, None, None]
        self.atom14_atom_exists = [N_RES, None]
        self.atom14_atom_is_ambiguous = [N_RES, None]
        self.atom14_gt_exists = [N_RES, None]
        self.atom14_gt_positions = [N_RES, None, None]
        self.atom37_atom_exists = [N_RES, None]
        self.frame_mask = [N_RES]
        self.true_frame_tensor = [N_RES, None, None]
        self.bert_mask = [N_MSA, N_RES]
        self.chi_angles_sin_cos = [N_RES, None, None]
        self.chi_mask = [N_RES, None]
        self.extra_msa_deletion_value = [N_EXTRA_MSA, N_RES]
        self.extra_msa_has_deletion = [N_EXTRA_MSA, N_RES]
        self.extra_msa = [N_EXTRA_MSA, N_RES]
        self.extra_msa_mask = [N_EXTRA_MSA, N_RES]
        self.extra_msa_row_mask = [N_EXTRA_MSA]
        self.is_distillation = []
        self.msa_feat = [N_MSA, N_RES, None]
        self.msa_mask = [N_MSA, N_RES]
        self.msa_chains = [N_MSA, None]
        self.msa_row_mask = [N_MSA]
        self.num_recycling_iters = []
        self.pseudo_beta = [N_RES, None]
        self.pseudo_beta_mask = [N_RES]
        self.residue_index = [N_RES]
        self.residx_atom14_to_atom37 = [N_RES, None]
        self.residx_atom37_to_atom14 = [N_RES, None]
        self.resolution = []
        self.rigidgroups_alt_gt_frames = [N_RES, None, None, None]
        self.rigidgroups_group_exists = [N_RES, None]
        self.rigidgroups_group_is_ambiguous = [N_RES, None]
        self.rigidgroups_gt_exists = [N_RES, None]
        self.rigidgroups_gt_frames = [N_RES, None, None, None]
        self.seq_length = []
        self.seq_mask = [N_RES]
        self.target_feat = [N_RES, None]
        self.template_aatype = [N_TPL, N_RES]
        self.template_all_atom_mask = [N_TPL, N_RES, None]
        self.template_all_atom_positions = [N_TPL, N_RES, None, None]
        self.template_alt_torsion_angles_sin_cos = [
            N_TPL,
            N_RES,
            None,
            None,
        ]

        self.template_frame_mask = [N_TPL, N_RES]
        self.template_frame_tensor = [N_TPL, N_RES, None, None]
        self.template_mask = [N_TPL]
        self.template_pseudo_beta = [N_TPL, N_RES, None]
        self.template_pseudo_beta_mask = [N_TPL, N_RES]
        self.template_sum_probs = [N_TPL, None]
        self.template_torsion_angles_mask = [N_TPL, N_RES, None]
        self.template_torsion_angles_sin_cos = [N_TPL, N_RES, None, None]
        self.true_msa = [N_MSA, N_RES]
        self.use_clamped_fape = []
        self.assembly_num_chains = [1]
        self.asym_id = [N_RES]
        self.sym_id = [N_RES]
        self.entity_id = [N_RES]
        self.num_sym = [N_RES]
        self.asym_len = [None]
        self.cluster_bias_mask = [N_MSA]


class SupervisedDataConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_clamped_fape_prob = 1.0
        self.supervised_features = [
            "all_atom_mask",
            "all_atom_positions",
            "resolution",
            "use_clamped_fape",
            "is_distillation",
        ]


class TrainDataConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_size = True
        self.subsample_templates = True
        self.block_delete_msa = True
        self.random_delete_msa = True
        self.masked_msa_replace_fraction = 0.15
        self.max_msa_clusters = 128
        self.max_templates = 4
        self.num_ensembles = 1
        self.crop = True
        self.crop_size = 256
        self.spatial_crop_prob = 0.5
        self.ca_ca_threshold = 10.0
        self.supervised = True
        self.use_clamped_fape_prob = 1.0
        self.max_distillation_msa_clusters = 1000
        self.biased_msa_by_chain = True
        self.share_mask = True


class EvalDataConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_size = True
        self.subsample_templates = False
        self.block_delete_msa = False
        self.random_delete_msa = True
        self.masked_msa_replace_fraction = 0.15
        self.max_msa_clusters = 128
        self.max_templates = 4
        self.num_ensembles = 1
        self.crop = False
        self.crop_size = None
        self.spatial_crop_prob = 0.5
        self.ca_ca_threshold = 10.0
        self.supervised = True
        self.biased_msa_by_chain = False
        self.share_mask = False


class PredictDataConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_size = True
        self.subsample_templates = False
        self.block_delete_msa = False
        self.random_delete_msa = True
        self.masked_msa_replace_fraction = 0.15
        self.max_msa_clusters = 128
        self.max_templates = 4
        self.num_ensembles = 2
        self.crop = False
        self.crop_size = None
        self.supervised = False
        self.biased_msa_by_chain = False
        self.share_mask = False
