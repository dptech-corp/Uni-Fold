from chanfig import Config


class LossConfig(Config):
    def __init__(self, *args, **kwargs):
        self.distogram = DistogramLossConfig()
        self.experimentally_resolved = ExperimentallyResolvedLossConfig()
        self.fape = FAPELossConfig()
        self.plddt = PLDDTLossConfig()
        self.masked_msa = MaskedMSALossConfig()
        self.supervised_chi = SupervisedChiLossConfig()
        self.violation = ViolationLossConfig()
        self.pae = PAELossConfig()
        self.repr_norm = ReprNormLossConfig()
        self.chain_centre_mass = ChainCentreMassLossConfig()
        super().__init__(*args, **kwargs)


class DistogramLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_bin = 2.3125
        self.max_bin = 21.6875
        self.num_bins = 64
        self.eps = 1e-6
        self.weight = 0.3


class ExperimentallyResolvedLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-8
        self.min_resolution = 0.1
        self.max_resolution = 3.0
        self.weight = 0.0


class FAPELossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = Config(
            clamp_distance=10.0,
            clamp_distance_between_chains=30.0,
            loss_unit_distance=10.0,
            loss_unit_distance_between_chains=20.0,
            weight=0.5,
            eps=1e-4,
        )
        self.sidechain = Config(
            clamp_distance=10.0,
            length_scale=10.0,
            weight=0.5,
            eps=1e-4,
        )
        self.weight = 1.0


class PLDDTLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_resolution = 0.1
        self.max_resolution = 3.0
        self.cutoff = 15.0
        self.num_bins = 50
        self.eps = 1e-10
        self.weight = 0.01


class MaskedMSALossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-8
        self.weight = 2.0


class SupervisedChiLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chi_weight = 0.5
        self.angle_norm_weight = 0.01
        self.eps = 1e-6
        self.weight = 1.0


class ViolationLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.violation_tolerance_factor = 12.0
        self.clash_overlap_tolerance = 1.5
        self.bond_angle_loss_weight = 0.3
        self.eps = 1e-6
        self.weight = 0.0


class PAELossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_bin = 31
        self.num_bins = 64
        self.min_resolution = 0.1
        self.max_resolution = 3.0
        self.eps = 1e-8
        self.weight = 0.0


class ReprNormLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = 0.01
        self.tolerance = 1.0


class ChainCentreMassLossConfig(Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = 0.0
        self.eps = 1e-8
