from chanfig import Config

from .variables import chunk_size, d_extra_msa, d_msa, d_pair, d_single, d_template, eps, inf, max_recycling_iters


class GlobalsConfig(Config):
    def __init__(self, *args, **kwargs):
        self.chunk_size = chunk_size
        self.block_size = None
        self.d_pair = d_pair
        self.d_msa = d_msa
        self.d_template = d_template
        self.d_extra_msa = d_extra_msa
        self.d_single = d_single
        self.eps = eps
        self.inf = inf
        self.max_recycling_iters = max_recycling_iters
        self.alphafold_original_mode = False
        super().__init__(*args, **kwargs)
