from typing import Any, Optional
from warnings import warn

from chanfig import Config

from .data import DataConfig
from .globals import GlobalsConfig
from .loss import LossConfig
from .model import ModelConfig


class UniFoldConfig(Config):
    def __init__(self, *args, **kwargs):
        self.globals = GlobalsConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.loss = LossConfig()
        super().__init__(*args, **kwargs)


def recursive_set(c: Config, key: str, value: Any, ignore: Optional[str] = None):
    with c.unlocked():
        for k, v in c.items():
            if ignore is not None and k == ignore:
                continue
            if isinstance(v, Config):
                recursive_set(v, key, value)
            elif k == key:
                c[k] = value


def base_config():
    deprecation_message = "`base_config` is deprecated.\nPlease call `UniFoldConfig()` instead"
    warn(deprecation_message, DeprecationWarning, stacklevel=2)
    return UniFoldConfig()


def model_config(name, train=False):
    c = UniFoldConfig()

    def model_2_v2(c):
        recursive_set(c, "v2_feature", True)
        recursive_set(c, "gumbel_sample", True)
        c.model.heads.masked_msa.d_out = 22
        c.model.structure_module.separate_kv = True
        c.model.structure_module.ipa_bias = False
        c.model.template.template_angle_embedder.d_in = 34
        return c

    def multimer(c):
        recursive_set(c, "is_multimer", True)
        recursive_set(c, "max_extra_msa", 1152)
        recursive_set(c, "max_msa_clusters", 128)
        recursive_set(c, "v2_feature", True)
        recursive_set(c, "gumbel_sample", True)
        c.model.template.template_angle_embedder.d_in = 34
        c.model.template.template_pair_stack.tri_attn_first = False
        c.model.template.template_pointwise_attention.enabled = False
        c.model.heads.pae.enabled = True
        # we forget to enable it in our training, so disable it here
        c.model.heads.pae.disable_enhance_head = True
        c.model.heads.masked_msa.d_out = 22
        c.model.structure_module.separate_kv = True
        c.model.structure_module.ipa_bias = False
        c.model.structure_module.trans_scale_factor = 20
        c.loss.pae.weight = 0.1
        c.model.input_embedder.tf_dim = 21
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.chain_centre_mass.weight = 1.0
        return c

    if name == "model_1":
        pass
    elif name == "model_1_ft":
        recursive_set(c, "max_extra_msa", 5120)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
    elif name == "model_1_af2":
        recursive_set(c, "max_extra_msa", 5120)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
    elif name == "model_2":
        pass
    elif name == "model_init":
        pass
    elif name == "model_init_af2":
        c.globals.alphafold_original_mode = True
        pass
    elif name == "model_2_ft":
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
    elif name == "model_2_af2":
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
    elif name == "model_2_v2":
        c = model_2_v2(c)
    elif name == "model_2_v2_ft":
        c = model_2_v2(c)
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
    elif name == "model_3_af2" or name == "model_4_af2":
        recursive_set(c, "max_extra_msa", 5120)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
        c.model.template.enabled = False
        c.model.template.embed_angles = False
        recursive_set(c, "use_templates", False)
        recursive_set(c, "use_template_torsion_angles", False)
    elif name == "model_5_af2":
        recursive_set(c, "max_extra_msa", 1024)
        recursive_set(c, "max_msa_clusters", 512)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.02
        c.loss.repr_norm.weight = 0
        c.model.heads.experimentally_resolved.enabled = True
        c.loss.experimentally_resolved.weight = 0.01
        c.globals.alphafold_original_mode = True
        c.model.template.enabled = False
        c.model.template.embed_angles = False
        recursive_set(c, "use_templates", False)
        recursive_set(c, "use_template_torsion_angles", False)
    elif name == "multimer":
        c = multimer(c)
    elif name == "multimer_ft":
        c = multimer(c)
        recursive_set(c, "max_extra_msa", 1152)
        recursive_set(c, "max_msa_clusters", 256)
        c.data.train.crop_size = 384
        c.loss.violation.weight = 0.5
    elif name == "multimer_af2":
        recursive_set(c, "max_extra_msa", 1152)
        recursive_set(c, "max_msa_clusters", 256)
        recursive_set(c, "is_multimer", True)
        recursive_set(c, "v2_feature", True)
        recursive_set(c, "gumbel_sample", True)
        c.model.template.template_angle_embedder.d_in = 34
        c.model.template.template_pair_stack.tri_attn_first = False
        c.model.template.template_pointwise_attention.enabled = False
        c.model.heads.pae.enabled = True
        c.model.heads.experimentally_resolved.enabled = True
        c.model.heads.masked_msa.d_out = 22
        c.model.structure_module.separate_kv = True
        c.model.structure_module.ipa_bias = False
        c.model.structure_module.trans_scale_factor = 20
        c.loss.pae.weight = 0.1
        c.loss.violation.weight = 0.5
        c.loss.experimentally_resolved.weight = 0.01
        c.model.input_embedder.tf_dim = 21
        c.globals.alphafold_original_mode = True
        c.data.train.crop_size = 384
        c.loss.repr_norm.weight = 0
        c.loss.chain_centre_mass.weight = 1.0
        recursive_set(c, "outer_product_mean_first", True)
    else:
        raise ValueError(f"invalid --model-name: {name}.")
    if train:
        c.globals.chunk_size = None
    recursive_set(c, "inf", 3e4)
    recursive_set(c, "eps", 1e-5, "loss")
    return c
