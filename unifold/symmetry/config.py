import ml_collections as mlc
from ..config import model_config, recursive_set

def uf_symmetry_config():
    config = model_config("multimer", train=False)

    config.data.common.features.symmetry_opers = [None, 3, 3]
    config.data.common.features.num_asym = [None]
    config.data.common.features.pseudo_residue_feat = [None]
    
    recursive_set(config, "max_msa_clusters", 256)
    
    config.model.heads.pae.enabled = True   # pTM is in development, not reliable.
    config.loss.pae.weight = 0.0
    config.model.heads.experimentally_resolved.enabled = True
    config.loss.experimentally_resolved.weight = 0.0
    
    config.model.pseudo_residue_embedder = mlc.ConfigDict({
        "d_in": 8,
        "d_hidden": 48,
        "d_out": 48,
        "num_blocks": 4,
    })
    
    config.model.input_embedder.pr_dim = 48
    config.model.heads.pae.disable_enhance_head = True
    
    return config
