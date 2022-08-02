import torch
import sys

from unifold.config import model_config
from unifold.modules.alphafold import AlphaFold

from scripts.translate_jax_params import (
    import_jax_weights_,
)

load_ckpt=sys.argv[1]
save_ckpt=sys.argv[2]
model_name = sys.argv[3]

config = model_config(model_name)
model = AlphaFold(config)
import_jax_weights_(model, load_ckpt, version=model_name)
state_dict = model.state_dict()
save_state_dict = {}
save_state_dict["ema"] = {}
save_state_dict["extra_state"] = {}
save_state_dict["extra_state"]["train_iterator"] = {}
save_state_dict["extra_state"]["train_iterator"]["epoch"] = 1
update_state_dict = {"model." + k:state_dict[k] for k in state_dict}
save_state_dict["ema"]["params"] = update_state_dict
torch.save(save_state_dict, save_ckpt)
