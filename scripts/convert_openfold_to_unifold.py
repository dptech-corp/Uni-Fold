import torch
import sys

def openfold2unifold(model_states):
    new_model_states = {}
    mul_projs = {}
    mul_gates = {}
    for key, value in model_states.items():
        new_key = key
        if "msa_att_col._msa_att" in key:
            new_key = new_key.replace("msa_att_col._msa_att", "msa_att_col")
        if "extra_msa_stack.stack" in key:
            new_key = new_key.replace("extra_msa_stack.stack", "extra_msa_stack")
        if "tri_mul" in key:
            if "linear_a_p" in key or "linear_b_p" in key:
                new_key = key.replace("linear_a_p", "linear_ab_p").replace(
                    "linear_b_p", "linear_ab_p"
                )
                mul_projs[new_key] = 1
                continue
            if "linear_a_g" in key or "linear_b_g" in key:
                new_key = key.replace("linear_a_g", "linear_ab_g").replace(
                    "linear_b_g", "linear_ab_g"
                )
                mul_gates[new_key] = 1
                continue
        if ".tm." in key:
            new_key = new_key.replace(".tm.", ".pae.")
        if ".core." in key:
            new_key = new_key.replace("core." ,"")
        new_model_states[new_key] = value

    for key in mul_projs:
        new_key = key
        k1 = key.replace("linear_ab_p", "linear_a_p")
        k2 = key.replace("linear_ab_p", "linear_b_p")
        weight = torch.cat([model_states[k1], model_states[k2]], dim=0)
        if ".core." in key:
            new_key = new_key.replace("core." ,"")
        new_model_states[new_key] = weight

    for key in mul_gates:
        new_key = key
        k1 = key.replace("linear_ab_g", "linear_a_g")
        k2 = key.replace("linear_ab_g", "linear_b_g")
        weight = torch.cat([model_states[k1], model_states[k2]], dim=0)
        if ".core." in key:
            new_key = new_key.replace("core." ,"")
        new_model_states[new_key] = weight

    return new_model_states


load_ckpt=sys.argv[1]
save_ckpt=sys.argv[2]
state_dict = torch.load(load_ckpt)
state_dict = openfold2unifold(state_dict)
save_state_dict = {}
save_state_dict["ema"] = {}
save_state_dict["extra_state"] = {}
save_state_dict["extra_state"]["train_iterator"] = {}
save_state_dict["extra_state"]["train_iterator"]["epoch"] = 1
update_state_dict = {"model." + k:state_dict[k] for k in state_dict}
save_state_dict["ema"]["params"] = update_state_dict
torch.save(save_state_dict, save_ckpt)
