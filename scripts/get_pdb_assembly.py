import os, sys

import glob
import json
from tqdm import tqdm
from multiprocessing import Pool
import requests
import time

input_dir = sys.argv[1]
output_dir = sys.argv[2]
input_files = glob.glob(input_dir + "*.cif") + glob.glob(input_dir + "*.cif.gz")
os.system("mkdir -p " + output_dir)

pdb_chain_mapper = json.load(open(sys.argv[3]))

rot_keys = [
    "matrix11",
    "matrix12",
    "matrix13",
    "matrix21",
    "matrix22",
    "matrix23",
    "matrix31",
    "matrix32",
    "matrix33",
]

trans_keys = ["vector1", "vector2", "vector3"]


def get_oper(cont):
    cont = cont["pdbx_struct_oper_list"]
    ret = {}
    for c in cont:
        id = c["id"]
        rot = []
        trans = []
        for k in rot_keys:
            rot.append(c[k])
        for k in trans_keys:
            trans.append(c[k])
        rot, trans = tuple(rot), tuple(trans)
        if rot == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) and trans == (
            0.0,
            0.0,
            0.0,
        ):
            ret[id] = "I"
        else:
            ret[id] = (rot, trans)
    ret["I"] = "I"
    return ret


# refer to https://data.rcsb.org/redoc/index.html
def get_pdb_meta_info(mmcif_path):
    name = os.path.split(mmcif_path)[-1].split(".")[0]
    url = f"https://data.rcsb.org/rest/v1/core/assembly/{name}/1"
    max_try_time = 10
    for i in range(max_try_time):
        try:
            out_path = os.path.join(output_dir, name + ".json")
            load = False
            if os.path.isfile(out_path):
                cont = json.load(open(out_path, "r"))
                load = True
            else:
                r = requests.get(url)
                cont = json.loads(r.text)
                json.dump(cont, open(out_path, "w"))
                load = r.ok
            if load:
                if "rcsb_struct_symmetry" not in cont:
                    break
                cur_mapper = pdb_chain_mapper[name]["to_auth_id"]
                for tt in cont["rcsb_struct_symmetry"]:
                    if tt["kind"] == "Global Symmetry":
                        symbol = tt["symbol"]
                        stoi = tt["stoichiometry"]
                        all_opers = get_oper(cont)
                        chains = []
                        opers = []
                        for c in tt["clusters"]:
                            for m in c["members"]:
                                chain_id = cur_mapper[m["asym_id"]]
                                if "pdbx_struct_oper_list_ids" in m:
                                    for op_idx in m["pdbx_struct_oper_list_ids"]:
                                        chains.append(chain_id)
                                        opers.append(all_opers[op_idx])
                                else:
                                    chains.append(chain_id)
                                    opers.append("I")
                        return name, {
                            "symbol": symbol,
                            "stoi": stoi,
                            "chains": chains,
                            "opers": opers,
                        }
                break
            elif cont["status"] == "404":
                break
        except Exception as e:
            print(name, e)
        time.sleep(2)
    return name, None


file_cnt = len(input_files)
meta_dict = {}
# failed = []
with Pool(64) as pool:
    for ret in tqdm(pool.imap(get_pdb_meta_info, input_files), total=file_cnt):
        name, res = ret
        if res:
            meta_dict[name] = res

json.dump(meta_dict, open("pdb_assembly.json", "w"), indent=2)
