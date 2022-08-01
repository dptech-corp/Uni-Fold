import os, sys

import shlex
import glob
import json
from tqdm import tqdm
from multiprocessing import Pool
from unifold.msa.mmcif import parse
import gzip


def parse_assembly(mmcif_path):
    name = os.path.split(mmcif_path)[-1].split(".")[0]
    if mmcif_path.endswith(".gz"):
        with gzip.open(mmcif_path, "rb") as f:
            mmcif_string = f.read().decode()
            mmcif_lines = mmcif_string.split("\n")
    else:
        with open(mmcif_path, "rb") as f:
            mmcif_string = f.read()
            mmcif_lines = mmcif_string.split("\n")
    parse_result = parse(file_id="", mmcif_string=mmcif_string)
    if "No protein chains found in this file." in parse_result.errors.values():
        return name, None
    mmcif_obj = parse_result.mmcif_object
    if mmcif_obj is None:
        print(name, parse_result.errors)
        return name, None
    mmcif_to_author_chain_id = mmcif_obj.mmcif_to_author_chain_id
    valid_chains = mmcif_obj.valid_chains.keys()
    valid_chains = list(set(valid_chains))  # valid chains is not auth_id
    return name, {"to_auth_id": mmcif_to_author_chain_id, "valid_chains": valid_chains}


input_dir = sys.argv[1]
output_file = sys.argv[2]
input_files = glob.glob(input_dir + "*.cif") + glob.glob(input_dir + "*.cif.gz")

file_cnt = len(input_files)
meta_dict = {}
failed = []
with Pool(64) as pool:
    for ret in tqdm(pool.imap(parse_assembly, input_files), total=file_cnt):
        name, res = ret
        if res:
            meta_dict[name] = res

json.dump(meta_dict, open(output_file, "w"), indent=2)
