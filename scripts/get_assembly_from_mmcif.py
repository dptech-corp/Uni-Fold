import os, sys

import shlex
import glob
import json
from tqdm import tqdm
from multiprocessing import Pool
from unifold.msa.mmcif import parse

rot_keys = """_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]""".split(
    "\n"
)
tran_keys = """_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.vector[3]""".split(
    "\n"
)


def process_block_to_dict(content):
    ret = {}
    lines = content.split("\n")
    if lines[0] == "loop_":
        keys = []
        values = []
        last_val = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("_"):
                keys.append(line)
            else:
                num_key = len(keys)
                cur_vals = shlex.split(line)
                last_val.extend(cur_vals)
                assert len(last_val) <= num_key, (
                    num_key,
                    len(last_val),
                    last_val,
                    cur_vals,
                )
                if len(last_val) == num_key:
                    values.append(last_val)
                    last_val = []
        if last_val:
            assert len(last_val) == num_key
            values.append(last_val)
        for i, k in enumerate(keys):
            ret[k] = [vals[i] for vals in values]
    else:
        last_val = []
        for line in lines:
            t = shlex.split(line)
            last_val.extend(t)
            if len(last_val) == 2:
                ret[last_val[0]] = [last_val[1]]
                last_val = []
        if last_val:
            assert len(last_val) == 2
            ret[last_val[0]] = [last_val[1]]
    return ret


def get_transform(data, idx):
    idx = int(idx) - 1
    rot = []
    for key in rot_keys:
        rot.append(float(data[key][idx]))
    trans = []
    for key in tran_keys:
        trans.append(float(data[key][idx]))
    rot, trans = tuple(rot), tuple(trans)
    if rot == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) and trans == (
        0.0,
        0.0,
        0.0,
    ):
        return "I"
    else:
        return rot, trans


def parse_assembly(mmcif_path):
    name = os.path.split(mmcif_path)[-1].split(".")[0]
    with open(mmcif_path) as f:
        mmcif_string = f.read()
        mmcif_lines = mmcif_string.split("\n")
    parse_result = parse(file_id="", mmcif_string=mmcif_string)
    if "No protein chains found in this file." in parse_result.errors.values():
        return name, [], [], [], "no protein"
    mmcif_obj = parse_result.mmcif_object
    if mmcif_obj is None:
        print(name, parse_result.errors)
        return name, [], [], [], "parse error"
    mmcif_to_author_chain_id = mmcif_obj.mmcif_to_author_chain_id
    valid_chains = mmcif_obj.valid_chains.keys()
    valid_chains = set(valid_chains)  # valid chains is not auth_id
    new_section = False
    is_loop = False
    cur_lines = []
    assembly = None
    assembly_gen = None
    oper = None
    error_type = None
    try:
        for line in mmcif_lines:
            line = line.strip().replace(";", "")
            if not line:
                continue
            if line == "#":
                cur_str = "\n".join(cur_lines)
                if "revision" in cur_str:
                    continue
                if "_pdbx_struct_assembly.id" in cur_str:
                    assembly = process_block_to_dict(cur_str)
                if "_pdbx_struct_assembly_gen.assembly_id" in cur_str:
                    assembly_gen = process_block_to_dict(cur_str)
                if "_pdbx_struct_oper_list.id" in cur_str:
                    oper = process_block_to_dict(cur_str)
                cur_lines = []
            else:
                cur_lines.append(line)
    except Exception as e:
        print(name, e)
        return name, [], [], [], "parse"
    if not (assembly is not None and assembly_gen is not None and oper is not None):
        return name, [], [], [], "miss"
    try:
        counts = assembly["_pdbx_struct_assembly.oligomeric_count"]
        asym_id = assembly_gen["_pdbx_struct_assembly_gen.assembly_id"]
        op_idx = assembly_gen["_pdbx_struct_assembly_gen.oper_expression"]
        assembly_chains = assembly_gen["_pdbx_struct_assembly_gen.asym_id_list"]
        chains = []
        chains_ops = []
        for i, j in enumerate(asym_id):
            if j == "1":
                sss = (
                    op_idx[i]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("'", "")
                    .replace('"', "")
                )
                if "-" in sss:
                    s, t = sss.split("-")
                    indices = range(int(s), int(t) + 1)
                else:
                    indices = sss.split(",")
                for idx in indices:
                    chains.append(assembly_chains[i].split(","))
                    chains_ops.append(get_transform(oper, idx))
        len_ops = len(chains)
        count = int(counts[0])
        all_chains = []
        all_chains_ops = []
        all_chains_label = []
        for i, cur_chains in enumerate(chains):
            for chain in cur_chains:
                if chain not in valid_chains:
                    continue
                auth_chain = mmcif_to_author_chain_id[chain]
                all_chains_label.append(chain)
                all_chains.append(auth_chain)
                all_chains_ops.append(chains_ops[i])
        return name, all_chains, all_chains_label, all_chains_ops, "success"
    except Exception as e:
        print(name, e)
        return name, [], [], [], "index"


input_dir = sys.argv[1]
output_file = sys.argv[2]
input_files = glob.glob(input_dir + "*.cif")

file_cnt = len(input_files)
meta_dict = {}
failed = []
with Pool(64) as pool:
    for ret in tqdm(pool.imap(parse_assembly, input_files), total=file_cnt):
        name, all_chains, all_chains_label, all_chains_ops, error_type = ret
        if all_chains:
            meta_dict[name] = {}
            meta_dict[name]["chains"] = all_chains
            meta_dict[name]["chains_label"] = all_chains_label
            meta_dict[name]["opers"] = all_chains_ops
        else:
            failed.append(name + " " + error_type)

json.dump(meta_dict, open(output_file, "w"), indent=2)


def write_list_to_file(a, file):
    with open(file, "w") as output:
        for x in a:
            output.write(str(x) + "\n")


write_list_to_file(failed, "failed_mmcif.txt")
