import os, sys

import shlex
import glob
import json
from tqdm import tqdm
from multiprocessing import Pool
from unifold.msa.mmcif import parse
import argparse
import gzip
from Bio.PDB import protein_letters_3to1
import numpy as np
from unifold.data.residue_constants import restype_order_with_x

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
            if line.startswith(";"):
                continue
            t = shlex.split(line)
            last_val.extend(t)
            if len(last_val) == 2:
                ret[last_val[0]] = [last_val[1]]
                last_val = []
            if len(last_val) > 2:
                last_val = []
        if last_val:
            assert len(last_val) == 2
            ret[last_val[0]] = [last_val[1]]
    return ret


def get_transform(data, idx):
    idx = data["_pdbx_struct_oper_list.id"].index(f"{idx}")
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


def mmcif_object_to_fasta(mmcif_object, auth_chain_id: str) -> str:
    residues = mmcif_object.seqres_to_structure[auth_chain_id]
    residue_names = [residues[t].name for t in range(len(residues))]
    residue_letters = [
        protein_letters_3to1[n] if n in protein_letters_3to1.keys() else "X" for n in residue_names
    ]
    # take care of cases where residue letters are of length 3
    # simply by replacing them as 'X' ('UNK')
    filter_out_triple_letters = lambda x: x if len(x) == 1 else "X"
    fasta_string = "".join([filter_out_triple_letters(n) for n in residue_letters])
    return fasta_string


def parse_assembly(mmcif_path):
    name = os.path.split(mmcif_path)[-1].split(".")[0]
    with gzip.open(mmcif_path, "rb") as f:
        mmcif_string = f.read().decode("utf8")
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
    valid_chains_set = set(valid_chains)  # valid chains is not auth_id

    resolution = np.array([mmcif_obj.header["resolution"]])
    if resolution > 9:
        return name, [], [], [], "resolution"

    invalid_chains = []
    for chain_id in mmcif_obj.chain_to_seqres:
        sequence = mmcif_object_to_fasta(mmcif_obj, chain_id)
        aatype_idx = np.array(
            [
                restype_order_with_x[rn]
                if rn in restype_order_with_x
                else restype_order_with_x["X"]
                for rn in sequence
            ]
        )
        seq_len = aatype_idx.shape[0]
        _, counts = np.unique(aatype_idx, return_counts=True)
        freqs = counts.astype(np.float32) / seq_len
        max_freq = np.max(freqs)
        if max_freq > 0.8:
            invalid_chains.append(chain_id)
    valid_chains = []
    for chain_id in valid_chains_set:
        if mmcif_to_author_chain_id[chain_id] not in invalid_chains:
            valid_chains.append(chain_id)
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
                if "revision" in cur_str and (
                    assembly is not None and assembly_gen is not None and oper is not None
                ):
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
            idxxxx = "1"
            if name in ["6lb3"]:
                # the first assembly consists of two polydeoxyribonucleotides
                idxxxx = "2"
            if j == idxxxx:
                sss = op_idx[i].replace("(", "").replace(")", "").replace("'", "").replace('"', "")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="/you/mmcif")
    parser.add_argument("--output-file", type=str, default="/you/mmcif_assembly3.json")
    args = parser.parse_args()
    print(args)

    input_dir = args.input_dir
    output_file = args.output_file

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, "*.cif.gz"))

    file_cnt = len(input_files)
    print(f"len(input_files): {file_cnt}")
    meta_dict = {}
    failed = {}

    with Pool(80) as pool:
        for ret in tqdm(pool.imap(parse_assembly, input_files, chunksize=10), total=file_cnt):
            name, all_chains, all_chains_label, all_chains_ops, error_type = ret
            if all_chains:
                meta_dict[name] = {}
                meta_dict[name]["chains"] = all_chains
                meta_dict[name]["chains_label"] = all_chains_label
                meta_dict[name]["opers"] = all_chains_ops
            else:
                # failed.append(name + " " + error_type)
                failed[name] = error_type

    json.dump(meta_dict, open(output_file, "w"), indent=2)
    # write_list_to_file(failed, "failed_mmcif.txt")
    root = os.path.splitext(output_file)[0]
    json.dump(failed, open(f"{root}.failed.json", "w"), indent=2)


def write_list_to_file(a, file):
    with open(file, "w") as output:
        for x in a:
            output.write(str(x) + "\n")


if __name__ == "__main__":
    main()
