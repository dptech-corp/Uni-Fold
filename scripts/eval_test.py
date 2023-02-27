import glob
import os
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import json
from scripts.eval_metrics import compute_multimer
from collections import defaultdict


def write_json(path, context):
    with open(path, "w") as f:
        json.dump(context, f, indent=4, separators=(",", ":"))


def open_json(path):
    with open(path, "r") as f:
        return json.load(f)


def process_one(args):
    target_name, tgt_src_dir, tgt_tgt_dir = args
    try:
        gt_pdb = glob.glob(os.path.join(tgt_src_dir, "label.pdb"))[0]
        plddt_fn = glob.glob(os.path.join(tgt_tgt_dir, "*multimer*_*_ptm.json"))[0]
        with open(plddt_fn, "r") as f:
            plddt = json.load(f)
        for key in plddt.keys():
            plddt[key] = float(plddt[key])
        sorted_results = sorted(plddt.keys(), key=lambda x: -plddt[x])
        top_result = sorted_results[0]
        pred_pdb = os.path.join(tgt_tgt_dir, f"{top_result}.pdb")

        entity_fn = os.path.join(tgt_src_dir, "sym_id.txt")
        with open(entity_fn, "r") as f:
            entity_list = f.read()
        entity_list = entity_list.split()
        entity_list = [int(x) for x in entity_list]

        result = compute_multimer(target_name, gt_pdb, pred_pdb, entity_list)

        save_fn = os.path.join(tgt_tgt_dir, "result.json")
        write_json(save_fn, result)
        return True, None, None
    except Exception as e:
        return False, target_name, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/vepfs/users/jinhua/data/multimer_inference/multimer_benchmark_paper_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/vepfs/users/jinhua/data/multimer_inference/multimer_benchmark_paper_v3_afv3m1_out",
    )
    args = parser.parse_args()
    print(args)
    target_paths = glob.glob(os.path.join(args.data_dir, "????"))

    def input_args():
        for target in target_paths:
            target_name = os.path.basename(os.path.normpath(target))
            tgt_src_dir = os.path.join(args.data_dir, target_name)
            tgt_tgt_dir = os.path.join(args.output_dir, target_name)
            yield target_name, tgt_src_dir, tgt_tgt_dir

    with Pool(32) as pool:
        for ret in tqdm(
            pool.imap_unordered(process_one, input_args(), chunksize=1),
            total=len(target_paths),
        ):
            if not ret[0]:
                print(ret)

    average_metrics = defaultdict(list)
    for target in target_paths:
        target_name = os.path.basename(os.path.normpath(target))
        result_fn = os.path.join(args.output_dir, target_name, "result.json")
        if os.path.exists(result_fn):
            result = open_json(result_fn)
            for k, v in result.items():
                average_metrics[k].append(v)

    last_result = dict()
    valid_cnt = None
    for k, v in average_metrics.items():
        if valid_cnt is None:
            valid_cnt = len(v)
        else:
            assert valid_cnt == len(v)
        last_result[k] = sum(v) / len(v)
    last_result["valid cnt"] = valid_cnt
    save_fn = os.path.join(args.output_dir, "result.json")
    write_json(save_fn, last_result)


if __name__ == "__main__":
    main()
