import glob
import subprocess
import os
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import numpy as np
import torch
import time
from pynvml import *
import shutil


def get_best_gpu():
    used = []
    nvmlInit()

    for i in range(8):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        used.append(info.used)
    idx = np.argmin(used)
    return idx


def process_one(args):
    cuda, param_path, data_dir, target_name, output_dir, model_name = args
    if cuda < 0:
        cuda = get_best_gpu()
    ptm_fns = glob.glob(os.path.join(output_dir, target_name, "*_plddt.json"))
    if len(ptm_fns) > 0:
        return True, cuda
    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    cmd = ["python", "unifold/inference.py"]
    args = [
        f"--model_name={model_name}",
        f"--param_path={param_path}",
        f"--data_dir={data_dir}",
        f"--target_name={target_name}",
        f"--output_dir={output_dir}",
    ]
    cmd = cmd + args
    print(f"Launch {cmd} on gpu {cuda}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env
    )
    stdout, stderr = process.communicate()
    retcode = process.wait()
    if retcode:
        return False, cuda, target_name, stderr.decode("utf-8")

    return True, cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param-path",
        type=str,
        default="/mnt/vepfs/users/guolin/unifold/aml_multimer_ft_run_3/checkpoint_best.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/vepfs/users/jinhua/data/multimer_inference/multimer_benchmark_paper_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/vepfs/users/jinhua/data/multimer_inference/multimer_benchmark_paper_v3_out",
    )
    parser.add_argument("--model-name", type=str, default="multimer_ft")
    args = parser.parse_args()
    print(args)
    target_paths = glob.glob(os.path.join(args.data_dir, "????"))

    def fil_func(path):
        target_name = os.path.basename(os.path.normpath(path))
        return (
            len(glob.glob(os.path.join(args.output_dir, target_name, "*_plddt.json")))
            == 0
        )

    target_paths = list(filter(fil_func, target_paths))

    for target_path in target_paths:
        target_name = os.path.basename(os.path.normpath(target_path))
        target_dir = os.path.join(args.output_dir, target_name)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

    free_queue = list(range(8))

    def input_args():

        for target in target_paths:
            target = os.path.basename(os.path.normpath(target))
            if len(free_queue) > 0:
                i = free_queue.pop(0)
            else:
                i = -1
            yield i, args.param_path, args.data_dir, target, args.output_dir, args.model_name

    with Pool(8) as pool:
        for ret in tqdm(
            pool.imap_unordered(process_one, input_args(), chunksize=1),
            total=len(target_paths),
        ):
            if not ret[0]:
                print(ret)


if __name__ == "__main__":
    main()
