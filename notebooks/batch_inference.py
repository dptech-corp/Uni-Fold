import warnings
import argparse
import os
import json
import time
from tqdm import tqdm
from unifold.colab.model import colab_inference
from unifold.colab.data import validate_input, get_features
warnings.filterwarnings("ignore")

MIN_SINGLE_SEQUENCE_LENGTH = 6
MAX_SINGLE_SEQUENCE_LENGTH = 5000
MAX_MULTIMER_LENGTH = 5000


def process_batch_json(tasks, jobname, output_dir_base):
    if isinstance(tasks, dict):
        new_tasks = []
        for k, v in tasks.items():
            v['id'] = k
            new_tasks.append(v)
        tasks = new_tasks
    
    # check the input.
    for idx, task in enumerate(tasks):
        if 'id' not in task.keys():
            task['id'] = idx
            
        if 'sequence' not in task.keys():
            raise KeyError(f"number {idx+1}-th 'sequence' not found in dict keys: {task.keys()} in json.")
        
        target_id = f"{jobname}_{task['id']}"
        input_sequences = task['sequence'].strip().split(';')
        
        task['target_id'] = target_id
        
        if 'symmetry' not in task.keys():
            task['symmetry'] = 'C1'
        
        symmetry_group = task['symmetry'] 
        # check the sequences
        sequences, is_multimer, symmetry_group = validate_input(
            input_sequences=input_sequences,
            symmetry_group=symmetry_group,
            min_length=MIN_SINGLE_SEQUENCE_LENGTH,
            max_length=MAX_SINGLE_SEQUENCE_LENGTH,
            max_multimer_length=MAX_MULTIMER_LENGTH)
        task['is_multimer'] = is_multimer
        
        # save features to `output_dir_base`
        feature_output_dir = get_features(
            jobname=jobname,
            target_id=target_id,
            sequences=sequences,
            output_dir_base=output_dir_base,
            is_multimer=is_multimer,
            msa_mode=args.msa_mode,
            use_templates=True if args.use_templates > 0 else False
            )
        
        task['feature_output_dir'] = feature_output_dir
        task['symmetry'] = task['symmetry'] if task['symmetry'] != 'C1' else None

    return tasks

def manual_operations():
    # developers may operate on the pickle files here
    # to customize the features for inference.
    pass

manual_operations()

        
def main(args):
    output_dir_base = args.out_dir
    os.makedirs(output_dir_base, exist_ok=True)
    
    input_json_path = args.input_json
    with open(input_json_path, encoding="utf-8") as fp:
        input_json = json.load(fp)

    all_tasks = process_batch_json(input_json, args.jobname, output_dir_base)
    
    for task in tqdm(all_tasks, desc='running Unifold'):
        start = time.time()
        best_result = colab_inference(
            target_id=task['target_id'],
            data_dir=task['feature_output_dir'],
            param_dir='.',
            output_dir=task['feature_output_dir'],
            symmetry_group=task['symmetry'],
            is_multimer=task['is_multimer'],
            max_recycling_iters=args.max_recycling_iters,
            num_ensembles=args.num_ensembles,
            times=args.times,
            manual_seed=args.manual_seed,
            device=args.device,                # do not change this on colab.
            bf16=args.bf16
        )
        
        task['best_plddt'] = best_result['plddt'].mean().item()
        task['pae'] = best_result['pae'].mean().item() if best_result['pae'] is not None else None
        task['best_results_path'] = best_result['best_results_path']
        task['run_time'] = (time.time() - start)/60
        
        # incase oom
        with open(os.path.join(output_dir_base, 'all_tasks_summary.json'), 'w') as f:
            json.dump(all_tasks, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_json', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, default="predictions")
    parser.add_argument('--jobname', type=str, default="jobname")
    parser.add_argument('--msa_mode', type=str, default="MMseqs2", choices=["MMseqs2","single_sequence"])
    parser.add_argument('--num_ensembles', type=int, default=2)
    parser.add_argument('--max_recycling_iters', type=int, default=3)
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--use_templates', type=int, default=1)
    parser.add_argument('--manual_seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bf16', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)