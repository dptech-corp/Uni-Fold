set -e 
fasta_path=$1
output_dir_base=$2
param_path=$3
lm_param_path=$4

echo "Starting generating reprentations from the further pre-trained protein language model"
python infer_lm.py \
    --input $fasta_path \
    --path $lm_param_path \
    --output_dir $output_dir_base \
    --repr_layers 36 --include per_tok --toks_per_batch 512 --fp16
    
echo "Starting preparing features for fasta sequences"
python unifold/pack_feat.py \
    --fasta_path=$fasta_path \
    --output_dir=$output_dir_base

echo "Starting prediction..."
fasta_file=$(basename $fasta_path)
target_name=${fasta_file%.fa*}
python unifold/inference.py \
    --model_name=single_multimer_3b_newemb \
    --param_path=$param_path \
    --data_dir=$output_dir_base \
    --target_name=$target_name \
    --output_dir=$output_dir_base
    
echo "done"
    