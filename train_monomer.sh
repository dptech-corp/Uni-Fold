[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${update_freq}" ] && update_freq=1
[ -z "${total_step}" ] && total_step=80000
[ -z "${warmup_step}" ] && warmup_step=1000
[ -z "${decay_step}" ] && decay_step=50000
[ -z "${decay_ratio}" ] && decay_ratio=0.95
[ -z "${sd_prob}" ] && sd_prob=0.75
[ -z "${lr}" ] && lr=1e-3
[ -z "${seed}" ] && seed=42
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $1
echo "save_dir" $2
echo "decay_step" $decay_step
echo "warmup_step" $warmup_step
echo "decay_ratio" $decay_ratio
echo "lr" $lr
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "data_folder:"
ls $1
echo "create folder for save"
mkdir -p $2
echo "start training"

model_name=$3

tmp_dir=`mktemp -d`

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       $(which unicore-train) $1 --user-dir unifold \
       --num-workers 4 --ddp-backend=no_c10d \
       --model-name $model_name \
       --task af2 --loss af2 --arch af2  --sd-prob $sd_prob  \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0  --per-sample-clip-norm 0.1 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr $lr --warmup-updates $warmup_step --decay-ratio $decay_ratio --decay-steps $decay_step --stair-decay --batch-size 1 \
       --update-freq $update_freq --seed $seed  --tensorboard-logdir $2/tsb/ \
       --max-update $total_step --max-epoch 1 --log-interval 10 --log-format simple \
       --save-interval-updates 500 --validate-interval-updates 500 --keep-interval-updates 40 --no-epoch-checkpoints  \
       --save-dir $2 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --bf16 --ema-decay 0.999 --data-buffer-size 32 --bf16-sr

rm -rf $tmp_dir
