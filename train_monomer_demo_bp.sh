ps -ef | grep "torch" | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep "unicore-train" | grep -v grep | awk '{print $2}' | xargs kill -9
export MASTER_IP=10.67.228.15
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${MASTER_PORT}" ] && MASTER_PORT=12345
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${PADDLE_TRAINERS_NUM}" ] && PADDLE_TRAINERS_NUM=1
[ -z "${PADDLE_TRAINER_ID}" ] && PADDLE_TRAINER_ID=0
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
mkdir -p $1
#n_gpu=4
tmp_dir=`mktemp -d`
#model_name=model_init_af2
model_name=model_1_af2
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT --nnodes=$PADDLE_TRAINERS_NUM --node_rank=$PADDLE_TRAINER_ID --master_addr=$MASTER_IP $(which unicore-train) ./example_data/ --user-dir unifold \
       --num-workers 8 --ddp-backend=no_c10d \
       --task af2 --loss af2 --arch af2 --model-name $model_name \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0  --per-sample-clip-norm 0.1 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr 1e-3 --warmup-updates 1000 --decay-ratio 0.95 --decay-steps 50000 --batch-size 1 \
       --update-freq 1 --seed 42  --tensorboard-logdir $1/tsb/ \
       --max-update 1000 --max-epoch 1 --log-interval 10 --log-format simple \
       --save-interval-updates 500 --validate-interval-updates 500 --keep-interval-updates 40 --no-epoch-checkpoints  \
       --save-dir $1 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --bf16 --ema-decay 0.999 --data-buffer-size 32 --bf16-sr --bp-degree 2
rm -rf $tmp_dir
#       --save-dir $1 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay 0.999 --data-buffer-size 32 --bp-degree 1
