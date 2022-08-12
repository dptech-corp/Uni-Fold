[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
mkdir -p $1
tmp_dir=`mktemp -d`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT $(which unicore-train) ./example_data/  --user-dir unifold \
       --num-workers 8 --ddp-backend=no_c10d \
       --task af2 --loss afm --arch af2 \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0  --per-sample-clip-norm 0.1 --allreduce-fp32-grad  \
       --lr-scheduler exponential_decay --lr 1e-3 --warmup-updates 1000 --decay-ratio 0.95 --decay-steps 50000 --batch-size 1 \
       --update-freq 1 --seed 42  --tensorboard-logdir $1/tsb/ \
       --max-update 1000 --max-epoch 1 --log-interval 1 --log-format simple \
       --save-interval-updates 100 --validate-interval-updates 100 --keep-interval-updates 5 --no-epoch-checkpoints  \
       --save-dir $1 --tmp-save-dir $tmp_dir --required-batch-size-multiple 1 --ema-decay 0.999 --model-name multimer --bf16 --bf16-sr # for V100 or older GPUs, you can disable --bf16 for faster speed.
rm -rf $tmp_dir