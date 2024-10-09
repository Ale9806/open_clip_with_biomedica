#!/bin/bash
#SBATCH --job-name=$train
#SBATCH --output=slurm_logs/train-%j-out.txt
#SBATCH --error=slurm_logs/train-%j-err.txt
#SBATCH --mem=40gb
#SBATCH -c 2
#SBATCH --gres=gpu:a100:2
#SBATCH -p pasteur
#SBATCH -A  pasteur
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodelist=pasteur-hgx-1

# 64k batchsize for 2.048e-3 lr
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_RUN_GROUP="R50_CT_from_webli"

#['openai', 'yfcc15m', 'cc12m']
#output-000001.tar
#--train-data '/pasteur2/u/ale9806/data/pmc-oa/{000000..000820}.tar' \
# --train-num-samples 820000 \
#torchrun -m  --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 open_clip_train.main \
srun torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id 1234 --rdzv_backend c10d --rdzv_endpoint localhost:1234  -m open_clip_train.main  \
    --save-most-recent \
    --train-data '/pasteur2/u/ale9806/data/pmc-oa/{000000..000002}.tar' \
    --train-num-samples 820000 \
    --accum-freq 2\
    --lr-scheduler 'cosine' \
    --dataset-type webdataset \
    --lr "2.048e-5" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 1000 \
    --wd 0.2 \
    --batch-size 512 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs=100\
    --workers=1 \
    --model RN50 \
    --precision 'fp32' \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 1 \
    --seed 0 \
    --logs ./logs_clip_rn50/ \
    --pretrained "openai" \
    --report-to "wandb" \
    --wandb-project-name "open-biomed-clip"
     #--force-image-size 224 \
     # --dataset-resampled \
