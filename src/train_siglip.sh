#!/bin/bash
#SBATCH --job-name=$train
#SBATCH --output=slurm_logs/train-siglip-%j-out.txt
#SBATCH --error=slurm_logs/train-siglip-%j-err.txt
#SBATCH --mem=80gb
#SBATCH -c 2
#SBATCH --gres=gpu:a100:2
#SBATCH -p pasteur
#SBATCH -A  pasteur
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodelist=pasteur-hgx-1


# 64k batchsize for 2.048e-3 lr
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_RUN_GROUP="SIGLIP_CT_from_webli"


#output-000001.tar --rdzv_backend c10d  --rdzv_endpoint 0.0.0.0:29502 
torchrun --nproc_per_node 2  -m open_clip_train.main \
    --save-frequency 1  \
    --dataset-resampled \
    --train-data '/pasteur2/u/ale9806/data/pmc-oa/{000000..000820}.tar' \
    --train-num-samples 820000 \
    --dataset-type webdataset \
    --lr "5.048e-4" \
    --accum-freq 2\
    --lr-scheduler 'cosine' \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 1000 \
    --wd 0.2 \
    --batch-size 256\
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs=30\
    --workers=1 \
    --model ViT-SO400M-14-SigLIP-384 \
    --siglip \
    --precision 'bf16' \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 1 \
    --seed 0 \
    --logs ./logs_siglip/ \
    --report-to "wandb" \
    --wandb-project-name "open-biomed-clip" \
    --pretrained "webli" 
     #--force-image-size 224 \
