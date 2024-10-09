#!/bin/bash
#SBATCH --job-name=$train
#SBATCH --output=slurm_logs/train-%j-out.txt
#SBATCH --error=slurm_logs/train-%j-err.txt
#SBATCH --mem=40gb
#SBATCH -c 2
#SBATCH --gres=gpu:a100
#SBATCH -p pasteur
#SBATCH -A  pasteur
#SBATCH --time=48:00:00
#SBATCH --ntasks=1

# 64k batchsize for 2.048e-3 lr
export CUDA_VISIBLE_DEVICES=0


#output-000001.tar
torchrun -m open_clip_train.main \
    --dataset-resampled \
    --save-frequency 1 \
    --save-most-recent \
    --train-data '/pasteur2/u/ale9806/data/pmc-oa/{000000..000820}.tar' \
    --train-num-samples 820000 \
    --dataset-type webdataset \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 1000 \
    --wd 0.2 \
    --batch-size 1024 \
    --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --epochs=100\
    --workers=1 \
    --model RN50 \
    --precision 'amp_bf16' \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --log-every-n-steps 1 \
    --seed 0 \
    --logs ./logs/ \
    --report-to "wandb" \
    --wandb-project-name "open-biomed-clip"
     #--force-image-size 224 \
