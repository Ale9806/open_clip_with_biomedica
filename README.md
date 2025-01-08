# How to Train a CLIP-like Model with OpenBioScience
This is a (very minor) adaptation of the OpenCLIP repository to train CLIP-style models using the Biomedical archive. All credits go to the original developers of [OpeCLIP](https://github.com/mlfoundations/open_clip). This repository also builds upon an original discussion on OpenCLIP's [GitHub](https://github.com/mlfoundations/open_clip/discussions/812).

### Introduction

OpenCLIP has gained recognition in both academic and industrial communities as an exceptional open-source framework for training CLIP-like models. However, the documentation can be lacking when it comes to fine-tuning these models for specific downstream tasks using custom datasets. For beginners, this can be overwhelming as they might not know where to begin. This guide outlines some key considerations and best practices for using OpenCLIP effectively.


### Step 1: Create a Virtual Environment

To begin, we need to set up a virtual environment. Based on my own testing, **Python 3.9** works well. You can create the environment using the following command:


```python
# Create env
conda create --name train_clip python=3.9

# Activate env
conda activate train_clip
```
---

### Step 2: Install environment
Check your CUDA version before installing torch and the corresponding packages， if we install the dependencies by directly using official command, we are very likely to encounter a series of errors caused by mismatched torch versions and CUDA versions. So install your environment according to the actual situation.

```python
# Check CUDA versaion
nvidia-smi
```

and we will get the driver version（Using my local device as an example):

```
NVIDIA-SMI 515.65.01 Driver Version: 515.65.01 CUDA Version: 11.7
```

Then visit torch official website to get a compatible  distribution. It is recommended to use  **pip** for installation. For example, for my version **CUDA 11.7** I used:


```python
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

Lastly, verify that the installation was successful:
```python
import torch
print(torch.cuda.is_available()) # verify it prints True
True 
```
---
### Step 3: Clone and install the open_clip 

```bash
# Clone repo
git clone https://github.com/mlfoundations/open_clip.git

# Enter the project root directory.
cd open_clip

# Install training dependcies 
pip install -r requirements-training.txt

# Install webdataset
git clone https://github.com/minwoosun/webdataset.git
cd webdataset
git checkout hf-token
pip install -e .

# Install wandb
pip install wandb

# Setup tokens
huggingface-cli login
wandb login
```

---
###  Step 4: Chose a suitable pre-trained model
OpenClip official provides quite a lot pre-trained models of the CLIP series for downloading and usage. You can use the following command to view the specific details of these models.

The first column represents the model’s name, which is also the parameter for text encoding in the model. The second column indicates either the provider of the model or the scale of training dataset used.


```python
import open_clip
open_clip.list_pretrained()

# [('RN50', 'openai'),
#  ('RN50', 'yfcc15m'),
#  ('RN50', 'cc12m'),
#   ...,
#  ('nllb-clip-large-siglip', 'v1')]
```



## 5. Train your model




To train CLIP-style models using a webdataset locally (e.g. biomedica_webdataset), first download the dataset locally. Then run the following commands:

## 5.A Training using webdataset without streaming

```python
# Enter the src folder of the open_clip repository
cd open_clip/src

# Create a bash file
vim train_openclip.sh

## Add the following:

# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# set the training args, Example:
torchrun --nproc_per_node 6 -m training.main \
    --batch-size 500 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="/path/to/your/local/logs" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data /path/to/your/local/training_dict.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-B-32 \
    --pretrained /path/to/your/local/model

```


For more detailed args explanation, please refer to：[https://github.com/mlfoundations/open_clip/blob/main/src/training/params.py](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/params.py)



#### Epochs

For larger datasets (eg Laion2B), we recommend setting `--train-num-samples` to a lower value than the full epoch, for example `--train-num-samples 135646078` to 1/16 of an epoch in conjunction with `--dataset-resampled` to do sampling with replacement. This allows having frequent checkpoints to evaluate more often.

#### Patch Dropout

<a href="https://arxiv.org/abs/2212.00794">Recent research</a> has shown that one can dropout half to three-quarters of the visual tokens, leading to up to 2-3x training speeds without loss of accuracy.

You can set this on your visual transformer config with the key `patch_dropout`.

In the paper, they also finetuned without the patch dropout at the end. You can do this with the command-line argument `--force-patch-dropout 0.`

#### Multiple data sources

OpenCLIP supports using multiple data sources, by separating different data paths with `::`.
For instance, to train on CC12M and on LAION, one might use `--train-data "/data/cc12m/cc12m-train-{0000..2175}.tar::/data/LAION-400M/{00000..41455}.tar"`.
Using `--dataset-resampled` is recommended for these cases.

By default, on expectation the amount of times the model will see a sample from each source is proportional to the size of the source.
For instance, when training on one data source with size 400M and one with size 10M, samples from the first source are 40x more likely to be seen in expectation.

#### Single-Node

We make use of `torchrun` to launch distributed jobs. The following launches a
a job on a node of 4 GPUs:

```bash
cd open_clip/src
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

#### Multi-Node
The same script above works, so long as users include information about the number
of nodes and host node.

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    --rdzv_endpoint=$HOSTE_NODE_ADDR \
    -m open_clip_train.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

#### SLURM

This is likely the easiest solution to utilize. The following script was used to
train our largest models:

```bash
#!/bin/bash -x
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=open_clip
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition PARTITION_NAME

eval "$(/path/to/conda/bin/conda shell.bash hook)" # init conda
conda activate open_clip
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12802

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd /shared/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -u src/open_clip_train/main.py \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="/path_to_biomedica_tars/{00000..41455}.tar" \
    --warmup 2000 \
    --batch-size=256 \
    --epochs=32 \
    --workers=8 \
    --model ViT-B-32 \
    --name "ViT-B-32-Vanilla" \
    --seed 0 \
    --local-loss \
    --gather-with-grad
```

### Resuming from a checkpoint:

```bash
python -m open_clip_train.main \
    --train-data="/path/to/train_data.csv" \
    --val-data="/path/to/validation_data.csv"  \
    --resume /path/to/checkpoints/epoch_K.pt
```


## Acknowledgments

We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this part of work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputer JUWELS Booster at Jülich Supercomputing Centre (JSC).

## The Team

Current development of this repository is led by [Ross Wightman](https://rwightman.com/), [Romain Beaumont](https://github.com/rom1504), [Cade Gordon](http://cadegordon.io/), and [Vaishaal Shankar](http://vaishaal.com/).

The original version of this repository is from a group of researchers at UW, Google, Stanford, Amazon, Columbia, and Berkeley.

[Gabriel Ilharco*](http://gabrielilharco.com/), [Mitchell Wortsman*](https://mitchellnw.github.io/), [Nicholas Carlini](https://nicholas.carlini.com/), [Rohan Taori](https://www.rohantaori.com/), [Achal Dave](http://www.achaldave.com/), [Vaishaal Shankar](http://vaishaal.com/), [John Miller](https://people.eecs.berkeley.edu/~miller_john/), [Hongseok Namkoong](https://hsnamkoong.github.io/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)

Special thanks to [Jong Wook Kim](https://jongwook.kim/) and [Alec Radford](https://github.com/Newmu) for help with reproducing CLIP!

## Citing

If you found this repository useful, please consider citing:
```bibtex
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```

```bibtex
@inproceedings{cherti2023reproducible,
  title={Reproducible scaling laws for contrastive language-image learning},
  author={Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2818--2829},
  year={2023}
}
```

```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

```bibtex
@inproceedings{schuhmann2022laionb,
  title={{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author={Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=M3Y74vmsMcY}
}
```

[![DOI](https://zenodo.org/badge/390536799.svg)](https://zenodo.org/badge/latestdoi/390536799)
