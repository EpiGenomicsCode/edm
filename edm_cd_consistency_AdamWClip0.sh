#!/bin/bash
#SBATCH --job-name=edm_cd_consistency
#SBATCH --account=bbse-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=30:00:00
#SBATCH --output=/u/xyou1/edm/slurm_logs/%x-%j.out
#SBATCH --error=/u/xyou1/edm/slurm_logs/%x-%j.err

unset NCCL_NET || true
unset FI_PROVIDER || true
unset NCCL_SOCKET_IFNAME || true
unset NCCL_NET_PLUGIN || true  # <--- Kills the broken Slurm ghost variable
export NCCL_P2P_DISABLE=0      # <--- Forces NCCL to use the internal NVLink

set -euo pipefail

# Ensure log directory exists
mkdir -p /u/xyou1/edm/slurm_logs

# Activate environment
CONDA_BASE=/sw/user/python/miniforge3-pytorch-2.5.0
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate edm2

# Move to working directory
cd /u/xyou1/edm

echo " CPU info:"
lscpu | egrep "Model name|Socket|Thread|NUMA|CPU\(s\)|MHz"
echo "------------------------------------------------------------"
echo " Memory info:"
free -h
echo "------------------------------------------------------------"
echo " NUMA info:"
numactl --show
echo "------------------------------------------------------------"
echo " GPU info:"
nvidia-smi -L
echo "------------------------------------------------------------"
echo " GPU topology:"
nvidia-smi topo -m
echo "------------------------------------------------------------"
echo " CUDA env:"
which nvcc || echo "nvcc not found"
nvcc --version || true
echo "============================================================"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export EDM_DDP_FIND_UNUSED_PARAMETERS=0

# Run training with torchrun and arguments per prompt
torchrun --standalone --nproc_per_node=4 train.py \
  --outdir=/work/hdd/bbse/xyou1/edm-training-runs/imagenet64-cd-s8/AdamWClip0 \
  --data=/work/nvme/bbse/vmathew/edm_training/edm/datasets/imagenet-64x64.zip \
  --cond=1 --arch=adm --precond=edm \
  --batch=2048 --batch-gpu=64 --fp16=True --ema=50 --lr=8e-5 --ema_rampup=0.05 --grad-clip=0.0 \
  --consistency=True \
  --sampling_mode=edm \
  --duration=410 \
  --terminal_anchor \
  --terminal_teacher_hop \
  --teacher=/work/nvme/bbse/vmathew/edm_training/edm-imagenet-64x64-cond-adm.pkl \
  --S=8 --T_start=64 --T_end=1280 --T_anneal_kimg=204800 \
  --rho=7 --sigma_min=0.002 --sigma_max=80 \
  --cd_loss=pseudo_huber --cd_weight_mode=sqrt_karras \
  --wandb=True --wandb_project=edm-cd --wandb_entity=vinaysmathew-penn-state \
  --wandb_run=imagenet64-cd-s8-live-AdamW-Clip0 --wandb_tags=imagenet,cd,s8 --wandb_mode=online \
  --val=1 \
  --val_teacher=False \
  --snap=20 \
  --dump=20 \
  --cd_target_mode=live \
  --val_ref=/work/nvme/bbse/vmathew/edm_training/edm/fid-refs/imagenet-64x64.npz \
  --val_steps=8 \
  --val_every=20 \
  --val_at_start=0 \
  --dropout=0.0 \
  --seed=1959836853 \
  --workers=4 \
  --resume=/u/xyou1/edm/training-runs/imagenet64-cd-s8/AdamWClip0/00001-imagenet-64x64-cond-adm-edm-gpus4-batch2048-fp16-cdS8-T64-1280/training-state-013316.pt \