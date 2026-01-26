# Multi-Step Consistency Distillation (EDM)

Implementation of [Multi-Step Consistency Models](https://arxiv.org/abs/2403.06807) (Heek et al., 2024) translated from the original VP/DDIM formulation to the [EDM](https://arxiv.org/abs/2206.00364) framework (Karras et al., 2022).

## Features

- **EDM-native consistency distillation** with Heun 2nd-order teacher hops and Karras noise schedule
- **Sigma-anchored segmentation** — teacher edges are partitioned by σ-intervals (not just index counts), with automatic filtering of duplicate σ values
- **Multiple target modes** — `live` (student self-consistency), `ema` (EMA network), or `teacher`
- **Built-in validation** — periodic FID evaluation during training without separate scripts
- **Teacher step annealing** — smooth transition from T_start to T_end over configurable kimg
- **W&B integration** — loss curves, FID, gradient norms, gain diagnostics, edge statistics
- **Mixed precision** — FP16 training with FP32 for numerically sensitive operations (invDDIM, Heun steps)
- **Multi-GPU support** — DDP with proper gradient synchronization

## Quick Start

**Prerequisites**: PyTorch 2.0+, CUDA. Install dependencies:
```bash
conda env create -f environment.yml
conda activate edm
```

**Training** (4 GPUs, ImageNet-64):
```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --outdir=training-runs --data=datasets/imagenet-64x64.zip \
  --cond=1 --arch=adm --precond=edm \
  --batch=2048 --batch-gpu=128 --fp16=True --ema=50 --lr=2e-5 \
  --consistency=True --teacher=edm-imagenet-64x64-cond-adm.pkl \
  --S=8 --T_start=64 --T_end=1280 --T_anneal_kimg=204800 \
  --cd_target_mode=ema \
  --val=1 --val_ref=fid-refs/imagenet-64x64.npz --val_every=20 \
  --wandb=True --wandb_project=edm-cd
```

**Generation** (8-step student):
```bash
torchrun --standalone --nproc_per_node=4 generate.py \
  --network=training-runs/your-snapshot.pkl \
  --outdir=samples --seeds=0-49999 --batch=128 --steps=8
```

**FID Evaluation**:
```bash
python fid.py calc --images=samples --ref=fid-refs/imagenet-64x64.npz
```

## Key Differences from Original MSCD

| Aspect | Original (VP) | This (EDM) |
|--------|---------------|------------|
| Noise schedule | Cosine | Karras (ρ=7) |
| Teacher sampler | aDDIM | Heun 2nd-order |
| Parameterization | ε-prediction | EDM preconditioning |
| Segmentation | Index-based | Sigma-anchored |

## Key Files

- `train.py` — main training script with CD options
- `training/loss_cd.py` — consistency distillation loss implementation
- `training/consistency_ops.py` — invDDIM, Heun hops, sigma grid utilities
- `training/validation.py` — built-in FID evaluation
- `generate.py` — multi-GPU image generation
- `fid.py` — FID computation

## References

- Heek, J., Hoogeboom, E., & Salimans, T. (2024). *Multistep Consistency Models*. arXiv:2403.06807
- Karras, T., et al. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. NeurIPS 2022.

## License

Inherits original EDM license. See `LICENSE.txt`.
