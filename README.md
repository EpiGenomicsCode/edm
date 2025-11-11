## Consistency Distillation for EDM (ImageNet-64, conditional) — In Progress

This repository is now focused on implementing Multistep Consistency Distillation (MCD) in the EDM framework for ImageNet-64 with class-conditioning. The legacy EDM README has been moved to docs (`docs/README-legacy-edm.md`)

### Status
- Target: S-step deterministic student (S ∈ {4, 8, 16}) distilled from a frozen EDM teacher using Heun integration over a denser Karras sigma grid, in EDM parameterization.

### What we’re building
- Consistency distillation in EDM space with:
  - Deterministic Heun teacher hops
  - Student single-eval Euler/DDIM forward
  - Segmenting the teacher grid and pushing to student boundaries
  - invDDIM targets and EDM-weighted Huber/L2 loss
  - Full DDP support, augmentation compatibility, and annealing of teacher edges

### Citations
- Multistep Consistency Models:
  - Heek, Jonathan, Emiel Hoogeboom, and Tim Salimans. “Multistep consistency models.” arXiv preprint arXiv:2403.06807 (2024).
- EDM:
  - Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. “Elucidating the Design Space of Diffusion-Based Generative Models.” NeurIPS 2022.

### Datasets and evaluation
- ImageNet-64 preprocessing and FID computation remain supported; see `dataset_tool.py` and `fid.py`.
- During development, we will periodically evaluate S-step deterministic sampling and report FID against `imagenet-64x64.npz`.

### License
This repo inherits the original EDM license terms. See `LICENSE.txt`.
