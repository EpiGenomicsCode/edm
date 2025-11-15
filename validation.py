# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
"""Distributed validation hook with FID evaluation during training."""

import os
import math
import json
import pickle
import numpy as np
import torch
import dnnlib
from typing import Optional, Dict, Any

from torch_utils import distributed as dist
from torch_utils import misc
import tqdm

# Reuse existing samplers & helpers.
from generate import edm_sampler, ablation_sampler, StackedRandomGenerator
from fid import calculate_fid_from_inception_stats, calculate_inception_stats

#----------------------------------------------------------------------------

def _load_inception_detector(device: torch.device):
    # Copied verbatim from fid.py to ensure parity.
    import dnnlib as _dnnlib
    import pickle as _pickle
    dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    with _dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = _pickle.load(f).to(device)
    return detector_net, detector_kwargs, 2048

#----------------------------------------------------------------------------

def _prepare_reference_stats(ref: Optional[str], ref_data: Optional[str], *, batch: int, device: torch.device, seed: int, cache_dir: Optional[str]):
    mu_ref = None
    sigma_ref = None
    if ref is not None:
        if dist.get_rank() == 0:
            with dnnlib.util.open_url(ref) as f:
                ref_npz = dict(np.load(f))
                mu_ref = ref_npz['mu']
                sigma_ref = ref_npz['sigma']
        # Ensure all ranks wait for rank0 to load.
        torch.distributed.barrier()
        # Broadcast via fileless approach: just re-open on other ranks as well for simplicity.
        if dist.get_rank() != 0:
            with dnnlib.util.open_url(ref) as f:
                ref_npz = dict(np.load(f))
                mu_ref = ref_npz['mu']
                sigma_ref = ref_npz['sigma']
        return mu_ref, sigma_ref
    if ref_data is not None:
        # Compute dataset stats once and optionally cache.
        mu_ref, sigma_ref = calculate_inception_stats(image_path=ref_data, max_batch_size=batch, seed=seed, device=device)
        if dist.get_rank() == 0 and cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            # Construct a simple filename from dataset path.
            base = os.path.basename(ref_data).replace('.zip', '').replace('.ZIP', '')
            dest = os.path.join(cache_dir, f'{base}.npz')
            np.savez(dest, mu=mu_ref, sigma=sigma_ref)
        torch.distributed.barrier()
        return mu_ref, sigma_ref
    raise RuntimeError('Validation requires either ref (npz/url) or ref_data (dataset path).')

#----------------------------------------------------------------------------

def run_fid_validation(
    net: torch.nn.Module,
    *,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    num_images: int = 50000,
    batch: int = 64,
    seed: int = 0,
    sampler: Dict[str, Any],
    labels: str = 'auto',  # 'auto'|'uniform'|'dataset'|'fixed:K'
    ref: Optional[str] = None,
    ref_data: Optional[str] = None,
    dump_images_dir: Optional[str] = None,
    overwrite: bool = False,
    step_kimg: Optional[int] = None,
    wandb_run: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute FID by generating images with the current EMA network."""
    device = torch.device('cuda')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    t0 = time.time() if 'time' in globals() else __import__('time').time()
    dist.print0(f'[VAL DEBUG] enter run_fid_validation: world={world_size} rank={rank} num_images={num_images} batch={batch}')

    # Short-circuit if snapshot for this step exists and overwrite is False.
    if (step_kimg is not None) and (rank == 0):
        step_json = os.path.join(run_dir, f'val_{int(step_kimg):06d}.json')
        if (not overwrite) and os.path.isfile(step_json):
            dist.print0(f'[VAL] Skipping validation at kimg={step_kimg} because results already exist.')
            torch.distributed.barrier()
            return {}
    # Note: barrier synchronization is performed before entering this function.

    # Ensure EMA is synchronized across ranks before validation.
    try:
        misc.check_ddp_consistency(net)
    except Exception:
        pass
    net = net.eval().requires_grad_(False).to(device)
    dist.print0(f'[VAL DEBUG] after DDP consistency check; elapsed {(__import__("time").time()-t0):.1f}s')

    # Reference stats.
    cache_dir = os.path.join(run_dir, 'fid-refs')
    dist.print0('[VAL DEBUG] loading reference stats...')
    mu_ref, sigma_ref = _prepare_reference_stats(ref, ref_data, batch=batch, device=device, seed=seed, cache_dir=cache_dir)
    dist.print0(f'[VAL DEBUG] reference loaded; elapsed {(__import__("time").time()-t0):.1f}s')

    # Inception on each rank, rank 0 goes first (mirror fid.py barrier pattern).
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    dist.print0('[VAL DEBUG] before inception load (rank0 will print the loader message next)')
    detector, detector_kwargs, feature_dim = _load_inception_detector(device)
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    dist.print0(f'[VAL DEBUG] inception ready; elapsed {(__import__("time").time()-t0):.1f}s')

    # Seed assignment and sharding.
    all_indices = torch.arange(num_images, device=torch.device('cpu'))
    num_batches = math.ceil(num_images / (batch * world_size)) * world_size
    all_batches = all_indices.tensor_split(num_batches)
    rank_batches = all_batches[rank :: world_size]

    if rank == 0:
        dist.print0(f'[VAL] Starting validation: num_images={num_images}, batches/world={num_batches}, batch_per_gpu={batch}')
        dist.print0(f'[VAL DEBUG] rank0 has {sum(1 for b in rank_batches if len(b)>0)} non-empty batches')

    # Accumulators in FP64.
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    # Label mode resolution.
    label_dim = getattr(net, 'label_dim', 0)
    use_labels = bool(label_dim and dataset_kwargs.get('use_labels', False))
    effective_mode = labels
    if labels == 'auto':
        effective_mode = 'uniform' if use_labels else 'none'
    if not use_labels:
        effective_mode = 'none'

    # Iterate rank-local batches.
    # Rank-local batch list and simple counters for explicit progress logs.
    rank_batches_list = list(rank_batches)
    non_empty_rank_batches = sum(1 for b in rank_batches_list if len(b) > 0)
    progress = tqdm.tqdm(rank_batches_list, unit='batch', disable=(rank != 0), ascii=True, mininterval=5.0, miniters=1, dynamic_ncols=False)
    local_batch_idx = 0
    for b_idxs in progress:
        torch.distributed.barrier()
        bsize = len(b_idxs)
        if bsize == 0:
            continue
        local_batch_idx += 1
        if rank == 0 and (local_batch_idx == 1 or (local_batch_idx % 10 == 0) or (local_batch_idx == non_empty_rank_batches)):
            pct = 100.0 * local_batch_idx / max(non_empty_rank_batches, 1)
            dist.print0(f'[VAL] Progress (rank0): {local_batch_idx}/{non_empty_rank_batches} ({pct:.1f}%)')
        # Per-sample seeds are base + index.
        seeds = (seed + b_idxs).tolist()
        rnd = StackedRandomGenerator(device, seeds)
        # Latents: match network interface.
        latents = rnd.randn([bsize, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if effective_mode != 'none':
            class_labels = torch.eye(label_dim, device=device)[rnd.randint(label_dim, size=[bsize], device=device)]
            if effective_mode.startswith('fixed:'):
                try:
                    k = int(effective_mode.split(':', 1)[1])
                except Exception:
                    k = 0
                class_labels[:, :] = 0
                class_labels[:, int(k) % label_dim] = 1
            # 'uniform' already satisfied by randint above.

        # Sampler selection mirroring generate.py.
        sampler_kind = (sampler or {}).get('kind', 'edm')
        sampler_kwargs = {k: v for k, v in (sampler or {}).items() if v is not None and k != 'kind'}
        have_ablation = (sampler_kind == 'ablate') or any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation else edm_sampler
        images = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Optional dump images for audit.
        if dump_images_dir is not None and rank == 0:
            import PIL.Image
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for idx, image_np in zip(b_idxs.tolist(), images_np):
                os.makedirs(dump_images_dir, exist_ok=True)
                image_path = os.path.join(dump_images_dir, f'{int(idx):06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        # Inception features and running stats.
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        # Match fid.py expectations: detector consumes uint8 0..255 tensors.
        images_u8 = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        feats = detector(images_u8.to(device), **detector_kwargs).to(torch.float64)
        mu += feats.sum(0)
        sigma += feats.T @ feats

    # Reduce and finalize.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= (num_images - 1)

    # Compute FID on rank 0.
    fid_value = None
    if dist.get_rank() == 0:
        fid_value = calculate_fid_from_inception_stats(mu.cpu().numpy(), sigma.cpu().numpy(), mu_ref, sigma_ref)
        # Write metrics.
        result = dict(
            kimg=int(step_kimg) if step_kimg is not None else None,
            fid=float(fid_value),
            num_images=int(num_images),
        )
        # Append jsonl.
        try:
            with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
                f.write(json.dumps(dict(result, timestamp=float(torch.tensor([]).new_zeros(1).cpu().numpy().item() if False else float(torch.tensor(0).item())))) + '\n')
        except Exception:
            # Fallback without timestamp.
            with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
                f.write(json.dumps(result) + '\n')
        # Snapshot full JSON.
        if step_kimg is not None:
            payload = dict(
                fid=float(fid_value),
                mu=mu.cpu().numpy().tolist(),
                sigma=sigma.cpu().numpy().tolist(),
                num_images=int(num_images),
                step_kimg=int(step_kimg),
                sampler=sampler or {},
                labels=labels,
                ref=ref,
                ref_data=ref_data,
            )
            with open(os.path.join(run_dir, f'val_{int(step_kimg):06d}.json'), 'wt') as f:
                json.dump(payload, f)
        # W&B logging if available.
        if wandb_run is not None:
            try:
                import wandb as _wandb
                _wandb.log({'val/fid': float(fid_value), 'val/num_images': int(num_images), 'progress_kimg': int(step_kimg) if step_kimg is not None else None}, commit=True)
            except Exception:
                pass
        dist.print0(f'[VAL] kimg={step_kimg} FID={fid_value:g}')
    torch.distributed.barrier()
    return {'fid': float(fid_value) if fid_value is not None else None}

#----------------------------------------------------------------------------

def maybe_validate(
    *,
    step_tick: int,
    step_kimg: int,
    net_ema: torch.nn.Module,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    validation_kwargs: Optional[Dict[str, Any]],
    wandb_run: Optional[Any],
):
    """Lightweight scheduler; call per tick."""
    if validation_kwargs is None:
        return
    if not validation_kwargs.get('enabled', True):
        return
    every = int(validation_kwargs.get('every', 0) or 0)
    at_start = bool(validation_kwargs.get('at_start', False))
    should_run = False
    # Only run at tick 0 if explicitly requested via at_start.
    if step_tick == 0:
        should_run = at_start
    else:
        if every > 0 and (step_tick % every == 0):
            should_run = True
    if not should_run:
        return
    # Broadcast a shared decision (bool -> int tensor).
    flag = torch.tensor([1 if should_run else 0], dtype=torch.int64, device=torch.device('cuda'))
    torch.distributed.broadcast(flag, src=0)
    if int(flag.item()) == 0:
        return
    # Unpack kwargs (keep names aligned with PRD).
    sampler_cfg = validation_kwargs.get('sampler', {}) or {}
    return run_fid_validation(
        net_ema,
        run_dir=run_dir,
        dataset_kwargs=dataset_kwargs,
        num_images=int(validation_kwargs.get('num_images', 50000)),
        batch=int(validation_kwargs.get('batch', 64)),
        seed=int(validation_kwargs.get('seed', 0)),
        sampler=sampler_cfg,
        labels=validation_kwargs.get('labels', 'auto'),
        ref=validation_kwargs.get('ref', None),
        ref_data=validation_kwargs.get('ref_data', None),
        dump_images_dir=validation_kwargs.get('dump_images_dir', None),
        overwrite=bool(validation_kwargs.get('overwrite', False)),
        step_kimg=int(step_kimg),
        wandb_run=wandb_run,
    )

#----------------------------------------------------------------------------

