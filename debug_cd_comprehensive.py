#!/usr/bin/env python3
"""
Comprehensive CD debugging harness with trajectory comparison and training debugging.

This script implements the debugging framework outlined in debug.md:
1. Trajectory debugging: Compare teacher vs student sampling step-by-step
2. Training debugging: Run a few training steps with extensive logging
3. Enhanced numeric checks: conditioning ratios, ill-conditioning, correlation analysis

Usage:
    python debug_cd_comprehensive.py \\
        --teacher=path/to/teacher.pkl \\
        --student=path/to/student.pkl \\
        --data=path/to/dataset.zip \\
        --mode=trajectory \\
        --outdir=cd_debug_comprehensive
"""

import os
import sys
import json
import math
import copy
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import click
import torch
import torch.nn.functional as F
import pickle
import numpy as np

import dnnlib
from training.loss_cd import EDMConsistencyDistillLoss, _save_image_grid, _huber_loss
from training.dataset import ImageFolderDataset
from training.consistency_ops import (
    make_karras_sigmas,
    heun_hop_edm,
    ddim_step_edm,
    inv_ddim_edm,
)
from torch_utils import misc


# ============================================================================
# TRAJECTORY DEBUGGING
# ============================================================================

def sample_student_trajectory(
    net,
    sigmas: torch.Tensor,
    seed: int,
    labels: Optional[torch.Tensor] = None,
    device: torch.device = torch.device('cuda'),
) -> List[torch.Tensor]:
    """
    Sample a full S-step trajectory using the student network.
    
    Args:
        net: Student network (EDMPrecond)
        sigmas: Noise schedule [S+1] with terminal 0
        seed: Random seed for noise
        labels: Optional class labels [N]
        device: Device
    
    Returns:
        List of length S containing intermediate states x at each step
    """
    torch.manual_seed(seed)
    N = 1 if labels is None else labels.shape[0]
    C, H, W = net.img_channels, net.img_resolution, net.img_resolution
    
    # Initial noise
    x = torch.randn(N, C, H, W, device=device) * sigmas[0].item()
    
    xs = []
    with torch.no_grad():
        for k in range(len(sigmas) - 1):
            sigma_t = sigmas[k]
            sigma_s = sigmas[k + 1]
            
            # Student denoises at sigma_t
            x_pred = net(x, sigma_t.reshape(1), labels)
            
            # DDIM step to sigma_s
            x = ddim_step_edm(x, x_pred, sigma_t, sigma_s)
            xs.append(x.clone())
    
    return xs


def sample_teacher_trajectory(
    teacher_net,
    sigmas: torch.Tensor,
    seed: int,
    labels: Optional[torch.Tensor] = None,
    device: torch.device = torch.device('cuda'),
    use_heun: bool = True,
) -> List[torch.Tensor]:
    """
    Sample a full S-step trajectory using the teacher network.
    
    Args:
        teacher_net: Teacher network (EDMPrecond)
        sigmas: Noise schedule [S+1] with terminal 0
        seed: Random seed for noise (must match student for fair comparison)
        labels: Optional class labels [N]
        device: Device
        use_heun: If True, use Heun sampler; otherwise use DDIM
    
    Returns:
        List of length S containing intermediate states x at each step
    """
    torch.manual_seed(seed)
    N = 1 if labels is None else labels.shape[0]
    C, H, W = teacher_net.img_channels, teacher_net.img_resolution, teacher_net.img_resolution
    
    # Initial noise (same seed as student)
    x = torch.randn(N, C, H, W, device=device) * sigmas[0].item()
    
    xs = []
    with torch.no_grad():
        for k in range(len(sigmas) - 1):
            sigma_t = sigmas[k]
            sigma_s = sigmas[k + 1]
            
            if use_heun and sigma_s > 0:
                # Heun sampler
                x = heun_hop_edm(teacher_net, x, sigma_t, sigma_s, labels)
            else:
                # DDIM sampler (fallback for sigma_s=0)
                x_pred = teacher_net(x, sigma_t.reshape(1), labels)
                x = ddim_step_edm(x, x_pred, sigma_t, sigma_s)
            
            xs.append(x.clone())
    
    return xs


def compute_trajectory_metrics(
    xs_teacher: List[torch.Tensor],
    xs_student: List[torch.Tensor],
) -> List[Dict[str, float]]:
    """
    Compute per-step metrics between teacher and student trajectories.
    
    Returns:
        List of dicts with metrics for each step
    """
    metrics = []
    
    for step_idx, (x_tch, x_std) in enumerate(zip(xs_teacher, xs_student)):
        diff = x_std - x_tch
        
        # RMS norms
        rms_teacher = float(torch.sqrt(torch.mean(x_tch.float() ** 2)).cpu())
        rms_student = float(torch.sqrt(torch.mean(x_std.float() ** 2)).cpu())
        rms_diff = float(torch.sqrt(torch.mean(diff.float() ** 2)).cpu())
        
        # Max abs difference
        max_abs_diff = float(diff.abs().max().cpu())
        
        # Cosine similarity (flatten and normalize)
        x_tch_flat = x_tch.view(x_tch.size(0), -1).float()
        x_std_flat = x_std.view(x_std.size(0), -1).float()
        
        # Mean-center
        x_tch_flat = x_tch_flat - x_tch_flat.mean(dim=1, keepdim=True)
        x_std_flat = x_std_flat - x_std_flat.mean(dim=1, keepdim=True)
        
        # Cosine similarity
        cos_sim = torch.sum(x_tch_flat * x_std_flat, dim=1) / (
            torch.sqrt(torch.sum(x_tch_flat ** 2, dim=1) + 1e-8) *
            torch.sqrt(torch.sum(x_std_flat ** 2, dim=1) + 1e-8)
        )
        cos_sim_mean = float(cos_sim.mean().cpu())
        
        metrics.append({
            'step': step_idx,
            'rms_teacher': rms_teacher,
            'rms_student': rms_student,
            'rms_diff': rms_diff,
            'max_abs_diff': max_abs_diff,
            'cosine_sim': cos_sim_mean,
        })
    
    return metrics


def visualize_trajectory(
    xs_teacher: List[torch.Tensor],
    xs_student: List[torch.Tensor],
    sigmas: torch.Tensor,
    seed: int,
    output_dir: str,
    sample_idx: int = 0,
):
    """
    Create a visualization grid comparing teacher vs student trajectories.
    
    Grid layout:
        Row 0: Teacher trajectory at steps 1...S
        Row 1: Student trajectory at steps 1...S
        Row 2: |Student - Teacher| heatmap
    """
    S = len(xs_teacher)
    
    def normalize_for_vis(t):
        """Normalize tensor to [0, 1] for visualization."""
        t = t.detach().cpu().float()
        t = torch.clamp(t, -1, 1)
        t = (t + 1) / 2  # [-1, 1] -> [0, 1]
        return t
    
    # Extract single sample
    teacher_imgs = [normalize_for_vis(x[sample_idx:sample_idx+1]) for x in xs_teacher]
    student_imgs = [normalize_for_vis(x[sample_idx:sample_idx+1]) for x in xs_student]
    
    # Compute difference heatmaps
    diff_imgs = []
    for x_tch, x_std in zip(xs_teacher, xs_student):
        diff = (x_std - x_tch).abs()
        diff_norm = normalize_for_vis(diff[sample_idx:sample_idx+1])
        diff_imgs.append(diff_norm)
    
    # Interleave for grid: teacher, student, diff for each step
    all_imgs = []
    for step_idx in range(S):
        all_imgs.extend([
            teacher_imgs[step_idx],
            student_imgs[step_idx],
            diff_imgs[step_idx],
        ])
    
    # Save grid
    img_path = os.path.join(output_dir, f'trajectory_seed_{seed}_sample_{sample_idx}.png')
    _save_image_grid(all_imgs, img_path, nrow=3)
    
    return img_path


def run_trajectory_debug(
    teacher_pkl: str,
    student_pkl: str,
    dataset_kwargs: Dict[str, Any],
    S: int,
    num_seeds: int,
    num_samples_per_seed: int,
    outdir: str,
    use_labels: bool,
    rho: float,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
):
    """
    Main trajectory debugging workflow.
    """
    os.makedirs(outdir, exist_ok=True)
    
    print("="*80)
    print("TRAJECTORY DEBUGGING")
    print("="*80)
    
    # Import training modules to support relative imports in checkpoints
    import training.networks
    import training.loss
    import training.loss_cd
    
    # Load networks
    print(f"\n1. Loading teacher from: {teacher_pkl}")
    with dnnlib.util.open_url(teacher_pkl, verbose=True) as f:
        teacher_data = pickle.load(f)
    teacher_net = teacher_data.get('ema', teacher_data.get('net', teacher_data))
    teacher_net = teacher_net.eval().requires_grad_(False).to(device)
    print(f"   Teacher loaded: {type(teacher_net).__name__}")
    
    print(f"\n2. Loading student from: {student_pkl}")
    with dnnlib.util.open_url(student_pkl, verbose=True) as f:
        student_data = pickle.load(f)
    student_net = student_data.get('ema', student_data.get('net', student_data))
    student_net = student_net.eval().requires_grad_(False).to(device)
    print(f"   Student loaded: {type(student_net).__name__}")
    
    # Build student grid
    print(f"\n3. Building sigma grid (S={S})")
    student_sigmas = make_karras_sigmas(
        num_nodes=S,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        round_fn=student_net.round_sigma,
    ).to(device)
    zero = torch.zeros(1, device=device, dtype=student_sigmas.dtype)
    student_sigmas = torch.cat([student_sigmas, zero], dim=0)
    print(f"   Sigmas: {[f'{s:.4f}' for s in student_sigmas.cpu().tolist()]}")
    
    # Load dataset
    print(f"\n4. Loading dataset")
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    print(f"   Dataset: {len(dataset)} images, resolution={dataset.resolution}")
    
    # Create report
    report_path = os.path.join(outdir, 'trajectory_report.md')
    report_lines = []
    report_lines.append("# Trajectory Debugging Report")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n## Configuration")
    report_lines.append(f"- Teacher: `{teacher_pkl}`")
    report_lines.append(f"- Student: `{student_pkl}`")
    report_lines.append(f"- S (steps): {S}")
    report_lines.append(f"- Number of seeds: {num_seeds}")
    report_lines.append(f"- Samples per seed: {num_samples_per_seed}")
    report_lines.append(f"- Sigma range: [{sigma_min}, {sigma_max}]")
    report_lines.append(f"- Rho: {rho}")
    report_lines.append("")
    
    # Run trajectory comparison for each seed
    print(f"\n5. Running trajectory comparison ({num_seeds} seeds)")
    all_metrics = []
    
    for seed_idx in range(num_seeds):
        seed = seed_idx * 1000  # Space out seeds
        print(f"\n   Seed {seed}...")
        
        # Get labels if needed
        labels = None
        if use_labels:
            # Sample random labels
            num_classes = dataset.label_dim if hasattr(dataset, 'label_dim') else 1000
            labels = torch.randint(0, num_classes, (num_samples_per_seed,), device=device)
        
        # Sample trajectories
        xs_teacher = sample_teacher_trajectory(
            teacher_net, student_sigmas, seed, labels, device, use_heun=True
        )
        xs_student = sample_student_trajectory(
            student_net, student_sigmas, seed, labels, device
        )
        
        # Compute metrics
        metrics = compute_trajectory_metrics(xs_teacher, xs_student)
        all_metrics.append({'seed': seed, 'metrics': metrics})
        
        # Visualize (one sample per seed)
        for sample_idx in range(min(num_samples_per_seed, 2)):
            img_path = visualize_trajectory(
                xs_teacher, xs_student, student_sigmas, seed, outdir, sample_idx
            )
            print(f"      Saved: {os.path.basename(img_path)}")
        
        # Add to report
        report_lines.append(f"### Seed {seed}")
        report_lines.append("")
        report_lines.append("| Step | σ | RMS Teacher | RMS Student | RMS Diff | Max Diff | Cos Sim |")
        report_lines.append("|------|---|-------------|-------------|----------|----------|---------|")
        for m in metrics:
            sigma_val = student_sigmas[m['step'] + 1].item()
            report_lines.append(
                f"| {m['step']} | {sigma_val:.4f} | {m['rms_teacher']:.4f} | "
                f"{m['rms_student']:.4f} | {m['rms_diff']:.4f} | "
                f"{m['max_abs_diff']:.4f} | {m['cosine_sim']:.4f} |"
            )
        report_lines.append("")
    
    # Summary statistics
    print(f"\n6. Computing summary statistics")
    report_lines.append("## Summary Statistics (Averaged Across Seeds)")
    report_lines.append("")
    
    # Average metrics across seeds for each step
    num_steps = len(all_metrics[0]['metrics'])
    report_lines.append("| Step | σ | Avg RMS Diff | Avg Max Diff | Avg Cos Sim |")
    report_lines.append("|------|---|--------------|--------------|-------------|")
    
    for step_idx in range(num_steps):
        sigma_val = student_sigmas[step_idx + 1].item()
        rms_diffs = [m['metrics'][step_idx]['rms_diff'] for m in all_metrics]
        max_diffs = [m['metrics'][step_idx]['max_abs_diff'] for m in all_metrics]
        cos_sims = [m['metrics'][step_idx]['cosine_sim'] for m in all_metrics]
        
        avg_rms_diff = np.mean(rms_diffs)
        avg_max_diff = np.mean(max_diffs)
        avg_cos_sim = np.mean(cos_sims)
        
        report_lines.append(
            f"| {step_idx} | {sigma_val:.4f} | {avg_rms_diff:.4f} | "
            f"{avg_max_diff:.4f} | {avg_cos_sim:.4f} |"
        )
    report_lines.append("")
    
    # Save report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Save JSON
    json_path = os.path.join(outdir, 'trajectory_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Trajectory debugging complete!")
    print(f"Report: {report_path}")
    print(f"Metrics: {json_path}")
    print(f"Images: {outdir}/*.png")
    print(f"{'='*80}\n")


# ============================================================================
# TRAINING DEBUGGING
# ============================================================================

def run_training_debug(
    teacher_pkl: str,
    student_pkl: str,
    dataset_kwargs: Dict[str, Any],
    S: int,
    T_start: int,
    T_end: int,
    T_anneal_kimg: int,
    num_steps: int,
    batch_size: int,
    lr: float,
    outdir: str,
    use_labels: bool,
    rho: float,
    sigma_min: float,
    sigma_max: float,
    loss_type: str,
    weight_mode: str,
    debug_snapshot_steps: List[int],
    device: torch.device,
):
    """
    Main training debugging workflow: run a few training steps with extensive logging.
    """
    os.makedirs(outdir, exist_ok=True)
    
    print("="*80)
    print("TRAINING DEBUGGING")
    print("="*80)
    
    # Import training modules to support relative imports in checkpoints
    import training.networks
    import training.loss
    import training.loss_cd
    
    # Load networks
    print(f"\n1. Loading teacher from: {teacher_pkl}")
    with dnnlib.util.open_url(teacher_pkl, verbose=True) as f:
        teacher_data = pickle.load(f)
    teacher_net = teacher_data.get('ema', teacher_data.get('net', teacher_data))
    teacher_net = teacher_net.eval().requires_grad_(False).to(device)
    print(f"   Teacher loaded: {type(teacher_net).__name__}")
    
    print(f"\n2. Loading student from: {student_pkl}")
    with dnnlib.util.open_url(student_pkl, verbose=True) as f:
        student_data = pickle.load(f)
    student_net = student_data.get('ema', student_data.get('net', student_data))
    student_net = student_net.train().requires_grad_(True).to(device)
    print(f"   Student loaded: {type(student_net).__name__}")
    
    # Create CD loss
    print(f"\n3. Creating CD loss (S={S}, T_start={T_start}, T_end={T_end})")
    loss_fn = EDMConsistencyDistillLoss(
        teacher_net=teacher_net,
        S=S,
        T_start=T_start,
        T_end=T_end,
        T_anneal_kimg=T_anneal_kimg,
        rho=rho,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        loss_type=loss_type,
        weight_mode=weight_mode,
        enable_stats=False,
        debug_invariants=True,
    )
    
    # Load dataset
    print(f"\n4. Loading dataset")
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    print(f"   Dataset: {len(dataset)} images, resolution={dataset.resolution}")
    
    sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=0)
    loader = iter(torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
    ))
    
    # Optimizer
    print(f"\n5. Creating optimizer (lr={lr})")
    opt = torch.optim.AdamW(student_net.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Training loop
    print(f"\n6. Running training debug loop ({num_steps} steps)")
    log_path = os.path.join(outdir, 'train_debug_log.jsonl')
    
    for step in range(num_steps):
        # Get batch
        images, labels = next(loader)
        images = images.to(device).to(torch.float32) / 127.5 - 1
        if use_labels:
            labels = labels.to(device)
        else:
            labels = None
        
        # Forward pass
        opt.zero_grad()
        loss_full = loss_fn(net=student_net, images=images, labels=labels, augment_pipe=None)
        loss_scalar = loss_full.mean()
        
        # Backward pass
        loss_scalar.backward()
        
        # Compute gradient norm
        total_grad_sq = 0.0
        for p in student_net.parameters():
            if p.grad is not None:
                total_grad_sq += float(p.grad.detach().pow(2).sum().cpu())
        grad_norm = math.sqrt(total_grad_sq)
        
        # Optimizer step
        opt.step()
        
        # Get edge stats
        edge_stats = loss_fn.get_edge_stats(reset=True)
        
        # Log
        log_entry = {
            'step': step,
            'loss_mean': float(loss_scalar.detach().cpu()),
            'loss_std': float(loss_full.detach().std().cpu()),
            'grad_norm': grad_norm,
            'lr': lr,
            **edge_stats,
        }
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"   Step {step:4d}: loss={loss_scalar.item():.6f}, grad_norm={grad_norm:.4f}, "
              f"terminal={edge_stats['terminal_pct']:.1f}%, boundary={edge_stats['boundary_match_pct']:.1f}%")
        
        # Debug snapshots
        if step in debug_snapshot_steps:
            print(f"      Running debug_batch snapshot...")
            snapshot_dir = os.path.join(outdir, f'debug_step_{step}')
            loss_fn.debug_batch(
                net=student_net,
                images=images[:8],  # Limit to 8 samples for visualization
                labels=labels[:8] if labels is not None else None,
                augment_pipe=None,
                output_dir=snapshot_dir,
                num_samples_visual=4,
                num_sigma_bins=3,
                run_teacher_self_test=True,
                global_step=step,
            )
    
    print(f"\n{'='*80}")
    print(f"Training debugging complete!")
    print(f"Log: {log_path}")
    print(f"Snapshots: {outdir}/debug_step_*/")
    print(f"{'='*80}\n")


# ============================================================================
# ENHANCED CD BATCH DEBUG (with conditioning analysis)
# ============================================================================

def run_enhanced_batch_debug(
    teacher_pkl: str,
    student_pkl: str,
    dataset_kwargs: Dict[str, Any],
    S: int,
    T_start: int,
    T_end: int,
    T_anneal_kimg: int,
    batch_size: int,
    outdir: str,
    use_labels: bool,
    rho: float,
    sigma_min: float,
    sigma_max: float,
    loss_type: str,
    weight_mode: str,
    device: torch.device,
):
    """
    Run an enhanced batch debug with conditioning analysis.
    """
    os.makedirs(outdir, exist_ok=True)
    
    print("="*80)
    print("ENHANCED BATCH DEBUGGING (with Conditioning Analysis)")
    print("="*80)
    
    # Import training modules to support relative imports in checkpoints
    import training.networks
    import training.loss
    import training.loss_cd
    
    # Load networks
    print(f"\n1. Loading teacher from: {teacher_pkl}")
    with dnnlib.util.open_url(teacher_pkl, verbose=True) as f:
        teacher_data = pickle.load(f)
    teacher_net = teacher_data.get('ema', teacher_data.get('net', teacher_data))
    teacher_net = teacher_net.eval().requires_grad_(False).to(device)
    
    print(f"\n2. Loading student from: {student_pkl}")
    with dnnlib.util.open_url(student_pkl, verbose=True) as f:
        student_data = pickle.load(f)
    student_net = student_data.get('ema', student_data.get('net', student_data))
    student_net = student_net.eval().requires_grad_(False).to(device)
    
    # Create CD loss
    print(f"\n3. Creating CD loss")
    loss_fn = EDMConsistencyDistillLoss(
        teacher_net=teacher_net,
        S=S,
        T_start=T_start,
        T_end=T_end,
        T_anneal_kimg=T_anneal_kimg,
        rho=rho,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        loss_type=loss_type,
        weight_mode=weight_mode,
        enable_stats=False,
        debug_invariants=True,
    )
    
    # Load dataset and get batch
    print(f"\n4. Loading dataset and batch")
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    images, labels = next(iter(dataloader))
    images = images.to(device)
    if use_labels:
        labels = labels.to(device)
    else:
        labels = None
    
    # Run debug_batch
    print(f"\n5. Running enhanced debug_batch")
    result = loss_fn.debug_batch(
        net=student_net,
        images=images,
        labels=labels,
        augment_pipe=None,
        output_dir=outdir,
        num_samples_visual=4,
        num_sigma_bins=3,
        run_teacher_self_test=True,
        global_step=0,
    )
    
    print(f"\n{'='*80}")
    print(f"Enhanced batch debugging complete!")
    print(f"Report: {result['report_path']}")
    print(f"Issues found: {result['num_problems']}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option('--teacher', 'teacher_pkl', help='Path/URL to teacher checkpoint', type=str, required=True)
@click.option('--student', 'student_pkl', help='Path/URL to student checkpoint (if None, clone teacher)', type=str, default=None)
@click.option('--data', 'data_path', help='Path to dataset', type=str, required=True)
@click.option('--mode', help='Debug mode: trajectory|training|batch', type=click.Choice(['trajectory', 'training', 'batch']), default='trajectory')
@click.option('--outdir', help='Output directory', type=str, default='cd_debug_comprehensive')
@click.option('--S', 's', help='Student steps', type=int, default=8)
@click.option('--T_start', 't_start', help='Initial teacher edges', type=int, default=256)
@click.option('--T_end', 't_end', help='Final teacher edges', type=int, default=1024)
@click.option('--T_anneal_kimg', 't_anneal_kimg', help='Teacher anneal horizon (kimg)', type=int, default=750)
@click.option('--rho', help='Karras rho', type=float, default=7.0)
@click.option('--sigma_min', help='Min sigma', type=float, default=0.002)
@click.option('--sigma_max', help='Max sigma', type=float, default=80.0)
@click.option('--loss_type', help='Loss type', type=click.Choice(['huber', 'l2']), default='huber')
@click.option('--weight_mode', help='Weight mode', type=click.Choice(['edm', 'vlike']), default='edm')
@click.option('--cond', help='Use class-conditional', type=bool, default=False)
@click.option('--batch', help='Batch size', type=int, default=16)
@click.option('--num_seeds', help='Number of seeds for trajectory mode', type=int, default=8)
@click.option('--num_samples', help='Samples per seed for trajectory mode', type=int, default=1)
@click.option('--num_steps', help='Training steps for training mode', type=int, default=50)
@click.option('--lr', help='Learning rate for training mode', type=float, default=1e-5)
@click.option('--seed', help='Random seed', type=int, default=0)
@click.option('--device', help='Device', type=str, default='cuda')
def main(**kwargs):
    """Comprehensive CD debugging harness."""
    opts = dnnlib.EasyDict(kwargs)
    
    # Set device
    device = torch.device(opts.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)
    
    # If no student provided, clone teacher
    if opts.student_pkl is None:
        print("No student checkpoint provided, will clone teacher")
        opts.student_pkl = opts.teacher_pkl
    
    # Construct dataset kwargs
    dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=opts.data_path,
        use_labels=opts.cond,
        xflip=False,
        cache=False,
    )
    
    # Run appropriate mode
    if opts.mode == 'trajectory':
        run_trajectory_debug(
            teacher_pkl=opts.teacher_pkl,
            student_pkl=opts.student_pkl,
            dataset_kwargs=dataset_kwargs,
            S=opts.s,
            num_seeds=opts.num_seeds,
            num_samples_per_seed=opts.num_samples,
            outdir=opts.outdir,
            use_labels=opts.cond,
            rho=opts.rho,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            device=device,
        )
    elif opts.mode == 'training':
        debug_snapshot_steps = [0, opts.num_steps // 2, opts.num_steps - 1]
        run_training_debug(
            teacher_pkl=opts.teacher_pkl,
            student_pkl=opts.student_pkl,
            dataset_kwargs=dataset_kwargs,
            S=opts.s,
            T_start=opts.t_start,
            T_end=opts.t_end,
            T_anneal_kimg=opts.t_anneal_kimg,
            num_steps=opts.num_steps,
            batch_size=opts.batch,
            lr=opts.lr,
            outdir=opts.outdir,
            use_labels=opts.cond,
            rho=opts.rho,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            loss_type=opts.loss_type,
            weight_mode=opts.weight_mode,
            debug_snapshot_steps=debug_snapshot_steps,
            device=device,
        )
    elif opts.mode == 'batch':
        run_enhanced_batch_debug(
            teacher_pkl=opts.teacher_pkl,
            student_pkl=opts.student_pkl,
            dataset_kwargs=dataset_kwargs,
            S=opts.s,
            T_start=opts.t_start,
            T_end=opts.t_end,
            T_anneal_kimg=opts.t_anneal_kimg,
            batch_size=opts.batch,
            outdir=opts.outdir,
            use_labels=opts.cond,
            rho=opts.rho,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            loss_type=opts.loss_type,
            weight_mode=opts.weight_mode,
            device=device,
        )


if __name__ == '__main__':
    main()

