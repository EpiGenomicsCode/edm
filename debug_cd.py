#!/usr/bin/env python3
"""
Standalone script to run the CD debugging harness on a single mini-batch.

Usage:
    python debug_cd.py --teacher=path/to/teacher.pkl --data=path/to/dataset.zip [options]

Example:
    python debug_cd.py \
        --teacher=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
        --data=datasets/imagenet-64x64.zip \
        --cond=1 \
        --batch=8 \
        --num_samples=4 \
        --output_dir=cd_debug_output
"""

import os
import sys
import click
import torch
import pickle
import dnnlib

from training.loss_cd import EDMConsistencyDistillLoss
from training.dataset import ImageFolderDataset


@click.command()
@click.option('--teacher', help='Path/URL to pre-trained teacher (EDM-precond UNet)', metavar='PKL|URL', type=str, required=True)
@click.option('--data', help='Path to the dataset', metavar='ZIP|DIR', type=str, required=True)
@click.option('--cond', help='Use class-conditional model', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--batch', help='Batch size for debug run', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--num_samples', help='Number of samples to visualize', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--num_bins', help='Number of noise bins (low/mid/high)', metavar='INT', type=click.IntRange(min=2), default=3, show_default=True)
@click.option('--output_dir', help='Output directory for report and images', metavar='DIR', type=str, default='cd_debug', show_default=True)
@click.option('--S', 'S', help='Student step count', metavar='INT', type=click.IntRange(min=2), default=8, show_default=True)
@click.option('--T_start', 'T_start', help='Initial teacher edges', metavar='INT', type=click.IntRange(min=2), default=256, show_default=True)
@click.option('--T_end', 'T_end', help='Final teacher edges', metavar='INT', type=click.IntRange(min=2), default=1024, show_default=True)
@click.option('--T_anneal_kimg', 'T_anneal_kimg', help='kimg horizon for linear anneal', metavar='INT', type=click.IntRange(min=0), default=750, show_default=True)
@click.option('--rho', help='Karras rho exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7.0, show_default=True)
@click.option('--sigma_min', help='Minimum sigma for grids', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max', help='Maximum sigma for grids', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=80.0, show_default=True)
@click.option('--cd_loss', help='Consistency loss type', metavar='huber|l2', type=click.Choice(['huber', 'l2']), default='huber', show_default=True)
@click.option(
    '--cd_weight_mode',
    help='Consistency weight mode',
    metavar='edm|vlike|flat|snr|snr+1|karras|truncated-snr|uniform',
    type=click.Choice(['edm', 'vlike', 'flat', 'snr', 'snr+1', 'karras', 'truncated-snr', 'uniform']),
    default='edm',
    show_default=True,
)
@click.option('--teacher_self_test', help='Run teacher-as-student self-consistency test', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--global_step', help='Optional: global step number for naming', metavar='INT', type=int, default=None)
@click.option('--seed', help='Random seed', metavar='INT', type=int, default=0, show_default=True)
@click.option('--device', help='Device to use', metavar='cuda|cpu', type=str, default='cuda', show_default=True)
def main(**kwargs):
    """
    Run CD debugging harness on a single mini-batch.
    
    This script:
    1. Loads a teacher network
    2. Creates a student network (initialized randomly or from teacher)
    3. Loads one mini-batch from the dataset
    4. Runs the debug_batch method to generate a comprehensive report
    """
    opts = dnnlib.EasyDict(kwargs)
    
    # Set device
    device = torch.device(opts.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)
    
    print("\n" + "="*80)
    print("CD Debug Harness")
    print("="*80)
    
    # Load teacher network
    print(f"\nLoading teacher from: {opts.teacher}")
    with dnnlib.util.open_url(opts.teacher, verbose=True) as f:
        teacher_data = pickle.load(f)
    
    if isinstance(teacher_data, dict) and ('ema' in teacher_data or 'net' in teacher_data):
        teacher_net = teacher_data['ema'] if 'ema' in teacher_data else teacher_data['net']
    else:
        teacher_net = teacher_data
    
    teacher_net = teacher_net.eval().requires_grad_(False).to(device)
    print(f"Teacher loaded successfully. Type: {type(teacher_net).__name__}")
    
    # Create student network (clone of teacher for debugging purposes)
    print("\nCreating student network (cloned from teacher)...")
    import copy
    student_net = copy.deepcopy(teacher_net).to(device)
    print(f"Student created. Type: {type(student_net).__name__}")
    
    # Create CD loss object
    print("\nInitializing CD loss...")
    cd_loss = EDMConsistencyDistillLoss(
        teacher_net=teacher_net,
        S=opts.S,
        T_start=opts.T_start,
        T_end=opts.T_end,
        T_anneal_kimg=opts.T_anneal_kimg,
        rho=opts.rho,
        sigma_min=opts.sigma_min,
        sigma_max=opts.sigma_max,
        loss_type=opts.cd_loss,
        weight_mode=opts.cd_weight_mode,
        enable_stats=False,  # Disable training stats for debug
    )
    print(f"CD loss initialized with S={opts.S}, T_start={opts.T_start}, T_end={opts.T_end}")
    
    # Load dataset
    print(f"\nLoading dataset from: {opts.data}")
    dataset = ImageFolderDataset(
        path=opts.data,
        use_labels=opts.cond,
        xflip=False,
        cache=False,
    )
    print(f"Dataset loaded: {len(dataset)} images, resolution={dataset.resolution}")
    
    # Create dataloader for one batch
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # Get one mini-batch
    print(f"\nLoading mini-batch (batch_size={opts.batch})...")
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device) if opts.cond else None
    
    print(f"Mini-batch loaded: images.shape={images.shape}")
    if labels is not None:
        print(f"  labels.shape={labels.shape}, unique labels={torch.unique(labels).tolist()}")
    
    # Run debug harness
    print("\n" + "="*80)
    print("Running debug harness...")
    print("="*80 + "\n")
    
    result = cd_loss.debug_batch(
        net=student_net,
        images=images,
        labels=labels,
        augment_pipe=None,  # No augmentation for debugging
        output_dir=opts.output_dir,
        num_samples_visual=opts.num_samples,
        num_sigma_bins=opts.num_bins,
        run_teacher_self_test=opts.teacher_self_test,
        global_step=opts.global_step,
    )
    
    print("\n" + "="*80)
    print("Debug harness completed!")
    print("="*80)
    print(f"\nReport written to: {result['report_path']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Number of issues detected: {result['num_problems']}")
    
    if result['num_problems'] > 0:
        print("\nIssues found:")
        for prob in result['problems']:
            print(f"  - {prob['type']}: {prob}")
    
    print("\nBin statistics summary:")
    for stat in result['bin_stats']:
        print(f"  Bin {stat['bin']} (σ_t={stat['sigma_t']:.4f}): loss={stat['loss_mean']:.6f}, rms_diff={stat['rms_diff']:.6f}")
    
    print(f"\nVisualization images saved in: {opts.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()

