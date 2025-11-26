"""
Example script demonstrating how to use the CD debug harness.
This is a Python version that can be easily converted to a notebook.
"""

import torch
import pickle
import copy
import os

# Import CD loss
from training.loss_cd import EDMConsistencyDistillLoss
from training.dataset import ImageFolderDataset


def main():
    print("="*80)
    print("EDM Consistency Distillation Debug Harness Example")
    print("="*80)
    
    # Configuration
    teacher_path = 'path/to/teacher.pkl'  # UPDATE THIS
    data_path = 'path/to/dataset.zip'     # UPDATE THIS
    use_labels = True  # Set to False for unconditional
    
    # 1. Load Teacher Network
    print("\n1. Loading teacher network...")
    with open(teacher_path, 'rb') as f:
        teacher_data = pickle.load(f)
    
    if isinstance(teacher_data, dict) and ('ema' in teacher_data or 'net' in teacher_data):
        teacher_net = teacher_data['ema'] if 'ema' in teacher_data else teacher_data['net']
    else:
        teacher_net = teacher_data
    
    teacher_net = teacher_net.eval().requires_grad_(False).cuda()
    print(f"   Teacher loaded: {type(teacher_net).__name__}")
    
    # 2. Create Student Network
    print("\n2. Creating student network...")
    student_net = copy.deepcopy(teacher_net).cuda()
    print(f"   Student created: {type(student_net).__name__}")
    
    # 3. Create CD Loss Object
    print("\n3. Creating CD loss object...")
    cd_loss = EDMConsistencyDistillLoss(
        teacher_net=teacher_net,
        S=8,
        T_start=256,
        T_end=1024,
        T_anneal_kimg=750,
        rho=7.0,
        sigma_min=0.002,
        sigma_max=80.0,
        loss_type='huber',
        weight_mode='edm',
        enable_stats=False,
    )
    print(f"   CD loss initialized with S={cd_loss.S}, T_edges={cd_loss._current_T_edges()}")
    
    # 4. Load Dataset and Get Mini-batch
    print("\n4. Loading dataset...")
    dataset = ImageFolderDataset(
        path=data_path,
        use_labels=use_labels,
        xflip=False,
        cache=False,
    )
    print(f"   Dataset: {len(dataset)} images, resolution={dataset.resolution}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )
    
    images, labels = next(iter(dataloader))
    images = images.cuda()
    labels = labels.cuda() if use_labels else None
    print(f"   Mini-batch: images.shape={images.shape}")
    
    # 5. Run Debug Harness
    print("\n5. Running debug harness...")
    result = cd_loss.debug_batch(
        net=student_net,
        images=images,
        labels=labels,
        augment_pipe=None,
        output_dir='cd_debug_example',
        num_samples_visual=4,
        num_sigma_bins=3,
        run_teacher_self_test=True,
        global_step=0,
    )
    
    # 6. View Results
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"\nReport: {result['report_path']}")
    print(f"Issues detected: {result['num_problems']}")
    
    print("\nBin Statistics:")
    for stat in result['bin_stats']:
        noise_level = ['high', 'mid', 'low'][stat['bin']] if stat['bin'] < 3 else 'other'
        print(f"\n  Bin {stat['bin']} ({noise_level} noise):")
        print(f"    Segment j={stat['j']}, Edge e={stat['e']}")
        print(f"    σ_t={stat['sigma_t']:.6f}, σ_s={stat['sigma_s']:.6f}, σ_bdry={stat['sigma_bdry']:.6f}")
        print(f"    Loss: {stat['loss_mean']:.6f}")
        print(f"    RMS difference: {stat['rms_diff']:.6f}")
    
    if result['num_problems'] > 0:
        print(f"\nIssues found:")
        for prob in result['problems']:
            print(f"  - {prob['type']}: {prob}")
    else:
        print("\nNo issues detected! ✓")
    
    print(f"\nFull report and visualizations saved to: {result['output_dir']}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
