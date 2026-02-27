#!/usr/bin/env python3
"""
Simple test to verify the CD debug harness works correctly.
Creates synthetic data and a minimal network to test the pipeline.
"""

import torch
import torch.nn as nn
import os
import sys


class MinimalEDMNet(nn.Module):
    """Minimal EDM-preconditioned network for testing."""
    
    def __init__(self, img_resolution=32, img_channels=3):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        
        # Simple conv network
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, img_channels, 3, padding=1),
        )
    
    def round_sigma(self, sigma):
        """EDM sigma rounding (identity for testing)."""
        return torch.as_tensor(sigma)
    
    def forward(self, x, sigma, class_labels=None, augment_labels=None):
        """
        EDM forward interface: D(x; σ) = denoised prediction.
        For testing, just apply the network.
        """
        # Ensure correct dtype
        x = x.float()
        # Simple denoising: x - noise_prediction
        noise_pred = self.net(x)
        return x - noise_pred * 0.1  # Small correction


def test_debug_harness():
    """Test the debug harness with synthetic data."""
    
    print("="*80)
    print("Testing CD Debug Harness")
    print("="*80)
    
    # Import after path setup
    from training.loss_cd import EDMConsistencyDistillLoss
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create minimal networks
    print("\nCreating teacher and student networks...")
    teacher_net = MinimalEDMNet(img_resolution=32, img_channels=3).to(device).eval()
    student_net = MinimalEDMNet(img_resolution=32, img_channels=3).to(device)
    
    # Create CD loss
    print("Creating CD loss object...")
    cd_loss = EDMConsistencyDistillLoss(
        teacher_net=teacher_net,
        S=4,  # Small for testing
        T_start=16,  # Small for testing
        T_end=32,
        T_anneal_kimg=100,
        rho=7.0,
        sigma_min=0.002,
        sigma_max=80.0,
        loss_type='huber',
        weight_mode='edm',
        enable_stats=False,
    )
    
    # Create synthetic data
    print("Creating synthetic mini-batch...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = None  # Unconditional for testing
    
    # Test regular forward pass first
    print("\nTesting regular forward pass...")
    with torch.no_grad():
        loss = cd_loss(student_net, images, labels, augment_pipe=None)
        print(f"  Loss shape: {loss.shape}")
        print(f"  Loss mean: {loss.mean().item():.6f}")
        print(f"  Loss is finite: {torch.isfinite(loss).all().item()}")
    
    # Test debug harness
    print("\nRunning debug harness...")
    output_dir = "cd_debug_test"
    
    result = cd_loss.debug_batch(
        net=student_net,
        images=images,
        labels=labels,
        augment_pipe=None,
        output_dir=output_dir,
        num_samples_visual=2,
        num_sigma_bins=3,
        run_teacher_self_test=True,
        global_step=0,
    )
    
    # Verify outputs
    print("\n" + "="*80)
    print("Debug Harness Results")
    print("="*80)
    
    print(f"\nReport path: {result['report_path']}")
    print(f"Report exists: {os.path.exists(result['report_path'])}")
    
    print(f"\nNumber of problems detected: {result['num_problems']}")
    if result['num_problems'] > 0:
        print("Problems:")
        for prob in result['problems']:
            print(f"  - {prob}")
    
    print(f"\nNumber of bins: {len(result['bin_stats'])}")
    print("Bin statistics:")
    for stat in result['bin_stats']:
        print(f"  Bin {stat['bin']}:")
        print(f"    sigma_t={stat['sigma_t']:.6f}, sigma_s={stat['sigma_s']:.6f}, sigma_bdry={stat['sigma_bdry']:.6f}")
        print(f"    loss_mean={stat['loss_mean']:.6f}, rms_diff={stat['rms_diff']:.6f}")
    
    # Check that images were created
    print("\nChecking visualization images...")
    num_images_expected = len(result['bin_stats']) * 2  # 2 samples per bin
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"  Expected images: {num_images_expected}")
    print(f"  Found images: {len(image_files)}")
    
    # Verify report content
    print("\nVerifying report content...")
    with open(result['report_path'], 'r') as f:
        report_content = f.read()
    
    required_sections = [
        "# CD Debug Report",
        "## 1. Config Snapshot",
        "## 2. Sigma Grids",
        "## 3. Segments",
        "## 4. Per-bin Statistics",
        "## 5. Potential Issues",
        "## 6. Teacher Self-Consistency Test",
        "## 7. Visualizations",
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in report_content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"  WARNING: Missing sections: {missing_sections}")
    else:
        print("  ✓ All required sections present")
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    success = True
    
    if not os.path.exists(result['report_path']):
        print("✗ Report file not created")
        success = False
    else:
        print("✓ Report file created")
    
    if len(image_files) != num_images_expected:
        print(f"✗ Image count mismatch (expected {num_images_expected}, got {len(image_files)})")
        success = False
    else:
        print("✓ All visualization images created")
    
    if missing_sections:
        print("✗ Report missing required sections")
        success = False
    else:
        print("✓ Report contains all required sections")
    
    if len(result['bin_stats']) != 3:
        print(f"✗ Bin stats count mismatch (expected 3, got {len(result['bin_stats'])})")
        success = False
    else:
        print("✓ Correct number of bins processed")
    
    # Check for numerical issues
    numerical_ok = True
    for stat in result['bin_stats']:
        if not all(isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or abs(v) == float('inf'))) 
                   for k, v in stat.items() if isinstance(v, (int, float))):
            numerical_ok = False
            break
    
    if not numerical_ok:
        print("✗ Numerical issues detected in statistics")
        success = False
    else:
        print("✓ All statistics are finite and valid")
    
    if success:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print(f"\nDebug output saved to: {output_dir}/")
        return 0
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(test_debug_harness())

