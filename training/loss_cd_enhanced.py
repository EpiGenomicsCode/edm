"""
Enhanced debug_batch additions for loss_cd.py

This module provides additional analysis functions for conditioning ratios,
ill-conditioning detection, and correlation analysis as outlined in debug.md.

These functions can be used standalone or integrated into the main debug_batch method.
"""

import torch
from typing import Dict, List, Tuple


def compute_conditioning_analysis(
    sigma_t_vec: torch.Tensor,
    sigma_s_vec: torch.Tensor,
    sigma_ref_vec: torch.Tensor,
    sigma_bdry_vec: torch.Tensor,
    is_terminal: torch.Tensor,
    is_boundary_snap: torch.Tensor,
) -> Dict[str, float]:
    """
    Analyze conditioning and ratios as per debug.md §3.1 and §3.2.
    
    This checks:
    - sigma_ref / sigma_t ratios (when close to 1, inv-DDIM is ill-conditioned)
    - denominator |1 - sigma_ref/sigma_t| (small = bad conditioning)
    - conditioning number 1 / |1 - sigma_ref/sigma_t|
    
    Args:
        sigma_t_vec: [N] sigma_t values
        sigma_s_vec: [N] sigma_s values  
        sigma_ref_vec: [N] sigma_ref values
        sigma_bdry_vec: [N] sigma_bdry values
        is_terminal: [N] terminal mask
        is_boundary_snap: [N] boundary snap mask
    
    Returns:
        Dict with conditioning metrics
    """
    N = sigma_t_vec.shape[0]
    
    # Compute ratios
    ratio_ref_t = (sigma_ref_vec / torch.clamp(sigma_t_vec, min=1e-12)).float()  # [N]
    ratio_s_bdry = (sigma_s_vec / torch.clamp(sigma_bdry_vec, min=1e-12)).float()  # [N]
    
    # Compute denominator for inv-DDIM
    denom = torch.abs(1.0 - ratio_ref_t)  # [N]
    
    # Conditioning number (1 / denom)
    cond = 1.0 / torch.clamp(denom, min=1e-12)  # [N]
    
    # Statistics
    stats = {
        'ratio_ref_t_mean': float(ratio_ref_t.mean().cpu()),
        'ratio_ref_t_max': float(ratio_ref_t.max().cpu()),
        'ratio_ref_t_min': float(ratio_ref_t.min().cpu()),
        'ratio_s_bdry_mean': float(ratio_s_bdry.mean().cpu()),
        'ratio_s_bdry_max': float(ratio_s_bdry.max().cpu()),
        'denom_min': float(denom.min().cpu()),
        'denom_mean': float(denom.mean().cpu()),
        'cond_max': float(cond.max().cpu()),
        'cond_mean': float(cond.mean().cpu()),
    }
    
    # Per-type breakdown
    is_general = (~is_terminal) & (~is_boundary_snap)
    
    for mask, name in [(is_terminal, 'terminal'), (is_boundary_snap, 'boundary'), (is_general, 'general')]:
        if mask.any():
            stats[f'{name}_count'] = int(mask.sum().item())
            stats[f'{name}_ratio_ref_t_mean'] = float(ratio_ref_t[mask].mean().cpu())
            stats[f'{name}_cond_max'] = float(cond[mask].max().cpu())
            stats[f'{name}_denom_min'] = float(denom[mask].min().cpu())
        else:
            stats[f'{name}_count'] = 0
            stats[f'{name}_ratio_ref_t_mean'] = float('nan')
            stats[f'{name}_cond_max'] = float('nan')
            stats[f'{name}_denom_min'] = float('nan')
    
    return stats


def compute_target_sanity_checks(
    y: torch.Tensor,
    x_ref_bdry: torch.Tensor,
    x_hat_t_star: torch.Tensor,
    x_hat_t: torch.Tensor,
    is_terminal: torch.Tensor,
    is_boundary_snap: torch.Tensor,
) -> Dict[str, float]:
    """
    Sanity checks on CD targets as per debug.md §3.3.
    
    This checks:
    - min/max/mean/std of targets
    - correlation between targets and clean images
    
    Args:
        y: [N, C, H, W] clean images
        x_ref_bdry: [N, C, H, W] boundary reference
        x_hat_t_star: [N, C, H, W] CD target at t
        x_hat_t: [N, C, H, W] student prediction at t
        is_terminal: [N] terminal mask
        is_boundary_snap: [N] boundary snap mask
    
    Returns:
        Dict with target statistics
    """
    
    def tensor_stats(t: torch.Tensor, name: str) -> Dict[str, float]:
        """Compute basic stats for a tensor."""
        return {
            f'{name}_min': float(t.min().cpu()),
            f'{name}_max': float(t.max().cpu()),
            f'{name}_mean': float(t.mean().cpu()),
            f'{name}_std': float(t.std().cpu()),
        }
    
    def correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute per-sample correlation between two tensors."""
        a_flat = a.view(a.size(0), -1).float()
        b_flat = b.view(b.size(0), -1).float()
        
        # Mean-center
        a_flat = a_flat - a_flat.mean(dim=1, keepdim=True)
        b_flat = b_flat - b_flat.mean(dim=1, keepdim=True)
        
        # Correlation
        numerator = torch.sum(a_flat * b_flat, dim=1)
        denominator = (
            torch.sqrt(torch.sum(a_flat ** 2, dim=1) + 1e-8) *
            torch.sqrt(torch.sum(b_flat ** 2, dim=1) + 1e-8)
        )
        return numerator / denominator
    
    stats = {}
    
    # Basic stats
    stats.update(tensor_stats(x_ref_bdry, 'x_ref_bdry'))
    stats.update(tensor_stats(x_hat_t_star, 'x_hat_t_star'))
    stats.update(tensor_stats(x_hat_t, 'x_hat_t'))
    
    # Correlation with clean image
    corr_y_ref = correlation(y, x_ref_bdry)
    corr_y_target = correlation(y, x_hat_t_star)
    corr_y_pred = correlation(y, x_hat_t)
    
    stats['corr_y_ref_mean'] = float(corr_y_ref.mean().cpu())
    stats['corr_y_ref_std'] = float(corr_y_ref.std().cpu())
    stats['corr_y_target_mean'] = float(corr_y_target.mean().cpu())
    stats['corr_y_target_std'] = float(corr_y_target.std().cpu())
    stats['corr_y_pred_mean'] = float(corr_y_pred.mean().cpu())
    stats['corr_y_pred_std'] = float(corr_y_pred.std().cpu())
    
    # Per-type breakdown
    is_general = (~is_terminal) & (~is_boundary_snap)
    
    for mask, name in [(is_terminal, 'terminal'), (is_boundary_snap, 'boundary'), (is_general, 'general')]:
        if mask.any():
            stats[f'{name}_corr_y_target_mean'] = float(corr_y_target[mask].mean().cpu())
            stats[f'{name}_corr_y_pred_mean'] = float(corr_y_pred[mask].mean().cpu())
        else:
            stats[f'{name}_corr_y_target_mean'] = float('nan')
            stats[f'{name}_corr_y_pred_mean'] = float('nan')
    
    return stats


def compute_per_type_loss_breakdown(
    loss: torch.Tensor,
    diff: torch.Tensor,
    is_terminal: torch.Tensor,
    is_boundary_snap: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute loss breakdown by edge type as per debug.md §3.4.
    
    Args:
        loss: [N, C, H, W] per-pixel loss
        diff: [N, C, H, W] x_hat_t - x_hat_t_star
        is_terminal: [N] terminal mask
        is_boundary_snap: [N] boundary snap mask
    
    Returns:
        Dict with per-type loss statistics
    """
    
    def rms(x: torch.Tensor) -> float:
        """Compute RMS norm."""
        return float(torch.sqrt(torch.mean(x.float() ** 2)).cpu())
    
    def masked_rms(x: torch.Tensor, m: torch.Tensor) -> float:
        """Compute RMS norm for masked subset."""
        if m.any():
            return rms(x[m])
        return float('nan')
    
    def masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
        """Compute mean for masked subset."""
        if m.any():
            return float(x[m].mean().cpu())
        return float('nan')
    
    is_general = (~is_terminal) & (~is_boundary_snap)
    
    # Per-sample loss
    loss_per_sample = loss.view(loss.size(0), -1).mean(dim=1)  # [N]
    
    stats = {
        # Overall
        'loss_mean': float(loss.mean().cpu()),
        'loss_std': float(loss_per_sample.std().cpu()),
        'rms_diff': rms(diff),
        
        # Terminal
        'terminal_count': int(is_terminal.sum().item()),
        'terminal_loss_mean': masked_mean(loss_per_sample, is_terminal),
        'terminal_rms_diff': masked_rms(diff, is_terminal),
        
        # Boundary snap
        'boundary_count': int(is_boundary_snap.sum().item()),
        'boundary_loss_mean': masked_mean(loss_per_sample, is_boundary_snap),
        'boundary_rms_diff': masked_rms(diff, is_boundary_snap),
        
        # General interior
        'general_count': int(is_general.sum().item()),
        'general_loss_mean': masked_mean(loss_per_sample, is_general),
        'general_rms_diff': masked_rms(diff, is_general),
    }
    
    return stats


def format_conditioning_report(cond_stats: Dict[str, float]) -> List[str]:
    """Format conditioning analysis for report."""
    lines = []
    lines.append("### Conditioning Analysis")
    lines.append("")
    lines.append("**Ratio σ_ref/σ_t (close to 1 = ill-conditioned):**")
    lines.append(f"- Mean: {cond_stats['ratio_ref_t_mean']:.6f}")
    lines.append(f"- Max: {cond_stats['ratio_ref_t_max']:.6f}")
    lines.append(f"- Min: {cond_stats['ratio_ref_t_min']:.6f}")
    lines.append("")
    lines.append("**Inverse-DDIM Denominator |1 - σ_ref/σ_t| (small = bad):**")
    lines.append(f"- Mean: {cond_stats['denom_mean']:.6f}")
    lines.append(f"- Min: {cond_stats['denom_min']:.6f}")
    lines.append("")
    lines.append("**Conditioning Number 1/|1 - σ_ref/σ_t| (large = bad):**")
    lines.append(f"- Mean: {cond_stats['cond_mean']:.2f}")
    lines.append(f"- Max: {cond_stats['cond_max']:.2f}")
    lines.append("")
    
    if cond_stats['cond_max'] > 100:
        lines.append("⚠️ **WARNING:** Maximum conditioning number is very large (>100). "
                    "This indicates ill-conditioned inverse-DDIM when σ_ref ≈ σ_t.")
        lines.append("")
    
    # Per-type breakdown
    lines.append("**Per-type Breakdown:**")
    lines.append("")
    lines.append("| Type | Count | Ratio σ_ref/σ_t | Cond Max | Denom Min |")
    lines.append("|------|-------|-----------------|----------|-----------|")
    for typ in ['terminal', 'boundary', 'general']:
        count = cond_stats.get(f'{typ}_count', 0)
        ratio = cond_stats.get(f'{typ}_ratio_ref_t_mean', float('nan'))
        cond_max = cond_stats.get(f'{typ}_cond_max', float('nan'))
        denom_min = cond_stats.get(f'{typ}_denom_min', float('nan'))
        lines.append(f"| {typ} | {count} | {ratio:.6f} | {cond_max:.2f} | {denom_min:.6f} |")
    lines.append("")
    
    return lines


def format_target_sanity_report(target_stats: Dict[str, float]) -> List[str]:
    """Format target sanity checks for report."""
    lines = []
    lines.append("### Target Sanity Checks")
    lines.append("")
    lines.append("**Target Statistics (x_hat_t_star):**")
    lines.append(f"- Min: {target_stats['x_hat_t_star_min']:.4f}")
    lines.append(f"- Max: {target_stats['x_hat_t_star_max']:.4f}")
    lines.append(f"- Mean: {target_stats['x_hat_t_star_mean']:.4f}")
    lines.append(f"- Std: {target_stats['x_hat_t_star_std']:.4f}")
    lines.append("")
    lines.append("**Correlation with Clean Image y:**")
    lines.append(f"- Target (x_hat_t_star): {target_stats['corr_y_target_mean']:.4f} ± {target_stats['corr_y_target_std']:.4f}")
    lines.append(f"- Prediction (x_hat_t): {target_stats['corr_y_pred_mean']:.4f} ± {target_stats['corr_y_pred_std']:.4f}")
    lines.append(f"- Reference (x_ref_bdry): {target_stats['corr_y_ref_mean']:.4f} ± {target_stats['corr_y_ref_std']:.4f}")
    lines.append("")
    
    if target_stats['corr_y_target_mean'] < 0.3:
        lines.append("⚠️ **WARNING:** Target has low correlation with clean image (<0.3). "
                    "This may indicate issues with how σ_ref or x_ref_bdry are constructed.")
        lines.append("")
    
    # Per-type correlation
    lines.append("**Per-type Correlation with y:**")
    lines.append("")
    lines.append("| Type | Target Corr | Prediction Corr |")
    lines.append("|------|-------------|-----------------|")
    for typ in ['terminal', 'boundary', 'general']:
        target_corr = target_stats.get(f'{typ}_corr_y_target_mean', float('nan'))
        pred_corr = target_stats.get(f'{typ}_corr_y_pred_mean', float('nan'))
        lines.append(f"| {typ} | {target_corr:.4f} | {pred_corr:.4f} |")
    lines.append("")
    
    return lines


def format_loss_breakdown_report(loss_stats: Dict[str, float]) -> List[str]:
    """Format per-type loss breakdown for report."""
    lines = []
    lines.append("### Per-Type Loss Breakdown")
    lines.append("")
    lines.append("| Type | Count | Loss Mean | RMS Diff |")
    lines.append("|------|-------|-----------|----------|")
    
    for typ in ['terminal', 'boundary', 'general']:
        count = loss_stats.get(f'{typ}_count', 0)
        loss_mean = loss_stats.get(f'{typ}_loss_mean', float('nan'))
        rms_diff = loss_stats.get(f'{typ}_rms_diff', float('nan'))
        lines.append(f"| {typ} | {count} | {loss_mean:.6f} | {rms_diff:.6f} |")
    
    lines.append("")
    
    # Check for anomalies
    terminal_loss = loss_stats.get('terminal_loss_mean', 0)
    boundary_loss = loss_stats.get('boundary_loss_mean', 0)
    general_loss = loss_stats.get('general_loss_mean', 0)
    
    if boundary_loss > 10 * general_loss and not math.isnan(boundary_loss) and not math.isnan(general_loss):
        lines.append("⚠️ **WARNING:** Boundary snap losses are >10× general losses. "
                    "This may indicate boundary snap edges are poisoning training.")
        lines.append("")
    
    return lines


def add_enhanced_debug_to_batch(
    y: torch.Tensor,
    x_ref_bdry: torch.Tensor,
    x_hat_t_star: torch.Tensor,
    x_hat_t: torch.Tensor,
    loss: torch.Tensor,
    diff: torch.Tensor,
    sigma_t_vec: torch.Tensor,
    sigma_s_vec: torch.Tensor,
    sigma_ref_vec: torch.Tensor,
    sigma_bdry_vec: torch.Tensor,
    is_terminal: torch.Tensor,
    is_boundary_snap: torch.Tensor,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Run all enhanced debug analyses and return stats + report lines.
    
    This is a convenience function that runs all the analyses and formats
    the results for inclusion in a debug report.
    
    Returns:
        (combined_stats, report_lines)
    """
    # Run analyses
    cond_stats = compute_conditioning_analysis(
        sigma_t_vec, sigma_s_vec, sigma_ref_vec, sigma_bdry_vec,
        is_terminal, is_boundary_snap
    )
    
    target_stats = compute_target_sanity_checks(
        y, x_ref_bdry, x_hat_t_star, x_hat_t,
        is_terminal, is_boundary_snap
    )
    
    loss_stats = compute_per_type_loss_breakdown(
        loss, diff, is_terminal, is_boundary_snap
    )
    
    # Combine stats
    combined_stats = {
        **cond_stats,
        **target_stats,
        **loss_stats,
    }
    
    # Format report
    report_lines = []
    report_lines.append("## Enhanced Debug Analysis")
    report_lines.append("")
    report_lines.extend(format_conditioning_report(cond_stats))
    report_lines.extend(format_target_sanity_report(target_stats))
    report_lines.extend(format_loss_breakdown_report(loss_stats))
    
    return combined_stats, report_lines


import math

