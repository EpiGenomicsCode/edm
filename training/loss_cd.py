import math
import os
import copy
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime

import torch
from torch_utils import persistence
from torch_utils import training_stats

from .consistency_ops import (
    make_karras_sigmas,
    partition_edges_into_segments,
    sample_segment_and_teacher_pair,
    heun_hop_edm,
    inv_ddim_edm,
    ddim_step_edm,
)


def _huber_loss(x: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.as_tensor(delta, device=x.device, dtype=x.dtype))
    # 0.5 * quad^2 + delta * (abs_x - quad)
    return 0.5 * (quad * quad) + (abs_x - quad) * delta


def _save_image_grid(tensors: List[torch.Tensor], path: str, nrow: int = 6):
    """
    Save a grid of images without requiring torchvision.
    tensors: List of [1, C, H, W] tensors
    """
    try:
        import torchvision.utils as vutils
        grid = torch.cat(tensors, dim=0)
        vutils.save_image(grid, path, nrow=nrow, padding=2, normalize=False)
    except ImportError:
        # Fallback: use PIL
        try:
            from PIL import Image
            import numpy as np
            
            # Convert tensors to numpy arrays
            images = []
            for t in tensors:
                img = t.squeeze(0).detach().cpu().float().numpy()
                # CHW -> HWC
                img = np.transpose(img, (1, 2, 0))
                # Clip to [0, 1]
                img = np.clip(img, 0, 1)
                # Convert to uint8
                img = (img * 255).astype(np.uint8)
                # Handle grayscale
                if img.shape[2] == 1:
                    img = img.squeeze(2)
                images.append(img)
            
            # Create grid
            n = len(images)
            h, w = images[0].shape[:2]
            grid_img = np.zeros((h, w * n, 3), dtype=np.uint8)
            
            for i, img in enumerate(images):
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=2)
                grid_img[:, i*w:(i+1)*w, :] = img
            
            # Save
            Image.fromarray(grid_img).save(path)
        except ImportError:
            # Last resort: save as numpy
            import numpy as np
            grid = torch.cat(tensors, dim=0).detach().cpu().numpy()
            np.save(path.replace('.png', '.npy'), grid)
            print(f"Warning: Could not save as PNG, saved as .npy instead: {path}")


@persistence.persistent_class
class EDMConsistencyDistillLoss:
    def __init__(
        self,
        teacher_net,                 # Frozen EDM-preconditioned UNet-compatible module
        S: int = 8,                  # Student steps
        T_start: int = 256,          # Initial teacher edges
        T_end: int = 1024,           # Final teacher edges
        T_anneal_kimg: int = 750,    # Linear anneal horizon (kimg)
        rho: float = 7.0,            # Karras exponent
        sigma_min: float = 2e-3,
        sigma_max: float = 80.0,
        loss_type: str = "huber",    # "huber" | "l2"
        weight_mode: str = "edm",    # "edm" | "vlike"
        sigma_data: float = 0.5,
        enable_stats: bool = True,
    ):
        assert S >= 2, "Student steps S must be >= 2"
        assert T_start >= 2 and T_end >= T_start
        assert loss_type in ("huber", "l2")
        assert weight_mode in ("edm", "vlike")
        self.teacher_net = teacher_net.eval().requires_grad_(False)
        self.S = int(S)
        self.T_start = int(T_start)
        self.T_end = int(T_end)
        self.T_anneal_kimg = float(T_anneal_kimg)
        self.rho = float(rho)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.loss_type = loss_type
        self.weight_mode = weight_mode
        self.sigma_data = float(sigma_data)
        self.enable_stats = enable_stats

        # Global kimg for teacher annealing; set externally by training loop.
        # Defaults to 0 if not explicitly set.
        self._global_kimg = 0.0
        
        # Diagnostic counters for edge type distribution
        self._count_terminal_edges = 0      # sigma_s == 0
        self._count_boundary_match = 0      # sigma_s == sigma_bdry (interior)
        self._count_general_edges = 0       # general interior edges
        self._count_total_calls = 0         # total __call__ invocations

    def set_global_kimg(self, kimg: float) -> None:
        """
        Set the global training progress in kimg for annealing teacher edges.
        The training loop is responsible for calling this once per tick with
        the current global kimg (including resume_kimg).
        """
        self._global_kimg = float(kimg)
    
    def get_edge_stats(self, reset: bool = True) -> dict:
        """
        Get diagnostic statistics about edge type distribution.
        
        Returns:
            dict with:
                - total_calls: total number of __call__ invocations
                - terminal_edges: count of terminal edges (sigma_s == 0)
                - boundary_match: count of interior edges where sigma_s == sigma_bdry
                - general_edges: count of general interior edges
                - terminal_pct: percentage of terminal edges
                - boundary_match_pct: percentage of boundary matches
        
        Args:
            reset: if True, reset counters after reading
        """
        total = max(self._count_total_calls, 1)  # avoid division by zero
        stats = {
            'total_calls': self._count_total_calls,
            'terminal_edges': self._count_terminal_edges,
            'boundary_match': self._count_boundary_match,
            'general_edges': self._count_general_edges,
            'terminal_pct': 100.0 * self._count_terminal_edges / total,
            'boundary_match_pct': 100.0 * self._count_boundary_match / total,
        }
        
        if reset:
            self._count_terminal_edges = 0
            self._count_boundary_match = 0
            self._count_general_edges = 0
            self._count_total_calls = 0
        
        return stats

    def _current_T_edges(self) -> int:
        """
        Compute current teacher edges T based on global training progress.
        If T_anneal_kimg <= 0, return T_end.

        Schedule: log-linear interpolation between T_start and T_end over
        T_anneal_kimg "kimg" of training, analogous in shape to the multistep
        paper's schedule N_teacher(i) = exp(log T_start + clip(i/H,0,1)*(log T_end-log T_start)),
        but expressed in terms of global kimg instead of raw optimizer steps.
        """
        if self.T_anneal_kimg <= 0:
            return self.T_end
        # Log-linear ramp from T_start to T_end over T_anneal_kimg (kimg),
        # using resume-aware global training progress.
        ratio = min(max(self._global_kimg / self.T_anneal_kimg, 0.0), 1.0)
        log_T_start = math.log(self.T_start)
        log_T_end = math.log(self.T_end)
        log_T_now = log_T_start + ratio * (log_T_end - log_T_start)
        T_now = int(round(math.exp(log_T_now)))
        T_now = max(self.T_start, min(self.T_end, T_now))
        return T_now

    def _build_student_grid(self, net, device: torch.device) -> torch.Tensor:
        # Student grid: S positive sigmas + terminal 0
        # net may be wrapped in DDP; grab underlying module's round_sigma if needed.
        round_fn = getattr(net, 'round_sigma', None)
        if round_fn is None and hasattr(net, 'module'):
            round_fn = getattr(net.module, 'round_sigma', None)
        sigmas_prepad = make_karras_sigmas(
            num_nodes=self.S,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            round_fn=round_fn,
        ).to(device)
        # Append terminal zero
        zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
        sigmas = torch.cat([sigmas_prepad, zero], dim=0)
        return sigmas

    def _build_teacher_grid(self, device: torch.device) -> torch.Tensor:
        # Teacher grid: T positive sigmas + terminal 0
        T_edges = self._current_T_edges()
        sigmas_prepad = make_karras_sigmas(
            num_nodes=T_edges,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            round_fn=self.teacher_net.round_sigma,
        ).to(device)
        # Append terminal zero
        zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
        sigmas = torch.cat([sigmas_prepad, zero], dim=0)
        return sigmas

    def _weight(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma expected shape [N,1,1,1]
        if self.weight_mode == "edm":
            return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # v-like: 1/σ^2 + 1
        return (1.0 / (sigma ** 2)) + 1.0

    def __call__(self, net, images, labels=None, augment_pipe=None):
        """
        Return per-sample loss tensor (same broadcasting semantics as other losses).
        """
        device = images.device
        batch_size = images.shape[0]

        # Optional augmentation (matches other losses).
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        # Build grids with terminal zeros.
        student_sigmas = self._build_student_grid(net=net, device=device)  # [S+1] with terminal 0
        teacher_sigmas = self._build_teacher_grid(device=device)           # [T+1] with terminal 0
        T_edges = teacher_sigmas.shape[0] - 1

        # Partition teacher edges into S segments.
        boundaries = partition_edges_into_segments(T=T_edges, S=self.S)

        # Sample (j, k_t, k_s, sigma_t, sigma_s, sigma_bdry, flags) once per call, then broadcast.
        sample_dict = sample_segment_and_teacher_pair(
            boundaries=boundaries,
            teacher_sigmas=teacher_sigmas,
            student_sigmas=student_sigmas,
            batch_size=1,
            device=device,
        )
        
        j = int(sample_dict["step_j"].item())
        sigma_t_scalar = sample_dict["sigma_t"].squeeze()
        sigma_s_scalar_teacher = sample_dict["sigma_s"].squeeze()
        sigma_bdry_scalar = sample_dict["sigma_bdry"].squeeze()
        is_terminal = bool(sample_dict["is_terminal"].item())
        is_boundary_snap = bool(sample_dict["is_boundary_snap"].item())

        # Choose effective teacher-hop sigma_s according to MSCD-style rule:
        # - terminal: use σ_s = 0 (handled separately)
        # - boundary snap (n_rel == 1, non-terminal): σ_s := σ_bdry(j)
        # - general interior: σ_s := max(σ_s_teacher, σ_bdry(j))
        if is_terminal:
            sigma_s_scalar = sigma_s_scalar_teacher  # should already be 0
        elif is_boundary_snap:
            sigma_s_scalar = sigma_bdry_scalar
        else:
            sigma_s_scalar = torch.maximum(sigma_s_scalar_teacher, sigma_bdry_scalar)

        # Broadcast scalars to per-sample sigma tensors as [N,1,1,1] to match other losses.
        sigma_t = sigma_t_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
        sigma_s = sigma_s_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
        sigma_bdry = sigma_bdry_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)

        # Sample noise and form x_t = y + sigma_t * eps.
        eps = torch.randn_like(y)
        x_t = y + sigma_t * eps

        # Update counters using MSCD-style classification.
        self._count_total_calls += 1
        if is_terminal:
            self._count_terminal_edges += 1
        elif is_boundary_snap:
            self._count_boundary_match += 1
        else:
            self._count_general_edges += 1

        # Handle terminal edge: σ_s = 0, σ_bdry = 0.
        tol = 1e-12
        if is_terminal:
            # Terminal edge: sigma_s = 0, sigma_bdry = 0
            # Skip teacher hop (can't evaluate at sigma=0)
            # Anchor directly to clean input for standard EDM denoising
            x_ref_bdry = y.to(torch.float32)
            sigma_ref_scalar = torch.as_tensor(0.0, device=device, dtype=sigma_t_scalar.dtype)
        else:
            with torch.no_grad():
                # Interior edge: run teacher hop
                x_s_teach = heun_hop_edm(
                    net=self.teacher_net,
                    x_t=x_t,
                    sigma_t=sigma_t_scalar,
                    sigma_s=sigma_s_scalar,
                    class_labels=labels,
                    augment_labels=augment_labels,
                )
            
            if is_boundary_snap:
                # Boundary snap: treat as degenerate MSCD-style teacher step.
                # Do not push with student; anchor directly to teacher hop at σ_s.
                x_ref_bdry = x_s_teach.to(torch.float32)
                sigma_ref_scalar = sigma_s_scalar
            else:
                # General interior: push from σ_s to σ_bdry using student.
                with torch.no_grad():
                    x_hat_s_ng = net(x_s_teach, sigma_s, labels, augment_labels=augment_labels).to(torch.float32)
                ratio_s_b = (sigma_bdry / torch.clamp(sigma_s, min=tol)).to(torch.float32)
                x_ref_bdry = x_hat_s_ng + ratio_s_b * (x_s_teach.to(torch.float32) - x_hat_s_ng)
                sigma_ref_scalar = sigma_bdry_scalar

        # Compute inv-DDIM target at t using appropriate reference sigma.
        sigma_ref = sigma_ref_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
        x_hat_t_star = inv_ddim_edm(
            x_ref=x_ref_bdry,
            x_t=x_t,
            sigma_t=sigma_t,
            sigma_ref=sigma_ref,
        ).to(torch.float32)

        # Student prediction at t.
        x_hat_t = net(x_t, sigma_t, labels, augment_labels=augment_labels).to(torch.float32)

        # Weighting and loss.
        weight = self._weight(sigma_t.to(torch.float32))
        diff = x_hat_t - x_hat_t_star
        if self.loss_type == "huber":
            per_elem = _huber_loss(diff)
        else:
            per_elem = diff * diff
        loss = weight * per_elem

        # Optional stats.
        if self.enable_stats:
            with torch.no_grad():
                training_stats.report('Loss/cd', loss)
                training_stats.report('CD/sigma_t', sigma_t_scalar)
                training_stats.report('CD/sigma_s', sigma_s_scalar)
                training_stats.report('CD/sigma_bdry', sigma_bdry_scalar)
                training_stats.report('CD/seg_id', torch.as_tensor(float(j), device=device))
                training_stats.report('CD/T_edges', torch.as_tensor(float(T_edges), device=device))

        return loss

    def debug_batch(
        self,
        net,
        images,
        labels=None,
        augment_pipe=None,
        output_dir: str = "cd_debug",
        num_samples_visual: int = 4,
        num_sigma_bins: int = 3,
        run_teacher_self_test: bool = True,
        global_step: Optional[int] = None,
    ):
        """
        Debug harness for EDM Consistency Distillation pipeline.
        
        Runs instrumented forward passes on a single mini-batch to:
        - Check sigma ordering and numerical sanity
        - Visualize key internal states across noise levels
        - Optionally test teacher self-consistency
        
        Args:
            net: Student network (EDM-preconditioned)
            images: Mini-batch of images [N, C, H, W]
            labels: Optional class labels [N]
            augment_pipe: Optional augmentation pipeline
            output_dir: Directory to write report and images
            num_samples_visual: Number of samples to visualize (default: 4)
            num_sigma_bins: Number of noise bins to sample (default: 3 for low/mid/high)
            run_teacher_self_test: Whether to run teacher-as-student test
            global_step: Current training step (for naming outputs)
        
        Returns:
            Dict with summary statistics and paths to generated files
        """
        device = images.device
        batch_size = images.shape[0]
        
        # Limit to num_samples_visual
        num_samples_visual = min(num_samples_visual, batch_size)
        images_vis = images[:num_samples_visual]
        labels_vis = labels[:num_samples_visual] if labels is not None else None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp for unique naming
        if global_step is not None:
            step_str = f"step_{global_step}"
        else:
            step_str = f"ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_path = os.path.join(output_dir, f"cd_debug_report_{step_str}.md")
        
        # Initialize report data structures
        report_lines = []
        problems = []
        bin_stats = []
        
        # Helper: RMS norm
        def rms(x: torch.Tensor) -> float:
            return float(torch.sqrt(torch.mean(x.float() ** 2)).cpu())
        
        # Helper: Check finiteness
        def check_finite(tensor: torch.Tensor, name: str, sigma_t: float, sigma_s: float, sigma_bdry: float):
            if not torch.isfinite(tensor).all():
                problems.append({
                    "type": "non_finite",
                    "tensor": name,
                    "sigma_t": sigma_t,
                    "sigma_s": sigma_s,
                    "sigma_bdry": sigma_bdry,
                })
        
        # Helper: Check sigma ordering
        def check_sigma_ordering(sigma_t: float, sigma_s: float, sigma_bdry: float, j: int, e: int):
            tol = 1e-9
            if not (sigma_t >= sigma_s - tol and sigma_s >= sigma_bdry - tol):
                problems.append({
                    "type": "sigma_ordering",
                    "j": j,
                    "e": e,
                    "sigma_t": sigma_t,
                    "sigma_s": sigma_s,
                    "sigma_bdry": sigma_bdry,
                })
            
            # Check ratios
            ratio_s_b = sigma_bdry / max(sigma_s, 1e-12)
            ratio_t_s = sigma_s / max(sigma_t, 1e-12)
            
            if ratio_s_b < 0 or ratio_s_b > 2.0 or ratio_t_s < 0 or ratio_t_s > 2.0:
                problems.append({
                    "type": "ratio_anomaly",
                    "j": j,
                    "e": e,
                    "ratio_s_b": ratio_s_b,
                    "ratio_t_s": ratio_t_s,
                    "sigma_t": sigma_t,
                    "sigma_s": sigma_s,
                    "sigma_bdry": sigma_bdry,
                })
        
        # Build grids with terminal zeros
        with torch.no_grad():
            student_sigmas = self._build_student_grid(net=net, device=device)
            teacher_sigmas = self._build_teacher_grid(device=device)
        
        T_edges = teacher_sigmas.shape[0] - 1
        boundaries = partition_edges_into_segments(T=T_edges, S=self.S)
        
        # === REPORT HEADER ===
        report_lines.append(f"# CD Debug Report ({step_str})")
        report_lines.append("")
        report_lines.append("## 1. Config Snapshot")
        report_lines.append("")
        report_lines.append(f"- S (student steps): {self.S}")
        report_lines.append(f"- T_start: {self.T_start}")
        report_lines.append(f"- T_end: {self.T_end}")
        report_lines.append(f"- Current T_edges: {T_edges}")
        report_lines.append(f"- T_anneal_kimg: {self.T_anneal_kimg}")
        report_lines.append(f"- rho: {self.rho}")
        report_lines.append(f"- sigma_min: {self.sigma_min}")
        report_lines.append(f"- sigma_max: {self.sigma_max}")
        report_lines.append(f"- sigma_data: {self.sigma_data}")
        report_lines.append(f"- loss_type: {self.loss_type}")
        report_lines.append(f"- weight_mode: {self.weight_mode}")
        report_lines.append(f"- global_kimg: {self._global_kimg}")
        report_lines.append("")
        
        # === SIGMA GRIDS ===
        report_lines.append("## 2. Sigma Grids")
        report_lines.append("")
        report_lines.append("### Student Sigmas")
        report_lines.append(f"- Total nodes: {len(student_sigmas)}")
        report_lines.append(f"- Range: [{student_sigmas[-1]:.6f}, {student_sigmas[0]:.6f}]")
        report_lines.append(f"- First 5: {[f'{s:.6f}' for s in student_sigmas[:5].cpu().tolist()]}")
        report_lines.append(f"- Last 5: {[f'{s:.6f}' for s in student_sigmas[-5:].cpu().tolist()]}")
        report_lines.append("")
        report_lines.append("### Teacher Sigmas")
        report_lines.append(f"- Total nodes: {len(teacher_sigmas)}")
        report_lines.append(f"- Range: [{teacher_sigmas[-1]:.6f}, {teacher_sigmas[0]:.6f}]")
        report_lines.append(f"- First 5: {[f'{s:.6f}' for s in teacher_sigmas[:5].cpu().tolist()]}")
        report_lines.append(f"- Last 5: {[f'{s:.6f}' for s in teacher_sigmas[-5:].cpu().tolist()]}")
        report_lines.append("")
        
        # === SEGMENTS ===
        report_lines.append("## 3. Segments")
        report_lines.append("")
        for j in range(self.S):
            start = boundaries[j].item()
            end = boundaries[j + 1].item()
            sigma_start = teacher_sigmas[start].item()
            sigma_end = teacher_sigmas[min(end, len(teacher_sigmas) - 1)].item()
            sigma_bdry_j = student_sigmas[j + 1].item()
            report_lines.append(f"- Segment {j}:")
            report_lines.append(f"  - edges: [{start}, {end})")
            report_lines.append(f"  - sigma range: [{sigma_end:.6f}, {sigma_start:.6f}]")
            report_lines.append(f"  - student right boundary: {sigma_bdry_j:.6f}")
        report_lines.append("")
        
        # === SELECT NOISE BINS ===
        # Choose representative edges across the teacher grid using Boltz sampling
        bin_edges_idx = torch.linspace(0, T_edges - 1, num_sigma_bins + 1).long()
        selected_configs = []
        
        for k in range(num_sigma_bins):
            # Pick midpoint of this bin
            start_idx = bin_edges_idx[k].item()
            end_idx = bin_edges_idx[k + 1].item()
            mid_idx = (start_idx + end_idx) // 2
            k_t = max(0, min(T_edges - 1, mid_idx))
            
            # Find corresponding segment for this k_t
            j = None
            for seg_idx in range(self.S):
                if boundaries[seg_idx] <= k_t < boundaries[seg_idx + 1]:
                    j = seg_idx
                    break
            
            if j is None:
                # Fallback: use last segment
                j = self.S - 1
                k_t = boundaries[j].item()
            
            selected_configs.append((k, j, k_t))
        
        # === RUN INSTRUMENTED FORWARD PASSES ===
        report_lines.append("## 4. Per-bin Statistics")
        report_lines.append("")
        
        # Optional augmentation (match training_loop.py semantics).
        # Dataset yields uint8 in [0,255]; training_loop converts to float32 in [-1,1].
        y, augment_labels = augment_pipe(images_vis) if augment_pipe is not None else (images_vis, None)
        if not torch.is_floating_point(y):
            # Convert uint8 -> float32 in [-1, 1], consistent with training.
            y = y.to(torch.float32) / 127.5 - 1.0
        else:
            # Ensure float32 for all math below.
            y = y.to(torch.float32)
        
        for bin_idx, j, k_t in selected_configs:
            k_s = k_t + 1
            sigma_t_scalar = teacher_sigmas[k_t]
            sigma_s_scalar = teacher_sigmas[k_s]
            sigma_bdry_scalar = student_sigmas[j + 1]  # Right boundary of segment j
            
            sigma_t_val = float(sigma_t_scalar.cpu())
            sigma_s_val = float(sigma_s_scalar.cpu())
            sigma_bdry_val = float(sigma_bdry_scalar.cpu())
            
            # Check sigma ordering
            check_sigma_ordering(sigma_t_val, sigma_s_val, sigma_bdry_val, j, k_t)
            
            # Broadcast sigmas
            sigma_t = sigma_t_scalar.reshape(1, 1, 1, 1).repeat(num_samples_visual, 1, 1, 1)
            sigma_s = sigma_s_scalar.reshape(1, 1, 1, 1).repeat(num_samples_visual, 1, 1, 1)
            sigma_bdry = sigma_bdry_scalar.reshape(1, 1, 1, 1).repeat(num_samples_visual, 1, 1, 1)
            
            # Sample noise and form x_t
            eps = torch.randn_like(y)
            x_t = y + sigma_t * eps
            
            # Check if at terminal edge
            tol = 1e-12
            at_terminal = torch.allclose(
                sigma_s_scalar,
                torch.tensor(0.0, device=device, dtype=sigma_s_scalar.dtype),
                atol=tol, rtol=0.0
            )
            
            if at_terminal:
                # Terminal edge: skip teacher hop, anchor to clean input
                x_ref_bdry = y.to(torch.float32)
                x_s_teach = None  # Not computed for terminal edge
            else:
                # Interior edge: run teacher hop
                with torch.no_grad():
                    x_s_teach = heun_hop_edm(
                        net=self.teacher_net,
                        x_t=x_t,
                        sigma_t=sigma_t_scalar,
                        sigma_s=sigma_s_scalar,
                        class_labels=labels_vis,
                        augment_labels=augment_labels,
                    )
                
                # Check if boundary coincides with teaching point
                equal_b_s = torch.allclose(
                    sigma_bdry_scalar, 
                    sigma_s_scalar, 
                    atol=tol, rtol=0.0
                )
                if equal_b_s:
                    x_ref_bdry = x_s_teach.to(torch.float32)
                else:
                    with torch.no_grad():
                        x_hat_s_ng = net(x_s_teach, sigma_s, labels_vis, augment_labels=augment_labels).to(torch.float32)
                    ratio_s_b = (sigma_bdry / torch.clamp(sigma_s, min=tol)).to(torch.float32)
                    x_ref_bdry = x_hat_s_ng + ratio_s_b * (x_s_teach.to(torch.float32) - x_hat_s_ng)
            
            # Inverse DDIM target
            x_hat_t_star = inv_ddim_edm(
                x_ref=x_ref_bdry,
                x_t=x_t,
                sigma_t=sigma_t,
                sigma_ref=sigma_bdry,
            ).to(torch.float32)
            
            # Student prediction
            with torch.no_grad():
                x_hat_t = net(x_t, sigma_t, labels_vis, augment_labels=augment_labels).to(torch.float32)
            
            # Weighting and loss
            weight = self._weight(sigma_t.to(torch.float32))
            diff = x_hat_t - x_hat_t_star
            if self.loss_type == "huber":
                per_elem = _huber_loss(diff)
            else:
                per_elem = diff * diff
            loss = weight * per_elem
            
            # Finiteness checks
            check_finite(x_t, "x_t", sigma_t_val, sigma_s_val, sigma_bdry_val)
            check_finite(x_s_teach, "x_s_teach", sigma_t_val, sigma_s_val, sigma_bdry_val)
            check_finite(x_ref_bdry, "x_ref_bdry", sigma_t_val, sigma_s_val, sigma_bdry_val)
            check_finite(x_hat_t_star, "x_hat_t_star", sigma_t_val, sigma_s_val, sigma_bdry_val)
            check_finite(x_hat_t, "x_hat_t", sigma_t_val, sigma_s_val, sigma_bdry_val)
            check_finite(loss, "loss", sigma_t_val, sigma_s_val, sigma_bdry_val)
            
            # Compute RMS norms
            rms_y = rms(y)
            rms_x_t = rms(x_t)
            rms_x_s_teach = rms(x_s_teach) if x_s_teach is not None else 0.0
            rms_x_ref_bdry = rms(x_ref_bdry)
            rms_x_hat_t_star = rms(x_hat_t_star)
            rms_x_hat_t = rms(x_hat_t)
            rms_diff = rms(x_hat_t - x_hat_t_star)
            loss_mean = float(loss.mean().cpu())
            weight_mean = float(weight.mean().cpu())
            
            # Store stats
            bin_stats.append({
                "bin": bin_idx,
                "j": j,
                "k_t": k_t,
                "k_s": k_s,
                "sigma_t": sigma_t_val,
                "sigma_s": sigma_s_val,
                "sigma_bdry": sigma_bdry_val,
                "rms_y": rms_y,
                "rms_x_t": rms_x_t,
                "rms_x_s_teach": rms_x_s_teach,
                "rms_x_ref_bdry": rms_x_ref_bdry,
                "rms_x_hat_t_star": rms_x_hat_t_star,
                "rms_x_hat_t": rms_x_hat_t,
                "rms_diff": rms_diff,
                "loss_mean": loss_mean,
                "weight_mean": weight_mean,
            })
            
            # Write to report
            sigma_range_desc = "high" if bin_idx == 0 else ("low" if bin_idx == num_sigma_bins - 1 else "mid")
            report_lines.append(f"### Bin {bin_idx} ({sigma_range_desc} noise)")
            report_lines.append(f"- Edge range: [{bin_edges_idx[bin_idx]}, {bin_edges_idx[bin_idx+1]})")
            report_lines.append(f"- Representative teacher edge k_t: {k_t}, k_s: {k_s}")
            report_lines.append(f"- Segment j: {j}")
            report_lines.append(f"- sigma_t: {sigma_t_val:.6f}")
            report_lines.append(f"- sigma_s: {sigma_s_val:.6f}")
            report_lines.append(f"- sigma_bdry: {sigma_bdry_val:.6f}")
            report_lines.append(f"- rms_y: {rms_y:.6f}")
            report_lines.append(f"- rms_x_t: {rms_x_t:.6f}")
            report_lines.append(f"- rms_x_s_teach: {rms_x_s_teach:.6f}")
            report_lines.append(f"- rms_x_ref_bdry: {rms_x_ref_bdry:.6f}")
            report_lines.append(f"- rms_x_hat_t_star: {rms_x_hat_t_star:.6f}")
            report_lines.append(f"- rms_x_hat_t: {rms_x_hat_t:.6f}")
            report_lines.append(f"- rms_diff: {rms_diff:.6f}")
            report_lines.append(f"- loss_mean: {loss_mean:.6f}")
            report_lines.append(f"- weight_mean: {weight_mean:.6f}")
            report_lines.append("")
            
            # === VISUALIZATIONS ===
            # Generate image grids for each sample
            for sample_idx in range(num_samples_visual):
                # Collect tensors for this sample
                if x_s_teach is not None:
                    tensors_to_vis = [
                        y[sample_idx:sample_idx+1],
                        x_t[sample_idx:sample_idx+1],
                        x_s_teach[sample_idx:sample_idx+1],
                        x_ref_bdry[sample_idx:sample_idx+1],
                        x_hat_t_star[sample_idx:sample_idx+1],
                        x_hat_t[sample_idx:sample_idx+1],
                    ]
                else:
                    # Terminal edge: no x_s_teach, show black placeholder
                    black_placeholder = torch.zeros_like(y[sample_idx:sample_idx+1])
                    tensors_to_vis = [
                        y[sample_idx:sample_idx+1],
                        x_t[sample_idx:sample_idx+1],
                        black_placeholder,
                        x_ref_bdry[sample_idx:sample_idx+1],
                        x_hat_t_star[sample_idx:sample_idx+1],
                        x_hat_t[sample_idx:sample_idx+1],
                    ]
                
                # Normalize to [0, 1] for visualization
                def normalize_for_vis(t):
                    t = t.detach().cpu().float()
                    t_min = t.min()
                    t_max = t.max()
                    if t_max > t_min:
                        return (t - t_min) / (t_max - t_min)
                    return t
                
                vis_tensors = [normalize_for_vis(t) for t in tensors_to_vis]
                
                # Save image grid
                img_path = os.path.join(output_dir, f"cd_debug_{step_str}_bin_{bin_idx}_sample_{sample_idx}.png")
                _save_image_grid(vis_tensors, img_path, nrow=6)
        
        # === POTENTIAL ISSUES ===
        report_lines.append("## 5. Potential Issues")
        report_lines.append("")
        if len(problems) == 0:
            report_lines.append("No issues detected.")
        else:
            for prob in problems:
                if prob["type"] == "sigma_ordering":
                    report_lines.append(f"- **[sigma_ordering]** j={prob['j']}, e={prob['e']}")
                    report_lines.append(f"  - sigma_t={prob['sigma_t']:.6f}, sigma_s={prob['sigma_s']:.6f}, sigma_bdry={prob['sigma_bdry']:.6f}")
                elif prob["type"] == "non_finite":
                    report_lines.append(f"- **[non_finite]** tensor={prob['tensor']}")
                    report_lines.append(f"  - sigma_t={prob['sigma_t']:.6f}, sigma_s={prob['sigma_s']:.6f}, sigma_bdry={prob['sigma_bdry']:.6f}")
                elif prob["type"] == "ratio_anomaly":
                    report_lines.append(f"- **[ratio_anomaly]** j={prob['j']}, e={prob['e']}")
                    report_lines.append(f"  - ratio_s_b={prob['ratio_s_b']:.6f}, ratio_t_s={prob['ratio_t_s']:.6f}")
                    report_lines.append(f"  - sigma_t={prob['sigma_t']:.6f}, sigma_s={prob['sigma_s']:.6f}, sigma_bdry={prob['sigma_bdry']:.6f}")
        report_lines.append("")
        
        # === TEACHER SELF-CONSISTENCY TEST ===
        if run_teacher_self_test:
            report_lines.append("## 6. Teacher Self-Consistency Test")
            report_lines.append("")
            
            # Clone teacher as student
            with torch.no_grad():
                student_clone = copy.deepcopy(self.teacher_net).eval().requires_grad_(False)
            
            # Run one forward pass with teacher as student (use mid bin)
            mid_bin_idx = num_sigma_bins // 2
            _, j_mid, k_t_mid = selected_configs[mid_bin_idx]
            
            k_s_mid = k_t_mid + 1
            sigma_t_scalar = teacher_sigmas[k_t_mid]
            sigma_s_scalar = teacher_sigmas[k_s_mid]
            sigma_bdry_scalar = student_sigmas[j_mid + 1]
            
            sigma_t = sigma_t_scalar.reshape(1, 1, 1, 1).repeat(num_samples_visual, 1, 1, 1)
            sigma_s = sigma_s_scalar.reshape(1, 1, 1, 1).repeat(num_samples_visual, 1, 1, 1)
            sigma_bdry = sigma_bdry_scalar.reshape(1, 1, 1, 1).repeat(num_samples_visual, 1, 1, 1)
            
            eps = torch.randn_like(y)
            x_t = y + sigma_t * eps
            
            with torch.no_grad():
                x_s_teach = heun_hop_edm(
                    net=self.teacher_net,
                    x_t=x_t,
                    sigma_t=sigma_t_scalar,
                    sigma_s=sigma_s_scalar,
                    class_labels=labels_vis,
                    augment_labels=augment_labels,
                )
                
                # Check if at terminal edge
                at_terminal = torch.allclose(
                    sigma_s_scalar,
                    torch.tensor(0.0, device=device, dtype=sigma_s_scalar.dtype),
                    atol=1e-12, rtol=0.0
                )
                
                if at_terminal:
                    x_ref_bdry = y.to(torch.float32)
                else:
                    equal_b_s = torch.allclose(
                        sigma_bdry_scalar, 
                        sigma_s_scalar, 
                        atol=1e-12, rtol=0.0
                    )
                    if equal_b_s:
                        x_ref_bdry = x_s_teach.to(torch.float32)
                    else:
                        x_hat_s_ng = student_clone(x_s_teach, sigma_s, labels_vis, augment_labels=augment_labels).to(torch.float32)
                        ratio_s_b = (sigma_bdry / torch.clamp(sigma_s, min=1e-12)).to(torch.float32)
                        x_ref_bdry = x_hat_s_ng + ratio_s_b * (x_s_teach.to(torch.float32) - x_hat_s_ng)
                
                x_hat_t_star = inv_ddim_edm(
                    x_ref=x_ref_bdry,
                    x_t=x_t,
                    sigma_t=sigma_t,
                    sigma_ref=sigma_bdry,
                ).to(torch.float32)
                
                x_hat_t = student_clone(x_t, sigma_t, labels_vis, augment_labels=augment_labels).to(torch.float32)
                
                weight = self._weight(sigma_t.to(torch.float32))
                diff = x_hat_t - x_hat_t_star
                if self.loss_type == "huber":
                    per_elem = _huber_loss(diff)
                else:
                    per_elem = diff * diff
                loss_self = weight * per_elem
                
                self_loss_mean = float(loss_self.mean().cpu())
                rms_self_diff = rms(x_hat_t - x_hat_t_star)
            
            report_lines.append(f"- Mean CD loss (teacher vs itself): {self_loss_mean:.6f}")
            report_lines.append(f"- RMS difference (teacher pred vs CD target): {rms_self_diff:.6f}")
            report_lines.append("")
            report_lines.append("**Interpretation:** If these values are large, it may indicate a design issue in the CD pipeline.")
            report_lines.append("")
        
        # === VISUALIZATIONS SUMMARY ===
        report_lines.append("## 7. Visualizations")
        report_lines.append("")
        report_lines.append("Image grids show (left to right):")
        report_lines.append("1. y (clean/augmented input)")
        report_lines.append("2. x_t (noisy input)")
        report_lines.append("3. x_s_teach (teacher hop)")
        report_lines.append("4. x_ref_bdry (boundary target)")
        report_lines.append("5. x_hat_t_star (CD target at t)")
        report_lines.append("6. x_hat_t (student prediction at t)")
        report_lines.append("")
        
        for bin_idx, _, _ in selected_configs:
            sigma_range_desc = "high" if bin_idx == 0 else ("low" if bin_idx == num_sigma_bins - 1 else "mid")
            report_lines.append(f"### Bin {bin_idx} ({sigma_range_desc} noise)")
            for sample_idx in range(num_samples_visual):
                img_filename = f"cd_debug_{step_str}_bin_{bin_idx}_sample_{sample_idx}.png"
                report_lines.append(f"- `{img_filename}`")
            report_lines.append("")
        
        # === WRITE REPORT ===
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[CD Debug] Report written to: {report_path}")
        print(f"[CD Debug] Found {len(problems)} potential issues")
        
        return {
            "report_path": report_path,
            "num_problems": len(problems),
            "problems": problems,
            "bin_stats": bin_stats,
            "output_dir": output_dir,
        }


