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
    partition_edges_by_sigma,
    filter_teacher_edges_by_sigma,
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
        S_churn: float = 0.0,        # Default to deterministic Heun (S_churn=0); stochastic churn breaks CD training
        S_min: float = 0.05,
        S_max: float = 50.0,
        S_noise: float = 1.003,
        loss_type: str = "huber",    # "huber" | "l2" (squared) | "l2_root" (euclidean)
        weight_mode: str = "edm",    # "edm" | "vlike" | "flat" | "snr" | "snr+1" | "karras" | "truncated-snr" | "uniform"
        sigma_data: float = 0.5,
        enable_stats: bool = True,
        debug_invariants: bool = False,  # Enable runtime invariant checks (PRD §5, R7)
        target_net = None,  # Optional: separate target network for sigma_s denoiser (OpenAI CM style EMA or teacher)
        anchor_by_sigma: bool = True,   # If True, segment teacher edges in sigma-space (closest-to-boundary first)
    ):
        assert S >= 2, "Student steps S must be >= 2"
        assert T_start >= 2 and T_end >= T_start
        assert loss_type in ("huber", "l2", "l2_root")
        assert weight_mode in (
            "edm",
            "vlike",
            "flat",
            # OpenAI consistency models style schedules (cm/karras_diffusion.get_weightings):
            "snr",
            "snr+1",
            "karras",
            "truncated-snr",
            "uniform",
        )
        self.teacher_net = teacher_net.eval().requires_grad_(False)
        self.S = int(S)
        self.T_start = int(T_start)
        self.T_end = int(T_end)
        self.T_anneal_kimg = float(T_anneal_kimg)
        self.rho = float(rho)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.S_churn = float(S_churn)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.S_noise = float(S_noise)
        self.loss_type = loss_type
        self.weight_mode = weight_mode
        self.sigma_data = float(sigma_data)
        self.enable_stats = enable_stats
        self.debug_invariants = debug_invariants
        # Optional target network for sigma_s denoiser (OpenAI CM style).
        # Three modes via training_loop's cd_target_mode:
        # - None (default): use live student weights at σ_s
        # - EMA copy of student: stabilizes training (OpenAI's approach, rate=0.95)
        # - Frozen teacher: for debugging
        self.target_net = target_net
        self.anchor_by_sigma = bool(anchor_by_sigma)

        # Global kimg for teacher annealing; set externally by training loop.
        # Defaults to 0 if not explicitly set.
        self._global_kimg = 0.0
        
        # Diagnostic counters for edge type distribution (per-sample)
        self._count_terminal_edges = 0      # total terminal edges sampled (per-sample)
        self._count_boundary_match = 0      # total boundary-snap edges sampled (per-sample)
        self._count_general_edges = 0       # total general interior edges sampled (per-sample)
        self._count_total_calls = 0         # total __call__ invocations (unchanged semantics)
        self._count_total_edges = 0         # NEW: total sampled edges across all calls (≈ batch_size × calls)

    def set_global_kimg(self, kimg: float) -> None:
        """
        Set the global training progress in kimg for annealing teacher edges.
        The training loop is responsible for calling this once per tick with
        the current global kimg (including resume_kimg).
        """
        self._global_kimg = float(kimg)
    
    def set_target_net(self, target_net) -> None:
        """
        Set or update the target network used for sigma_s denoiser in general edges.
        The training loop is responsible for maintaining this EMA copy and updating
        it via this method.
        """
        self.target_net = target_net
    
    def get_edge_stats(self, reset: bool = True) -> dict:
        """
        Get diagnostic statistics about edge type distribution.
        
        Returns:
            dict with:
                - total_calls: total number of __call__ invocations (unchanged semantics)
                - total_edges: NEW: total sampled edges across all calls (≈ batch_size × calls)
                - terminal_edges: count of terminal edges (sigma_s == 0)
                - boundary_match: count of interior edges where sigma_s == sigma_bdry
                - general_edges: count of general interior edges
                - terminal_pct: percentage of terminal edges (over total_edges)
                - boundary_match_pct: percentage of boundary matches (over total_edges)
        
        Args:
            reset: if True, reset counters after reading
        """
        total_edges = max(self._count_total_edges, 1)  # avoid division by zero
        stats = {
            'total_calls': self._count_total_calls,
            'total_edges': self._count_total_edges,
            'terminal_edges': self._count_terminal_edges,
            'boundary_match': self._count_boundary_match,
            'general_edges': self._count_general_edges,
            'terminal_pct': 100.0 * self._count_terminal_edges / total_edges,
            'boundary_match_pct': 100.0 * self._count_boundary_match / total_edges,
        }
        
        if reset:
            self._count_terminal_edges = 0
            self._count_boundary_match = 0
            self._count_general_edges = 0
            self._count_total_calls = 0
            self._count_total_edges = 0
        
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
        """
        Per-sample weighting as a function of sigma_t.

        This function is applied exactly once per sampled teacher timestep σ_t,
        and the resulting per-sample scalar is broadcast over BCHW to weight
        the per-pixel residuals, matching the training dynamics used in:

        - EDM teacher training (Karras et al. 2022), via `weight_mode="edm"`.
        - OpenAI consistency models (cm/karras_diffusion.py), via the
          weight_schedules "snr", "snr+1", "karras", "truncated-snr", "uniform".
        """
        # sigma is expected to be shape [N,1,1,1] (or broadcastable to that).
        if self.weight_mode == "edm":
            # EDM-style weighting (inverse SNR^2), matches original teacher training:
            #   w(σ) ∝ 1/σ^2 + 1/σ_data^2  (up to a global scale 1/σ_data^2).
            return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        if self.weight_mode == "vlike":
            # v-prediction-like weighting: 1/σ^2 + 1
            return (1.0 / (sigma ** 2)) + 1.0

        if self.weight_mode == "flat":
            # "flat": no weighting; all timesteps contribute equally.
            return torch.ones_like(sigma)

        # OpenAI consistency models style schedules (cm/karras_diffusion.get_weightings).
        # Define SNR(σ) = 1/σ^2, then:
        #   "snr"          -> w = snr
        #   "snr+1"        -> w = snr + 1
        #   "karras"       -> w = snr + 1/σ_data^2
        #   "truncated-snr"-> w = max(snr, 1)
        #   "uniform"      -> w = 1
        snr = 1.0 / (sigma ** 2 + 1e-20)
        if self.weight_mode == "snr":
            return snr
        if self.weight_mode == "snr+1":
            return snr + 1.0
        if self.weight_mode == "karras":
            return snr + (1.0 / (self.sigma_data ** 2))
        if self.weight_mode == "truncated-snr":
            return torch.clamp(snr, min=1.0)
        # "uniform": OpenAI's name; identical to our "flat" schedule.
        assert self.weight_mode == "uniform"
        return torch.ones_like(sigma)

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

        # Full teacher grid from Karras schedule
        teacher_sigmas_full = self._build_teacher_grid(device=device)      # [T+1] with terminal 0

        if self.anchor_by_sigma:
            # Build CD-specific teacher grid with duplicates removed
            teacher_sigmas, terminal_k = filter_teacher_edges_by_sigma(
                student_sigmas=student_sigmas,
                teacher_sigmas=teacher_sigmas_full,
            )
        else:
            teacher_sigmas = teacher_sigmas_full
            terminal_k = teacher_sigmas.shape[0] - 2  # last positive before terminal 0

        # Now everything below uses the (maybe filtered) teacher_sigmas
        T_edges = teacher_sigmas.shape[0] - 1  # number of edges in CD grid

        # Index-based segment boundaries (for index-anchored path)
        boundaries = partition_edges_into_segments(T=T_edges, S=self.S)

        sigma_bounds = None
        if self.anchor_by_sigma:
            sigma_bounds = partition_edges_by_sigma(
                student_sigmas=student_sigmas,
                teacher_sigmas=teacher_sigmas,  # already filtered & strictly descending
            )

        # Sample per-sample edges: each element in batch gets independent (j, k_t, k_s, sigmas).
        sample_dict = sample_segment_and_teacher_pair(
            boundaries=boundaries,
            teacher_sigmas=teacher_sigmas,
            student_sigmas=student_sigmas,
            batch_size=batch_size,         # per-sample sampling (PRD §4.2.1)
            device=device,
            anchor_by_sigma=self.anchor_by_sigma,
            sigma_bounds=sigma_bounds,
            terminal_k=terminal_k,
        )
        
        # Extract per-sample vectors (all shape [N])
        j = sample_dict["step_j"].long()               # [N]
        n_rel = sample_dict["n_rel"].long()            # [N]
        sigma_t_vec = sample_dict["sigma_t"]           # [N]
        sigma_s_teacher_vec = sample_dict["sigma_s"]   # [N]
        sigma_bdry_vec = sample_dict["sigma_bdry"]     # [N]
        is_terminal = sample_dict["is_terminal"].bool()        # [N]
        is_boundary_snap = sample_dict["is_boundary_snap"].bool()  # [N]

        # Vectorized MSCD sigma_s refinement (PRD §4.2.2)
        # Apply per-element MSCD rules:
        # - Terminal: keep sigma_s_eff = sigma_s_teacher_vec (already 0)
        # - Boundary snap: sigma_s_eff = sigma_bdry_vec
        # - General interior: sigma_s_eff = max(sigma_s_teacher_vec, sigma_bdry_vec)
        sigma_s_eff = sigma_s_teacher_vec.clone()
        
        # Boundary snap override
        sigma_s_eff = torch.where(
            is_boundary_snap,
            sigma_bdry_vec,
            sigma_s_eff,
        )
        
        # General interior: max(sigma_s_teacher, sigma_bdry)
        is_general = (~is_terminal) & (~is_boundary_snap)
        sigma_s_eff = torch.where(
            is_general,
            torch.maximum(sigma_s_teacher_vec, sigma_bdry_vec),
            sigma_s_eff,
        )

        # Broadcast per-sample sigmas to [N,1,1,1] for BCHW operations (PRD §4.2.3).
        #   - sigma_t_vec / sigma_s_eff / sigma_bdry_vec come from Karras grids and are float64 by default.
        #   - y is float32 (images converted to float32 in training_loop).
        #   - If we leave sigmas as float64, x_t = y + sigma_t * eps becomes float64 and
        #     x_ref_bdry (zeros_like(x_t)) is also float64, while y[...] is float32.
        #   - On the NCSA GH200 nodes this caused:
        #         RuntimeError: Index put requires the source and destination dtypes match,
        #         got Double for the destination and Float for the source.
        #   - We fix this by explicitly casting sigmas to y.dtype before broadcasting so that
        #     x_t and all downstream tensors stay in float32.
        sigma_t = sigma_t_vec.to(y.dtype).view(batch_size, 1, 1, 1)
        sigma_s = sigma_s_eff.to(y.dtype).view(batch_size, 1, 1, 1)
        sigma_bdry = sigma_bdry_vec.to(y.dtype).view(batch_size, 1, 1, 1)

        # Optional runtime invariant checks (PRD §5, R7)
        if self.debug_invariants:
            # Ordering invariants
            assert (sigma_t_vec >= sigma_s_eff - 1e-8).all(), "Ordering: sigma_t >= sigma_s_eff violated"
            assert (sigma_s_eff >= sigma_bdry_vec - 1e-8).all(), "Ordering: sigma_s_eff >= sigma_bdry violated"
            assert (sigma_s_eff[is_terminal] == 0).all(), "Terminal edges must have sigma_s_eff == 0"
            
            # Edge-type mask disjointness
            assert (is_terminal & is_boundary_snap).sum() == 0, "Terminal and boundary_snap must be disjoint"
            
            # Sampler-specific invariants
            assert (n_rel[is_boundary_snap] == 1).all(), "Boundary snap edges must have n_rel == 1"

        # Sample noise and form x_t = y + sigma_t * eps (PRD §4.2.4)
        eps = torch.randn_like(y)
        x_t = y + sigma_t * eps

        # Mixed terminal/non-terminal teacher hop logic (PRD §4.2.5)
        # Define per-sample masks
        non_terminal = ~is_terminal          # [N]
        boundary_mask = is_boundary_snap     # [N]
        general_mask = (~is_terminal) & (~is_boundary_snap)  # [N]
        
        # Allocate containers
        x_s_teach = torch.zeros_like(x_t)          # [N, C, H, W]
        x_ref_bdry = torch.zeros_like(x_t)         # [N, C, H, W]
        sigma_ref_vec = torch.zeros_like(sigma_t_vec)  # [N]
        tol = 1e-12
        
        # Teacher hop only for non-terminal samples
        if non_terminal.any():
            idx = non_terminal
            with torch.no_grad():
                x_s_teach_nt = heun_hop_edm(
                    net=self.teacher_net,
                    x_t=x_t[idx],
                    sigma_t=sigma_t_vec[idx],     # [N_nt]
                    sigma_s=sigma_s_eff[idx],     # [N_nt], all > 0 here
                    class_labels=labels[idx] if labels is not None else None,
                    augment_labels=augment_labels[idx] if augment_labels is not None else None,
                )
            x_s_teach[idx] = x_s_teach_nt
        
        # Terminal edges: anchor to clean input, σ_ref = 0
        x_ref_bdry[is_terminal] = y[is_terminal].to(torch.float32)
        sigma_ref_vec[is_terminal] = 0.0
        
        # Boundary snap edges: reference is teacher hop, σ_ref = σ_s_eff
        x_ref_bdry[boundary_mask] = x_s_teach[boundary_mask].to(torch.float32)
        sigma_ref_vec[boundary_mask] = sigma_s_eff[boundary_mask]
        
        # General interior edges: push from σ_s to σ_bdry using a denoiser at σ_s.
        # Two modes (controlled via training_loop's cd_target_mode):
        # 1. target_net is provided: use TARGET (EMA or teacher) at σ_s (OpenAI CM style)
        # 2. default: use live STUDENT at σ_s (standard CD)
        if general_mask.any():
            idx_g = general_mask
            with torch.no_grad():
                if self.target_net is not None:
                    # Use target network (EMA or teacher) at σ_s.
                    # Note: if target_net is the teacher, we pass sigma_s_eff (1D) to match
                    # the teacher's EDMPrecond interface; if it's the student EMA, we pass
                    # sigma_s (4D broadcast). Both work because EDMPrecond accepts both shapes.
                    sigma_for_target = sigma_s_eff[idx_g] if self.target_net is self.teacher_net else sigma_s[idx_g]
                    x_hat_s_ng = self.target_net(
                        x_s_teach[idx_g],
                        sigma_for_target,
                        labels[idx_g] if labels is not None else None,
                        augment_labels=augment_labels[idx_g] if augment_labels is not None else None,
                    ).to(torch.float32)
                else:
                    # Standard path: use live STUDENT at σ_s.
                    x_hat_s_ng = net(
                        x_s_teach[idx_g],
                        sigma_s[idx_g],              # [N_g,1,1,1]
                        labels[idx_g] if labels is not None else None,
                        augment_labels=augment_labels[idx_g] if augment_labels is not None else None,
                    ).to(torch.float32)
            
            ratio_s_b = (sigma_bdry[idx_g] / torch.clamp(sigma_s[idx_g], min=tol)).to(torch.float32)
            x_ref_bdry[idx_g] = x_hat_s_ng + ratio_s_b * (x_s_teach[idx_g].to(torch.float32) - x_hat_s_ng)
            sigma_ref_vec[idx_g] = sigma_bdry_vec[idx_g]

        # Gain diagnostics: σ_ref / σ_t and CD gain 1 / (1 - σ_ref/σ_t).
        # This is purely diagnostic and does not affect training dynamics.
        ratio_ref = sigma_ref_vec / torch.clamp(sigma_t_vec, min=1e-12)
        gain = 1.0 / torch.clamp(1.0 - ratio_ref, min=1e-6)  # [N]

        # Compute inv-DDIM target at t using per-sample sigma_ref (PRD §4.2.6)
        try:
            x_hat_t_star = inv_ddim_edm(
                x_ref=x_ref_bdry,
                x_t=x_t,
                sigma_t=sigma_t_vec,      # [N], standardize on 1D vectors
                sigma_ref=sigma_ref_vec,  # [N]
            ).to(torch.float32)
        except ValueError as e:
            # Add sampling context to the error message
            error_msg = str(e) + "\n\n  Sampling context for affected samples (first 5):\n"
            bad_idx = (torch.abs(sigma_ref_vec - sigma_t_vec) < 1e-8).nonzero(as_tuple=False).view(-1)[:5]
            for idx in bad_idx:
                i = int(idx.item())
                error_msg += (
                    f"    Sample {i}: seg_j={j[i].item()}, n_rel={n_rel[i].item()}, "
                    f"terminal={is_terminal[i].item()}, boundary_snap={is_boundary_snap[i].item()}\n"
                    f"              sigma_t={sigma_t_vec[i].item():.9f}, "
                    f"sigma_s_eff={sigma_s_eff[i].item():.9f}, "
                    f"sigma_bdry={sigma_bdry_vec[i].item():.9f}, "
                    f"sigma_ref={sigma_ref_vec[i].item():.9f}\n"
                )
            raise ValueError(error_msg) from e

        # Student prediction at t (PRD §4.2.7)
        x_hat_t = net(x_t, sigma_t, labels, augment_labels=augment_labels).to(torch.float32)

        # Weighting and loss
        # _weight expects 1D sigma vector [N]
        weight = self._weight(sigma_t_vec.to(torch.float32))  # [N]
        weight = weight.view(batch_size, 1, 1, 1)              # [N,1,1,1] for broadcast
        diff = x_hat_t - x_hat_t_star
        if self.loss_type == "huber":
            per_elem = _huber_loss(diff)
            loss = weight * per_elem
        elif self.loss_type == "l2_root":
            # Euclidean (L2) distance across all channels/pixels per sample.
            # Produce shape [N,1,1,1] to keep reporting semantics.
            per_sample = torch.sqrt(torch.clamp((diff * diff).sum(dim=[1, 2, 3]), min=1e-12))
            per_elem = per_sample.view(batch_size, 1, 1, 1)
            loss = weight * per_elem  # weight broadcast per sample
        else:
            # "l2": elementwise squared error (matches prior behavior)
            per_elem = diff * diff
            loss = weight * per_elem

        # Update edge statistics (per-sample counts) (PRD §4.2.8)
        num_terminal = int(is_terminal.sum().item())
        num_boundary = int(is_boundary_snap.sum().item())
        num_general = int((~is_terminal & ~is_boundary_snap).sum().item())
        num_edges = num_terminal + num_boundary + num_general  # should equal batch_size
        
        # Optional debug check: mask coverage
        if self.debug_invariants:
            assert num_edges == batch_size, f"Edge counts don't sum to batch_size: {num_edges} != {batch_size}"
        
        self._count_total_calls += 1                  # preserve "# of calls" semantics
        self._count_total_edges += num_edges          # new: "# of edges"
        self._count_terminal_edges += num_terminal
        self._count_boundary_match += num_boundary
        self._count_general_edges += num_general

        # Training stats reporting: batch means (PRD §4.2.9)
        if self.enable_stats:
            with torch.no_grad():
                # Core loss and sigma diagnostics.
                training_stats.report('Loss/cd', loss)
                training_stats.report('CD/sigma_t', sigma_t_vec.mean())
                training_stats.report('CD/sigma_s', sigma_s_eff.mean())
                training_stats.report('CD/sigma_bdry', sigma_bdry_vec.mean())
                training_stats.report('CD/seg_id', j.float().mean())
                training_stats.report('CD/T_edges', torch.as_tensor(float(T_edges), device=device))

                # Gain statistics (overall).
                training_stats.report('CD/gain_mean', gain.mean())
                training_stats.report('CD/gain_max', gain.max())
                training_stats.report('CD/gain_95p', gain.quantile(0.95))
                training_stats.report('CD/gain_99p', gain.quantile(0.99))

                # Gain statistics per edge type.
                gain_all = gain
                gain_terminal = gain[is_terminal]
                gain_boundary = gain[is_boundary_snap]
                gain_general = gain[general_mask]

                # Use empty lists when there are no samples for a given type so that
                # the set and ordering of statistic names remain consistent.
                training_stats.report('CD/gain_all', gain_all)
                training_stats.report(
                    'CD/gain_terminal_mean',
                    gain_terminal.mean() if gain_terminal.numel() > 0 else [],
                )
                training_stats.report(
                    'CD/gain_boundary_mean',
                    gain_boundary.mean() if gain_boundary.numel() > 0 else [],
                )
                training_stats.report(
                    'CD/gain_general_mean',
                    gain_general.mean() if gain_general.numel() > 0 else [],
                )

                # Correlate gain with loss magnitude.
                loss_mean_per_sample = loss.mean(dim=(1, 2, 3))  # [N]
                training_stats.report('CD/loss_mean', loss_mean_per_sample.mean())
                training_stats.report('CD/loss_gain_corr', (gain * loss_mean_per_sample).mean())

                # Cache last-step diagnostics for optional per-optimizer-step logging.
                try:
                    self._last_step_metrics = {
                        'cd_gain_mean': float(gain.mean().detach().cpu()),
                        'cd_gain_max': float(gain.max().detach().cpu()),
                        'cd_gain_95p': float(gain.quantile(0.95).detach().cpu()),
                        'cd_gain_99p': float(gain.quantile(0.99).detach().cpu()),
                        'cd_gain_terminal_mean': float(gain_terminal.mean().detach().cpu()) if gain_terminal.numel() > 0 else None,
                        'cd_gain_boundary_mean': float(gain_boundary.mean().detach().cpu()) if gain_boundary.numel() > 0 else None,
                        'cd_gain_general_mean': float(gain_general.mean().detach().cpu()) if gain_general.numel() > 0 else None,
                        'cd_loss_mean': float(loss_mean_per_sample.mean().detach().cpu()),
                        'cd_loss_gain_corr': float((gain * loss_mean_per_sample).mean().detach().cpu()),
                    }
                except Exception:
                    # Diagnostics are best-effort only; do not break training if something goes wrong.
                    self._last_step_metrics = getattr(self, '_last_step_metrics', None)

        return loss


