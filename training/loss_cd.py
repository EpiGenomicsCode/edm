import math
import os
import copy
from typing import Optional, Tuple, Dict, List, Any

import torch
from torch_utils import persistence
from torch_utils import training_stats

from .consistency_ops import (
    make_karras_sigmas,
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


def _pseudo_huber_vector_norm(diff: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Pseudo-Huber applied to the Euclidean vector norm (per sample).

    Computes  sqrt(||diff||^2 + eps^2) - eps  where ||·|| is the L2 norm
    across all spatial/channel dims.  For ||diff|| >> eps this equals ||diff||;
    near zero it smoothly transitions to a quadratic (differentiable everywhere).

    This matches the MSCD paper's "huber epsilon of 1e-4" formulation.

    Args:
        diff: [N, C, H, W] tensor of per-pixel differences.
        eps:  Huber smoothing parameter (paper uses 1e-4 for ImageNet).

    Returns:
        [N] tensor of per-sample pseudo-Huber norms.
    """
    norm_sq = (diff * diff).sum(dim=[1, 2, 3])  # [N]
    return torch.sqrt(norm_sq + eps * eps) - eps


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
        loss_type: str = "huber",    # "huber" | "l2" (squared) | "l2_root" (euclidean) | "pseudo_huber" (MSCD paper)
        weight_mode: str = "edm",    # "edm" | "vlike" | "flat" | "snr" | "snr+1" | "karras" | "sqrt_karras" | "truncated-snr" | "uniform"
        sigma_data: float = 0.5,
        enable_stats: bool = True,
        debug_invariants: bool = False,  # Enable runtime invariant checks (PRD §5, R7)
        target_net = None,  # Optional: separate target network for sigma_s denoiser (OpenAI CM style EMA or teacher)
        sampling_mode: str = "vp",  # Edge sampling: "uniform" | "vp" (MSCD uniform-t) | "edm" (log-normal)
        terminal_anchor: bool = True,  # Anchor terminal edge to 1/T probability (matches MSCD paper)
        terminal_teacher_hop: bool = False,  # Use teacher Euler hop for terminal edge instead of clean image y
    ):
        assert S >= 2, "Student steps S must be >= 2"
        assert T_start >= 2 and T_end >= T_start
        assert loss_type in ("huber", "l2", "l2_root", "pseudo_huber")
        assert weight_mode in (
            "edm",
            "vlike",
            "flat",
            # OpenAI consistency models style schedules (cm/karras_diffusion.get_weightings):
            "snr",
            "snr+1",
            "karras",
            "sqrt_karras",   # sqrt of Karras weight — correct pairing for linear losses (l2_root / pseudo_huber)
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
        assert sampling_mode in ("uniform", "vp", "edm"), f"Invalid sampling_mode: {sampling_mode}"
        self.sampling_mode = sampling_mode
        self.terminal_anchor = bool(terminal_anchor)
        self.terminal_teacher_hop = bool(terminal_teacher_hop)

        # Global kimg for teacher annealing; set externally by training loop.
        # Defaults to 0 if not explicitly set.
        self._global_kimg = 0.0
        
        # Diagnostic counters for edge type distribution (per-sample)
        self._count_terminal_edges = 0      # total terminal edges sampled (per-sample)
        self._count_boundary_match = 0      # total boundary-snap edges sampled (per-sample)
        self._count_general_edges = 0       # total general interior edges sampled (per-sample)
        self._count_total_calls = 0         # total __call__ invocations (unchanged semantics)
        self._count_total_edges = 0         # NEW: total sampled edges across all calls (≈ batch_size × calls)
        self._filter_cache = {}             # {target_T: (filtered_teacher_sigmas, terminal_k)}

    def set_run_dir(self, run_dir: str) -> None:
        """Set the training run directory for debug logging."""
        self._run_dir = run_dir

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

    def _build_teacher_grid(self, student_sigmas: torch.Tensor, device: torch.device):
        """Build filtered teacher grid with at least _current_T_edges() effective edges.

        Teacher edges whose sigma matches a student interior sigma are removed
        (prevents sigma_t == segment upper boundary).  To ensure the effective
        edge count never drops below the target from the annealing schedule, the
        raw grid size is inflated until the post-filter count >= target.

        Returns:
            teacher_sigmas: 1D tensor [T_eff+1] (descending, terminal 0)
            terminal_k: index of the terminal edge in the filtered grid
        """
        target_T = self._current_T_edges()

        # Check cache: avoid recomputing for the same target_T.
        if target_T in self._filter_cache:
            cached_sigmas, cached_terminal_k = self._filter_cache[target_T]
            return cached_sigmas.to(device), cached_terminal_k

        raw_T = target_T
        while True:
            sigmas_prepad = make_karras_sigmas(
                num_nodes=raw_T,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                round_fn=self.teacher_net.round_sigma,
            ).to(device)
            zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
            teacher_full = torch.cat([sigmas_prepad, zero], dim=0)

            teacher_filtered, terminal_k = filter_teacher_edges_by_sigma(
                student_sigmas=student_sigmas,
                teacher_sigmas=teacher_full,
            )
            T_eff = teacher_filtered.shape[0] - 1
            if T_eff >= target_T:
                break
            raw_T += 1

        self._filter_cache[target_T] = (teacher_filtered.clone(), terminal_k)
        return teacher_filtered.to(device), terminal_k

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
        if self.weight_mode == "sqrt_karras":
            # Square root of Karras weight: sqrt((σ² + σ_data²) / (σ·σ_data)²).
            # Correct pairing for linear losses (l2_root, pseudo_huber) so that
            # c_out(σ) · w(σ) = 1, giving uniform contribution across σ.
            # Equivalent to 1 / c_out(σ) where c_out = σ·σ_data / sqrt(σ² + σ_data²).
            return torch.sqrt(sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)
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

        # Teacher grid: filtered to remove edges where sigma_t == student boundary.
        # Raw grid is inflated so effective edge count >= target from annealing schedule,
        # preventing the non-monotonic T_edges oscillation that plagued the old approach.
        teacher_sigmas, terminal_k = self._build_teacher_grid(
            student_sigmas=student_sigmas, device=device,
        )
        T_edges = teacher_sigmas.shape[0] - 1

        sigma_bounds = partition_edges_by_sigma(
            student_sigmas=student_sigmas,
            teacher_sigmas=teacher_sigmas,
        )

        sample_dict = sample_segment_and_teacher_pair(
            sigma_bounds=sigma_bounds,
            teacher_sigmas=teacher_sigmas,
            student_sigmas=student_sigmas,
            batch_size=batch_size,
            device=device,
            terminal_k=terminal_k,
            sampling_mode=self.sampling_mode,
            rho=self.rho,
            terminal_anchor=self.terminal_anchor,
        )

        # -------------------------------------------------------------------------------------
        # Option A: Resample only degenerate edges (invDDIM requires σ_ref < σ_t).
        #
        # We pre-screen (σ_t, σ_ref) pairs *before* sampling noise x_t so that any resampled
        # elements get a consistent x_t drawn for their new σ_t.
        #
        # Extract per-sample vectors (all shape [N]).
        j = sample_dict["step_j"].long()               # [N]
        n_rel = sample_dict["n_rel"].long()            # [N]
        sigma_t_vec = sample_dict["sigma_t"]           # [N]
        sigma_s_teacher_vec = sample_dict["sigma_s"]   # [N]
        sigma_bdry_vec = sample_dict["sigma_bdry"]     # [N]
        is_terminal = sample_dict["is_terminal"].bool()        # [N]
        is_boundary_snap = sample_dict["is_boundary_snap"].bool()  # [N]
        is_general = (~is_terminal) & (~is_boundary_snap)

        # Compute σ_s_eff based on edge type:
        # - boundary_snap: σ_s_eff = σ_bdry (snap to student boundary)
        # - general: σ_s_eff = max(σ_s_teacher, σ_bdry) (don't cross below boundary)
        # - terminal: σ_s_eff = 0 (handled separately, but set for consistency)
        sigma_s_eff = sigma_s_teacher_vec.clone()
        sigma_s_eff = torch.where(is_boundary_snap, sigma_bdry_vec, sigma_s_eff)
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

        # Ground-truth ordering invariants (always asserted).
        tol = 1e-8
        # sigma_t must be strictly below the segment's upper boundary, except for
        # two structural endpoints: (a) segment 0 where sigma_t = sigma_max is
        # unavoidable since both grids start there, and (b) the terminal edge.
        sigma_upper_vec = student_sigmas[j]
        interior_mask = (j > 0) & (~is_terminal)
        if interior_mask.any():
            assert (sigma_t_vec[interior_mask] < sigma_upper_vec[interior_mask] - tol).all(), (
                "Ordering: sigma_t must be strictly < segment upper boundary (interior segments)"
            )
        assert (sigma_t_vec > sigma_s_teacher_vec + tol).all(), (
            "Ordering: sigma_t must be strictly > sigma_s (raw)"
        )
        non_term = ~is_terminal
        if non_term.any():
            assert (sigma_t_vec[non_term] > sigma_s_eff[non_term] + tol).all(), (
                "Ordering: sigma_t must be strictly > sigma_s_eff for non-terminal"
            )
        assert (sigma_s_eff >= sigma_bdry_vec - tol).all(), (
            "Ordering: sigma_s_eff must be >= sigma_bdry"
        )

        # Optional extra invariant checks (PRD §5, R7) when debug_invariants enabled.
        if self.debug_invariants:
            assert (sigma_s_eff[is_terminal] == 0).all(), "Terminal edges must have sigma_s_eff == 0"
            assert (is_terminal & is_boundary_snap).sum() == 0, "Terminal and boundary_snap must be disjoint"
            assert (n_rel[is_boundary_snap] == 1).all(), "Boundary snap edges must have n_rel == 1"
            if is_boundary_snap.any():
                assert (sigma_t_vec[is_boundary_snap] > sigma_bdry_vec[is_boundary_snap] + tol).all(), (
                    "Boundary snap: sigma_t must be strictly > sigma_bdry"
                )

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
        
        # Terminal edges: σ_ref = 0.
        # Default: anchor to ground-truth clean image y.
        # With terminal_teacher_hop: use teacher's Euler step from σ_min→0,
        # which equals D_teacher(x_t, σ_min). This matches the actual sampling
        # chain (generate.py: last step is Euler-only since Heun's 2nd-order
        # correction divides by t_next=0).
        if self.terminal_teacher_hop and is_terminal.any():
            with torch.no_grad():
                x_ref_bdry[is_terminal] = self.teacher_net(
                    x_t[is_terminal],
                    sigma_t_vec[is_terminal],
                    labels[is_terminal] if labels is not None else None,
                    augment_labels=augment_labels[is_terminal] if augment_labels is not None else None,
                ).to(torch.float32)
        else:
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
                    # Disable dropout/label_dropout for the target reference so the
                    # inv-DDIM target is deterministic (matches ema/teacher modes
                    # which are already .eval()).  Only GroupNorm is used (no running
                    # stats), so eval/train toggle is safe.
                    net.eval()
                    x_hat_s_ng = net(
                        x_s_teach[idx_g],
                        sigma_s[idx_g],              # [N_g,1,1,1]
                        labels[idx_g] if labels is not None else None,
                        augment_labels=augment_labels[idx_g] if augment_labels is not None else None,
                    ).to(torch.float32)
                    net.train()
            
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

        # Student prediction at t (net is back in .train() mode → dropout active)
        x_hat_t = net(x_t, sigma_t, labels, augment_labels=augment_labels).to(torch.float32)

        # Weighting and loss
        # _weight expects 1D sigma vector [N]
        weight = self._weight(sigma_t_vec.to(torch.float32))  # [N]
        weight = weight.view(batch_size, 1, 1, 1)              # [N,1,1,1] for broadcast
        diff = x_hat_t - x_hat_t_star
        if self.loss_type == "huber":
            per_elem = _huber_loss(diff)
            loss = weight * per_elem
        elif self.loss_type == "pseudo_huber":
            # Pseudo-Huber on the vector norm (MSCD paper formulation).
            # sqrt(||diff||^2 + eps^2) - eps, with eps = 1e-4.
            per_sample = _pseudo_huber_vector_norm(diff, eps=1e-4)
            per_elem = per_sample.view(batch_size, 1, 1, 1)
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

                # =========================================================================
                # DIAGNOSTIC 0: Loss spike analysis
                # =========================================================================
                # Detect per-sample outliers whose loss is far above the batch median.
                # We use median + 5*MAD (median absolute deviation) as a robust threshold;
                # any sample exceeding this is a "spike".  We then log the composition
                # of spikes (edge type, sigma_t, gain, etc.) so we can correlate.
                batch_median_loss = loss_mean_per_sample.median()
                batch_mad = (loss_mean_per_sample - batch_median_loss).abs().median().clamp(min=1e-12)
                spike_threshold = batch_median_loss + 5.0 * batch_mad
                is_spike = loss_mean_per_sample > spike_threshold  # [N] bool
                num_spikes = int(is_spike.sum().item())
                
                training_stats.report('CD/spike_count', torch.as_tensor(float(num_spikes), device=device))
                training_stats.report('CD/spike_threshold', spike_threshold)
                training_stats.report('CD/spike_frac', torch.as_tensor(float(num_spikes) / max(batch_size, 1), device=device))
                
                if num_spikes > 0:
                    spike_losses = loss_mean_per_sample[is_spike]
                    training_stats.report('CD/spike_loss_mean', spike_losses.mean())
                    training_stats.report('CD/spike_loss_max', spike_losses.max())
                    
                    # What edge types are the spikes?
                    spike_terminal = (is_spike & is_terminal).sum().float()
                    spike_boundary = (is_spike & is_boundary_snap).sum().float()
                    spike_general  = (is_spike & general_mask).sum().float()
                    training_stats.report('CD/spike_pct_terminal', spike_terminal / max(num_spikes, 1))
                    training_stats.report('CD/spike_pct_boundary', spike_boundary / max(num_spikes, 1))
                    training_stats.report('CD/spike_pct_general',  spike_general  / max(num_spikes, 1))
                    
                    # Sigma and gain profile of spikes
                    training_stats.report('CD/spike_sigma_t_mean', sigma_t_vec[is_spike].mean())
                    training_stats.report('CD/spike_sigma_t_min',  sigma_t_vec[is_spike].min())
                    training_stats.report('CD/spike_sigma_t_max',  sigma_t_vec[is_spike].max())
                    training_stats.report('CD/spike_gain_mean', gain[is_spike].mean())
                    training_stats.report('CD/spike_gain_max',  gain[is_spike].max())
                    
                    # What segment ids are producing spikes?
                    training_stats.report('CD/spike_seg_id_mean', j[is_spike].float().mean())
                    
                    # Weight (EDM weighting) of spike samples — are high-weight samples spiking?
                    training_stats.report('CD/spike_weight_mean', weight.view(-1)[is_spike].mean())
                else:
                    # Report zeros / empty so stat names stay consistent
                    training_stats.report('CD/spike_loss_mean', [])
                    training_stats.report('CD/spike_loss_max', [])
                    training_stats.report('CD/spike_pct_terminal', [])
                    training_stats.report('CD/spike_pct_boundary', [])
                    training_stats.report('CD/spike_pct_general', [])
                    training_stats.report('CD/spike_sigma_t_mean', [])
                    training_stats.report('CD/spike_sigma_t_min', [])
                    training_stats.report('CD/spike_sigma_t_max', [])
                    training_stats.report('CD/spike_gain_mean', [])
                    training_stats.report('CD/spike_gain_max', [])
                    training_stats.report('CD/spike_seg_id_mean', [])
                    training_stats.report('CD/spike_weight_mean', [])

                # =========================================================================
                # DIAGNOSTIC 1: Per-edge breakdown of consistency error (L2)
                # =========================================================================
                # Compute per-sample L2 error (squared norm across CHW)
                per_sample_l2 = (diff * diff).sum(dim=[1, 2, 3])  # [N]
                per_sample_l2_sqrt = torch.sqrt(per_sample_l2.clamp(min=1e-12))  # [N]
                
                # L2 error by edge type
                l2_terminal = per_sample_l2_sqrt[is_terminal]
                l2_boundary = per_sample_l2_sqrt[is_boundary_snap]
                l2_general = per_sample_l2_sqrt[general_mask]
                
                training_stats.report('CD/l2_error_all', per_sample_l2_sqrt.mean())
                training_stats.report(
                    'CD/l2_error_terminal',
                    l2_terminal.mean() if l2_terminal.numel() > 0 else [],
                )
                training_stats.report(
                    'CD/l2_error_boundary',
                    l2_boundary.mean() if l2_boundary.numel() > 0 else [],
                )
                training_stats.report(
                    'CD/l2_error_general',
                    l2_general.mean() if l2_general.numel() > 0 else [],
                )
                
                # Also track variance to see if error is consistent or spiky
                training_stats.report('CD/l2_error_std', per_sample_l2_sqrt.std() if per_sample_l2_sqrt.numel() > 1 else [])
                training_stats.report(
                    'CD/l2_error_boundary_max',
                    l2_boundary.max() if l2_boundary.numel() > 0 else [],
                )
                
                # =========================================================================
                # DIAGNOSTIC 2 & 3: Teacher vs Student output comparison
                # =========================================================================
                # Compute teacher's denoised estimate at sigma_t for ALL samples (single forward pass)
                with torch.no_grad():
                    x_hat_t_teacher = self.teacher_net(
                        x_t,
                        sigma_t_vec,
                        labels,
                        augment_labels=augment_labels,
                    ).to(torch.float32)
                
                # Student-Teacher divergence at sigma_t (per-sample)
                diff_st = x_hat_t - x_hat_t_teacher
                st_divergence = torch.sqrt((diff_st * diff_st).sum(dim=[1, 2, 3]).clamp(min=1e-12))  # [N]
                
                # Overall divergence stats
                training_stats.report('CD/student_teacher_divergence', st_divergence.mean())
                training_stats.report('CD/student_teacher_divergence_max', st_divergence.max())
                
                # Divergence by edge type
                training_stats.report(
                    'CD/st_div_terminal',
                    st_divergence[is_terminal].mean() if is_terminal.any() else [],
                )
                training_stats.report(
                    'CD/st_div_boundary',
                    st_divergence[is_boundary_snap].mean() if is_boundary_snap.any() else [],
                )
                training_stats.report(
                    'CD/st_div_general',
                    st_divergence[general_mask].mean() if general_mask.any() else [],
                )
                
                # Additional boundary-specific diagnostics (using the already-computed x_hat_t_teacher)
                if is_boundary_snap.any():
                    idx_b = is_boundary_snap
                    
                    # Boundary-specific divergence (reuse from above)
                    training_stats.report('CD/student_teacher_div_boundary', st_divergence[idx_b].mean())
                    training_stats.report('CD/student_teacher_div_boundary_max', st_divergence[idx_b].max())
                    
                    # Teacher's output to ground truth (x_0 = y) for boundary edges
                    diff_teacher_gt = x_hat_t_teacher[idx_b] - y[idx_b]
                    teacher_gt_error = torch.sqrt((diff_teacher_gt * diff_teacher_gt).sum(dim=[1, 2, 3]).clamp(min=1e-12))
                    training_stats.report('CD/teacher_gt_error_boundary', teacher_gt_error.mean())
                    
                    # Student's output to ground truth for boundary edges
                    diff_student_gt = x_hat_t[idx_b] - y[idx_b]
                    student_gt_error = torch.sqrt((diff_student_gt * diff_student_gt).sum(dim=[1, 2, 3]).clamp(min=1e-12))
                    training_stats.report('CD/student_gt_error_boundary', student_gt_error.mean())
                else:
                    training_stats.report('CD/student_teacher_div_boundary', [])
                    training_stats.report('CD/student_teacher_div_boundary_max', [])
                    training_stats.report('CD/teacher_gt_error_boundary', [])
                    training_stats.report('CD/student_gt_error_boundary', [])
                
                # =========================================================================
                # DIAGNOSTIC 4: Gradient contribution analysis
                # =========================================================================
                # Compute weighted loss contribution per edge type (proxy for gradient contribution)
                weighted_loss_per_sample = (weight.view(-1) * per_sample_l2).detach()  # [N]
                
                total_weighted_loss = weighted_loss_per_sample.sum()
                terminal_contrib = weighted_loss_per_sample[is_terminal].sum() if is_terminal.any() else torch.tensor(0.0, device=device)
                boundary_contrib = weighted_loss_per_sample[is_boundary_snap].sum() if is_boundary_snap.any() else torch.tensor(0.0, device=device)
                general_contrib = weighted_loss_per_sample[general_mask].sum() if general_mask.any() else torch.tensor(0.0, device=device)
                
                # Fraction of gradient from each edge type
                eps_frac = 1e-10
                training_stats.report('CD/grad_frac_terminal', terminal_contrib / (total_weighted_loss + eps_frac))
                training_stats.report('CD/grad_frac_boundary', boundary_contrib / (total_weighted_loss + eps_frac))
                training_stats.report('CD/grad_frac_general', general_contrib / (total_weighted_loss + eps_frac))
                
                # =========================================================================
                # DIAGNOSTIC 5: Target consistency check
                # =========================================================================
                # For boundary snap: is the target (x_hat_t_star) close to what teacher would produce?
                # The target is computed from inv_ddim using x_ref = teacher's hop result
                # If student is drifting, target quality might degrade
                if is_boundary_snap.any():
                    idx_b = is_boundary_snap
                    # Target-Teacher divergence: how different is our target from teacher's direct estimate?
                    diff_target_teacher = x_hat_t_star[idx_b] - x_hat_t_teacher[idx_b]
                    target_teacher_div = torch.sqrt((diff_target_teacher * diff_target_teacher).sum(dim=[1, 2, 3]).clamp(min=1e-12))
                    training_stats.report('CD/target_teacher_div_boundary', target_teacher_div.mean())
                    
                    # Target-GT divergence
                    diff_target_gt = x_hat_t_star[idx_b] - y[idx_b]
                    target_gt_div = torch.sqrt((diff_target_gt * diff_target_gt).sum(dim=[1, 2, 3]).clamp(min=1e-12))
                    training_stats.report('CD/target_gt_div_boundary', target_gt_div.mean())
                else:
                    training_stats.report('CD/target_teacher_div_boundary', [])
                    training_stats.report('CD/target_gt_div_boundary', [])
                
                # =========================================================================
                # DIAGNOSTIC 6: Gradient direction conflict analysis
                # =========================================================================
                # Check if boundary and general edges are pushing the student in conflicting directions
                # We use the student-target difference (diff) as a proxy for the gradient direction
                # Positive cosine similarity = aligned, Negative = conflicting
                if is_boundary_snap.any() and general_mask.any():
                    # Flatten diff to [N, D] where D = C*H*W
                    diff_flat = diff.view(batch_size, -1)  # [N, D]
                    
                    # Mean gradient direction for boundary edges
                    boundary_grad_dir = diff_flat[is_boundary_snap].mean(dim=0)  # [D]
                    boundary_grad_norm = torch.sqrt((boundary_grad_dir * boundary_grad_dir).sum().clamp(min=1e-12))
                    
                    # Mean gradient direction for general edges
                    general_grad_dir = diff_flat[general_mask].mean(dim=0)  # [D]
                    general_grad_norm = torch.sqrt((general_grad_dir * general_grad_dir).sum().clamp(min=1e-12))
                    
                    # Cosine similarity between boundary and general gradient directions
                    cosine_sim = (boundary_grad_dir * general_grad_dir).sum() / (boundary_grad_norm * general_grad_norm + 1e-12)
                    training_stats.report('CD/grad_conflict_boundary_general', cosine_sim)
                    
                    # Also track the relative magnitude of boundary vs general gradients
                    training_stats.report('CD/grad_norm_boundary', boundary_grad_norm)
                    training_stats.report('CD/grad_norm_general', general_grad_norm)
                    training_stats.report('CD/grad_norm_ratio_boundary_general', boundary_grad_norm / (general_grad_norm + 1e-12))
                else:
                    training_stats.report('CD/grad_conflict_boundary_general', [])
                    training_stats.report('CD/grad_norm_boundary', [])
                    training_stats.report('CD/grad_norm_general', [])
                    training_stats.report('CD/grad_norm_ratio_boundary_general', [])

                # =========================================================================
                # DIAGNOSTIC 7: Per-sigma-bucket denoising quality (H2 test)
                # =========================================================================
                # Measures whether the student loses denoising quality at specific sigma
                # ranges during training.  ||D(z_t, σ_t) - x|| per sigma bucket.
                denoise_err = torch.sqrt(((x_hat_t - y) ** 2).sum(dim=[1, 2, 3]).clamp(min=1e-12))  # [N]
                sigma_flat = sigma_t_vec  # [N], 1-D
                bucket_lo = torch.tensor([0.0, 0.1, 1.0, 10.0], device=device)
                bucket_hi = torch.tensor([0.1, 1.0, 10.0, 81.0], device=device)
                bucket_names = ['0_01', '01_1', '1_10', '10_80']
                for bname, lo, hi in zip(bucket_names, bucket_lo, bucket_hi):
                    mask_b = (sigma_flat >= lo) & (sigma_flat < hi)
                    if mask_b.any():
                        training_stats.report(f'CD/denoise_q_{bname}', denoise_err[mask_b].mean())
                    else:
                        training_stats.report(f'CD/denoise_q_{bname}', [])

                # Overall denoising quality
                training_stats.report('CD/denoise_quality_all', denoise_err.mean())

                # =========================================================================
                # DIAGNOSTIC 8: DDIM ratio for general edges (H3 test)
                # =========================================================================
                # σ_bdry/σ_s for general edges — how self-referential is the target?
                # Small ratio means target ≈ student prediction (weak teacher influence).
                if general_mask.any():
                    ddim_ratio_gen = (sigma_bdry[general_mask] / torch.clamp(sigma_s[general_mask], min=1e-12)).squeeze()
                    training_stats.report('CD/ddim_ratio_gen_mean', ddim_ratio_gen.mean())
                    training_stats.report('CD/ddim_ratio_gen_min', ddim_ratio_gen.min())
                    training_stats.report('CD/ddim_ratio_gen_median', ddim_ratio_gen.median() if ddim_ratio_gen.numel() > 0 else [])
                    # Fraction of general edges with ratio < 0.1 (highly self-referential)
                    frac_self_ref = (ddim_ratio_gen < 0.1).float().mean()
                    training_stats.report('CD/ddim_frac_self_ref', frac_self_ref)
                else:
                    training_stats.report('CD/ddim_ratio_gen_mean', [])
                    training_stats.report('CD/ddim_ratio_gen_min', [])
                    training_stats.report('CD/ddim_ratio_gen_median', [])
                    training_stats.report('CD/ddim_frac_self_ref', [])

                # =========================================================================
                # DIAGNOSTIC 9: Edge type fractions and T value (H1 context)
                # =========================================================================
                training_stats.report('CD/frac_terminal', torch.as_tensor(float(num_terminal) / max(num_edges, 1), device=device))
                training_stats.report('CD/frac_boundary', torch.as_tensor(float(num_boundary) / max(num_edges, 1), device=device))
                training_stats.report('CD/frac_general', torch.as_tensor(float(num_general) / max(num_edges, 1), device=device))

                # Cache last-step diagnostics for optional per-optimizer-step logging.
                try:
                    self._last_step_metrics = {
                        # Original gain metrics
                        'cd_gain_mean': float(gain.mean().detach().cpu()),
                        'cd_gain_max': float(gain.max().detach().cpu()),
                        'cd_gain_95p': float(gain.quantile(0.95).detach().cpu()),
                        'cd_gain_99p': float(gain.quantile(0.99).detach().cpu()),
                        'cd_gain_terminal_mean': float(gain_terminal.mean().detach().cpu()) if gain_terminal.numel() > 0 else None,
                        'cd_gain_boundary_mean': float(gain_boundary.mean().detach().cpu()) if gain_boundary.numel() > 0 else None,
                        'cd_gain_general_mean': float(gain_general.mean().detach().cpu()) if gain_general.numel() > 0 else None,
                        'cd_loss_mean': float(loss_mean_per_sample.mean().detach().cpu()),
                        'cd_loss_gain_corr': float((gain * loss_mean_per_sample).mean().detach().cpu()),
                        
                        # DIAGNOSTIC 1: Per-edge L2 error
                        'cd_l2_error_all': float(per_sample_l2_sqrt.mean().detach().cpu()),
                        'cd_l2_error_terminal': float(l2_terminal.mean().detach().cpu()) if l2_terminal.numel() > 0 else None,
                        'cd_l2_error_boundary': float(l2_boundary.mean().detach().cpu()) if l2_boundary.numel() > 0 else None,
                        'cd_l2_error_general': float(l2_general.mean().detach().cpu()) if l2_general.numel() > 0 else None,
                        
                        # DIAGNOSTIC 3: Student-Teacher divergence
                        'cd_st_divergence': float(st_divergence.mean().detach().cpu()),
                        'cd_st_divergence_max': float(st_divergence.max().detach().cpu()),
                        'cd_st_div_terminal': float(st_divergence[is_terminal].mean().detach().cpu()) if is_terminal.any() else None,
                        'cd_st_div_boundary': float(st_divergence[is_boundary_snap].mean().detach().cpu()) if is_boundary_snap.any() else None,
                        'cd_st_div_general': float(st_divergence[general_mask].mean().detach().cpu()) if general_mask.any() else None,
                        
                        # DIAGNOSTIC 4: Gradient fraction
                        'cd_grad_frac_terminal': float((terminal_contrib / (total_weighted_loss + eps_frac)).detach().cpu()),
                        'cd_grad_frac_boundary': float((boundary_contrib / (total_weighted_loss + eps_frac)).detach().cpu()),
                        'cd_grad_frac_general': float((general_contrib / (total_weighted_loss + eps_frac)).detach().cpu()),
                    }
                    
                    # DIAGNOSTIC 0: Loss spike analysis
                    self._last_step_metrics['cd_spike_count'] = num_spikes
                    self._last_step_metrics['cd_spike_frac'] = float(num_spikes) / max(batch_size, 1)
                    if num_spikes > 0:
                        self._last_step_metrics['cd_spike_loss_mean'] = float(spike_losses.mean().detach().cpu())
                        self._last_step_metrics['cd_spike_loss_max'] = float(spike_losses.max().detach().cpu())
                        spike_terminal = (is_spike & is_terminal).sum().float()
                        spike_boundary = (is_spike & is_boundary_snap).sum().float()
                        spike_general  = (is_spike & general_mask).sum().float()
                        self._last_step_metrics['cd_spike_pct_terminal'] = float(spike_terminal / max(num_spikes, 1))
                        self._last_step_metrics['cd_spike_pct_boundary'] = float(spike_boundary / max(num_spikes, 1))
                        self._last_step_metrics['cd_spike_pct_general'] = float(spike_general / max(num_spikes, 1))
                        self._last_step_metrics['cd_spike_sigma_t_mean'] = float(sigma_t_vec[is_spike].mean().detach().cpu())
                        self._last_step_metrics['cd_spike_gain_mean'] = float(gain[is_spike].mean().detach().cpu())
                        self._last_step_metrics['cd_spike_gain_max'] = float(gain[is_spike].max().detach().cpu())
                        self._last_step_metrics['cd_spike_seg_id_mean'] = float(j[is_spike].float().mean().detach().cpu())
                        self._last_step_metrics['cd_spike_weight_mean'] = float(weight.view(-1)[is_spike].mean().detach().cpu())
                    else:
                        self._last_step_metrics['cd_spike_loss_mean'] = None
                        self._last_step_metrics['cd_spike_loss_max'] = None
                        self._last_step_metrics['cd_spike_pct_terminal'] = None
                        self._last_step_metrics['cd_spike_pct_boundary'] = None
                        self._last_step_metrics['cd_spike_pct_general'] = None
                        self._last_step_metrics['cd_spike_sigma_t_mean'] = None
                        self._last_step_metrics['cd_spike_gain_mean'] = None
                        self._last_step_metrics['cd_spike_gain_max'] = None
                        self._last_step_metrics['cd_spike_seg_id_mean'] = None
                        self._last_step_metrics['cd_spike_weight_mean'] = None

                    # DIAGNOSTIC 6: Gradient conflict (only if both boundary and general exist)
                    if is_boundary_snap.any() and general_mask.any():
                        diff_flat = diff.view(batch_size, -1)
                        boundary_grad_dir = diff_flat[is_boundary_snap].mean(dim=0)
                        boundary_grad_norm = torch.sqrt((boundary_grad_dir * boundary_grad_dir).sum().clamp(min=1e-12))
                        general_grad_dir = diff_flat[general_mask].mean(dim=0)
                        general_grad_norm = torch.sqrt((general_grad_dir * general_grad_dir).sum().clamp(min=1e-12))
                        cosine_sim = (boundary_grad_dir * general_grad_dir).sum() / (boundary_grad_norm * general_grad_norm + 1e-12)
                        self._last_step_metrics['cd_grad_conflict'] = float(cosine_sim.detach().cpu())
                        self._last_step_metrics['cd_grad_norm_boundary'] = float(boundary_grad_norm.detach().cpu())
                        self._last_step_metrics['cd_grad_norm_general'] = float(general_grad_norm.detach().cpu())
                        self._last_step_metrics['cd_grad_norm_ratio'] = float((boundary_grad_norm / (general_grad_norm + 1e-12)).detach().cpu())
                    else:
                        self._last_step_metrics['cd_grad_conflict'] = None
                        self._last_step_metrics['cd_grad_norm_boundary'] = None
                        self._last_step_metrics['cd_grad_norm_general'] = None
                        self._last_step_metrics['cd_grad_norm_ratio'] = None

                    # DIAGNOSTIC 7: Per-sigma-bucket denoising quality
                    self._last_step_metrics['cd_denoise_quality_all'] = float(denoise_err.mean().detach().cpu())
                    for bname, lo, hi in zip(bucket_names, bucket_lo, bucket_hi):
                        mask_b = (sigma_flat >= lo) & (sigma_flat < hi)
                        self._last_step_metrics[f'cd_denoise_q_{bname}'] = float(denoise_err[mask_b].mean().detach().cpu()) if mask_b.any() else None

                    # DIAGNOSTIC 8: DDIM ratio for general edges
                    if general_mask.any():
                        ddim_ratio_gen = (sigma_bdry[general_mask] / torch.clamp(sigma_s[general_mask], min=1e-12)).squeeze()
                        self._last_step_metrics['cd_ddim_ratio_gen_mean'] = float(ddim_ratio_gen.mean().detach().cpu())
                        self._last_step_metrics['cd_ddim_ratio_gen_min'] = float(ddim_ratio_gen.min().detach().cpu())
                        self._last_step_metrics['cd_ddim_frac_self_ref'] = float((ddim_ratio_gen < 0.1).float().mean().detach().cpu())
                    else:
                        self._last_step_metrics['cd_ddim_ratio_gen_mean'] = None
                        self._last_step_metrics['cd_ddim_ratio_gen_min'] = None
                        self._last_step_metrics['cd_ddim_frac_self_ref'] = None

                    # DIAGNOSTIC 9: Edge type fractions
                    self._last_step_metrics['cd_frac_terminal'] = float(num_terminal) / max(num_edges, 1)
                    self._last_step_metrics['cd_frac_boundary'] = float(num_boundary) / max(num_edges, 1)
                    self._last_step_metrics['cd_frac_general'] = float(num_general) / max(num_edges, 1)
                except Exception:
                    # Diagnostics are best-effort only; do not break training if something goes wrong.
                    self._last_step_metrics = getattr(self, '_last_step_metrics', None)

        return loss


