import math
from typing import Optional, Tuple

import torch
from torch_utils import persistence
from torch_utils import training_stats

from .consistency_ops import (
    make_karras_sigmas,
    partition_edges_into_segments,
    heun_hop_edm,
    inv_ddim_edm,
    ddim_step_edm,
)


def _select_segment_and_edge(segments: list, rng_device: torch.device) -> Tuple[int, int]:
    """
    Uniformly sample a segment id j and an adjacent edge index e inside that segment.
    segments[j] = (start, end) over teacher edges, using [start, end) convention.
    Returns (j, e) with e in [start, end-1].
    """
    assert len(segments) > 0
    S = len(segments)
    # Choose segment uniformly.
    j = int(torch.randint(low=0, high=S, size=(1,), device=rng_device).item())
    start, end = segments[j]
    # Handle degenerate empty segments by resampling a non-empty one.
    attempts = 0
    while (end - start) <= 0 and attempts < 8:
        j = int(torch.randint(low=0, high=S, size=(1,), device=rng_device).item())
        start, end = segments[j]
        attempts += 1
    if (end - start) <= 0:
        # Fallback: collapse to the closest valid edge index.
        # This should be exceedingly rare unless S >> T.
        j = 0
        start, end = segments[j]
    # Sample adjacent edge e within [start, end-1].
    # If only one edge in the segment, this becomes deterministic.
    e = int(torch.randint(low=start, high=max(start + 1, end), size=(1,), device=rng_device).item())
    return j, e


def _huber_loss(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.as_tensor(delta, device=x.device, dtype=x.dtype))
    # 0.5 * quad^2 + delta * (abs_x - quad)
    return 0.5 * (quad * quad) + (abs_x - quad) * delta


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

    def set_global_kimg(self, kimg: float) -> None:
        """
        Set the global training progress in kimg for annealing teacher edges.
        The training loop is responsible for calling this once per tick with
        the current global kimg (including resume_kimg).
        """
        self._global_kimg = float(kimg)

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
        # Student requires S+1 descending positive sigmas; conceptual 0 boundary handled separately.
        # net may be wrapped in DDP; grab underlying module's round_sigma if needed.
        round_fn = getattr(net, 'round_sigma', None)
        if round_fn is None and hasattr(net, 'module'):
            round_fn = getattr(net.module, 'round_sigma', None)
        sigmas = make_karras_sigmas(
            num_nodes=self.S + 1,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            round_fn=round_fn,
        ).to(device)
        return sigmas

    def _build_teacher_grid(self, device: torch.device) -> torch.Tensor:
        # Teacher edges T => T+1 nodes.
        T_edges = self._current_T_edges()
        sigmas = make_karras_sigmas(
            num_nodes=T_edges + 1,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            round_fn=self.teacher_net.round_sigma,
        ).to(device)
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

        # Build grids.
        student_sigmas = self._build_student_grid(net=net, device=device)  # [S+1], descending
        teacher_sigmas = self._build_teacher_grid(device=device)           # [T+1], descending
        T_edges = teacher_sigmas.shape[0] - 1

        # Partition teacher edges into S segments.
        segments = partition_edges_into_segments(T=T_edges, S=self.S)

        # Sample a segment j and an adjacent teacher edge e inside it.
        j, e = _select_segment_and_edge(segments, rng_device=device)
        sigma_t_scalar = teacher_sigmas[e]
        sigma_s_scalar = teacher_sigmas[e + 1]

        # Determine right student boundary for this segment.
        # For segments j = 0..S-2, boundary is student_sigmas[j+1]; for last j=S-1, conceptual boundary is 0.
        if j < self.S - 1:
            sigma_bdry_scalar = student_sigmas[j + 1]
        else:
            sigma_bdry_scalar = torch.as_tensor(0.0, device=device, dtype=teacher_sigmas.dtype)

        # Broadcast scalars to per-sample sigma tensors as [N,1,1,1] to match other losses.
        sigma_t = sigma_t_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
        sigma_s = sigma_s_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
        sigma_bdry = sigma_bdry_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)

        # Sample noise and form x_t = y + sigma_t * eps.
        eps = torch.randn_like(y)
        x_t = y + sigma_t * eps

        # Deterministic teacher hop with Heun: x_s_teach.
        x_s_teach = heun_hop_edm(
            net=self.teacher_net,
            x_t=x_t,
            sigma_t=sigma_t_scalar,  # Pass scalar to allow rounding within the op.
            sigma_s=sigma_s_scalar,
            class_labels=labels,
            augment_labels=augment_labels,
        )

        # Push to right student boundary at s using student net at σ_s with stopgrad prediction.
        # Guard: if σ_bdry == σ_s (within tol), set x_ref_bdry = x_s_teach.
        # Use float32 math internally.
        tol = 1e-12
        equal_b_s_scalar = torch.isclose(
            sigma_bdry_scalar.to(torch.float32), sigma_s_scalar.to(torch.float32), atol=1e-12, rtol=0.0
        ).item()
        if equal_b_s_scalar:
            # Take the teacher hop as the boundary directly; do not run a student Euler/DDIM step.
            x_ref_bdry = x_s_teach.to(torch.float32)
        else:
            with torch.no_grad():
                x_hat_s_ng = net(x_s_teach, sigma_s, labels, augment_labels=augment_labels).to(torch.float32)
            ratio_s_b = (sigma_bdry / torch.clamp(sigma_s, min=tol)).to(torch.float32)
            x_ref_bdry = x_hat_s_ng + ratio_s_b * (x_s_teach.to(torch.float32) - x_hat_s_ng)

        # Backsolve a target at t via inverse-DDIM in EDM space.
        # If σ_bdry == σ_t (within tol), resample a few times or skip.
        attempts = 0
        while attempts < 4 and torch.isclose(sigma_bdry_scalar.to(torch.float32), sigma_t_scalar.to(torch.float32), atol=1e-12, rtol=0.0).item():
            # Resample a new pair inside the same segment j if possible; else resample segment.
            # Prefer changing e within the same segment.
            start, end = segments[j]
            if (end - start) > 1:
                e = int(torch.randint(low=start, high=end, size=(1,), device=device).item())
                sigma_t_scalar = teacher_sigmas[e]
                sigma_s_scalar = teacher_sigmas[e + 1]
            else:
                j, e = _select_segment_and_edge(segments, rng_device=device)
                sigma_t_scalar = teacher_sigmas[e]
                sigma_s_scalar = teacher_sigmas[e + 1]
                if j < self.S - 1:
                    sigma_bdry_scalar = student_sigmas[j + 1]
                else:
                    sigma_bdry_scalar = torch.as_tensor(0.0, device=device, dtype=teacher_sigmas.dtype)
            sigma_t = sigma_t_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
            sigma_s = sigma_s_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
            sigma_bdry = sigma_bdry_scalar.reshape(1, 1, 1, 1).repeat(batch_size, 1, 1, 1)
            # Recompute x_t and teacher hop for new pair.
            eps = torch.randn_like(y)
            x_t = y + sigma_t * eps
            x_s_teach = heun_hop_edm(
                net=self.teacher_net,
                x_t=x_t,
                sigma_t=sigma_t_scalar,
                sigma_s=sigma_s_scalar,
                class_labels=labels,
                augment_labels=augment_labels,
            )
            equal_b_s_scalar = torch.isclose(
                sigma_bdry_scalar.to(torch.float32), sigma_s_scalar.to(torch.float32), atol=1e-12, rtol=0.0
            ).item()
            if equal_b_s_scalar:
                x_ref_bdry = x_s_teach.to(torch.float32)
            else:
                with torch.no_grad():
                    x_hat_s_ng = net(x_s_teach, sigma_s, labels, augment_labels=augment_labels).to(torch.float32)
                ratio_s_b = (sigma_bdry / torch.clamp(sigma_s, min=tol)).to(torch.float32)
                x_ref_bdry = x_hat_s_ng + ratio_s_b * (x_s_teach.to(torch.float32) - x_hat_s_ng)
            attempts += 1

        # Now compute inv-DDIM target at t.
        x_hat_t_star = inv_ddim_edm(
            x_ref=x_ref_bdry,
            x_t=x_t,
            sigma_t=sigma_t,
            sigma_ref=sigma_bdry,
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


