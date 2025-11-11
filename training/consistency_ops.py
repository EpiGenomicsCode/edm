import math
from typing import Callable, List, Tuple

import torch


@torch.no_grad()
def make_karras_sigmas(
    num_nodes: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    round_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Construct a monotonically descending Karras noise grid (length = num_nodes) in EDM space.
    Does not append 0; consumers treat 0 as a conceptual boundary.
    Each sigma is passed through `round_fn` (e.g., `net.round_sigma`) to align with network rounding.

    Returns a 1D tensor [σ_0 > σ_1 > ... > σ_{num_nodes-1}].
    """
    assert num_nodes >= 1, "num_nodes must be >= 1"
    # Build on the same device/dtype end-to-end so that round_fn preserves placement.
    # Use float64 for schedule arithmetic (matches generate.py), then rely on round_fn
    # and downstream casts. This avoids excessive drift while staying consistent.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    step_indices = torch.arange(num_nodes, dtype=torch.float64, device=device)
    sigma_min_r = float(sigma_min)
    sigma_max_r = float(sigma_max)
    rho_r = float(rho)
    sigmas = (sigma_max_r ** (1.0 / rho_r) + step_indices / max(num_nodes - 1, 1) * (sigma_min_r ** (1.0 / rho_r) - sigma_max_r ** (1.0 / rho_r))) ** rho_r
    # Apply network's rounding to keep consistency with training/inference.
    sigmas = round_fn(sigmas)
    return sigmas


def partition_edges_into_segments(T: int, S: int) -> List[Tuple[int, int]]:
    """
    Partition T teacher edges (indices 0..T-1) into S contiguous [start, end) ranges using:
        k_b(j) = floor(j*T/S + 1/2), for j in {0..S}
    Returns a list of S tuples (start_inclusive, end_exclusive).
    """
    assert T >= 1 and S >= 1, "T and S must be >= 1"
    bounds = []
    for j in range(S + 1):
        kb = int(math.floor(j * T / S + 0.5))
        # Clamp to [0, T] to avoid any floating corner cases.
        kb = max(0, min(T, kb))
        bounds.append(kb)
    segments: List[Tuple[int, int]] = []
    for j in range(S):
        start = bounds[j]
        end = bounds[j + 1]
        start = max(0, min(T, start))
        end = max(0, min(T, end))
        segments.append((start, end))
    return segments


def _expand_sigma_to_bchw(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Ensure sigma broadcasts over BCHW. Accepts scalar/(), (N,), or already-broadcast shapes.
    Returns tensor on like.device, float32.
    """
    s = torch.as_tensor(sigma, device=like.device, dtype=torch.float32)
    if s.ndim == 0:
        s = s.reshape(1, 1, 1, 1)
    elif s.ndim == 1:
        # (N,) -> (N,1,1,1)
        s = s.reshape(-1, 1, 1, 1)
    return s


def ddim_step_edm(
    x_t: torch.Tensor,
    x_pred_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
) -> torch.Tensor:
    """
    EDM-space DDIM/Euler step:
        x_s = x̂_t + (σ_s/σ_t) * (x_t - x̂_t)
    """
    assert x_t.shape == x_pred_t.shape, "x_t and x_pred_t must have the same shape"
    # Work in float32 for stability; return in input dtype.
    out_dtype = x_t.dtype
    x_t32 = x_t.to(torch.float32)
    x_pred_t32 = x_pred_t.to(torch.float32)
    sigma_t_b = _expand_sigma_to_bchw(sigma_t, x_t32)
    sigma_s_b = _expand_sigma_to_bchw(sigma_s, x_t32)
    # Guard against division by zero at σ=0 (should not be called with σ_t == 0).
    if torch.any(sigma_t_b == 0):
        raise ValueError("ddim_step_edm received sigma_t == 0. Avoid σ=0 in DDIM steps.")
    ratio = sigma_s_b / sigma_t_b
    x_s = x_pred_t32 + ratio * (x_t32 - x_pred_t32)
    return x_s.to(out_dtype)


def inv_ddim_edm(
    x_ref: torch.Tensor,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_ref: torch.Tensor,
) -> torch.Tensor:
    """
    EDM-space inverse-DDIM:
        x̂*_t = [x_ref - (σ_ref/σ_t) x_t] / [1 - (σ_ref/σ_t)]
    """
    assert x_ref.shape == x_t.shape, "x_ref and x_t must have the same shape"
    out_dtype = x_t.dtype
    x_ref32 = x_ref.to(torch.float32)
    x_t32 = x_t.to(torch.float32)
    sigma_t_b = _expand_sigma_to_bchw(sigma_t, x_t32)
    sigma_ref_b = _expand_sigma_to_bchw(sigma_ref, x_t32)
    # Guards: avoid division by zero or degenerate denominator (σ_ref == σ_t).
    if torch.any(sigma_t_b == 0):
        raise ValueError("inv_ddim_edm received sigma_t == 0. Avoid σ=0 when backsolving.")
    ratio = sigma_ref_b / sigma_t_b
    denom = 1.0 - ratio
    if torch.any(denom == 0):
        raise ValueError("inv_ddim_edm denominator is zero (σ_ref == σ_t). Drop or resample this pair.")
    x_hat_star_t = (x_ref32 - ratio * x_t32) / denom
    return x_hat_star_t.to(out_dtype)


@torch.no_grad()
def heun_hop_edm(
    net,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    class_labels: torch.Tensor = None,
    augment_labels: torch.Tensor = None,
) -> torch.Tensor:
    """
    Deterministic Heun hop (teacher) from σ_t -> σ_s in EDM PF-ODE:
        k1 = (x_t - D(x_t; σ_t))/σ_t
        x_eul = x_t + (σ_s - σ_t) * k1
        k2 = (x_eul - D(x_eul; σ_s))/σ_s
        x_s = x_t + 0.5*(σ_s - σ_t)*(k1 + k2)
    Never evaluates the net at σ=0. Assumes `net` has EDM interface: net(x, σ, labels, augment_labels).
    """
    assert isinstance(x_t, torch.Tensor)
    out_dtype = x_t.dtype
    x32 = x_t.to(torch.float32)

    # Round sigmas using network's policy and reshape/broadcast.
    # Maintain the same device placement throughout.
    sigma_t_r = net.round_sigma(torch.as_tensor(sigma_t, device=x32.device))
    sigma_s_r = net.round_sigma(torch.as_tensor(sigma_s, device=x32.device))
    sigma_t_b = _expand_sigma_to_bchw(sigma_t_r, x32)
    sigma_s_b = _expand_sigma_to_bchw(sigma_s_r, x32)

    # Never evaluate at σ=0.
    if torch.any(sigma_t_b == 0) or torch.any(sigma_s_b == 0):
        raise ValueError("heun_hop_edm received σ=0. Avoid σ=0 in teacher hops.")

    # k1 at (x_t, σ_t).
    denoised_t = net(x32, sigma_t_r, class_labels=class_labels, augment_labels=augment_labels).to(torch.float32)
    k1 = (x32 - denoised_t) / sigma_t_b

    # Euler proposal to σ_s.
    x_eul = x32 + (sigma_s_b - sigma_t_b) * k1

    # k2 at (x_eul, σ_s).
    denoised_s = net(x_eul, sigma_s_r, class_labels=class_labels, augment_labels=augment_labels).to(torch.float32)
    k2 = (x_eul - denoised_s) / sigma_s_b

    # Heun update.
    x_s = x32 + 0.5 * (sigma_s_b - sigma_t_b) * (k1 + k2)
    return x_s.to(out_dtype)


