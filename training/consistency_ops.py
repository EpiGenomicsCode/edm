import math
from typing import Callable, Dict

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


def partition_edges_into_segments(T: int, S: int) -> torch.Tensor:
    """
    Student-anchored segment boundaries for consistency distillation (index-based).
    
    Returns boundaries[j] = round(j * T / S) with ties going up,
    for j in {0, ..., S}, as a LongTensor of shape (S+1,).
    
    Guarantees:
    - boundaries[0] = 0
    - boundaries[S] = T
    - boundaries strictly increasing
    - all segments non-empty
    
    Each segment j spans teacher edges [boundaries[j], boundaries[j+1}).
    """
    assert T >= 1 and S >= 1, "T and S must be >= 1"
    assert S <= T, f"S ({S}) must be <= T ({T})"
    
    # Nearest-int with ties up: floor(x + 0.5)
    j = torch.arange(0, S + 1, dtype=torch.float64)
    kb = torch.floor(j * float(T) / float(S) + 0.5).to(torch.long)
    
    # Guards
    kb[0] = 0
    kb[-1] = T
    
    # Validate strictly increasing
    if not torch.all(kb[1:] > kb[:-1]):
        raise AssertionError("boundaries must be strictly increasing")
    
    # Validate non-empty segments
    seg_len = kb[1:] - kb[:-1]
    if not torch.all(seg_len >= 1):
        raise AssertionError("all segments must be non-empty")
    
    if int(seg_len.sum().item()) != T:
        raise AssertionError("sum of segment lengths must equal T")
    
    return kb


def filter_teacher_edges_by_sigma(
    student_sigmas: torch.Tensor,
    teacher_sigmas: torch.Tensor,
    eps: float = 1e-5,
) -> (torch.Tensor, int):
    """
    Build a CD-specific teacher grid by removing teacher edges whose upper sigma
    matches an interior student sigma (within tolerance), except for the terminal edge.
    Always keep the first edge (sigma_max) and the terminal edge (last positive -> 0).

    Returns:
        teacher_sigmas_cd: 1D tensor of length T_cd+1, descending, last entry 0
        terminal_k_cd: index of the terminal teacher edge (last positive -> 0) in this CD grid
    """
    assert student_sigmas.ndim == 1 and teacher_sigmas.ndim == 1
    T = len(teacher_sigmas) - 1
    assert T >= 1

    # Find terminal edge in the original full grid: last k with sigma_t > 0 and sigma_{k+1} == 0
    terminal_k_full = None
    for k in range(T - 1, -1, -1):
        if teacher_sigmas[k] > 0 and teacher_sigmas[k + 1] == 0:
            terminal_k_full = k
            break
    if terminal_k_full is None:
        terminal_k_full = T - 1  # fallback

    student_interior = student_sigmas[1:-1]  # exclude sigma_max and 0

    kept = []
    for k in range(T):
        sigma_k = teacher_sigmas[k]
        # Always keep very first edge and the original terminal edge
        if k == 0 or k == terminal_k_full:
            kept.append(k)
            continue
        # Drop if matches any interior student sigma (relative tolerance)
        match = False
        for s in student_interior:
            if torch.isclose(
                sigma_k,
                s,
                rtol=eps,
                atol=eps * max(1.0, float(abs(sigma_k)), float(abs(s))),
            ):
                match = True
                break
        if match:
            continue
        kept.append(k)

    kept_idx = torch.tensor(kept, dtype=torch.long, device=teacher_sigmas.device)

    # Build compact CD teacher grid: kept sigmas + final zero
    teacher_sigmas_cd = torch.cat(
        [teacher_sigmas[kept_idx], teacher_sigmas[-1:].clone()], dim=0
    )

    # In this compact grid, terminal edge is the last positive -> 0 transition
    # That's the second-to-last index (index T_cd - 1 where T_cd = len(kept))
    terminal_k_cd = len(kept) - 1  # last kept index position, pointing to last positive before zero

    return teacher_sigmas_cd, int(terminal_k_cd)


def partition_edges_by_sigma(student_sigmas: torch.Tensor, teacher_sigmas: torch.Tensor) -> torch.Tensor:
    """
    Sigma-anchored segmentation: segment j collects all teacher edges whose upper sigma
    lies in (sigma_s[j+1], sigma_s[j]].
    
    Args:
        student_sigmas: Float tensor shape (S+1,) descending, terminal 0
        teacher_sigmas: Float tensor shape (T+1,) descending, terminal 0
    Returns:
        bounds: LongTensor of shape (S, 2) with [k_start, k_end] inclusive per segment.
                Each segment is contiguous in teacher index space.
                If a segment would be empty (no teacher edges in that sigma range),
                we fall back to the nearest teacher index to the lower boundary.
    """
    assert student_sigmas.ndim == 1 and teacher_sigmas.ndim == 1
    S = len(student_sigmas) - 1
    T = len(teacher_sigmas) - 1
    bounds = []
    # ensure descending
    if not torch.all(student_sigmas[:-1] >= student_sigmas[1:]):
        raise AssertionError("student_sigmas must be descending")
    if not torch.all(teacher_sigmas[:-1] >= teacher_sigmas[1:]):
        raise AssertionError("teacher_sigmas must be descending")
    for j in range(S):
        upper = student_sigmas[j]
        lower = student_sigmas[j + 1]
        # teacher edges are indexed by their upper sigma teacher_sigmas[k]
        mask = (teacher_sigmas[:-1] <= upper) & (teacher_sigmas[:-1] > lower)
        idx = mask.nonzero(as_tuple=False).view(-1)
        if len(idx) == 0:
            # Fallback: pick the nearest teacher index to the lower boundary
            # (in value space) to avoid empty segment.
            diffs = (teacher_sigmas[:-1] - lower).abs()
            k_near = int(torch.argmin(diffs).item())
            bounds.append((k_near, k_near))
        else:
            k_start = int(idx.min().item())
            k_end = int(idx.max().item())
            bounds.append((k_start, k_end))
    return torch.tensor(bounds, dtype=torch.long, device=student_sigmas.device)


def sample_segment_and_teacher_pair(
    boundaries: torch.Tensor,
    teacher_sigmas: torch.Tensor,
    student_sigmas: torch.Tensor,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator = None,
    anchor_by_sigma: bool = True,
    sigma_bounds: torch.Tensor = None,
    terminal_k: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample (j, k_t, k_s) for consistency distillation using MSCD-style SNT logic.
    
    **Per-sample edge sampling**: Each element in the batch independently draws its own
    edge (j, k_t, k_s) and corresponding sigmas. All returned tensors have shape [batch_size]
    with independent draws per element (no replication).

    Args:
        boundaries: LongTensor of shape (S+1,) from partition_edges_into_segments
        teacher_sigmas: FloatTensor of shape (T+1,) with terminal 0
        student_sigmas: FloatTensor of shape (S+1,) with terminal 0
        batch_size: number of samples (each gets an independent edge)
        device: torch device
        generator: optional RNG

    Returns:
        Dict with:
            step_j: segment indices [batch_size]
            n_rel: relative edge index within segment [batch_size]
            k_t, k_s: teacher edge indices [batch_size]
            sigma_t, sigma_s, sigma_bdry: noise levels [batch_size]
            is_terminal: bool mask [batch_size]
            is_boundary_snap: bool mask [batch_size]

    MSCD-style semantics:
        - terminal: k_s == T (σ_s = 0)
        - boundary_snap: first edge in segment (n_rel == 1), not last segment, not terminal
        - general interior: all other edges

    If anchor_by_sigma is True, segments are defined in sigma-space using sigma_bounds
    (precomputed via partition_edges_by_sigma). Segment lengths may differ. n_rel=1
    corresponds to the teacher edge closest to the student boundary (lowest sigma in
    the segment), except for the terminal segment where terminal edges are flagged.
    
    Note: Changing batch_size from 1 to N alters RNG consumption (N independent draws per call
    vs 1 draw per call), which affects reproducibility vs legacy single-edge-per-batch behavior.
    """
    S = len(boundaries) - 1
    T = len(teacher_sigmas) - 1
    
    boundaries = boundaries.to(device)
    
    # Sample segment j uniformly from {0, ..., S-1}
    step_j = torch.randint(low=0, high=S, size=(batch_size,), device=device, dtype=torch.long, generator=generator)

    if anchor_by_sigma:
        assert sigma_bounds is not None and sigma_bounds.shape == (S, 2)
        sigma_bounds = sigma_bounds.to(device)
        k0 = sigma_bounds[step_j, 0]
        k1 = sigma_bounds[step_j, 1]
        seg_len_j = (k1 - k0 + 1).clamp(min=1)

        # Sample n_rel uniformly in {1, ..., seg_len_j}
        u = torch.empty(batch_size, device=device, dtype=torch.float32)
        if generator is not None:
            u.uniform_(0.0, 1.0, generator=generator)
        else:
            u.uniform_(0.0, 1.0)
        n_rel = torch.floor(u * seg_len_j.float() + 1.0).to(torch.long)
        n_rel = torch.minimum(n_rel, seg_len_j)

        # n_rel = 1 → closest to boundary (lowest sigma in segment) → k_t = k1
        k_t = k1 - (n_rel - 1)
        k_s = (k_t + 1).clamp(max=T)  # guard, though k_t<T by construction
    else:
        # Index-anchored (original MSCD) segmentation.
        seg_len = boundaries[1:] - boundaries[:-1]  # shape (S,)
        seg_len_j = seg_len[step_j]  # shape (batch_size,)

        u = torch.empty(batch_size, device=device, dtype=torch.float32)
        if generator is not None:
            u.uniform_(0.0, 1.0, generator=generator)
        else:
            u.uniform_(0.0, 1.0)

        n_rel = torch.floor(u * seg_len_j.float() + 1.0).to(torch.long)
        n_rel = torch.clamp(n_rel, min=1)
        n_rel = torch.minimum(n_rel, seg_len_j)

        k_t = boundaries[step_j] + (n_rel - 1)
        k_s = k_t + 1

    # Gather sigmas
    sigma_t = teacher_sigmas[k_t]
    sigma_s = teacher_sigmas[k_s]
    sigma_bdry = student_sigmas[step_j + 1]

    # Classification flags
    if terminal_k is None:
        terminal_k = T - 1
    is_terminal = (step_j == (S - 1)) & (k_t == terminal_k)
    is_boundary_snap = (~is_terminal) & (n_rel == 1) & (step_j < (S - 1))

    return {
        "step_j": step_j,
        "k_t": k_t,
        "k_s": k_s,
        "sigma_t": sigma_t,
        "sigma_s": sigma_s,
        "sigma_bdry": sigma_bdry,
        "n_rel": n_rel,
        "is_terminal": is_terminal,
        "is_boundary_snap": is_boundary_snap,
    }


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
        bad_idx = (sigma_t_b == 0).nonzero(as_tuple=False)[:, 0].unique()
        sigma_t_flat = sigma_t if sigma_t.ndim <= 1 else sigma_t_b[:, 0, 0, 0]
        bad_vals = [(int(i.item()), float(sigma_t_flat[i].item())) for i in bad_idx[:5]]
        raise ValueError(
            f"inv_ddim_edm received sigma_t == 0. Avoid σ=0 when backsolving.\n"
            f"  Affected samples (first 5): {bad_vals}"
        )
    ratio = sigma_ref_b / sigma_t_b
    denom = 1.0 - ratio
    if torch.any(denom.abs() < 1e-8):
        bad_idx = (denom.abs() < 1e-8).nonzero(as_tuple=False)[:, 0].unique()
        sigma_t_flat = sigma_t if sigma_t.ndim <= 1 else sigma_t_b[:, 0, 0, 0]
        sigma_ref_flat = sigma_ref if sigma_ref.ndim <= 1 else sigma_ref_b[:, 0, 0, 0]
        bad_vals = [
            (int(i.item()), float(sigma_t_flat[i].item()), float(sigma_ref_flat[i].item()))
            for i in bad_idx[:5]
        ]
        raise ValueError(
            f"inv_ddim_edm denominator is zero (σ_ref ≈ σ_t). Drop or resample this pair.\n"
            f"  Affected samples (first 5, format: [idx, sigma_t, sigma_ref]):\n"
            f"    {bad_vals}\n"
            f"  This indicates sigma_ref and sigma_t are nearly equal, making the invDDIM formula degenerate."
        )
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


# Backward compatibility alias for old checkpoints
@torch.no_grad()
def heun_hop_edm_stochastic(
    net,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    class_labels: torch.Tensor = None,
    augment_labels: torch.Tensor = None,
    **kwargs  # Ignore any churn parameters from old code
) -> torch.Tensor:
    """
    Backward compatibility wrapper for old checkpoints.
    Now just calls deterministic heun_hop_edm (churn removed).
    """
    return heun_hop_edm(net, x_t, sigma_t, sigma_s, class_labels, augment_labels)


