#!/usr/bin/env python3
# Unified, fair symmetry discovery (general matrix Lie algebra, no system-specific actions)
# - One discovery method for ALL systems
# - Uses ONLY general matrix exponential and linear state action x -> g x
# - Discovers both subspace and its dimension (adaptive k)
# - Generates publication-ready plots and LaTeX tables
# - No SE(2) or any special-case action formulas anywhere

import os
import math
import time
import random
import json
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Publication-style defaults
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# CPU-only, threads, precision
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
torch.set_float32_matmul_precision("high")
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.set_num_interop_threads(max(1, (os.cpu_count() or 1) // 2))
try:
    import resource
    def get_max_rss():
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except Exception:
    def get_max_rss():
        return None

# ============ Utils ============

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

def get_device() -> str:
    return "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def matrix_exp(A: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.linalg, "matrix_exp"):
        return torch.linalg.matrix_exp(A)
    return torch.matrix_exp(A)

# ============ Integrator (differentiable RK4) ============

def rk4_step(f, x: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout(f, x0: torch.Tensor, t0: float, dt: float, steps: int) -> torch.Tensor:
    batched = (x0.dim() == 2)
    x = x0 if batched else x0.unsqueeze(0)
    B, D = x.shape
    xs = torch.empty(B, steps + 1, D, device=x.device, dtype=x.dtype)
    xs[:, 0, :] = x
    t = torch.tensor(t0, dtype=x.dtype, device=x.device)
    for s in range(steps):
        x = rk4_step(f, x, t, dt)
        t = t + dt
        xs[:, s + 1, :] = x
    return xs if batched else xs.squeeze(0)

# ============ Dynamics model (Neural ODE field) ============

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, depth=3, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(d_in, d_hidden), act()]
        for _ in range(depth - 2):
            layers += [nn.Linear(d_hidden, d_hidden), act()]
        layers += [nn.Linear(d_hidden, d_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class NeuralODEField(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.dim = dim
        self.f = MLP(dim, hidden, dim, depth)
    def forward(self, t, x):
        return self.f(x)

# ============ General matrix Lie subspace helpers ============

def skewify(W: torch.Tensor) -> torch.Tensor:
    return 0.5 * (W - W.transpose(-1, -2))

def orthonormalize_basis(Bs: torch.Tensor) -> torch.Tensor:
    # Frobenius-QR on flattened matrices
    k, D, _ = Bs.shape
    U = Bs.reshape(k, -1).T  # [D*D, k]
    Q, _ = torch.linalg.qr(U, mode='reduced')
    return Q.T.reshape(k, D, D)

def A_from_xi(xi: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (xi.view(-1,1,1) * B).sum(dim=0)

def A_unit_from_xi(xi: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = A_from_xi(xi, B)
    n = torch.linalg.norm(A, ord="fro")
    return A / (n + eps)

def sample_xi(k: int, M: int, device: str) -> torch.Tensor:
    return torch.randn(M, k, device=device)

# Structure constants and Killing form (post-hoc certification)
def structure_constants_and_killing(B: torch.Tensor) -> Dict[str, Any]:
    k = B.shape[0]
    c = torch.zeros(k, k, k, dtype=B.dtype)
    r = torch.zeros(k, k, dtype=B.dtype)
    for i in range(k):
        for j in range(k):
            comm = B[i] @ B[j] - B[j] @ B[i]
            for l in range(k):
                c[l, i, j] = torch.tensordot(B[l], comm, dims=2)
            recon = sum(c[l, i, j] * B[l] for l in range(k))
            r[i, j] = torch.linalg.norm(comm - recon, ord="fro")
    K = torch.zeros(k, k, dtype=B.dtype)
    for i in range(k):
        for j in range(k):
            s = 0.0
            for a in range(k):
                for b in range(k):
                    s += c[a, i, b] * c[b, j, a]
            K[i, j] = s
    evals = torch.linalg.eigvalsh(K)
    return {
        "c_ijk": c.detach().cpu().tolist(),
        "closure_residuals": r.detach().cpu().tolist(),
        "killing_eigs": evals.detach().cpu().tolist(),
        "killing_matrix": K.detach().cpu().tolist(),
    }

# ============ General, system-agnostic integral loss (linear action only) ============

def integral_equiv_loss_general(f_field: nn.Module,
                                x0s: torch.Tensor,
                                g: torch.Tensor,
                                dt: float, steps: int) -> torch.Tensor:
    # Evaluate trajectory-level commutation error with a fixed group element g
    xs = rollout(f_field, x0s, t0=0.0, dt=dt, steps=steps)
    x0s_g = x0s @ g.T
    xs_g0 = rollout(f_field, x0s_g, t0=0.0, dt=dt, steps=steps)
    xs_g  = xs @ g.T
    diff = (xs_g0 - xs_g).pow(2).sum(dim=-1).mean()  # [B,T+1] -> scalar
    return dt * diff

def expected_integral_loss_general(f_field: nn.Module,
                                   x0s: torch.Tensor,
                                   W: torch.Tensor,          # params: [k,D,D]
                                   dt: float, steps: int, eps: float,
                                   mult_eps=(0.5,1.0,2.0), M: int = 8,
                                   mode: str = "full") -> Tuple[torch.Tensor, torch.Tensor]:
    # mode: "full" -> general gl(d); "skew" -> so(d) (optional)
    if mode == "skew":
        Braw = skewify(W)
    else:
        Braw = W
    B = orthonormalize_basis(Braw)
    k = B.shape[0]; device = x0s.device
    xis = sample_xi(k, M, device)
    L = 0.0
    for m in range(M):
        xi = xis[m]
        A = A_unit_from_xi(xi, B)
        for s in mult_eps:
            g = matrix_exp((s * eps) * A)
            L += integral_equiv_loss_general(f_field, x0s, g, dt, steps)
    return L / (M * len(mult_eps)), B

# ============ Baselines (pointwise and bracket), general linear action ============

@torch.no_grad()
def pointwise_equiv_loss_linear_states(f_field: nn.Module,
                                       states: torch.Tensor,   # [N,D]
                                       xi: torch.Tensor, B: torch.Tensor,
                                       eps: float, mult_eps=(0.5,1.0,2.0)) -> float:
    A = A_unit_from_xi(xi, B)
    acc = 0.0
    for m in mult_eps:
        g = matrix_exp((m * eps) * A)
        gx = states @ g.T
        fx = f_field(0.0, states)
        fgx = f_field(0.0, gx)
        lhs = fx @ g.T
        acc += (lhs - fgx).pow(2).sum(dim=-1).mean()
    return float((acc / len(mult_eps)).cpu())

def _jvp_f(f_field: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    fx = f_field(0.0, x)
    g = torch.autograd.grad(fx, x, grad_outputs=v, retain_graph=False, create_graph=False, allow_unused=False)[0]
    return g

@torch.no_grad()
def bracket_loss_linear_states(f_field: nn.Module,
                               states: torch.Tensor,     # [N,D]
                               B: torch.Tensor, xi: torch.Tensor) -> float:
    A = A_unit_from_xi(xi, B)
    errs = []
    batch = min(32, states.shape[0])
    for i in range(0, states.shape[0], batch):
        xb = states[i:i+batch].detach()
        vA = (xb @ A.T)
        jfv = []
        for j in range(xb.shape[0]):
            jfv.append(_jvp_f(f_field, xb[j:j+1], vA[j:j+1]))
        jfv = torch.cat(jfv, dim=0)
        fx = f_field(0.0, xb)
        comm = jfv - (fx @ A.T)
        errs.append(comm.pow(2).sum(dim=-1))
    return float(torch.cat(errs).mean().cpu())

# ============ General symmetry discovery (no assumptions) ============

@dataclass
class DiscoveryConfig:
    max_k: int = 10
    iters_per_k: int = 1500
    lr: float = 5e-3
    eps: float = 1e-2
    M_expectation: int = 8
    mult_eps: Tuple[float, ...] = (0.5, 1.0, 2.0)
    param_mode: str = "full"  # "full" (gl(d)) or "skew" (so(d))

def discover_symmetry_general(f_field: nn.Module,
                              trajs: torch.Tensor,
                              dt: float, steps: int,
                              cfg: DiscoveryConfig,
                              device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Dimension-agnostic discovery over general matrix Lie algebras.
    - No system-specific parametrizations
    - Only uses matrix exponential and linear action x -> g x
    """
    N = trajs.shape[0]
    Bsize = min(32, N)
    d = trajs.shape[-1]
    best_loss = float('inf')
    best_B = None
    best_k = None
    history = []

    for k in range(1, min(cfg.max_k, d*d) + 1):
        W = (0.01 * torch.randn(k, d, d, device=device)).requires_grad_(True)
        opt = optim.Adam([W], lr=cfg.lr)
        k_losses = []
        for it in range(1, cfg.iters_per_k + 1):
            idx = torch.randint(0, N, (Bsize,), device=device)
            x0s = trajs[idx, 0, :]
            L, B = expected_integral_loss_general(
                f_field, x0s, W, dt, steps, cfg.eps,
                mult_eps=cfg.mult_eps, M=cfg.M_expectation, mode=cfg.param_mode
            )
            opt.zero_grad(); L.backward()
            nn.utils.clip_grad_norm_([W], 5.0)
            opt.step()
            k_losses.append(float(L.detach().cpu()))
            if it % 200 == 0:
                print(f"  [General] k={k:2d} iter {it:4d} | E[Lint] {float(L.detach().cpu()):.3e}")
        final_loss = float(L.detach().cpu())
        history.append({"k": k, "final_loss": final_loss, "loss_curve": k_losses})
        if final_loss < best_loss:
            best_loss = final_loss
            best_B = B.detach().clone()
            best_k = k
        # Adaptive stop: if not improving substantially vs previous k
        if len(history) >= 2 and final_loss > 0.9 * history[-2]["final_loss"]:
            print(f"  [General] Early stop at k={k}, improvement plateaued.")
            break

    return best_B, {"best_k": best_k, "best_loss": best_loss, "history": history}

# ============ Plots (publication-ready, general) ============

def plot_equivariance_overlay_linear(f_field, A_dir, trajs, dt, steps, eps, outdir: Path, title: str, dims=(0,1)):
    outdir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        B = min(5, trajs.shape[0])
        x0 = trajs[:B,0,:]
        xs = rollout(f_field, x0, t0=0.0, dt=dt, steps=steps)
        g = matrix_exp(eps * A_dir)
        x0_g = x0 @ g.T
        xs_g0 = rollout(f_field, x0_g, t0=0.0, dt=dt, steps=steps)
        xs_g = xs @ g.T
    plt.figure(figsize=(6,5))
    for i in range(B):
        plt.plot(xs[i,:,dims[0]].cpu(), xs[i,:,dims[1]].cpu(), 'b-', alpha=0.8, label='flow' if i==0 else "")
        plt.plot(xs_g0[i,:,dims[0]].cpu(), xs_g0[i,:,dims[1]].cpu(), 'r--', alpha=0.8, label='transform-then-flow' if i==0 else "")
        plt.plot(xs_g[i,:,dims[0]].cpu(), xs_g[i,:,dims[1]].cpu(), 'g-.', alpha=0.8, label='flow-then-transform' if i==0 else "")
    plt.legend()
    plt.title(f"{title} overlay"); plt.xlabel(f"dim {dims[0]}"); plt.ylabel(f"dim {dims[1]}")
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.tight_layout(); plt.savefig(outdir / f"{title.replace(' ','_')}_overlay.png", dpi=300)

    err_time = (xs_g0 - xs_g).pow(2).sum(dim=-1).mean(dim=0).cpu().numpy()
    t = (torch.arange(steps+1, dtype=torch.float32) * dt).cpu().numpy()
    plt.figure(figsize=(6,4)); plt.plot(t, err_time, 'k-')
    plt.yscale('log'); plt.grid(True, which='both', axis='both', alpha=0.3)
    plt.xlabel("time"); plt.ylabel("equivariance error")
    plt.title(f"{title} timewise equivariance error")
    plt.tight_layout(); plt.savefig(outdir / f"{title.replace(' ','_')}_time_err.png", dpi=300); plt.close('all')

def plot_noise_robustness(f_field, B, trajs, dt, steps, eps, outpath: Path, mult_eps=(0.5,1.0,2.0)):
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]
    vals = []
    x0s_base = trajs[:min(64, trajs.shape[0]) ,0,:]
    for s in sigmas:
        noisy = trajs + (s * torch.randn_like(trajs) if s > 0 else 0.0)
        x0s = noisy[:min(64, noisy.shape[0]) ,0,:]
        # Evaluate with random xi-average
        mean, std = expected_loss_general_eval(f_field, B, x0s, dt, steps, eps, mult_eps, M=16)
        vals.append(mean)
    plt.figure(figsize=(6,4)); plt.plot(sigmas, vals, 'ko-', markersize=4)
    plt.yscale('log'); plt.xlabel("noise sigma"); plt.ylabel("E[Lint]"); plt.title("Noise robustness")
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()
    return sigmas, vals

def expected_loss_general_eval(f_field, B: torch.Tensor, x0s: torch.Tensor,
                               dt: float, steps: int, eps: float,
                               mult_eps=(0.5,1.0,2.0), M: int = 24) -> Tuple[float,float]:
    k = B.shape[0]
    Ls = []
    with torch.no_grad():
        for _ in range(M):
            xi = torch.randn(k, device=x0s.device)
            A = A_unit_from_xi(xi, B)
            acc = 0.0
            for m in mult_eps:
                g = matrix_exp((m * eps) * A)
                acc += float(integral_equiv_loss_general(f_field, x0s, g, dt, steps).cpu())
            Ls.append(acc / len(mult_eps))
    return float(np.mean(Ls)), float(np.std(Ls))

def plot_data_efficiency(f_field, B, trajs, dt, steps, eps, outdir: Path, mult_eps=(0.5,1.0,2.0)):
    B_list = [16, 32, 64, min(128, trajs.shape[0])]
    T_list = [max(20, steps//3), max(40, steps//2), steps]
    vals_B, vals_T = [], []
    with torch.no_grad():
        for b in B_list:
            x0s = trajs[:b, 0, :]
            m, _ = expected_loss_general_eval(f_field, B, x0s, dt, steps, eps, mult_eps, M=12)
            vals_B.append(m)
        for tsub in T_list:
            x0s = trajs[:min(64, trajs.shape[0]), 0, :]
            m, _ = expected_loss_general_eval(f_field, B, x0s, dt, tsub, eps, mult_eps, M=12)
            vals_T.append(m)
    plt.figure(figsize=(6,4)); plt.plot(B_list, vals_B, 'bo-', markersize=4)
    plt.yscale('log'); plt.xlabel("trajectories (B)"); plt.ylabel("E[Lint]"); plt.title("Data-efficiency vs B")
    plt.tight_layout(); plt.savefig(outdir/"data_efficiency_B.png", dpi=300); plt.close()
    plt.figure(figsize=(6,4)); plt.plot(T_list, vals_T, 'ro-', markersize=4)
    plt.yscale('log'); plt.xlabel("horizon (T)"); plt.ylabel("E[Lint]"); plt.title("Data-efficiency vs T")
    plt.tight_layout(); plt.savefig(outdir/"data_efficiency_T.png", dpi=300); plt.close()
    return {"B": B_list, "E_Lint_B": vals_B, "T": T_list, "E_Lint_T": vals_T}

def plot_composition_heatmap(f_field, B: torch.Tensor, trajs, dt, steps, eps, outpath: Path, grid=(5,5)):
    k = B.shape[0]; device = trajs.device
    x0s = trajs[:min(64, trajs.shape[0]), 0, :]
    eps1_list = np.linspace(0.5*eps, 2.0*eps, grid[0])
    eps2_list = np.linspace(0.5*eps, 2.0*eps, grid[1])
    heat = np.zeros((grid[0], grid[1]), dtype=np.float64)
    with torch.no_grad():
        for i, e1 in enumerate(eps1_list):
            for j, e2 in enumerate(eps2_list):
                xi1 = torch.randn(k, device=device); xi2 = torch.randn(k, device=device)
                A1 = A_unit_from_xi(xi1, B); A2 = A_unit_from_xi(xi2, B)
                g = matrix_exp(e2 * A2) @ matrix_exp(e1 * A1)
                xs = rollout(f_field, x0s, t0=0.0, dt=dt, steps=steps)
                xs_g  = xs @ g.T
                x0s_g = x0s @ g.T
                xs_g0 = rollout(f_field, x0s_g, t0=0.0, dt=dt, steps=steps)
                diff = (xs_g0 - xs_g).pow(2).sum(dim=-1).mean()
                heat[i, j] = float((dt * diff).cpu())
    plt.figure(figsize=(5.2,4.2))
    im = plt.imshow(heat, origin='lower', extent=[eps2_list[0], eps2_list[-1], eps1_list[0], eps1_list[-1]], aspect='auto', cmap='viridis')
    plt.colorbar(im, label="E[Lint] (composed)")
    ax = plt.gca()
    ax.set_xticks(np.round(np.linspace(eps2_list[0], eps2_list[-1], 5), 3))
    ax.set_yticks(np.round(np.linspace(eps1_list[0], eps1_list[-1], 5), 3))
    plt.xlabel("eps2"); plt.ylabel("eps1")
    plt.title("Composition equivariance (general)")
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()
    return {"eps1": eps1_list.tolist(), "eps2": eps2_list.tolist(), "E_Lint_composed": heat.tolist()}

def plot_eps_sweep(f_field, B: torch.Tensor, trajs, dt, steps, eps_list: List[float], outdir: Path, M:int=16):
    x0s = trajs[:min(64, trajs.shape[0]), 0, :]
    raw, norm = [], []
    with torch.no_grad():
        for e in eps_list:
            m, _ = expected_loss_general_eval(f_field, B, x0s, dt, steps, e, (1.0,), M=M)
            raw.append(m); norm.append(m / (e**2))
    plt.figure(figsize=(6,4))
    plt.plot(eps_list, raw, 'ko-', label='E[Lint]')
    plt.plot(eps_list, norm, 'rs--', label='E[Lint]/eps$^2$')
    plt.yscale('log'); plt.xlabel('eps'); plt.ylabel('loss')
    plt.legend()
    plt.tight_layout(); plt.savefig(outdir/"eps_sweep.png", dpi=300); plt.close()
    return {"eps": eps_list, "E_Lint": raw, "E_Lint_over_eps2": norm}

def plot_T_dt_heatmap(f_field, B: torch.Tensor, trajs, dt_list: List[float], T_list: List[int],
                      eps: float, outpath: Path):
    x0s = trajs[:min(64, trajs.shape[0]), 0, :]
    heat = np.zeros((len(dt_list), len(T_list)), dtype=np.float64)
    with torch.no_grad():
        for i, d in enumerate(dt_list):
            for j, Tsub in enumerate(T_list):
                m, _ = expected_loss_general_eval(f_field, B, x0s, d, Tsub, eps, (0.5,1.0,2.0), M=12)
                heat[i, j] = m
    plt.figure(figsize=(5.2,4.2))
    im = plt.imshow(heat, origin='lower', aspect='auto', cmap='viridis',
               extent=[T_list[0], T_list[-1], dt_list[0], dt_list[-1]])
    plt.colorbar(im, label="E[Lint]")
    ax = plt.gca()
    ax.set_xticks(T_list); ax.set_yticks(dt_list)
    plt.xlabel("T"); plt.ylabel("dt")
    plt.title("Sensitivity heatmap (general)")
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()
    return {"dt": dt_list, "T": T_list, "E_Lint": heat.tolist()}

def plot_discovered_generators_grid(B: torch.Tensor, out_path: Path, title: str = "discovered generators"):
    # Show discovered matrices
    k, D, _ = B.shape
    cols = min(6, k)
    rows = int(math.ceil(k / cols))
    vmax = float(torch.max(torch.abs(B)).cpu().item())
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else vmax
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.4, rows*2.4))
    axes = np.atleast_2d(axes)
    for i in range(rows*cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]; ax.axis("off")
        if i < k:
            im = ax.imshow(B[i].detach().cpu().numpy(), cmap="bwr", vmin=-vmax, vmax=vmax, aspect='equal')
            ax.set_title(f"#{i}", fontsize=9); ax.axis("on")
    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-vmax, vmax=vmax), cmap='bwr')
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, label="value")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close(fig)

def plot_closure_killing(struct: Dict[str, Any], outdir: Path, title_prefix: str):
    # closure residuals heatmap
    r = np.array(struct["closure_residuals"])
    plt.figure(figsize=(4.4,4))
    im = plt.imshow(r, cmap='magma', aspect='auto')
    plt.colorbar(im, label="closure residual (Fro)")
    plt.xlabel("j"); plt.ylabel("i")
    plt.title(f"{title_prefix} closure residuals")
    plt.tight_layout(); plt.savefig(outdir/"closure_residuals_heatmap.png", dpi=300); plt.close()
    # Killing eigenvalues
    evals = np.array(struct["killing_eigs"])
    plt.figure(figsize=(5,3))
    plt.bar(list(range(1,len(evals)+1)), evals, color="#55A868")
    plt.axhline(0., color='k', linewidth=0.8)
    plt.ylabel("eigenvalue")
    plt.title(f"{title_prefix} Killing eigenvalues")
    plt.tight_layout(); plt.savefig(outdir/"killing_eigs_bar.png", dpi=300); plt.close()

def plot_pointwise_vs_integral_scatter(f_field, B: torch.Tensor, trajs, dt, steps, eps, outdir: Path):
    k = B.shape[0]
    states = trajs.reshape(-1, trajs.shape[-1])
    states = states[torch.randperm(states.shape[0])[:1024]]
    x0s = trajs[:min(64, trajs.shape[0]), 0, :]
    xs_pw, ys_int = [], []
    with torch.no_grad():
        for _ in range(64):
            xi = torch.randn(k, device=x0s.device)
            pw = pointwise_equiv_loss_linear_states(f_field, states[:512], xi, B, eps, mult_eps=(0.5,1.0,2.0))
            A = A_unit_from_xi(xi, B)
            acc = 0.0
            for m in (0.5,1.0,2.0):
                g = matrix_exp((m * eps) * A)
                acc += float(integral_equiv_loss_general(f_field, x0s, g, dt, steps).cpu())
            L = acc / 3.0
            xs_pw.append(pw); ys_int.append(L)
    plt.figure(figsize=(5.5,4))
    plt.scatter(xs_pw, ys_int, s=16, alpha=0.7, edgecolors='none')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("pointwise loss"); plt.ylabel("integral loss")
    plt.title("Pointwise small, trajectory large?")
    plt.tight_layout(); plt.savefig(outdir / "pointwise_satisfy_but_fail.png", dpi=300); plt.close()

# ============ Systems (true dynamics generators ONLY for data) ============

@dataclass
class System:
    name: str
    dim: int
    f: Callable[[float, torch.Tensor], torch.Tensor]

def make_harmonic2d(omega=1.5, device="cpu") -> System:
    A = torch.tensor([[0., -omega],[omega, 0.]], device=device)
    def f(t, x): return x @ A.T
    return System("Harmonic2D", 2, f)

def make_nonlinear_rot2d(omega0=0.8, beta=0.4, device="cpu") -> System:
    J = torch.tensor([[0., -1.],[1., 0.]], device=device)
    def omega(x): return omega0 + beta * (x.pow(2).sum(dim=-1, keepdim=True))
    def f(t, x): return omega(x) * (x @ J.T)
    return System("NonlinearRot2D", 2, f)

def make_so3_equivariant3d(scale=0.7, device="cpu") -> System:
    def f(t, x): return scale * x
    return System("SO3-Equivariant3D", 3, f)

def make_so4_equivariant4d(scale=0.5, device="cpu") -> System:
    def f(t, x): return scale * x
    return System("SO4-Equivariant4D", 4, f)

def make_kepler2d(mu=1.0, device="cpu") -> System:
    def f(t, x):
        pos = x[..., :2]; vel = x[..., 2:]
        r = torch.norm(pos, dim=-1, keepdim=True).clamp_min(1e-6)
        acc = -mu * pos / (r**3)
        return torch.cat([vel, acc], dim=-1)
    return System("Kepler2D", 4, f)

def make_se2_unicycle(v=1.0, omega=0.2, device="cpu") -> System:
    # This system has inherently non-linear group action in its true symmetry (SE(2)).
    # We include it to DEMONSTRATE FAILURE under our linear action assumption.
    def f(t, x):
        th = x[..., 2:3]
        vx = v * torch.cos(th); vy = v * torch.sin(th)
        vth = torch.full_like(th, omega)
        return torch.cat([vx, vy, vth], dim=-1)
    return System("SE2-Unicycle (FAIL expected under linear action)", 3, f)

# ============ Training ============

def sample_trajs(system: System, B: int, T: int, dt: float, x_scale: float, device: str) -> torch.Tensor:
    D = system.dim
    if "Kepler2D" in system.name:
        pos = torch.randn(B, 2, device=device) * x_scale
        vel = torch.randn(B, 2, device=device) * (x_scale * 0.2)
        x0 = torch.cat([pos, vel], dim=-1)
    elif "SE2" in system.name:
        xy = torch.randn(B, 2, device=device) * x_scale
        th = (torch.rand(B, 1, device=device) - 0.5) * 2 * math.pi
        x0 = torch.cat([xy, th], dim=-1)
    else:
        x0 = torch.randn(B, D, device=device) * x_scale
    xs = rollout(system.f, x0, t0=0.0, dt=dt, steps=T)
    return xs

def train_stage1(trajs: torch.Tensor,
                 dim: int,
                 steps: int,
                 dt: float,
                 iters: int = 800,
                 lr: float = 1e-3,
                 device: str = "cpu") -> nn.Module:
    hidden = 256 if dim == 4 else 128
    depth = 4 if dim == 4 else 3
    model = NeuralODEField(dim=dim, hidden=hidden, depth=depth).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    N, T1, _ = trajs.shape
    Bsize = min(64, N)
    for it in range(1, iters + 1):
        idx = torch.randint(0, N, (Bsize,), device=device)
        x0 = trajs[idx, 0, :]
        target = trajs[idx, :, :]
        pred = rollout(model, x0, t0=0.0, dt=dt, steps=T1 - 1)
        loss = (pred - target).pow(2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 200 == 0:
            print(f"  [Stage1] iter {it:4d} | MSE {loss.item():.6e}")
    return model

# ============ Runner ============

@dataclass
class Variant:
    name: str
    seed: int = 42

def run_experiment(system: System,
                   device: str,
                   out_root: Path,
                   variant: Variant,
                   B: int = 128, T: int = 120, dt: float = 0.02,
                   s1_iters: int = 800,
                   discover_cfg: DiscoveryConfig = DiscoveryConfig()) -> None:
    set_seed(variant.seed)
    print(f"\n=== {system.name} | dim={system.dim} | variant={variant.name} | seed={variant.seed} ===")
    outdir = out_root / f"{system.name.replace(' ','_')}__{variant.name}"
    ensure_dir(outdir)

    # Tweak horizon for Kepler
    if "Kepler2D" in system.name:
        dt, T = 0.01, 200

    trajs = sample_trajs(system, B=B, T=T, dt=dt, x_scale=1.0, device=device)
    steps = T

    print(" Training Stage 1 (learn dynamics)...")
    t0 = time.time()
    f_model = train_stage1(trajs, dim=system.dim, steps=steps, dt=dt, iters=s1_iters, device=device)
    stage1_time = time.time() - t0
    rss1 = get_max_rss()
    print(f"  Stage 1 time: {stage1_time:.1f}s")

    compiled = False
    try:
        f_model = torch.compile(f_model)
        compiled = True
    except Exception:
        pass
    _ = rollout(f_model, trajs[:2, 0, :], t0=0.0, dt=dt, steps=min(4, steps))

    with torch.no_grad():
        idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
        pred = rollout(f_model, trajs[idx,0,:], t0=0.0, dt=dt, steps=steps)
        mse = (pred - trajs[idx]).pow(2).mean().item()
        print(f"  [Eval] Stage 1 rollout MSE: {mse:.3e}")

    print(" Stage 2 (General discovery, dimension-agnostic)...")
    t1 = time.time()
    Bdisc, disc_info = discover_symmetry_general(
        f_model, trajs, dt, steps, cfg=discover_cfg, device=device
    )
    stage2_time = time.time() - t1
    rss2 = get_max_rss()
    k_best = disc_info["best_k"]
    print(f"  Best k={k_best}, E[Lint]={disc_info['best_loss']:.3e}")

    # Per-basis expected integral loss
    with torch.no_grad():
        idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
        x0t = trajs[idx, 0, :]
        E_mean, E_std = expected_loss_general_eval(f_model, Bdisc, x0t, dt, steps, discover_cfg.eps, discover_cfg.mult_eps, M=24)
    print(f"  E[Lint] ~ {E_mean:.3e} ± {E_std:.3e}")

    # Overlay, plots, diagnostics
    plot_discovered_generators_grid(Bdisc, outdir/"generators_discovered.png", "discovered generators")
    Arep = Bdisc[0]
    plot_equivariance_overlay_linear(f_model, Arep, trajs, dt, steps, discover_cfg.eps, outdir, system.name)

    # Structure constants, closure, Killing
    lin_struct = structure_constants_and_killing(Bdisc)
    plot_closure_killing(lin_struct, outdir, title_prefix="General")

    # Noise robustness, data-efficiency
    sigmas, vals = plot_noise_robustness(f_model, Bdisc, trajs, dt, steps, discover_cfg.eps, outdir/"noise_curve.png", mult_eps=discover_cfg.mult_eps)
    deff = plot_data_efficiency(f_model, Bdisc, trajs, dt, steps, discover_cfg.eps, outdir, mult_eps=discover_cfg.mult_eps)

    # Baselines on sampled states (for reference only)
    states = trajs.reshape(-1, system.dim)
    states = states[torch.randperm(states.shape[0])[:1024]]
    baselines = {}
    try:
        pw = []
        for _ in range(8):
            xi = torch.randn(Bdisc.shape[0], device=states.device)
            pw.append(pointwise_equiv_loss_linear_states(f_model, states, xi, Bdisc, discover_cfg.eps, mult_eps=discover_cfg.mult_eps))
        baselines["pointwise_mean"] = float(np.mean(pw)); baselines["pointwise_std"] = float(np.std(pw))
    except Exception as e:
        baselines["pointwise_error"] = str(e)
    try:
        br = []
        for _ in range(4):
            xi = torch.randn(Bdisc.shape[0], device=states.device)
            br.append(bracket_loss_linear_states(f_model, states[:64], Bdisc, xi))
        baselines["bracket_mean"] = float(np.mean(br)); baselines["bracket_std"] = float(np.std(br))
    except Exception as e:
        baselines["bracket_error"] = str(e)

    # Composition, eps sweeps, sensitivity, scatter
    comp = plot_composition_heatmap(f_model, Bdisc, trajs, dt, steps, discover_cfg.eps, outdir/"composition_heatmap.png", grid=(5,5))
    eps_sweep = plot_eps_sweep(f_model, Bdisc, trajs, dt, steps, eps_list=[5e-3, 1e-2, 2e-2, 5e-2], outdir=outdir, M=12)
    sens = plot_T_dt_heatmap(f_model, Bdisc, trajs, dt_list=[0.005,0.01,0.02,0.05], T_list=[60,120,200], eps=discover_cfg.eps, outpath=outdir/"sensitivity_heatmap.png")
    plot_pointwise_vs_integral_scatter(f_model, Bdisc, trajs, dt, steps, discover_cfg.eps, outdir)

    # Save results
    result = {
        "system": system.name,
        "dim": system.dim,
        "variant": variant.name,
        "seed": variant.seed,
        "type": "general",
        "k_discovered": int(Bdisc.shape[0]),
        "basis_discovered": Bdisc.detach().cpu().tolist(),
        "E_Lint_mean": E_mean,
        "E_Lint_std": E_std,
        "mse_rollout": float(mse),
        "stage1_time_sec": float(stage1_time),
        "stage2_time_sec": float(stage2_time),
        "max_rss": {"stage1": rss1, "stage2": rss2},
        "params": {
            "B": int(B), "T": int(T), "dt": float(dt),
            "eps": float(discover_cfg.eps), "mult_eps": list(discover_cfg.mult_eps),
            "M_samples": int(discover_cfg.M_expectation),
            "s2_iters_per_k": int(discover_cfg.iters_per_k),
            "max_k": int(discover_cfg.max_k),
            "param_mode": str(discover_cfg.param_mode),
            "seed": int(variant.seed), "compile_used": bool(compiled),
            "device": device, "threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads())
        },
        "loss_history": disc_info["history"],
        "structure_constants": lin_struct,
        "baselines": baselines,
        "noise": {"sigmas": sigmas, "E_Lint": vals},
        "data_efficiency": deff,
        "composition": comp,
        "eps_sweep": eps_sweep,
        "sensitivity_T_dt": sens,
    }
    results_path = out_root / "integral_results.json"
    all_results = []
    if results_path.exists():
        try: all_results = json.loads(results_path.read_text())
        except Exception: pass
    all_results.append(result)
    results_path.write_text(json.dumps(all_results, indent=2))

    plot_convergence([lc for h in disc_info["history"] for lc in h["loss_curve"]], outdir/"convergence.png", f"{system.name} Stage-2 convergence")

# ============ Aggregation (tables + a few figures) ============

def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = np.array([x for x in xs if x is not None], dtype=float)
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std())

def _safe(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

def load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of results")
    return data

def group_by_system_variant(entries: List[Dict[str, Any]]) -> Dict[Tuple[str,str], List[Dict[str,Any]]]:
    g: Dict[Tuple[str,str], List[Dict[str,Any]]] = {}
    for e in entries:
        sysn = e.get("system")
        var = e.get("variant","base")
        if not sysn: continue
        g.setdefault((sysn,var), []).append(e)
    return g

def summarize_group(sysn: str, var: str, runs: List[Dict[str,Any]]) -> Dict[str, Any]:
    EL_means = [r.get("E_Lint_mean") for r in runs if r.get("E_Lint_mean") is not None]
    EL_stds  = [r.get("E_Lint_std") for r in runs if r.get("E_Lint_std") is not None]
    mse_list = [r.get("mse_rollout") for r in runs if r.get("mse_rollout") is not None]
    s1_times = [r.get("stage1_time_sec") for r in runs if r.get("stage1_time_sec") is not None]
    s2_times = [r.get("stage2_time_sec") for r in runs if r.get("stage2_time_sec") is not None]
    k_list   = [r.get("k_discovered") for r in runs if r.get("k_discovered") is not None]

    EL_m, EL_sd = _mean_std(EL_means), _mean_std(EL_stds)
    mse_m, _ = _mean_std(mse_list), (None, None)
    s1_m, _ = _mean_std(s1_times)
    s2_m, _ = _mean_std(s2_times)
    k_m, _  = _mean_std(k_list)

    return {
        "system": sysn, "variant": var, "type": "general",
        "E_Lint_mean_mean": EL_m[0], "E_Lint_mean_std": EL_m[1],
        "mse_rollout_mean": mse_m[0],
        "stage1_time_mean": s1_m, "stage2_time_mean": s2_m,
        "k_discovered_mean": k_m,
        "n_runs": len(runs),
    }

def write_main_table_tex(summaries: List[Dict[str,Any]], out_tex: Path):
    lines = ["\\begin{table}[h]","\\centering","\\caption{General symmetry discovery across systems (mean±std over seeds).}",
             "\\label{tab:general_results}","\\small","\\begin{tabular}{l c c c c}","\\toprule",
             "System & $k$ & $\\bar{\\cL}_{\\mathrm{int}}$ ($\\times 10^{-3}$) & MSE & S2 (s) \\\\",
             "\\midrule"]
    for s in summaries:
        EL = s["E_Lint_mean_mean"]; ELsd = s["E_Lint_mean_std"]
        el_str = "--" if EL is None else f"{(EL*1e3):.2f}" + (f" $\\pm$ {(ELsd*1e3):.2f}" if ELsd is not None else "")
        mse_str = "--" if s["mse_rollout_mean"] is None else f"{s['mse_rollout_mean']:.2e}"
        s2_str  = "--" if s["stage2_time_mean"] is None else f"{s['stage2_time_mean']:.1f}"
        k_str   = "--" if s["k_discovered_mean"] is None else f"{s['k_discovered_mean']:.1f}"
        lines.append(f"{s['system']} & {k_str} & {el_str} & {mse_str} & {s2_str} \\\\")
    lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
    out_tex.write_text("\n".join(lines))

def write_summary_csv(summaries: List[Dict[str,Any]], out_csv: Path):
    import csv
    keys = [
        "system","variant","type","n_runs",
        "k_discovered_mean","E_Lint_mean_mean","E_Lint_mean_std",
        "mse_rollout_mean","stage1_time_mean","stage2_time_mean"
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in keys})

def plot_compute_vs_accuracy(entries: List[Dict[str,Any]], out_path: Path):
    xs, ys = [], []
    for e in entries:
        t = e.get("stage2_time_sec"); L = e.get("E_Lint_mean")
        if isinstance(t,(float,int)) and isinstance(L,(float,int)):
            xs.append(t/60.0); ys.append(L)
    if not xs: return
    plt.figure(figsize=(5.0,3.8))
    plt.scatter(xs, ys, c='b', s=28)
    plt.xlabel("Total compute (min, Stage 2)"); plt.ylabel("$\\bar{\\cL}_{\\text{int}}$")
    plt.yscale('log')
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def generate_all(json_path: Path, out_dir: Path):
    entries = load_json(json_path)

    # Summaries
    groups = group_by_system_variant(entries)
    summaries = []
    for (sysn, var), runs in sorted(groups.items(), key=lambda kv: kv[0][0]):
        if all(r.get("E_Lint_mean") is None for r in runs):
            continue
        summaries.append(summarize_group(sysn, var, runs))

    write_main_table_tex(summaries, out_dir / "main_results_table.tex")
    write_summary_csv(summaries, out_dir / "summary.csv")
    plot_compute_vs_accuracy(entries, out_dir / "compute_vs_accuracy.png")

    # Copy discussion figs up one level too
    try:
        for fn in ("compute_vs_accuracy.png",):
            src = out_dir / fn
            if src.exists():
                shutil.copyfile(src, out_dir.parent / fn)
    except Exception:
        pass

    print(f"[OK] Wrote tables and figures to {out_dir}")

# ============ Main ============

def plot_convergence(loss_hist: List[float], outpath: Path, title: str):
    if not loss_hist:
        return
    plt.figure(figsize=(6,4))
    ys = np.array(loss_hist, dtype=float)
    xs = np.arange(1, len(ys)+1)
    plt.plot(xs, ys, 'b-')
    plt.yscale('log'); plt.xlabel('iter'); plt.ylabel('E[Lint]')
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

def main():
    device = get_device()
    print(f"Using device: {device}")
    out_root = Path("results/paper_integral_full"); ensure_dir(out_root)

    # Data generators (no action hacks)
    systems = [
        make_harmonic2d(omega=1.5, device=device),
        make_nonlinear_rot2d(omega0=0.8, beta=0.4, device=device),
        make_so3_equivariant3d(scale=0.7, device=device),
        make_so4_equivariant4d(scale=0.5, device=device),
        make_kepler2d(mu=1.0, device=device),
        make_se2_unicycle(v=1.0, omega=0.2, device=device),  # Expect failure under linear action
    ]
    var = Variant(name="base", seed=42)

    # Fair discovery config (no structure beyond matrix exponential and linear action)
    discover_cfg = DiscoveryConfig(
        max_k=10, iters_per_k=1200, lr=5e-3, eps=1e-2, M_expectation=8,
        mult_eps=(0.5,1.0,2.0), param_mode="full"  # "full" = general gl(d)
    )

    for sysn in systems:
        run_experiment(sysn, device=device, out_root=out_root, variant=var,
                       B=128, T=120, dt=0.02, s1_iters=800, discover_cfg=discover_cfg)

    # Post-process: generate tables/figures
    out_dir = out_root / "heatmaps"; ensure_dir(out_dir)
    try:
        generate_all(out_root / "integral_results.json", out_dir)
    except Exception as e:
        print(f"[WARN] post-processing skipped: {e}")

if __name__ == "__main__":
    main()