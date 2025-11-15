#!/usr/bin/env python3
# Integral-only Symmetry Discovery (full runner + baselines + ablations + paper tables)
# Systems: SO(2) (Harmonic2D), Nonlinear SO(2), SO(3) full, SO(4) full, Kepler (SO2 block), SE(2) (full twists)
# Plots: overlays, timewise error, noise, eps-sweep (+ normalized), data-efficiency (B, T), composition heatmap,
#        closure/Killing, coefficients heatmap, principal-angles bar, sensitivity (SE(2)), convergence
# JSON: integral/pointwise/infinitesimal results, ablations, params, timings, loss histories, alignment metrics
# Tables: main_results_table.tex, baselines_table.tex, ablations_table.tex written under results/paper_integral_full/heatmaps

import os
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Callable, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ Environment ============
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TORCH_LOGS", "")
torch.set_float32_matmul_precision("high")
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.set_num_interop_threads(max(1, (os.cpu_count() or 1) // 2))

try:
	def get_max_rss():
		import resource
		return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except Exception:
	def get_max_rss():
		return None

# ============ Utils ============
def set_seed(seed: int = 42):
	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

def get_device() -> str:
	return "cpu"

def ensure_dir(p: Path):
	p.mkdir(parents=True, exist_ok=True)

def matrix_exp(A: torch.Tensor) -> torch.Tensor:
	if hasattr(torch.linalg, "matrix_exp"):
		return torch.linalg.matrix_exp(A)
	return torch.matrix_exp(A)

def to_cpu_np(x: torch.Tensor) -> np.ndarray:
	return x.detach().cpu().numpy()

def json_safe(x):
	import numpy as _np
	import torch as _torch
	if isinstance(x, _np.ndarray):
		return x.tolist()
	if isinstance(x, (_np.floating, _np.integer, _np.bool_)):
		return x.item()
	if isinstance(x, _torch.Tensor):
		return x.detach().cpu().tolist()
	if isinstance(x, dict):
		return {k: json_safe(v) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [json_safe(v) for v in x]
	return x

def append_result(out_path: Path, entry: Dict[str, Any]) -> None:
	entry = json_safe(entry)
	data: List[Dict[str, Any]] = []
	if out_path.exists():
		try:
			data = json.loads(out_path.read_text())
		except Exception:
			data = []
	data.append(entry)
	out_path.write_text(json.dumps(data, indent=2))

def safe_fname(s: str) -> str:
	return "".join(c if c.isalnum() or c in "._-()" else "_" for c in s)

# Plot axis helpers
def set_log_ylim(ax, y: np.ndarray, y_min=1e-12, y_max=1e-3):
	y = np.asarray(y, dtype=float)
	if not np.any(np.isfinite(y)):
		ax.set_yscale("log"); ax.set_ylim(y_min, y_max); return
	finite = y[np.isfinite(y) & (y > 0)]
	lo = y_min if finite.size == 0 else max(y_min, np.nanmin(finite)/1.5)
	hi = y_max if finite.size == 0 else max(y_min*10, min(y_max, np.nanmax(finite)*1.5))
	ax.set_yscale("log"); ax.set_ylim(lo, hi)

def set_log_xlim(ax, x: np.ndarray):
	x = np.asarray(x, dtype=float)
	finite = x[np.isfinite(x) & (x > 0)]
	if finite.size == 0:
		ax.set_xscale("log"); ax.set_xlim(1e-6, 1.0); return
	lo = np.nanmin(finite)/1.5
	hi = np.nanmax(finite)*1.5
	ax.set_xscale("log"); ax.set_xlim(lo, hi)

# ============ Integrator ============
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

# ============ Dynamics model ============
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

# ============ Bases and helpers ============
def rotation_basis_R2() -> torch.Tensor:
	J = torch.tensor([[0., -1.],[1., 0.]], dtype=torch.float32)
	return J.unsqueeze(0)

def rotation_basis_R3() -> torch.Tensor:
	Bx = torch.tensor([[0., 0., 0.],[0., 0., -1.],[0., 1., 0.]], dtype=torch.float32)
	By = torch.tensor([[0., 0., 1.],[0., 0., 0.],[-1., 0., 0.]], dtype=torch.float32)
	Bz = torch.tensor([[0., -1., 0.],[1.,  0., 0.],[0.,  0., 0.]], dtype=torch.float32)
	return torch.stack([Bx, By, Bz], dim=0)

def so_basis(n: int) -> torch.Tensor:
	Bs = []
	for i in range(n):
		for j in range(i+1, n):
			M = torch.zeros(n, n, dtype=torch.float32)
			M[i, j] = 1.0; M[j, i] = -1.0
			Bs.append(M)
	return torch.stack(Bs, dim=0) if Bs else torch.zeros(0, n, n)

def blockdiag_so2_two_blocks() -> torch.Tensor:
	J = torch.tensor([[0., -1.],[1., 0.]], dtype=torch.float32)
	A = torch.zeros(4, 4, dtype=torch.float32)
	A[:2,:2] = J; A[2:,2:] = J
	return A.unsqueeze(0)

def orthonormalize_basis(Bs: torch.Tensor) -> torch.Tensor:
	k, D, _ = Bs.shape
	U = Bs.reshape(k, -1).T
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

# Principal angles and alignment
def orthonormalize_frobenius(B: torch.Tensor) -> torch.Tensor:
	k, D, _ = B.shape
	U = B.reshape(k, -1).T
	Q, _ = torch.linalg.qr(U, mode='reduced')
	return Q.T.reshape(k, D, D)

def principal_angles_and_alignment(E: Optional[torch.Tensor], D: torch.Tensor) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[torch.Tensor]]:
	if E is None or D is None:
		return None, None, None
	Eo = orthonormalize_frobenius(E)
	Do = orthonormalize_frobenius(D)
	kE, Ddim, _ = Eo.shape
	Ef = Eo.reshape(kE, -1)
	Df = Do.reshape(Do.shape[0], -1)
	C = Ef @ Df.T
	U, S, Vt = torch.linalg.svd(C, full_matrices=False)
	S = torch.clamp(S, -1.0, 1.0)
	angles = torch.arccos(S).detach().cpu().numpy() * (180.0 / math.pi)
	r = S.shape[0]
	O = (U[:, :r] @ Vt[:r, :]).detach().cpu().numpy()
	D_aligned = torch.einsum('ij,jab->iab', torch.from_numpy(O).to(Do.device, Do.dtype), D)
	return angles, O, D_aligned

# Structure constants / Killing (linear)
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

# ============ Linear losses (cached) ============
def integral_equiv_loss_linear_cached(xs: torch.Tensor,
                                      f_field: nn.Module,
                                      x0s: torch.Tensor,
                                      xi: torch.Tensor, B: torch.Tensor,
                                      dt: float, steps: int, eps: float,
                                      mult_eps: Tuple[float, ...],
                                      normalize_A: bool = True) -> torch.Tensor:
	A = A_unit_from_xi(xi, B) if normalize_A else A_from_xi(xi, B)
	err_acc = 0.0
	for m in mult_eps:
		g = matrix_exp((m * eps) * A)
		x0s_g = x0s @ g.T
		xs_g0 = rollout(f_field, x0s_g, t0=0.0, dt=dt, steps=steps)
		xs_g = xs @ g.T
		diff = (xs_g0 - xs_g).pow(2).sum(dim=-1)
		err_acc += dt * diff.mean()
	return err_acc / len(mult_eps)

def expected_integral_loss_linear_basis(f_field: nn.Module,
                                        x0s: torch.Tensor,
                                        W: torch.Tensor,
                                        dt: float, steps: int, eps: float,
                                        mult_eps=(0.5,1.0,2.0), M: int = 6,
                                        normalize_A: bool = True,
                                        use_skew: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
	if use_skew:
		Bskew = 0.5 * (W - W.transpose(-1, -2))
		B = orthonormalize_basis(Bskew)
	else:
		B = orthonormalize_basis(W)
	with torch.no_grad():
		xs = rollout(f_field, x0s, t0=0.0, dt=dt, steps=steps).detach()
	k = B.shape[0]; device = x0s.device
	xis = sample_xi(k, M, device)
	L = 0.0
	for m in range(M):
		L += integral_equiv_loss_linear_cached(xs, f_field, x0s, xis[m], B, dt, steps, eps, mult_eps=mult_eps, normalize_A=normalize_A)
	return L / M, B

# ============ SE(2) action-agnostic ============
class VFMLP(nn.Module):
	def __init__(self, d_in: int, hidden: int = 64, depth: int = 3):
		super().__init__()
		self.net = MLP(d_in, hidden, d_in, depth)
	def forward(self, x):
		return self.net(x)

def orthonormalize_field_values(Fvals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	# Fvals: [k,B,d]
	k, B, d = Fvals.shape
	G = torch.zeros(k, k, device=Fvals.device, dtype=Fvals.dtype)
	for i in range(k):
		for j in range(k):
			G[i, j] = (Fvals[i].reshape(B, d) * Fvals[j].reshape(B, d)).sum(dim=1).mean()
	evals, Q = torch.linalg.eigh(G + 1e-9 * torch.eye(k, device=G.device, dtype=G.dtype))
	evals = torch.clamp(evals, min=1e-9)
	Gm12 = Q @ torch.diag(evals.rsqrt()) @ Q.T
	Dvals = torch.einsum("ij,jbd->ibd", Gm12, Fvals)
	return Dvals, Gm12

def flow_of_field(vfun: Callable[[torch.Tensor], torch.Tensor],
                  x: torch.Tensor, eps: float, steps: int = 8) -> torch.Tensor:
	dt = eps / float(steps)
	z = x
	for _ in range(steps):
		k1 = vfun(z)
		k2 = vfun(z + 0.5 * dt * k1)
		k3 = vfun(z + 0.5 * dt * k2)
		k4 = vfun(z + dt * k3)
		z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
	return z

def integral_equiv_loss_se2_agnostic_cached(xs: torch.Tensor,
                                            f_field: nn.Module,
                                            x0s: torch.Tensor,
                                            VF: List[nn.Module],
                                            dt: float, steps: int, eps: float,
                                            mult_eps: Tuple[float, ...] = (0.5,1.0,2.0)) -> torch.Tensor:
	with torch.no_grad():
		Fvals = torch.stack([VF[i](x0s) for i in range(3)], dim=0)  # [3,B,3]
	Dvals, Mmix = orthonormalize_field_values(Fvals)

	def v_dir(x, xi):
		u = xi / (xi.norm() + 1e-12)
		Fb = torch.stack([VF[0](x), VF[1](x), VF[2](x)], dim=0)
		D = torch.einsum("ij,jnd->ind", Mmix, Fb)
		return torch.einsum('i,ind->nd', u, D)

	L = 0.0
	B = x0s.shape[0]
	for m in mult_eps:
		xi = torch.randn(3, device=x0s.device)
		epsm = float(m * eps)
		def vfun0(z): return v_dir(z, xi)
		x0s_TF = flow_of_field(vfun0, x0s, epsm, steps=8)
		xs_TF = rollout(f_field, x0s_TF, t0=0.0, dt=dt, steps=steps)
		xs_flat = xs.reshape((B*(steps+1), -1))
		xs_FT_flat = flow_of_field(vfun0, xs_flat, epsm, steps=8)
		xs_FT = xs_FT_flat.reshape(B, steps+1, -1)
		diff = (xs_TF - xs_FT).pow(2).sum(dim=-1)
		L += dt * diff.mean()
	return L / len(mult_eps)

# ============ Baselines (pointwise, infinitesimal FD) ============
def pointwise_equiv_loss_linear_states(f_field: nn.Module,
                                       states: torch.Tensor,
                                       xi: torch.Tensor, B: torch.Tensor,
                                       eps: float, mult_eps=(0.5,1.0,2.0),
                                       normalize_A: bool = True) -> torch.Tensor:
	A = A_unit_from_xi(xi, B) if normalize_A else A_from_xi(xi, B)
	acc = 0.0
	for m in mult_eps:
		g = matrix_exp((m * eps) * A)
		gx = states @ g.T
		fx = f_field(0.0, states)
		fgx = f_field(0.0, gx)
		lhs = fx @ g.T
		acc = acc + (lhs - fgx).pow(2).sum(dim=-1).mean()
	return acc / len(mult_eps)

def _jvp_f(f_field: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	x = x.detach().requires_grad_(True)
	fx = f_field(0.0, x)
	g = torch.autograd.grad(fx, x, grad_outputs=v, retain_graph=False, create_graph=False, allow_unused=False)[0]
	return g

@torch.no_grad()
def bracket_loss_linear_states(f_field: nn.Module,
                               states: torch.Tensor,
                               B: torch.Tensor, xi: torch.Tensor,
                               normalize_A: bool = True) -> float:
	A = A_unit_from_xi(xi, B) if normalize_A else A_from_xi(xi, B)
	errs = []
	for j in range(states.shape[0]):
		xj = states[j:j+1].detach()
		vA = (xj @ A.T)
		jfv = _jvp_f(f_field, xj, vA)
		fx = f_field(0.0, xj)
		comm = jfv - (fx @ A.T)
		errs.append(comm.pow(2).sum(dim=-1))
	return float(torch.cat(errs).mean().cpu())

def discover_linear_basis_pointwise(f_field: nn.Module,
                                    trajs: torch.Tensor,
                                    dt: float, steps: int, dim: int,
                                    k: int, iters: int = 1000, lr: float = 5e-3,
                                    eps: float = 1e-2, M: int = 6,
                                    mult_eps=(0.5,1.0,2.0), device: str = "cpu",
                                    at_final_only: bool = False,
                                    normalize_A: bool = True,
                                    early_stop: bool = True, patience: int = 300, min_delta_rel: float = 1e-4,
                                    use_skew: bool = True
                                    ) -> Tuple[torch.Tensor, List[float]]:
	W = (0.01 * torch.randn(k, dim, dim, device=device)).requires_grad_(True)
	opt = optim.Adam([W], lr=lr)
	sched = CosineAnnealingLR(opt, T_max=iters, eta_min=lr*0.1)
	N = trajs.shape[0]; Bsize = min(64, N)
	best_B = None; best_loss = None; bad = 0; losses = []
	for it in range(1, iters+1):
		idx = torch.randint(0, N, (Bsize,), device=device)
		if at_final_only:
			states = trajs[idx, -1, :]
		else:
			states = trajs[idx, 0, :]
		if use_skew:
			Bcand = 0.5 * (W - W.transpose(-1, -2))
		else:
			Bcand = W
		Borth = orthonormalize_basis(Bcand)
		L = 0.0
		for _ in range(M):
			xi = torch.randn(k, device=device)
			L = L + pointwise_equiv_loss_linear_states(f_field, states, xi, Borth, eps, mult_eps=mult_eps, normalize_A=normalize_A)
		L = L / float(M)
		opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_([W], 5.0); opt.step(); sched.step()
		Lf = float(L.detach().cpu()); losses.append(Lf)
		if best_B is None or Lf < (best_loss if best_loss is not None else float("inf")):
			best_B, best_loss = Borth.detach().clone(), Lf
		if best_loss is None or Lf < best_loss * (1.0 - min_delta_rel):
			best_loss, bad = Lf, 0
		else:
			bad += 1
		if it % 200 == 0:
			print(f"  [Pointwise] iter {it:4d} | loss {Lf:.3e}")
		if early_stop and bad >= patience:
			print(f"  [Pointwise] early stop at {it}, best {best_loss:.3e}")
			break
	return best_B, losses

def discover_linear_basis_infinitesimal_fd(f_field: nn.Module,
                                           trajs: torch.Tensor,
                                           dt: float, steps: int, dim: int,
                                           k: int, iters: int = 1000, lr: float = 5e-3,
                                           h_fd: float = 1e-3, M: int = 8,
                                           device: str = "cpu",
                                           normalize_A: bool = True,
                                           early_stop: bool = True, patience: int = 300, min_delta_rel: float = 1e-4,
                                           use_skew: bool = True
                                           ) -> Tuple[torch.Tensor, List[float]]:
	W = (0.01 * torch.randn(k, dim, dim, device=device)).requires_grad_(True)
	opt = optim.Adam([W], lr=lr)
	sched = CosineAnnealingLR(opt, T_max=iters, eta_min=lr*0.1)
	N = trajs.shape[0]; Bsize = min(32, N)
	best_B = None; best_loss = None; bad = 0; losses = []
	for it in range(1, iters+1):
		idx = torch.randint(0, N, (Bsize,), device=device)
		x0s = trajs[idx, 0, :]
		L = 0.0
		Bcand = (0.5 * (W - W.transpose(-1, -2))) if use_skew else W
		for _ in range(M):
			xi = torch.randn(k, device=device)
			Borth = orthonormalize_basis(Bcand)
			A = A_unit_from_xi(xi, Borth) if normalize_A else A_from_xi(xi, Borth)
			vA = x0s @ A.T
			fx = f_field(0.0, x0s)
			f_plus = f_field(0.0, x0s + h_fd * vA)
			f_minus = f_field(0.0, x0s - h_fd * vA)
			jfv_fd = (f_plus - f_minus) / (2.0 * h_fd)
			comm = jfv_fd - (fx @ A.T)
			L = L + comm.pow(2).sum(dim=-1).mean()
		L = L / float(M)
		opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_([W], 5.0); opt.step(); sched.step()
		Lf = float(L.detach().cpu()); losses.append(Lf)
		with torch.no_grad():
			Bflat = ((0.5 * (W - W.transpose(-1, -2))) if use_skew else W).reshape(k, -1).T
			Q, _ = torch.linalg.qr(Bflat, mode='reduced')
			Borth2 = Q.T.reshape(k, dim, dim).detach().clone()
		if best_B is None or Lf < (best_loss if best_loss is not None else float("inf")):
			best_B, best_loss = Borth2, Lf
		if best_loss is None or Lf < best_loss * (1.0 - min_delta_rel):
			best_loss, bad = Lf, 0
		else:
			bad += 1
		if it % 200 == 0:
			print(f"  [Inf-FD] iter {it:4d} | bracket FD loss {Lf:.3e}")
		if early_stop and bad >= patience:
			print(f"  [Inf-FD] early stop at {it}, best {best_loss:.3e}")
			break
	return best_B, losses

# ============ Linear joint-basis discovery (with optional projector) ============

def project_to_se2(W: torch.Tensor) -> torch.Tensor:
	# Enforce se(2) structured generator in homogeneous coords (3x3): [[skew2, t],[0, 0]]
	if W.shape[-1] != 3 or W.shape[-2] != 3:
		return W
	A = W.clone()
	A[..., 2, :] = 0.0
	M = A[..., :2, :2]
	A[..., :2, :2] = 0.5 * (M - M.transpose(-1, -2))
	return A

def project_to_su3_real(W: torch.Tensor) -> torch.Tensor:
	# Project 6x6 real matrix onto realification of su(3): blocks [[R, -I],[I,R]] with R skew, I symmetric traceless
	if W.shape[-1] != 6 or W.shape[-2] != 6:
		return W
	A = W.clone()
	A11 = A[..., :3, :3]
	A12 = A[..., :3, 3:]
	A21 = A[..., 3:, :3]
	A22 = A[..., 3:, 3:]
	Ru = 0.5 * (A11 + A22)
	Iu = 0.5 * (A21 - A12)
	R = 0.5 * (Ru - Ru.transpose(-1, -2))  # skew
	Is = 0.5 * (Iu + Iu.transpose(-1, -2))  # symmetric
	eye3 = torch.eye(3, dtype=W.dtype, device=W.device).expand_as(R)
	trI = Is.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / 3.0
	Is = Is - trI[..., None] * eye3
	top = torch.cat([R, -Is], dim=-1)
	bot = torch.cat([Is,  R], dim=-1)
	return torch.cat([top, bot], dim=-2)

def discover_linear_basis_joint(f_field: nn.Module,
                                trajs: torch.Tensor,
                                dt: float, steps: int, dim: int,
                                k: int, iters: int = 1400, lr: float = 5e-3,
                                eps: float = 1e-2, M: int = 8,
                                mult_eps=(0.5,1.0,2.0), device: str = "cpu",
                                normalize_A: bool = True,
                                early_stop: bool = True, patience: int = 300, min_delta_rel: float = 1e-4,
                                use_skew: bool = True,
                                projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                                ) -> Tuple[torch.Tensor, List[float]]:
	W = (0.01 * torch.randn(k, dim, dim, device=device)).requires_grad_(True)
	opt = optim.Adam([W], lr=lr)
	sched = CosineAnnealingLR(opt, T_max=iters, eta_min=lr*0.1)
	N = trajs.shape[0]; Bsize = min(32, N)
	best = None; losses = []; best_loss = None; bad = 0
	for it in range(1, iters+1):
		idx = torch.randint(0, N, (Bsize,), device=device)
		x0s = trajs[idx, 0, :]
		L, B = expected_integral_loss_linear_basis(f_field, x0s, W, dt, steps, eps, mult_eps=mult_eps, M=M, normalize_A=normalize_A, use_skew=use_skew)
		opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_([W], 5.0); opt.step()
		if projector is not None:
			with torch.no_grad():
				W.copy_(projector(W))
		sched.step()
		Lf = float(L.detach().cpu()); losses.append(Lf)
		if best is None or Lf < best[0]: best = (Lf, B.detach().clone())
		if best_loss is None or Lf < best_loss * (1.0 - min_delta_rel):
			best_loss = Lf; bad = 0
		else:
			bad += 1
		if it % 200 == 0: print(f"  [Joint-Basis] iter {it:4d} | E[Lint] {Lf:.3e}")
		if early_stop and bad >= patience:
			print(f"  [Joint-Basis] early stop at {it}, best {best_loss:.3e}")
			break
	return best[1], losses

def discover_se2_basis_agnostic(f_field: nn.Module,
                                trajs: torch.Tensor,
                                dt: float, steps: int,
                                iters: int = 1800, lr: float = 5e-3,
                                eps: float = 1e-2, M: int = 12,
                                mult_eps=(0.5,1.0,2.0), device: str = "cpu",
                                early_stop: bool = True, patience: int = 400, min_delta_rel: float = 1e-4
                                ) -> Tuple[List[nn.Module], List[float]]:
	VF = [VFMLP(3, hidden=64, depth=3).to(device) for _ in range(3)]
	params = []
	for v in VF: params += list(v.parameters())
	opt = optim.Adam(params, lr=lr)
	sched = CosineAnnealingLR(opt, T_max=iters, eta_min=lr*0.1)
	N = trajs.shape[0]; Bsize = min(32, N)
	best = None; losses = []; best_loss = None; bad = 0
	for it in range(1, iters+1):
		idx = torch.randint(0, N, (Bsize,), device=device)
		x0s = trajs[idx, 0, :]
		with torch.no_grad():
			xs = rollout(f_field, x0s, t0=0.0, dt=dt, steps=steps).detach()
		L = 0.0
		for _ in range(M):
			L += integral_equiv_loss_se2_agnostic_cached(xs, f_field, x0s, VF, dt, steps, eps, mult_eps=mult_eps)
		L = L / M
		opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step()
		Lf = float(L.detach().cpu()); losses.append(Lf)
		if best is None or Lf < best[0]:
			best = (Lf, [v.state_dict() for v in VF])
		if best_loss is None or Lf < best_loss * (1.0 - min_delta_rel):
			best_loss = Lf; bad = 0
		else:
			bad += 1
		if it % 200 == 0: print(f"  [SE2-agn] iter {it:4d} | E[Lint] {Lf:.3e}")
		if early_stop and bad >= patience:
			print(f"  [SE2-agn] early stop at {it}, best {best_loss:.3e}")
			break
	if best is not None:
		for v, sd in zip(VF, best[1]): v.load_state_dict(sd)
	return VF, losses

# ============ Systems ============
@dataclass
class System:
	name: str
	dim: int
	f: Callable[[float, torch.Tensor], torch.Tensor]
	expected_basis: Optional[torch.Tensor]

def make_harmonic2d(omega=1.5, device="cpu") -> System:
	A = torch.tensor([[0., -omega],[omega, 0.]], device=device)
	def f(t, x): return x @ A.T
	return System("Harmonic2D (SO2,1D)", 2, f, rotation_basis_R2().to(device))

def make_nonlinear_rot2d(omega0=0.8, beta=0.4, device="cpu") -> System:
	J = torch.tensor([[0., -1.],[1., 0.]], device=device)
	def omega(x): return omega0 + beta * (x.pow(2).sum(dim=-1, keepdim=True))
	def f(t, x): return omega(x) * (x @ J.T)
	return System("NonlinearRot2D (SO2,1D)", 2, f, rotation_basis_R2().to(device))

def make_so3_equivariant3d(scale=0.7, device="cpu") -> System:
	def f(t, x): return scale * x
	return System("SO3-Equivariant3D (full 3D)", 3, f, rotation_basis_R3().to(device))

def make_so4_equivariant4d(scale=0.5, device="cpu") -> System:
	def f(t, x): return scale * x
	return System("SO4-Equivariant4D (full 6D)", 4, f, so_basis(4).to(device))

def make_kepler2d(mu=1.0, device="cpu") -> System:
	def f(t, x):
		pos = x[..., :2]; vel = x[..., 2:]
		r = torch.norm(pos, dim=-1, keepdim=True).clamp_min(1e-6)
		acc = -mu * pos / (r**3)
		return torch.cat([vel, acc], dim=-1)
	return System("Kepler2D (SO2_block,1D)", 4, f, blockdiag_so2_two_blocks().to(device))

def make_se2_unicycle(v=1.0, omega=0.2, device="cpu") -> System:
	def f(t, x):
		th = x[..., 2:3]
		vx = v * torch.cos(th); vy = v * torch.sin(th)
		vth = torch.full_like(th, omega)
		return torch.cat([vx, vy, vth], dim=-1)
	return System("SE2-Unicycle (SE2,3D)", 3, f, None)

# Exact linear SE(2) in homogeneous coords (x,y,1): canonical se(2) basis.
def make_se2_linear(scale=0.7, device="cpu") -> System:
	R = torch.tensor([[0., -1., 0.],[1.,  0., 0.],[0.,  0., 0.]], dtype=torch.float32, device=device)
	Tx= torch.tensor([[0.,  0., 1.],[0.,  0., 0.],[0.,  0., 0.]], dtype=torch.float32, device=device)
	Ty= torch.tensor([[0.,  0., 0.],[0.,  0., 1.],[0.,  0., 0.]], dtype=torch.float32, device=device)
	B = torch.stack([R, Tx, Ty], dim=0)
	def f(t, x):
		xy = x[..., :2]; one = x[..., 2:3]
		return torch.cat([scale*xy, torch.zeros_like(one)], dim=-1)
	return System("SE2-LinearHom (SE2,3D)", 3, f, B)

# SU(3): 8 realified generators (6x6 real) via Gell-Mann
def _gell_mann() -> List[np.ndarray]:
	lam = []
	lam.append(np.array([[0,1,0],[1,0,0],[0,0,0]],dtype=np.float64))
	lam.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]],dtype=np.complex128))
	lam.append(np.array([[1,0,0],[0,-1,0],[0,0,0]],dtype=np.float64))
	lam.append(np.array([[0,0,1],[0,0,0],[1,0,0]],dtype=np.float64))
	lam.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]],dtype=np.complex128))
	lam.append(np.array([[0,0,0],[0,0,1],[0,1,0]],dtype=np.float64))
	lam.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]],dtype=np.complex128))
	lam.append((1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]],dtype=np.float64))
	return lam

def _complex_to_real_block(A: np.ndarray) -> np.ndarray:
	Ar = np.real(A); Ai = np.imag(A)
	top = np.concatenate([Ar, -Ai], axis=1)
	bot = np.concatenate([Ai,  Ar], axis=1)
	return np.concatenate([top, bot], axis=0)

def su3_real_basis_torch(device: str) -> torch.Tensor:
	lam = _gell_mann()
	B = []
	for L in lam:
		A = 1j * L
		R = _complex_to_real_block(A)
		B.append(torch.tensor(R, dtype=torch.float32, device=device))
	return torch.stack(B, dim=0)

def make_su3_equivariant6d(scale=0.5, device="cpu") -> System:
	def f(t, x): return scale * x
	return System("SU3-Equivariant6D (full 8D)", 6, f, su3_real_basis_torch(device))

# ============ Training ============
def sample_trajs(system: System, B: int, T: int, dt: float, x_scale: float, device: str) -> torch.Tensor:
	D = system.dim
	if "Kepler2D" in system.name:
		pos = torch.randn(B, 2, device=device) * x_scale
		vel = torch.randn(B, 2, device=device) * (x_scale * 0.2)
		x0 = torch.cat([pos, vel], dim=-1)
	elif "SE2" in system.name:
		if "LinearHom" in system.name:
			xy = torch.randn(B, 2, device=device) * x_scale
			one = torch.ones(B, 1, device=device)
			x0 = torch.cat([xy, one], dim=-1)
		else:
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
                 device: str = "cpu",
                 early_stop: bool = True, patience: int = 300, min_delta_rel: float = 1e-4, check_every: int = 50
                 ) -> nn.Module:
	hidden = 256 if dim == 4 else 128
	depth = 4 if dim == 4 else 3
	model = NeuralODEField(dim=dim, hidden=hidden, depth=depth).to(device)
	opt = optim.Adam(model.parameters(), lr=lr)
	N, T1, _ = trajs.shape
	Bsize = min(64, N)
	best_loss = None
	best_sd = None
	bad = 0
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
		if it % check_every == 0:
			Lf = float(loss.detach().cpu())
			if best_loss is None or Lf < best_loss * (1.0 - min_delta_rel):
				best_loss = Lf
				best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
				bad = 0
			else:
				bad += check_every
			if early_stop and bad >= patience:
				print(f"  [Stage1] early stop at {it}, best MSE {best_loss:.3e}")
				break
	if best_sd is not None:
		model.load_state_dict(best_sd)
	model.eval()
	return model

# ============ Plotting ============
def plot_noise_curve(vals: List[float], sigmas: List[float], out_path: Path, title: str):
	xs = np.array(sigmas, dtype=float); ys = np.array(vals, dtype=float)
	fig, ax = plt.subplots(figsize=(5.5,4))
	ax.plot(xs, ys, 'ko-')
	ax.set_xlabel("noise sigma"); ax.set_ylabel("E[Lint]"); ax.set_title(title)
	set_log_ylim(ax, ys, y_min=1e-12, y_max=1e-3)
	ax.grid(True, which='both', alpha=0.2)
	fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_eps_sweep_curve(eps_list, E_raw, E_over_eps2, out_dir: Path, title: str):
	xs = np.array(eps_list, dtype=float); ys = np.array(E_raw, dtype=float)
	fig, ax = plt.subplots(figsize=(5.5,4))
	ax.plot(xs, ys, 'ko-', label='E[Lint]')
	set_log_ylim(ax, ys, y_min=1e-12, y_max=1e-3)
	set_log_xlim(ax, xs)
	ax.set_xlabel("ε"); ax.set_ylabel("E[Lint]"); ax.set_title(title); ax.legend()
	fig.tight_layout(); fig.savefig(out_dir/"eps_sweep.png"); plt.close(fig)
	ys2 = np.array(E_over_eps2, dtype=float)
	fig, ax = plt.subplots(figsize=(5.5,4))
	ax.plot(xs, ys2, 'rs--', label='E[Lint]/ε²')
	set_log_ylim(ax, ys2, y_min=1e-12, y_max=1e-3)
	set_log_xlim(ax, xs)
	ax.set_xlabel("ε"); ax.set_ylabel("E[Lint]/ε²"); ax.set_title(title + " (normalized)"); ax.legend()
	fig.tight_layout(); fig.savefig(out_dir/"eps_sweep_over_eps2.png"); plt.close(fig)

def plot_coefficients_heatmap(coeff: np.ndarray, out_path: Path, title: str):
	fig, ax = plt.subplots(figsize=(4.5,4))
	im = ax.imshow(coeff, cmap='coolwarm', vmin=-1.0, vmax=1.0)
	fig.colorbar(im, ax=ax, label="coeff")
	ax.set_title(title); ax.set_xlabel("discovered"); ax.set_ylabel("expected")
	fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_principal_angles_bar(angles_deg: np.ndarray, out_path: Path, title: str):
	fig, ax = plt.subplots(figsize=(5,3))
	ax.bar(list(range(1, len(angles_deg)+1)), angles_deg)
	ax.set_ylabel("deg"); ax.set_title(title)
	fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_time_error(tvec: np.ndarray, err_time: np.ndarray, out_path: Path, title: str):
	fig, ax = plt.subplots(figsize=(6,4))
	ax.plot(tvec, err_time, 'k-')
	set_log_ylim(ax, err_time, 1e-12, 1e-3)
	ax.set_xlabel("time"); ax.set_ylabel("equivariance error"); ax.set_title(title)
	fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def plot_basis_grid(B: torch.Tensor, out_path: Path, title: str, vmax: float = 1.0):
	k = int(B.shape[0]); cols = min(k, 6); rows = int(math.ceil(k/cols))
	fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 3.0*rows))
	if rows==1 and cols==1: axes = np.array([[axes]])
	elif rows==1: axes = np.array([axes])
	for i in range(rows*cols):
		r, c = divmod(i, cols)
		ax = axes[r, c]
		if i < k:
			im = ax.imshow(B[i].detach().cpu().numpy(), cmap='coolwarm', vmin=-vmax, vmax=vmax)
			ax.set_title(f"B[{i}]"); ax.set_xticks([]); ax.set_yticks([])
		else:
			ax.axis('off')
	fig.suptitle(title)
	fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

# ============ Variants and runner ============
@dataclass
class Variant:
	name: str
	mult_eps: Tuple[float, ...] = (0.5,1.0,2.0)
	seed: int = 42
	no_normalize: bool = False
	dt_override: Optional[float] = None
	T_override: Optional[int] = None
	pointwise_final_only: bool = False

def run_experiment(system: System,
                   device: str,
                   out_root: Path,
                   variant: Variant,
                   B: int = 128, T: int = 120, dt: float = 0.02,
                   s1_iters: int = 800,
                   s2_lr: float = 5e-3,
                   use_se2_agnostic: bool = True) -> None:
	set_seed(variant.seed)
	print(f"\n=== {system.name} | dim={system.dim} | variant={variant.name} | seed={variant.seed} ===")
	outdir = out_root / f"{system.name.replace(' ','_')}__{variant.name}"
	ensure_dir(outdir)

	if "Kepler2D" in system.name:
		dt, T = 0.01, 200

	if variant.dt_override is not None:
		dt = variant.dt_override
	if variant.T_override is not None:
		T = variant.T_override

	trajs = sample_trajs(system, B=B, T=T, dt=dt, x_scale=1.0, device=device)
	steps = T

	print(" Training Stage 1 (learn dynamics)...")
	t0 = time.time()
	s1_local_iters = max(s1_iters, 5000) if "Kepler2D" in system.name else s1_iters
	f_model = train_stage1(trajs, dim=system.dim, steps=steps, dt=dt, iters=s1_local_iters, device=device)
	stage1_time = time.time() - t0
	rss1 = get_max_rss()
	print(f"  Stage 1 time: {stage1_time:.1f}s")

	with torch.no_grad():
		idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
		pred = rollout(f_model, trajs[idx,0,:], t0=0.0, dt=dt, steps=steps)
		mse = (pred - trajs[idx]).pow(2).mean().item()
		print(f"  [Eval] Stage 1 rollout MSE: {mse:.3e}")

	# keep last homogeneous coord fixed for SE2-LinearHom during Stage-2
	if "SE2-LinearHom" in system.name:
		class _ClampLast(nn.Module):
			def __init__(self, base): super().__init__(); self.base = base
			def forward(self, t, x):
				y = self.base(t, x)
				y = y.clone()
				y[..., -1] = 0.0
				return y
		f_model = _ClampLast(f_model)

	# Evaluation batch for restart selection (does not use expected basis)
	with torch.no_grad():
		idx_eval = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
		x0_eval = trajs[idx_eval, 0, :]
		xs_eval = rollout(f_model, x0_eval, t0=0.0, dt=dt, steps=steps).detach()

	eps_small = 1e-2
	# Group-specific ε
	if "SE2-LinearHom" in system.name:
		eps_small = 5e-3
	elif "SU3-Equivariant6D" in system.name:
		eps_small = 3e-3

	if "SE2-Unicycle" in system.name:
		print(f" Training Stage 2 (SE(2), {'agnostic' if use_se2_agnostic else 'family-conditioned'})...")
		t1 = time.time()
		if use_se2_agnostic:
			VF, loss_hist = discover_se2_basis_agnostic(
				f_model, trajs, dt, steps, iters=1800, lr=s2_lr,
				eps=eps_small, M=12, mult_eps=variant.mult_eps, device=device,
				early_stop=True, patience=400, min_delta_rel=1e-4
			)
			stage2_time = time.time() - t1
			rss2 = get_max_rss()

			with torch.no_grad():
				idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
				x0t = trajs[idx, 0, :]
				xs = rollout(f_model, x0t, t0=0.0, dt=dt, steps=steps).detach()
			Ls = []
			for _ in range(24):
				Ls.append(float(integral_equiv_loss_se2_agnostic_cached(xs, f_model, x0t, VF, dt, steps, eps_small, mult_eps=variant.mult_eps).cpu()))
			E_mean, E_std = float(np.mean(Ls)), float(np.std(Ls))
			L_per = [E_mean, E_mean, E_mean]
			print(f"  E[Lint] ~ {E_mean:.3e} ± {E_std:.3e}")

			with torch.no_grad():
				Bplot = min(5, trajs.shape[0])
				x0 = trajs[:Bplot,0,:]
				xs_base = rollout(f_model, x0, t0=0.0, dt=dt, steps=steps)
				xi = torch.randn(3, device=x0.device)
				Fprobe = torch.stack([VF[0](x0), VF[1](x0), VF[2](x0)], dim=0)
				_, Mmix = orthonormalize_field_values(Fprobe)
				def v_dir(x):
					u = xi / (xi.norm() + 1e-12)
					Fb = torch.stack([VF[0](x), VF[1](x), VF[2](x)], dim=0)
					D = torch.einsum("ij,jnd->ind", Mmix, Fb)
					return torch.einsum('i,ind->nd', u, D)
				x0_g = flow_of_field(v_dir, x0, eps_small, steps=8)
				xs_g0 = rollout(f_model, x0_g, t0=0.0, dt=dt, steps=steps)
				xs_g = flow_of_field(v_dir, xs_base.reshape(Bplot*(steps+1), -1), eps_small, steps=8).reshape(Bplot, steps+1, -1)
			fig, ax = plt.subplots(figsize=(6,5))
			for i in range(Bplot):
				ax.plot(to_cpu_np(xs_base[i,:,0]), to_cpu_np(xs_base[i,:,1]), 'b-', alpha=0.7, label='flow' if i==0 else "")
				ax.plot(to_cpu_np(xs_g0[i,:,0]), to_cpu_np(xs_g0[i,:,1]), 'r--', alpha=0.7, label='transform-then-flow' if i==0 else "")
				ax.plot(to_cpu_np(xs_g[i,:,0]),  to_cpu_np(xs_g[i,:,1]),  'g-.', alpha=0.7, label='flow-then-transform' if i==0 else "")
			ax.legend(); ax.set_title(f"{system.name} overlay (xy)"); ax.set_xlabel("x"); ax.set_ylabel("y")
			fig.tight_layout(); fig.savefig(outdir / f"{system.name.replace(' ','_')}_overlay_xy.png"); plt.close(fig)

			err_time = (xs_g0 - xs_g).pow(2).sum(dim=-1).mean(dim=0).cpu().numpy()
			tvec = (torch.arange(steps+1, dtype=torch.float32) * dt).cpu().numpy()
			plot_time_error(tvec, err_time, outdir / f"{system.name.replace(' ','_')}_time_err.png", f"{system.name} timewise equivariance error")

			sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]; vals = []
			for s in sigmas:
				noisy = trajs + (s * torch.randn_like(trajs) if s > 0 else 0.0)
				with torch.no_grad():
					x0n = noisy[:min(64, noisy.shape[0]), 0, :]
					xsn = rollout(f_model, x0n, t0=0.0, dt=dt, steps=steps).detach()
				tmp = []
				for _ in range(16):
					tmp.append(float(integral_equiv_loss_se2_agnostic_cached(xsn, f_model, x0n, VF, dt, steps, eps_small, mult_eps=variant.mult_eps).cpu()))
				vals.append(float(np.mean(tmp)))
			plot_noise_curve(vals, sigmas, outdir/"noise_curve.png", "Noise robustness (SE2)")

			eps_list = [5e-3, 1e-2, 2e-2, 5e-2]; raw, norm = [], []
			with torch.no_grad():
				x0e = trajs[:min(64, trajs.shape[0]), 0, :]; xse = rollout(f_model, x0e, t0=0.0, dt=dt, steps=steps).detach()
			for e in eps_list:
				tmp = []
				for _ in range(12):
					tmp.append(float(integral_equiv_loss_se2_agnostic_cached(xse, f_model, x0e, VF, dt, steps, e, mult_eps=(1.0,)).cpu()))
				m = float(np.mean(tmp)); raw.append(m); norm.append(m/(e*e))
			plot_eps_sweep_curve(eps_list, raw, norm, outdir, "ε-sweep (SE2)")

			dt_list = [0.005,0.01,0.02,0.05]; T_list = [60,120,200]
			heat = np.zeros((len(dt_list), len(T_list)), dtype=np.float64)
			for i, d in enumerate(dt_list):
				with torch.no_grad():
					x0s = trajs[:min(64, trajs.shape[0]), 0, :]
					xs2 = rollout(f_model, x0s, t0=0.0, dt=d, steps=T_list[-1]).detach()
				for j, Tsub in enumerate(T_list):
					tmp = []
					for _ in range(12):
						tmp.append(float(integral_equiv_loss_se2_agnostic_cached(xs2[:, :Tsub+1], f_model, x0s, VF, d, Tsub, eps_small, mult_eps=variant.mult_eps).cpu()))
					heat[i, j] = float(np.mean(tmp))
			fig, ax = plt.subplots(figsize=(5,4))
			im = ax.imshow(heat, origin='lower', aspect='auto', cmap='viridis',
			               extent=[T_list[0], T_list[-1], dt_list[0], dt_list[-1]])
			fig.colorbar(im, ax=ax, label="E[Lint]"); ax.set_xlabel("T"); ax.set_ylabel("dt"); ax.set_title("Sensitivity heatmap (SE2)")
			fig.tight_layout(); fig.savefig(outdir/"sensitivity_heatmap.png"); plt.close(fig)

			result = {
				"system": system.name, "dim": system.dim, "variant": variant.name, "seed": variant.seed,
				"type": "SE2-agnostic",
				"method": "integral",
				"twists": None,
				"per_basis_Lint": L_per,
				"E_Lint_mean": E_mean, "E_Lint_std": E_std,
				"mse_rollout": float(mse),
				"stage1_time_sec": float(stage1_time),
				"stage2_time_sec": float(stage2_time),
				"max_rss": {"stage1": rss1, "stage2": rss2},
				"params": {
					"B": int(B), "T": int(T), "dt": float(dt),
					"eps": float(eps_small), "mult_eps": list(variant.mult_eps),
					"M_samples": 12, "s2_iters": 1800, "s2_lr": float(s2_lr),
					"seed": int(variant.seed), "compile_used": False,
					"device": device, "threads": int(torch.get_num_threads()),
					"interop_threads": int(torch.get_num_interop_threads())
				},
				"stage1": {"iters_used": int(s1_local_iters), "lr": 1e-3, "dim": int(system.dim)},
				"loss_history": loss_hist,
				"se2_structure_constants": None,
				"baselines": {},
				"noise": {"sigmas": sigmas, "E_Lint": vals},
				"eps_sweep": {"eps": eps_list, "E_Lint": raw, "E_Lint_over_eps2": norm},
				"sensitivity_T_dt": {"dt": dt_list, "T": T_list, "E_Lint": heat.tolist()},
			}
			results_path = out_root / "integral_results.json"
			append_result(results_path, result)

			ys = np.array(loss_hist, dtype=float); xs = np.arange(1, len(ys)+1)
			fig, ax = plt.subplots(figsize=(6,4)); ax.plot(xs, ys, 'b-'); ax.set_yscale('log')
			ax.set_xlabel('iter'); ax.set_ylabel('E[Lint]'); ax.set_title(f"{system.name} Stage-2 convergence")
			fig.tight_layout(); fig.savefig(outdir/"convergence.png"); plt.close(fig)
			return
		else:
			raise NotImplementedError("Family-conditioned SE(2) disabled. Use agnostic path.")

	# Linear groups
	if system.expected_basis is not None:
		k = int(system.expected_basis.shape[0])
	else:
		if "SO3-Equivariant3D" in system.name:
			k = 3
		elif "SO4-Equivariant4D" in system.name:
			k = 6
		elif "SU3-Equivariant6D" in system.name:
			k = 8
		elif "SE2-LinearHom" in system.name:
			k = 3
		else:
			k = min(3, system.dim * (system.dim - 1) // 2)

	# Enforce skew for SO(n), but not for SU(3) (realified) or SE(2) homogeneous
	use_skew = not (("SU3-Equivariant6D" in system.name) or ("SE2-LinearHom" in system.name))

	# Group-specific Stage‑2 and mult‑ε overrides
	local_iters = 1800 if k > 1 else 1200
	local_M = 12 if k > 1 else 6
	local_lr = s2_lr
	local_mult_eps = variant.mult_eps
	if "SU3-Equivariant6D" in system.name:
		local_iters = 3000
		local_M = 16
		local_lr = 3e-3
		local_mult_eps = (0.75, 1.0, 1.5)
	if "SE2-LinearHom" in system.name:
		local_iters = 2200
		local_M = 12
		local_lr = 3e-3
		local_mult_eps = (0.5, 1.0, 1.5)

	# Optional projectors
	projector = None
	if "SE2-LinearHom" in system.name:
		projector = project_to_se2
	elif "SU3-Equivariant6D" in system.name:
		projector = project_to_su3_real

	# Restarts (selection by E[Lint] on eval batch; no use of expected basis)
	restarts = 2 if (("SU3-Equivariant6D" in system.name) or ("SE2-LinearHom" in system.name)) else 1
	best_B = None; best_loss_hist = None; best_E = float("inf")
	for r in range(restarts):
		set_seed(variant.seed + 100 * r)
		Bcand, loss_hist_cand = discover_linear_basis_joint(
			f_model, trajs, dt, steps, dim=system.dim, k=k,
			iters=local_iters, lr=local_lr, eps=eps_small, M=local_M,
			mult_eps=local_mult_eps, device=device, normalize_A=(not variant.no_normalize),
			use_skew=use_skew, projector=projector
		)
		with torch.no_grad():
			Ls_eval = []
			for _ in range(12):
				xi = torch.randn(k, device=x0_eval.device)
				Ls_eval.append(float(integral_equiv_loss_linear_cached(
					xs_eval, f_model, x0_eval, xi, Bcand, dt, steps, eps_small, mult_eps=local_mult_eps,
					normalize_A=(not variant.no_normalize)
				).detach().cpu()))
			E_eval = float(np.mean(Ls_eval))
		if E_eval < best_E:
			best_E = E_eval; best_B = Bcand; best_loss_hist = loss_hist_cand

	Bjoint = best_B; loss_hist = best_loss_hist

	stage2_time = time.time() - t0
	rss2 = get_max_rss()

	with torch.no_grad():
		idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
		x0t = trajs[idx, 0, :]
		xs = rollout(f_model, x0t, t0=0.0, dt=dt, steps=steps).detach()
		Ls = []
		for _ in range(24):
			xi = torch.randn(k, device=x0t.device)
			Ls.append(float(integral_equiv_loss_linear_cached(xs, f_model, x0t, xi, Bjoint, dt, steps, eps_small, mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)).detach().cpu()))
		E_mean, E_std = float(np.mean(Ls)), float(np.std(Ls))
		L_per = []
		for i in range(k):
			xi = torch.zeros(k, device=x0t.device); xi[i] = 1.0
			L_per.append(float(integral_equiv_loss_linear_cached(xs, f_model, x0t, xi, Bjoint, dt, steps, eps_small, mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)).detach().cpu()))
	print(f"  Per-basis Lint: {['%.3e'%v for v in L_per]} | E[Lint] ~ {E_mean:.3e} ± {E_std:.3e}")

	with torch.no_grad():
		Bplot = min(5, trajs.shape[0])
		x0 = trajs[:Bplot,0,:]
		xs_base = rollout(f_model, x0, t0=0.0, dt=dt, steps=steps)
		g = matrix_exp(eps_small * Bjoint[0])
		x0_g = x0 @ g.T
		xs_g0 = rollout(f_model, x0_g, t0=0.0, dt=dt, steps=steps)
		xs_g = xs_base @ g.T

	fig, ax = plt.subplots(figsize=(6,5))
	for i in range(Bplot):
		ax.plot(to_cpu_np(xs_base[i,:,0]), to_cpu_np(xs_base[i,:,1]), 'b-', alpha=0.7, label='flow' if i==0 else "")
		ax.plot(to_cpu_np(xs_g0[i,:,0]),   to_cpu_np(xs_g0[i,:,1]),   'r--', alpha=0.7, label='transform-then-flow' if i==0 else "")
		ax.plot(to_cpu_np(xs_g[i,:,0]),    to_cpu_np(xs_g[i,:,1]),    'g-.', alpha=0.7, label='flow-then-transform' if i==0 else "")
	ax.legend(); ax.set_title(f"{system.name} overlay"); ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1")
	fig.tight_layout(); fig.savefig(outdir / f"{system.name.replace(' ','_')}_overlay.png"); plt.close(fig)

	err_time_lin = (xs_g0 - xs_g).pow(2).sum(dim=-1).mean(dim=0).cpu().numpy()
	tvec = (torch.arange(steps+1, dtype=torch.float32) * dt).cpu().numpy()
	plot_time_error(tvec, err_time_lin, outdir / f"{system.name.replace(' ','_')}_time_err.png", f"{system.name} timewise equivariance error")

	lin_struct = structure_constants_and_killing(Bjoint)
	fig, ax = plt.subplots(figsize=(4,4))
	im = ax.imshow(np.array(lin_struct["closure_residuals"]), cmap='magma')
	fig.colorbar(im, ax=ax, label="closure residual (Fro)"); ax.set_title("Linear closure residuals")
	fig.tight_layout(); fig.savefig(outdir/"closure_residuals_heatmap.png"); plt.close(fig)
	evals = np.array(lin_struct["killing_eigs"])
	fig, ax = plt.subplots(figsize=(5,3)); ax.bar(list(range(1,len(evals)+1)), evals); ax.axhline(0., color='k', linewidth=0.8)
	ax.set_title("Linear Killing eigenvalues"); fig.tight_layout(); fig.savefig(outdir/"killing_eigs_bar.png"); plt.close(fig)

	sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]; vals = []
	with torch.no_grad():
		x0n = trajs[:min(64, trajs.shape[0]), 0, :]
		xsn = rollout(f_model, x0n, t0=0.0, dt=dt, steps=steps).detach()
	for s in sigmas:
		noisy = trajs + (s * torch.randn_like(trajs) if s > 0 else 0.0)
		with torch.no_grad():
			x0s = noisy[:min(64, noisy.shape[0]), 0, :]
			xs2 = rollout(f_model, x0s, t0=0.0, dt=dt, steps=steps).detach()
		tmp = []
		for _ in range(16):
			xi = torch.randn(k, device=x0s.device)
			tmp.append(float(integral_equiv_loss_linear_cached(xs2, f_model, x0s, xi, Bjoint, dt, steps, eps_small, mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)).cpu()))
		vals.append(float(np.mean(tmp)))
	plot_noise_curve(vals, sigmas, outdir/"noise_curve.png", "Noise robustness (linear)")

	def expected_loss_linear_subset(Basis, trajs, Bsub, Tsub):
		x0s = trajs[:Bsub, 0, :]
		with torch.no_grad():
			xs3 = rollout(f_model, x0s, t0=0.0, dt=dt, steps=Tsub).detach()
		tmp = []
		for _ in range(12):
			xi = torch.randn(k, device=x0s.device)
			tmp.append(float(integral_equiv_loss_linear_cached(xs3, f_model, x0s, xi, Basis, dt, Tsub, eps_small, mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)).cpu()))
		return float(np.mean(tmp))
	B_list = [16, 32, 64, min(128, trajs.shape[0])]; T_list = [max(20, steps//3), max(40, steps//2), steps]
	vals_B = [expected_loss_linear_subset(Bjoint, trajs, b, steps) for b in B_list]
	vals_T = [expected_loss_linear_subset(Bjoint, trajs, min(64, trajs.shape[0]), t) for t in T_list]
	fig, ax = plt.subplots(figsize=(6,4)); ax.plot(B_list, vals_B, 'bo-'); set_log_ylim(ax, np.array(vals_B), 1e-12, 1e-3)
	ax.set_xlabel("B"); ax.set_ylabel("E[Lint]"); ax.set_title("Data-efficiency vs B (linear)"); fig.tight_layout(); fig.savefig(outdir/"data_efficiency_B.png"); plt.close(fig)
	fig, ax = plt.subplots(figsize=(6,4)); ax.plot(T_list, vals_T, 'ro-'); set_log_ylim(ax, np.array(vals_T), 1e-12, 1e-3)
	ax.set_xlabel("T"); ax.set_ylabel("E[Lint]"); ax.set_title("Data-efficiency vs T (linear)"); fig.tight_layout(); fig.savefig(outdir/"data_efficiency_T.png"); plt.close(fig)

	eps_list = [5e-3, 1e-2, 2e-2, 5e-2]; raw, norm = [], []
	with torch.no_grad():
		x0e = trajs[:min(64, trajs.shape[0]), 0, :]; xse = rollout(f_model, x0e, t0=0.0, dt=dt, steps=steps).detach()
	for e in eps_list:
		tmp = []
		for _ in range(12):
			xi = torch.randn(k, device=x0e.device)
			tmp.append(float(integral_equiv_loss_linear_cached(xse, f_model, x0e, xi, Bjoint, dt, steps, e, mult_eps=(1.0,), normalize_A=(not variant.no_normalize)).detach().cpu()))
		m = float(np.mean(tmp)); raw.append(m); norm.append(m/(e*e))
	plot_eps_sweep_curve(eps_list, raw, norm, outdir, "ε-sweep (linear)")

	if ("SO3-Equivariant3D (full 3D)" in system.name) and (variant.name == "base"):
		sigmas_cert = [0.0, 0.05, 0.10, 0.15]
		cert_max = {"integral": [], "pointwise": [], "infinitesimal": []}
		for s in sigmas_cert:
			noisy = trajs + (s * torch.randn_like(trajs) if s > 0 else 0.0)
			Bi, _ = discover_linear_basis_joint(
				f_model, noisy, dt, steps, dim=system.dim, k=k,
				iters=1000 if k>1 else 800, lr=s2_lr, eps=eps_small, M=8 if k>1 else 6,
				mult_eps=variant.mult_eps, device=device, normalize_A=True, use_skew=True
			)
			ai, _, _ = principal_angles_and_alignment(system.expected_basis, Bi)
			cert_max["integral"].append(float(np.max(ai)) if ai is not None else float("nan"))
			Bp, _ = discover_linear_basis_pointwise(
				f_model, noisy, dt, steps, dim=system.dim, k=k,
				iters=700 if k>1 else 600, lr=s2_lr, eps=eps_small, M=6,
				mult_eps=variant.mult_eps, device=device, at_final_only=variant.pointwise_final_only,
				normalize_A=True, early_stop=True, patience=250, min_delta_rel=1e-4, use_skew=True
			)
			ap, _, _ = principal_angles_and_alignment(system.expected_basis, Bp)
			cert_max["pointwise"].append(float(np.max(ap)) if ap is not None else float("nan"))
			Bf, _ = discover_linear_basis_infinitesimal_fd(
				f_model, noisy, dt, steps, dim=system.dim, k=k,
				iters=700 if k>1 else 600, lr=s2_lr, h_fd=1e-3, M=6, device=device,
				normalize_A=True, early_stop=True, patience=250, min_delta_rel=1e-4, use_skew=True
			)
			af, _, _ = principal_angles_and_alignment(system.expected_basis, Bf)
			cert_max["infinitesimal"].append(float(np.max(af)) if af is not None else float("nan"))
		fig, ax = plt.subplots(figsize=(6,4))
		ax.plot(sigmas_cert, cert_max["integral"], 'b-o', label="Integral (Ours)")
		ax.plot(sigmas_cert, cert_max["pointwise"], 'r--s', label="Pointwise")
		ax.plot(sigmas_cert, cert_max["infinitesimal"], 'g:^', label="Infinitesimal")
		ax.set_xlabel("noise sigma"); ax.set_ylabel("Max principal angle (deg)")
		ax.grid(True, which='both', alpha=0.3); ax.legend()
		ax.set_title("SO(3) noise robustness (certificate)")
		fig.tight_layout(); fig.savefig(outdir/"so3_noise_certificate.png"); plt.close(fig)

	angles_deg, Omap, B_aligned = principal_angles_and_alignment(system.expected_basis, Bjoint)
	if angles_deg is not None:
		plot_principal_angles_bar(angles_deg, outdir/"principal_angles_bar.png", "Principal angles (expected vs discovered)")
		if Omap is not None:
			plot_coefficients_heatmap(Omap, outdir/"coefficients_heatmap.png", "Coefficients (expected → discovered)")

	# Basis grids for side-by-side visual comparison
	if ("SO4-Equivariant4D" in system.name) or ("SU3-Equivariant6D" in system.name) or ("SE2-LinearHom" in system.name):
		plot_basis_grid(system.expected_basis, outdir/"basis_expected_grid.png", f"{system.name}: expected basis", vmax=1.0)
		plot_basis_grid(Bjoint, outdir/"basis_discovered_grid.png", f"{system.name}: discovered basis", vmax=1.0)

	result = {
		"system": system.name,
		"dim": system.dim,
		"variant": variant.name,
		"seed": variant.seed,
		"type": "linear",
		"method": "integral",
		"k_discovered": int(Bjoint.shape[0]),
		"basis_discovered": Bjoint.detach().cpu().tolist(),
		"per_basis_Lint": L_per,
		"E_Lint_mean": E_mean,
		"E_Lint_std": E_std,
		"mse_rollout": float(mse),
		"stage1_time_sec": float(stage1_time),
		"stage2_time_sec": float(stage2_time),
		"max_rss": {"stage1": rss1, "stage2": rss2},
		"expected_basis_dim": int(system.expected_basis.shape[0]) if system.expected_basis is not None else None,
		"coefficients_expected_to_discovered": Omap.tolist() if Omap is not None else None,
		"principal_angles_deg": angles_deg.tolist() if angles_deg is not None else None,
		"alignment_alpha_1d": None,
		"params": {
			"B": int(B), "T": int(T), "dt": float(dt),
			"eps": float(eps_small), "mult_eps": list(local_mult_eps),
			"M_samples": int(local_M),
			"s2_iters": int(local_iters), "s2_lr": float(local_lr),
			"k_requested": int(k), "seed": int(variant.seed), "compile_used": False,
			"device": device, "threads": int(torch.get_num_threads()),
			"interop_threads": int(torch.get_num_interop_threads()),
			"normalize_A": (not variant.no_normalize),
			"use_skew": use_skew
		},
		"stage1": {"iters_used": int(s1_local_iters), "lr": 1e-3, "dim": int(system.dim)},
		"loss_history": loss_hist,
		"structure_constants": lin_struct,
		"baselines": {},
		"noise": {"sigmas": sigmas, "E_Lint": vals},
		"data_efficiency": {"B": B_list, "E_Lint_B": vals_B, "T": T_list, "E_Lint_T": vals_T},
		"eps_sweep": {"eps": eps_list, "E_Lint": raw, "E_Lint_over_eps2": norm},
	}
	results_path = out_root / "integral_results.json"
	append_result(results_path, result)

	ys = np.array(loss_hist, dtype=float); xs = np.arange(1, len(ys)+1)
	fig, ax = plt.subplots(figsize=(6,4)); ax.plot(xs, ys, 'b-'); ax.set_yscale('log')
	ax.set_xlabel('iter'); ax.set_ylabel('E[Lint]'); ax.set_title(f"{system.name} Stage-2 convergence")
	fig.tight_layout(); fig.savefig(outdir/"convergence.png"); plt.close(fig)

	print(" Running pointwise baseline...")
	t1p = time.time()
	Bpt, loss_hist_pt = discover_linear_basis_pointwise(
		f_model, trajs, dt, steps, dim=system.dim, k=k,
		iters=800 if k>1 else 600, lr=s2_lr, eps=eps_small, M=6,
		mult_eps=local_mult_eps, device=device, at_final_only=variant.pointwise_final_only,
		normalize_A=(not variant.no_normalize),
		early_stop=True, patience=300, min_delta_rel=1e-4,
		use_skew=use_skew
	)
	stage2_pt_time = time.time() - t1p
	with torch.no_grad():
		idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
		x0t = trajs[idx, 0, :]
		xs = rollout(f_model, x0t, t0=0.0, dt=dt, steps=steps).detach()
		Ls = []
		for _ in range(16):
			xi = torch.randn(k, device=x0t.device)
			Ls.append(float(integral_equiv_loss_linear_cached(xs, f_model, x0t, xi, Bpt, dt, steps, eps_small, mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)).detach().cpu()))
	Ep_mean, Ep_std = float(np.mean(Ls)), float(np.std(Ls))
	print(f"  [Pointwise] E[Lint] ~ {Ep_mean:.3e} ± {Ep_std:.3e}")

	angles_deg_pt, Omap_pt, _ = principal_angles_and_alignment(system.expected_basis, Bpt)
	rss2_pt = get_max_rss()

	sigmas_pt = [0.0, 0.005, 0.01, 0.02, 0.05]; vals_pt = []
	with torch.no_grad():
		x0b = trajs[:min(64, trajs.shape[0]), 0, :]
		xsb = rollout(f_model, x0b, t0=0.0, dt=dt, steps=steps).detach()
	for s in sigmas_pt:
		noisy = trajs + (s * torch.randn_like(trajs) if s > 0 else 0.0)
		with torch.no_grad():
			x0s = noisy[:min(64, noisy.shape[0]), 0, :]
			xs2 = rollout(f_model, x0s, t0=0.0, dt=dt, steps=steps).detach()
		tmp = []
		for _ in range(16):
			xi = torch.randn(k, device=x0s.device)
			tmp.append(float(integral_equiv_loss_linear_cached(
				xs2, f_model, x0s, xi, Bpt, dt, steps, eps_small,
				mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)
			).cpu()))
		vals_pt.append(float(np.mean(tmp)))

	result_pt = {
		"system": system.name, "dim": system.dim, "variant": variant.name, "seed": variant.seed,
		"type": "linear", "method": "pointwise",
		"k_discovered": int(Bpt.shape[0]),
		"basis_discovered": Bpt.detach().cpu().tolist(),
		"E_Lint_mean": Ep_mean, "E_Lint_std": Ep_std,
		"mse_rollout": float(mse),
		"stage1_time_sec": float(stage1_time),
		"stage2_time_sec": float(stage2_pt_time),
		"principal_angles_deg": angles_deg_pt.tolist() if angles_deg_pt is not None else None,
		"coefficients_expected_to_discovered": Omap_pt.tolist() if Omap_pt is not None else None,
		"max_rss": {"stage1": rss1, "stage2": rss2_pt},
		"params": { "B": int(B), "T": int(T), "dt": float(dt),
		            "eps": float(eps_small), "mult_eps": list(local_mult_eps),
		            "s2_iters": 800 if k>1 else 600, "s2_lr": float(s2_lr),
		            "k_requested": int(k), "seed": int(variant.seed), "compile_used": False,
		            "normalize_A": (not variant.no_normalize),
		            "use_skew": use_skew },
		"loss_history": loss_hist_pt,
		"structure_constants": structure_constants_and_killing(Bpt),
		"noise": {"sigmas": sigmas_pt, "E_Lint": vals_pt}
	}
	append_result(results_path, result_pt)

	print(" Running infinitesimal (FD bracket) baseline...")
	t1i = time.time()
	Binf, loss_hist_inf = discover_linear_basis_infinitesimal_fd(
		f_model, trajs, dt, steps, dim=system.dim, k=k,
		iters=800 if k>1 else 600, lr=s2_lr, h_fd=1e-3, M=6, device=device, normalize_A=(not variant.no_normalize),
		early_stop=True, patience=300, min_delta_rel=1e-4, use_skew=use_skew
	)
	stage2_inf_time = time.time() - t1i
	with torch.no_grad():
		idx = torch.randint(0, trajs.shape[0], (min(64, trajs.shape[0]),), device=device)
		x0t = trajs[idx, 0, :]
		xs = rollout(f_model, x0t, t0=0.0, dt=dt, steps=steps).detach()
		Ls = []
		for _ in range(16):
			xi = torch.randn(k, device=x0t.device)
			Ls.append(float(integral_equiv_loss_linear_cached(xs, f_model, x0t, xi, Binf, dt, steps, eps_small, mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)).detach().cpu()))
	Ei_mean, Ei_std = float(np.mean(Ls)), float(np.std(Ls))
	print(f"  [Inf-FD] E[Lint] ~ {Ei_mean:.3e} ± {Ei_std:.3e}")
	lin_struct_inf = structure_constants_and_killing(Binf)

	angles_deg_inf, Omap_inf, _ = principal_angles_and_alignment(system.expected_basis, Binf)
	rss2_inf = get_max_rss()

	sigmas_inf = [0.0, 0.005, 0.01, 0.02, 0.05]; vals_inf = []
	with torch.no_grad():
		x0b = trajs[:min(64, trajs.shape[0]), 0, :]
		xsb = rollout(f_model, x0b, t0=0.0, dt=dt, steps=steps).detach()
	for s in sigmas_inf:
		noisy = trajs + (s * torch.randn_like(trajs) if s > 0 else 0.0)
		with torch.no_grad():
			x0s = noisy[:min(64, noisy.shape[0]), 0, :]
			xs2 = rollout(f_model, x0s, t0=0.0, dt=dt, steps=steps).detach()
		tmp = []
		for _ in range(16):
			xi = torch.randn(k, device=x0s.device)
			tmp.append(float(integral_equiv_loss_linear_cached(
				xs2, f_model, x0s, xi, Binf, dt, steps, eps_small,
				mult_eps=local_mult_eps, normalize_A=(not variant.no_normalize)
			).cpu()))
		vals_inf.append(float(np.mean(tmp)))

	result_inf = {
		"system": system.name, "dim": system.dim, "variant": variant.name, "seed": variant.seed,
		"type": "linear", "method": "infinitesimal",
		"k_discovered": int(Binf.shape[0]),
		"basis_discovered": Binf.detach().cpu().tolist(),
		"E_Lint_mean": Ei_mean, "E_Lint_std": Ei_std,
		"mse_rollout": float(mse),
		"stage1_time_sec": float(stage1_time),
		"stage2_time_sec": float(stage2_inf_time),
		"principal_angles_deg": angles_deg_inf.tolist() if angles_deg_inf is not None else None,
		"coefficients_expected_to_discovered": Omap_inf.tolist() if Omap_inf is not None else None,
		"max_rss": {"stage1": rss1, "stage2": rss2_inf},
		"params": { "B": int(B), "T": int(T), "dt": float(dt),
		            "eps": float(eps_small), "mult_eps": list(local_mult_eps),
		            "s2_iters": 800 if k>1 else 600, "s2_lr": float(s2_lr),
		            "k_requested": int(k), "seed": int(variant.seed), "compile_used": False,
		            "normalize_A": (not variant.no_normalize),
		            "use_skew": use_skew },
		"loss_history": loss_hist_inf,
		"structure_constants": lin_struct_inf,
		"baselines": {},
		"noise": {"sigmas": sigmas_inf, "E_Lint": vals_inf}
	}
	append_result(results_path, result_inf)

# ============ Table writers ============
def _summarize_integral(entries: List[Dict[str,Any]]) -> Dict[str, Dict[str, Any]]:
	summary: Dict[str, Dict[str, Any]] = {}
	for e in entries:
		if e.get("method") != "integral": continue
		sys = e.get("system"); var = e.get("variant","base")
		if var != "base": continue
		item = {
			"E_Lint_mean": e.get("E_Lint_mean"),
			"E_Lint_std": e.get("E_Lint_std"),
			"principal_angles_deg": e.get("principal_angles_deg"),
			"closure_max": None,
			"mse_rollout": e.get("mse_rollout"),
			"stage2_time_sec": e.get("stage2_time_sec"),
		}
		sc = e.get("structure_constants")
		if sc and "closure_residuals" in sc:
			crm = np.array(sc["closure_residuals"])
			item["closure_max"] = float(np.max(crm))
		summary[sys] = item
	return summary

def _lookup_method(entries: List[Dict[str,Any]], system: str, method: str, variant: str="base") -> Optional[Dict[str,Any]]:
	for e in entries:
		if e.get("system")==system and e.get("method")==method and e.get("variant")==variant:
			return e
	return None

def write_main_results_table(entries: List[Dict[str,Any]], out_dir: Path):
	data = _summarize_integral(entries)
	lines = [
		"\\begin{table}[h]",
		"\\centering",
		"\\caption{Main results across systems (integral, base variant).}",
		"\\label{tab:main_results}",
		"\\small",
		"\\begin{tabular}{lcccc}",
		"\\toprule",
		"System & $\\bar{\\mathcal{L}}_{\\text{int}}$ & Max Principal Angle (\\degree) & Max Closure Residual & Stage-2 Time (s) \\\\",
		"\\midrule"
	]
	for sys in sorted(set([e.get("system") for e in entries if e.get("method")=="integral"])):
		e = _lookup_method(entries, sys, "integral", "base")
		if e is None: continue
		angles = e.get("principal_angles_deg")
		ang_str = f"{max(angles):.2f}" if angles else "--"
		sc = e.get("structure_constants")
		clo = None
		if sc and "closure_residuals" in sc:
			clo = float(np.max(np.array(sc["closure_residuals"])))
		clo_str = f"{clo:.2e}" if clo is not None else "--"
		lines.append(f"{sys} & {e.get('E_Lint_mean'):.2e} $\\pm$ {e.get('E_Lint_std'):.1e} & {ang_str} & {clo_str} & {e.get('stage2_time_sec'):.1f} \\\\")
	lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
	(out_dir / "main_results_table.tex").write_text("\n".join(lines))

def write_baselines_table(entries: List[Dict[str,Any]], out_dir: Path):
	systems = sorted(set([e.get("system") for e in entries if e.get("method")=="integral"]))
	lines = [
		"\\begin{table}[h]",
		"\\centering",
		"\\caption{Integral vs pointwise vs infinitesimal baselines (base variant).}",
		"\\label{tab:baselines}",
		"\\small",
		"\\begin{tabular}{lccc}",
		"\\toprule",
		"System & Integral $\\bar{\\mathcal{L}}_{\\text{int}}$ & Pointwise $\\bar{\\mathcal{L}}_{\\text{int}}$ & Infinitesimal $\\bar{\\mathcal{L}}_{\\text{int}}$ \\\\",
		"\\midrule"
	]
	for sys in systems:
		ei = _lookup_method(entries, sys, "integral", "base")
		ep = _lookup_method(entries, sys, "pointwise", "base")
		ef = _lookup_method(entries, sys, "infinitesimal", "base")
		if ei is None: continue
		ip = f"{ep.get('E_Lint_mean'):.2e}" if ep else "--"
		ifd = f"{ef.get('E_Lint_mean'):.2e}" if ef else "--"
		lines.append(f"{sys} & {ei.get('E_Lint_mean'):.2e} & {ip} & {ifd} \\\\")
	lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
	(out_dir / "baselines_table.tex").write_text("\n".join(lines))

def write_ablations_table(entries: List[Dict[str,Any]], out_dir: Path):
	vars_of_interest = ["base", "no_mult_eps", "no_normalize", "short_T", "large_dt"]
	systems = sorted(set([e.get("system") for e in entries if e.get("method")=="integral"]))
	# Only SO(3) ablations
	systems = [s for s in systems if "SO3-Equivariant3D" in s]
	lines = [
		"\\begin{table}[h]",
		"\\centering",
		"\\caption{Ablations (SO(3) only, integral). Missing entries shown as --.}",
		"\\label{tab:ablations}",
		"\\small",
		"\\begin{tabular}{lccccc}",
		"\\toprule",
		"System & base & no-mult-$\\varepsilon$ & no-normalize & short-$T$ & large-$\\Delta t$ \\\\",
		"\\midrule"
	]
	for sys in systems:
		row = [sys]
		for vname in vars_of_interest:
			e = _lookup_method(entries, sys, "integral", vname)
			if e is None:
				row.append("--")
			else:
				row.append(f"{e.get('E_Lint_mean'):.2e}")
		lines.append(" & ".join(row) + " \\\\")
	lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
	(out_dir / "ablations_table.tex").write_text("\n".join(lines))

def write_ablations_per_system(entries: List[Dict[str,Any]], out_dir: Path):
	vars_of_interest = ["base", "no_mult_eps", "no_normalize", "short_T", "large_dt"]
	systems = sorted(set([e.get("system") for e in entries if e.get("method")=="integral"]))
	# Only SO(3) per-system ablations
	systems = [s for s in systems if "SO3-Equivariant3D" in s]
	for sys in systems:
		lines = [
			"\\begin{table}[h]",
			"\\centering",
			f"\\caption{{Ablations for {sys} (integral).}}",
			f"\\label{{tab:abl_{safe_fname(sys)}}}",
			"\\small",
			"\\begin{tabular}{lc}",
			"\\toprule",
			"Variant & $\\bar{\\mathcal{L}}_{\\text{int}}$ \\\\",
			"\\midrule"
		]
		for vname in vars_of_interest:
			e = _lookup_method(entries, sys, "integral", vname)
			val = f"{e.get('E_Lint_mean'):.2e}" if e else "--"
			lines.append(f"{vname} & {val} \\\\")
		lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
		(out_dir / f"ablations_{safe_fname(sys)}.tex").write_text("\n".join(lines))

# ============ Main ============
def main():
	device = get_device()
	print(f"Using device: {device}")
	out_root = Path("results/paper_integral_full"); ensure_dir(out_root)
	out_tables = out_root / "heatmaps"; ensure_dir(out_tables)

	systems = [
		# make_harmonic2d(omega=1.5, device=device),
		# make_nonlinear_rot2d(omega0=0.8, beta=0.4, device=device),
		# make_so3_equivariant3d(scale=0.7, device=device),
		# make_so4_equivariant4d(scale=0.5, device=device),
		make_su3_equivariant6d(scale=0.5, device=device),
		make_se2_linear(scale=0.7, device=device),
		# make_se2_unicycle(v=1.0, omega=0.2, device=device),
	]

	var_base      = Variant(name="base",          mult_eps=(0.5,1.0,2.0), seed=42, no_normalize=False)
	var_no_eps    = Variant(name="no_mult_eps",   mult_eps=(1.0,),        seed=42, no_normalize=False)
	var_no_norm   = Variant(name="no_normalize",  mult_eps=(0.5,1.0,2.0), seed=42, no_normalize=True)
	var_shortT    = Variant(name="short_T",       mult_eps=(0.5,1.0,2.0), seed=42, T_override=20)
	var_large_dt  = Variant(name="large_dt",      mult_eps=(0.5,1.0,2.0), seed=42, dt_override=0.1)

	variants_linear = [var_base, var_no_eps, var_no_norm, var_shortT, var_large_dt]
	variants_se2    = [var_base]

	for sys in systems:
		if "SE2-Unicycle" in sys.name:
			for var in variants_se2:
				run_experiment(sys, device=device, out_root=out_root, variant=var, B=128, T=120, dt=0.02, s1_iters=800, s2_lr=5e-3, use_se2_agnostic=True)
		elif "SO3-Equivariant3D" in sys.name:
			# SO(3): run all ablations
			for var in variants_linear:
				run_experiment(sys, device=device, out_root=out_root, variant=var, B=128, T=120, dt=0.02, s1_iters=800, s2_lr=5e-3, use_se2_agnostic=True)
		else:
			# All other linear groups: base only
			run_experiment(sys, device=device, out_root=out_root, variant=var_base, B=128, T=120, dt=0.02, s1_iters=800, s2_lr=5e-3, use_se2_agnostic=True)

	results_path = out_root / "integral_results.json"
	try:
		entries = json.loads(results_path.read_text())
	except Exception:
		entries = []
	write_main_results_table(entries, out_tables)
	write_baselines_table(entries, out_tables)
	write_ablations_table(entries, out_tables)
	write_ablations_per_system(entries, out_tables)
	print(f"Tables written to {out_tables}")

if __name__ == "__main__":
	main()