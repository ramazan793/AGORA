"""
CPU Benchmark for SVD+MLP deformation inference.

Benchmarks the CPU-runnable components of the SVD+MLP pipeline separately:
  1. MLP forward: [W(512), flame_55] → 4K PCA coefficients
  2. Linear combination: coeff @ U_basis + base for all 4 attributes
  3. Geometric constraints: clamp scale, normalize rotation, clamp opacity

3DGS rasterization (gsplat, requires CUDA) is excluded.
"""

from dataclasses import dataclass
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)

from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR

FFHQ_FUSED_PARAMS = f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy'

ATTR_KEYS = ['xyz', 'scale', 'rotation', 'opacity']


# ─────────────────────────────────────── MLP (must match training) ────────────

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int = 567, hidden1: int = 256, hidden2: int = 512,
                 out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ────────────────────────────────────────────────────── args ──────────────────

import tyro

@dataclass
class Args:
    svd_data_path: str = ''       # path to svd_basis.pt
    mlp_path: str = ''            # path to mlp_regressor.pt
    n_frames: int = 1000          # number of frames to benchmark
    batch_size: int = 1           # batch size per inference call (1 = real-time)
    warmup: int = 50              # warmup iterations before timing
    opacity_overshoot: float = 0.0  # opacity_overshoot value from model config


# ──────────────────────────────────────────────── benchmark helpers ────────────

def load_mlp_inputs(svd_data: dict, n_frames: int) -> torch.Tensor:
    """
    Load or cycle 567-dim MLP inputs [W(512), flame_55] to fill n_frames.
    Uses the first w_code from svd_data as a representative identity latent.
    """
    if 'flame_inputs' in svd_data:
        flame_base = svd_data['flame_inputs']          # [N, 55]
    else:
        ffhq_data = np.load(FFHQ_FUSED_PARAMS)        # [N, 364]
        exp    = torch.tensor(ffhq_data[:, 300:350], dtype=torch.float32)
        jaw    = torch.tensor(ffhq_data[:, 353:356], dtype=torch.float32)
        eyelid = torch.tensor(ffhq_data[:, 356:358], dtype=torch.float32)
        flame_base = torch.cat([exp, jaw, eyelid], dim=-1)  # [N, 55]

    # Use first w_code from dataset as representative identity latent
    if 'w_codes' in svd_data:
        w_code = svd_data['w_codes'][0:1].float()     # [1, 512]
    else:
        w_code = torch.zeros(1, 512)                  # fallback: zero w_code

    # Cycle flame inputs to fill n_frames
    repeats = (n_frames + len(flame_base) - 1) // len(flame_base)
    flame_all = flame_base.repeat(repeats, 1)[:n_frames].float()  # [n_frames, 55]
    w_all = w_code.expand(n_frames, -1)                           # [n_frames, 512]
    return torch.cat([w_all, flame_all], dim=-1)                  # [n_frames, 567]


def _stats(times: list) -> dict:
    arr = np.array(times) * 1000.0   # convert to ms
    return {
        'mean':   float(np.mean(arr)),
        'std':    float(np.std(arr)),
        'min':    float(np.min(arr)),
        'max':    float(np.max(arr)),
        'median': float(np.median(arr)),
    }


def print_table(results: dict, batch_size: int, n_frames: int):
    header = f"{'Component':<22}  {'mean ms':>9}  {'std ms':>9}  {'min ms':>9}  {'median ms':>10}  {'max ms':>9}"
    sep    = '-' * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, s in results.items():
        print(f"{name:<22}  {s['mean']:>9.3f}  {s['std']:>9.3f}  {s['min']:>9.3f}  {s['median']:>10.3f}  {s['max']:>9.3f}")
    print(sep)

    total_mean = results['Total (sum)']['mean']
    fps = 1000.0 / total_mean / batch_size if total_mean > 0 else float('inf')
    print(f"\nBatch size    : {batch_size}")
    print(f"Total mean    : {total_mean:.3f} ms / batch")
    print(f"Throughput    : {fps:.1f} frames/s  (CPU-only components, batch_size={batch_size})")
    print()


# ──────────────────────────────────────────────────────────── main ─────────────

def main(args: Args):
    assert args.svd_data_path, "Provide --svd-data-path"
    assert args.mlp_path,      "Provide --mlp-path"

    device = torch.device('cpu')

    # ── Load SVD data ──────────────────────────────────────────────────────────
    print(f"Loading SVD data from {args.svd_data_path} ...")
    svd_data = torch.load(args.svd_data_path, map_location='cpu')
    K     = int(svd_data['U_xyz'].shape[0])
    G_num = int(svd_data['G'])
    print(f"  K={K}, G={G_num}")
    for k in ['U_xyz', 'U_scale', 'U_rotation', 'U_opacity']:
        if k in svd_data:
            print(f"  {k}: {tuple(svd_data[k].shape)}")

    # Pre-move SVD tensors to CPU (already there, but explicit)
    # v2 has no base_* keys — use zero bases as stand-in for timing purposes
    Us = {k: svd_data[f'U_{k}'] for k in ATTR_KEYS}     # [K, G*C]
    attr_shapes = {'xyz': (G_num, 3), 'scale': (G_num, 3),
                   'rotation': (G_num, 4), 'opacity': (G_num, 1)}
    bases = {k: torch.zeros(1, *attr_shapes[k]) for k in ATTR_KEYS}   # [1, G, C]

    # ── Load MLP ──────────────────────────────────────────────────────────────
    print(f"Loading MLP from {args.mlp_path} ...")
    mlp_ckpt = torch.load(args.mlp_path, map_location='cpu')
    mlp = MLPRegressor(
        in_dim=mlp_ckpt['in_dim'],
        hidden1=mlp_ckpt['hidden1'],
        hidden2=mlp_ckpt['hidden2'],
        out_dim=mlp_ckpt['out_dim'],
    )
    mlp.load_state_dict(mlp_ckpt['state_dict'])
    mlp.eval()
    print(f"  MLP: {mlp_ckpt['in_dim']}→{mlp_ckpt['hidden1']}→{mlp_ckpt['hidden2']}"
          f"→{mlp_ckpt['out_dim']}  (K={K})")

    # ── Prepare inputs ────────────────────────────────────────────────────────
    all_inputs = load_mlp_inputs(svd_data, args.n_frames + args.warmup)
    print(f"  MLP inputs shape: {tuple(all_inputs.shape)}")

    ov = args.opacity_overshoot

    # ── Benchmark loop ────────────────────────────────────────────────────────
    t_mlp   = []
    t_lincomb = []
    t_constraint = []

    total_iters = args.warmup + args.n_frames
    n_batches   = (total_iters + args.batch_size - 1) // args.batch_size

    print(f"\nRunning {args.warmup} warmup + {args.n_frames} timed iterations "
          f"(batch_size={args.batch_size}) ...")

    frame_idx = 0
    with torch.no_grad():
        for bi in range(n_batches):
            batch = all_inputs[frame_idx: frame_idx + args.batch_size].float()
            frame_idx += len(batch)
            is_warmup = (bi * args.batch_size) < args.warmup

            # ── 1. MLP forward ───────────────────────────────────────────────
            t0 = time.perf_counter()
            c_all = mlp(batch)   # [B, 4K]  (batch is [W, flame_55] concatenated)
            t1 = time.perf_counter()

            c_parts = {
                'xyz':      c_all[:, 0*K:1*K],
                'scale':    c_all[:, 1*K:2*K],
                'rotation': c_all[:, 2*K:3*K],
                'opacity':  c_all[:, 3*K:4*K],
            }

            # ── 2. Linear combination ────────────────────────────────────────
            t2 = time.perf_counter()
            phys = {}
            for key in ATTR_KEYS:
                base  = bases[key]              # [1, G, C]
                U     = Us[key]                 # [K, G*C]
                Gn    = base.shape[1]
                Cn    = base.shape[2]
                delta = (c_parts[key] @ U).view(-1, Gn, Cn)   # [B, G, C]
                phys[key] = base + delta
            t3 = time.perf_counter()

            # ── 3. Geometric constraints ─────────────────────────────────────
            t4 = time.perf_counter()
            phys['scale']    = torch.clamp(phys['scale'], min=1e-5)
            phys['rotation'] = F.normalize(phys['rotation'], p=2, dim=-1)
            phys['opacity']  = torch.clamp(phys['opacity'],
                                           min=-ov + 1e-6,
                                           max=1.0 + ov - 1e-6)
            t5 = time.perf_counter()

            if not is_warmup:
                t_mlp.append(t1 - t0)
                t_lincomb.append(t3 - t2)
                t_constraint.append(t5 - t4)

            if frame_idx >= total_iters:
                break

    # ── Report ────────────────────────────────────────────────────────────────
    results = {
        'MLP forward':         _stats(t_mlp),
        'Linear combination':  _stats(t_lincomb),
        'Geometric constraints': _stats(t_constraint),
        'Total (sum)': _stats([a + b + c
                               for a, b, c in zip(t_mlp, t_lincomb, t_constraint)]),
    }

    print_table(results, args.batch_size, args.n_frames)


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
