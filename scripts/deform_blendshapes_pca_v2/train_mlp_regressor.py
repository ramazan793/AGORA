"""
Step 2: Train MLP regressor  [W(512) + FLAME-55] → PCA-4K coefficients (v2).

Generates its OWN training dataset by sampling n_identities × n_per_id from FFHQ
and projecting residuals onto the existing SVD basis (U matrices from Step 1).
This allows training on a much broader identity distribution than Step 1 used,
which improves MLP generalization to unseen identities.

The dataset is cached at:
  {svd_data_dir}/mlp_dataset_M{n_id}_Nper{n_per}_seed{start}.pt
so re-training with different hyperparameters (arch, lr, epochs) is fast.

MLP architecture (configurable via --mlp-hidden):
    Linear(567, hidden1) → LeakyReLU(0.2)
    Linear(hidden1, hidden2) → LeakyReLU(0.2)
    Linear(hidden2, 4*K)

Input: concat(W[512], flame_55) = 567-dim
Output: 4*K PCA coefficients (xyz, scale, rotation, opacity blocks of size K each)

Saves:
  data/{mlp_tag}/mlp_regressor.pt          — MLP weights + metadata
  vis/{mlp_tag}/loss_plot.png              — train/val loss curves
  vis/{mlp_tag}/mlp_reconstruction_grid_train.png  — GT vs MLP+SVD (training ids)
  vis/{mlp_tag}/mlp_reconstruction_grid_ood.png    — GT vs MLP+SVD (OOD ids)
"""

from dataclasses import dataclass
from typing import Optional
import tyro
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
os.environ.setdefault('GGHEAD_MODELS_PATH', '/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/')

from dreifus.image import Img
from src.gghead.model_manager.finder import find_model_manager
from src.gghead.config.gaussian_attribute import GaussianAttribute
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR

ATTR_KEYS = [
    ('xyz',      GaussianAttribute.POSITION),
    ('scale',    GaussianAttribute.SCALE),
    ('rotation', GaussianAttribute.ROTATION),
    ('opacity',  GaussianAttribute.OPACITY),
]

FFHQ_FUSED_PARAMS = f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy'


# ─────────────────────────────────────── MLP architecture ─────────────────────

class MLPRegressor(nn.Module):
    """Maps concat(W[512], flame_55) = 567-dim → 4*K PCA coefficients."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────── identity / collection helpers ─────────

def setup_identity(G, rng, ffhq_shapes, c_front, device):
    """Deterministic identity setup from a seeded rng. Returns (z, fixed_shape, neutral_flame)."""
    z = torch.randn((1, G._config.z_dim), device=device, generator=rng)
    flame_shape_id = torch.randint(0, ffhq_shapes.shape[0], (1,),
                                   generator=rng, device=device).cpu().item()
    fixed_shape = torch.tensor(ffhq_shapes[flame_shape_id, :300],
                               dtype=torch.float32, device=device)
    neutral_flame = torch.zeros(1, 358, device=device, dtype=torch.float32)
    neutral_flame[:, :300] = fixed_shape
    return z, fixed_shape, neutral_flame


def get_physical_attrs(G, z, c_front, flame_params, resolution,
                       cache_backbone=False, use_cached_backbone=False):
    """Forward pass → dict of post-activation physical Gaussian attributes."""
    ws = G.mapping(z, c_front, truncation_psi=0.7, flame_params=flame_params)
    output = G.synthesis(
        ws, c_front, flame_params,
        neural_rendering_resolution=resolution,
        cache_backbone=cache_backbone,
        use_cached_backbone=use_cached_backbone,
        noise_mode='const',
        sh_ref_cam=c_front,
        return_gaussian_attributes=True,
    )
    ga_raw = output.returned_gaussian_attributes
    return {
        GaussianAttribute.POSITION: ga_raw[GaussianAttribute.POSITION],
        GaussianAttribute.SCALE:    G._gaussian_model.scaling_activation(
                                        ga_raw[GaussianAttribute.SCALE]),
        GaussianAttribute.ROTATION: G._gaussian_model.rotation_activation(
                                        ga_raw[GaussianAttribute.ROTATION]),
        GaussianAttribute.OPACITY:  G._gaussian_model.opacity_activation(
                                        ga_raw[GaussianAttribute.OPACITY]),
        GaussianAttribute.COLOR:    ga_raw[GaussianAttribute.COLOR],
    }


# ─────────────────────────────────────── render helpers ───────────────────────

def render_from_physical_attrs(G, z, c_front, flame_params,
                                phys_dict, zero_deltas, resolution):
    """
    Render using precomputed physical Gaussian attrs via post_act_blendshapes
    trick (base = reconstructed physical attrs, deltas = zero).
    """
    post_act_bs = {
        'base_xyz':       phys_dict['xyz'],
        'base_scale':     phys_dict['scale'],
        'base_rotation':  phys_dict['rotation'],
        'base_opacity':   phys_dict['opacity'],
        'base_color':     phys_dict['color'],
        'delta_xyz':      zero_deltas['xyz'],
        'delta_scale':    zero_deltas['scale'],
        'delta_rotation': zero_deltas['rotation'],
        'delta_opacity':  zero_deltas['opacity'],
        'delta_color':    zero_deltas['color'],
    }
    ws = G.mapping(z, c_front, truncation_psi=0.7, flame_params=flame_params)
    output = G.synthesis(
        ws, c_front, flame_params,
        neural_rendering_resolution=resolution,
        cache_backbone=False,
        use_cached_backbone=True,
        noise_mode='const',
        sh_ref_cam=c_front,
        post_act_blendshapes=post_act_bs,
    )
    img = output['image'][0]
    return Img.from_normalized_torch(img).to_numpy().img[..., :3]


def reconstruct_physical_attrs(mlp, w_code, flame_55, svd_data,
                                base_gaussians_id, base_color, device, K):
    """Run MLP(concat(w_code, flame_55)) → PCA coeffs → physical attrs via SVD bases."""
    mlp_input = torch.cat([w_code, flame_55], dim=-1)  # [B, 567]
    c_all = mlp(mlp_input)                              # [B, 4K]
    c_parts = {
        'xyz':      c_all[:, 0*K:1*K],
        'scale':    c_all[:, 1*K:2*K],
        'rotation': c_all[:, 2*K:3*K],
        'opacity':  c_all[:, 3*K:4*K],
    }
    phys = {}
    for key_str, _ in ATTR_KEYS:
        Gn   = base_gaussians_id[key_str].shape[1]
        Cn   = base_gaussians_id[key_str].shape[2]
        U    = svd_data[f'U_{key_str}'].to(device)
        base = base_gaussians_id[key_str].to(device)
        delta = (c_parts[key_str] @ U).view(-1, Gn, Cn)
        phys[key_str] = base + delta
    phys['color'] = base_color.to(device)
    return phys


def apply_geometric_constraints(phys, ov):
    """In-place clamp/normalize physical attrs."""
    phys['scale']    = torch.clamp(phys['scale'], min=1e-5)
    phys['rotation'] = torch.nn.functional.normalize(phys['rotation'], p=2, dim=-1)
    opa_min, opa_max = -ov + 1e-6, 1.0 + ov - 1e-6
    phys['opacity']  = torch.clamp(phys['opacity'], min=opa_min, max=opa_max)
    return phys


# ─────────────────────────────────────────────────── args ─────────────────────

@dataclass
class Args:
    svd_data_path: str = ''          # path to svd_basis.pt from Step 1
    DEVICE: str = 'cuda:0'
    # ── MLP training data generation ──────────────────────────────────────────
    n_identities: int = 200          # identities for MLP training (can be >> PCA n_identities)
    n_per_id: int = 50               # FFHQ expressions per identity
    id_seed_start: int = 0           # starting seed for identity sampling
    collect_resolution: int = 256    # resolution for fast collection forward passes
    # ── MLP architecture ──────────────────────────────────────────────────────
    mlp_hidden: str = '256,512'      # comma-separated hidden layer sizes, e.g. "256,512"
    # ── training ──────────────────────────────────────────────────────────────
    epochs: int = 3000
    lr: float = 1e-3
    batch_size: int = 256
    val_fraction: float = 0.1
    # ── model / visualization ─────────────────────────────────────────────────
    run_name: str = 'DGGHEAD-158'
    checkpoint: int = 20500
    cam_scale: float = 8.0
    vis_resolution: int = 512
    n_vis_train: int = 3             # training identities to visualize (n_vis_samples each)
    n_vis_samples: int = 10          # FFHQ expressions per training identity
    n_ood_vis: int = 30              # OOD identities (not in training), 1 expression each


# ─────────────────────────────────────────────────── main ─────────────────────

def main(args: Args):
    device = torch.device(args.DEVICE)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse mlp_hidden
    hidden_parts = [int(x.strip()) for x in args.mlp_hidden.split(',')]
    assert len(hidden_parts) == 2, "--mlp-hidden must be two comma-separated ints, e.g. '256,512'"
    hidden1, hidden2 = hidden_parts

    # ── load SVD basis ────────────────────────────────────────────────────────
    assert args.svd_data_path, "Provide --svd-data-path pointing to svd_basis.pt"
    print(f"Loading SVD basis from {args.svd_data_path}...")
    svd_data = torch.load(args.svd_data_path, map_location='cpu')

    K         = int(svd_data['U_xyz'].shape[0])   # number of PCA components
    G_num     = int(svd_data['G'])
    run_name  = svd_data['run_name']
    checkpoint = svd_data['checkpoint']

    for k in ['U_xyz', 'U_scale', 'U_rotation', 'U_opacity']:
        print(f"  {k}: {tuple(svd_data[k].shape)}")

    # ── output dirs ───────────────────────────────────────────────────────────
    mlp_tag = (f"{run_name}_{checkpoint}"
               f"_M{args.n_identities}_Nper{args.n_per_id}"
               f"_K{K}_h{hidden1}_{hidden2}")
    data_dir = os.path.join(script_dir, 'data', mlp_tag)
    vis_dir  = os.path.join(script_dir, 'vis', mlp_tag)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    # ── load FFHQ distribution (needed for both generation and visualization) ─
    ffhq_data   = np.load(FFHQ_FUSED_PARAMS)
    ffhq_cams   = ffhq_data[:, -6:]
    ffhq_shapes = ffhq_data[:, :300]
    ffhq_exp    = ffhq_data[:, 300:350]
    ffhq_jaw    = ffhq_data[:, 353:356]
    ffhq_eyelid = ffhq_data[:, 356:358]

    mean_cam = np.mean(ffhq_cams, axis=0)
    c_front  = torch.tensor(mean_cam, dtype=torch.float32).unsqueeze(0).to(device)
    c_front[0, 3] = args.cam_scale

    # ── MLP training dataset: check cache or generate ─────────────────────────
    svd_data_dir  = os.path.dirname(os.path.abspath(args.svd_data_path))
    dataset_cache = os.path.join(
        svd_data_dir,
        f"mlp_dataset_M{args.n_identities}_Nper{args.n_per_id}_seed{args.id_seed_start}.pt")

    G = None  # loaded on demand

    if os.path.exists(dataset_cache):
        print(f"Loading cached MLP dataset from {dataset_cache} ...")
        ds = torch.load(dataset_cache, map_location='cpu')
        w_codes_all      = ds['w_codes']       # [N_total, 512]
        flame_inputs_all = ds['flame_inputs']  # [N_total, 55]
        c_all_dict = {k: ds[f'c_{k}'] for k, _ in ATTR_KEYS}
        N_total = int(ds['n_identities']) * int(ds['n_per_id'])
        print(f"  Loaded {len(w_codes_all)} samples (K={K})")
    else:
        # ── Load model for data generation ────────────────────────────────────
        print(f"Loading {run_name} checkpoint {checkpoint} for data generation...")
        model_manager = find_model_manager(run_name)
        ckpt_obj = model_manager._resolve_checkpoint_id(checkpoint)
        G = model_manager.load_checkpoint(ckpt_obj, load_ema=True).to(device)
        G.eval()
        G._config.use_flame_rasterization = 0

        U_dev = {k: svd_data[f'U_{k}'].to(device) for k, _ in ATTR_KEYS}

        rng    = torch.Generator(device)
        rng_np = np.random.RandomState(42)

        N_total = args.n_identities * args.n_per_id
        all_w_codes      = []
        all_flame_inputs = []
        c_all_dict       = {k: [] for k, _ in ATTR_KEYS}

        print(f"Generating MLP dataset: {args.n_identities} identities × "
              f"{args.n_per_id} expressions = {N_total} samples")

        with torch.no_grad():
            for id_idx in tqdm(range(args.n_identities), desc="Identities"):
                seed = args.id_seed_start + id_idx
                rng.manual_seed(seed)
                z, fixed_shape, neutral_flame = setup_identity(
                    G, rng, ffhq_shapes, c_front, device)

                # Extract W code (identity-only, no expression dependence)
                ws_neutral = G.mapping(z, c_front, truncation_psi=0.7,
                                       flame_params=neutral_flame)
                backbone_ws, _ = ws_neutral
                w_code = backbone_ws[:, 0, :]   # [1, 512]

                # Base gaussians + cache backbone for this identity
                base_phys = get_physical_attrs(
                    G, z, c_front, neutral_flame, args.collect_resolution,
                    cache_backbone=True, use_cached_backbone=False)
                base_gaus = {k: base_phys[ae].detach() for k, ae in ATTR_KEYS}

                # Sample n_per_id expressions
                idx = rng_np.randint(0, len(ffhq_data), size=args.n_per_id)
                exp_id    = ffhq_exp[idx]
                jaw_id    = ffhq_jaw[idx]
                eyelid_id = ffhq_eyelid[idx]
                flame_55_id = np.concatenate(
                    [exp_id, jaw_id, eyelid_id], axis=1).astype(np.float32)

                for i in range(args.n_per_id):
                    fp = neutral_flame.clone()
                    fp[0, 300:350] = torch.tensor(exp_id[i],    dtype=torch.float32, device=device)
                    fp[0, 353:356] = torch.tensor(jaw_id[i],    dtype=torch.float32, device=device)
                    fp[0, 356:358] = torch.tensor(eyelid_id[i], dtype=torch.float32, device=device)

                    attrs_i = get_physical_attrs(
                        G, z, c_front, fp, args.collect_resolution,
                        cache_backbone=False, use_cached_backbone=True)

                    for key_str, attr_enum in ATTR_KEYS:
                        Gn = base_gaus[key_str].shape[1]
                        Cn = base_gaus[key_str].shape[2]
                        res = attrs_i[attr_enum].detach() - base_gaus[key_str]  # [1, G, C]
                        res_flat = res.reshape(1, Gn * Cn).float()
                        c_k = (res_flat @ U_dev[key_str].T).cpu()               # [1, K]
                        c_all_dict[key_str].append(c_k)

                all_w_codes.append(w_code.cpu().expand(args.n_per_id, -1).clone())
                all_flame_inputs.append(torch.tensor(flame_55_id))

        w_codes_all      = torch.cat(all_w_codes, dim=0)       # [N_total, 512]
        flame_inputs_all = torch.cat(all_flame_inputs, dim=0)  # [N_total, 55]
        c_all_dict = {k: torch.cat(v, dim=0) for k, v in c_all_dict.items()}

        # Save dataset cache
        cache_save = {
            'w_codes':       w_codes_all,
            'flame_inputs':  flame_inputs_all,
            'n_identities':  args.n_identities,
            'n_per_id':      args.n_per_id,
            'id_seed_start': args.id_seed_start,
            'K':             K,
            'G':             G_num,
        }
        for k, _ in ATTR_KEYS:
            cache_save[f'c_{k}'] = c_all_dict[k]
        torch.save(cache_save, dataset_cache)
        print(f"Dataset cache saved → {dataset_cache}  ({N_total} samples)")

    # ── build X / Y tensors ───────────────────────────────────────────────────
    X = torch.cat([w_codes_all, flame_inputs_all], dim=-1).float()  # [N_total, 567]
    Y = torch.cat([
        c_all_dict['xyz'],
        c_all_dict['scale'],
        c_all_dict['rotation'],
        c_all_dict['opacity'],
    ], dim=-1).float()                                               # [N_total, 4K]
    N_total = len(X)

    print(f"  Dataset: N_total={N_total}, K={K}")
    print(f"  X={tuple(X.shape)}, Y={tuple(Y.shape)}")

    # ── train / val split ─────────────────────────────────────────────────────
    n_val   = max(1, int(N_total * args.val_fraction))
    n_train = N_total - n_val
    dataset = TensorDataset(X, Y)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── model, optimizer ──────────────────────────────────────────────────────
    mlp = MLPRegressor(in_dim=567, hidden1=hidden1, hidden2=hidden2, out_dim=4 * K).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"Training MLP [{hidden1}→{hidden2}] for {args.epochs} epochs  "
          f"(train={n_train}, val={n_val})...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = mlp(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / n_train)

        mlp.eval()
        with torch.no_grad():
            v_loss = sum(nn.functional.mse_loss(mlp(xb.to(device)), yb.to(device)).item() * len(xb)
                         for xb, yb in val_loader) / n_val
        val_losses.append(v_loss)
        scheduler.step()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state    = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

        if epoch % 100 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={train_losses[-1]:.6f}  val={val_losses[-1]:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ── save MLP ──────────────────────────────────────────────────────────────
    mlp.load_state_dict(best_state)   # restore best checkpoint
    mlp_save = {
        'state_dict':    best_state,
        'K':             K,
        'in_dim':        567,
        'hidden1':       hidden1,
        'hidden2':       hidden2,
        'out_dim':       4 * K,
        'val_loss':      best_val_loss,
        'run_name':      run_name,
        'checkpoint':    checkpoint,
        'n_identities':  args.n_identities,
        'id_seed_start': args.id_seed_start,
    }
    mlp_path = os.path.join(data_dir, 'mlp_regressor.pt')
    torch.save(mlp_save, mlp_path)
    print(f"MLP saved → {mlp_path}  (best val_loss={best_val_loss:.6f})")

    # ── loss plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, label='train')
    ax.semilogy(val_losses,   label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title(f'MLP regressor (v2: W+flame → PCA, hidden={hidden1},{hidden2})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    loss_plot_path = os.path.join(vis_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path, dpi=120)
    plt.close()
    print(f"Loss plot → {loss_plot_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Visualization: FFHQ-based GT-vs-approx side-by-side grids
    # ══════════════════════════════════════════════════════════════════════════
    print("\nLoading model for visualization...")
    if G is None:
        model_manager = find_model_manager(run_name)
        ckpt_obj = model_manager._resolve_checkpoint_id(checkpoint)
        G = model_manager.load_checkpoint(ckpt_obj, load_ema=True).to(device)
        G.eval()
        G._config.use_flame_rasterization = 0

    ov = getattr(G._config, 'opacity_overshoot', 0.0)
    mlp.eval()

    zero_deltas = None
    rng_t = torch.Generator(device)

    def setup_identity_vis(seed):
        nonlocal zero_deltas
        rng_t.manual_seed(seed)
        z_id = torch.randn((1, G._config.z_dim), device=device, generator=rng_t)
        fsid = torch.randint(0, ffhq_shapes.shape[0], (1,),
                             generator=rng_t, device=device).cpu().item()
        fixed_shape = torch.tensor(ffhq_shapes[fsid, :300], dtype=torch.float32, device=device)

        neutral_fl = torch.zeros(1, 358, device=device, dtype=torch.float32)
        neutral_fl[:, :300] = fixed_shape

        ws_id = G.mapping(z_id, c_front, truncation_psi=0.7, flame_params=neutral_fl)
        backbone_ws, _ = ws_id
        w_code_id = backbone_ws[:, 0, :]   # [1, 512]

        output_base = G.synthesis(
            ws_id, c_front, neutral_fl,
            neural_rendering_resolution=args.vis_resolution,
            cache_backbone=True, use_cached_backbone=False,
            noise_mode='const', sh_ref_cam=c_front,
            return_gaussian_attributes=True,
        )
        ga_raw = output_base.returned_gaussian_attributes
        base_gaus = {
            'xyz':      ga_raw[GaussianAttribute.POSITION].detach(),
            'scale':    G._gaussian_model.scaling_activation(ga_raw[GaussianAttribute.SCALE]).detach(),
            'rotation': G._gaussian_model.rotation_activation(ga_raw[GaussianAttribute.ROTATION]).detach(),
            'opacity':  G._gaussian_model.opacity_activation(ga_raw[GaussianAttribute.OPACITY]).detach(),
        }
        base_col = ga_raw[GaussianAttribute.COLOR].detach()

        if zero_deltas is None:
            C_color = base_col.shape[-1]
            zero_deltas = {
                'xyz':      torch.zeros(59, G_num, 3,       device=device),
                'scale':    torch.zeros(59, G_num, 3,       device=device),
                'rotation': torch.zeros(59, G_num, 4,       device=device),
                'opacity':  torch.zeros(59, G_num, 1,       device=device),
                'color':    torch.zeros(59, G_num, C_color, device=device),
            }

        return z_id, neutral_fl, w_code_id, base_gaus, base_col

    def render_pair(z_id, neutral_fl, w_code_id, base_gaus, base_col,
                    exp_row, jaw_row, eyelid_row):
        """Render one [GT | MLP+SVD] pair for given FFHQ expression row."""
        fp = neutral_fl.clone()
        fp[0, 300:350] = torch.tensor(exp_row,    dtype=torch.float32, device=device)
        fp[0, 353:356] = torch.tensor(jaw_row,    dtype=torch.float32, device=device)
        fp[0, 356:358] = torch.tensor(eyelid_row, dtype=torch.float32, device=device)

        # GT: full deformation network
        ws = G.mapping(z_id, c_front, truncation_psi=0.7, flame_params=fp)
        out_gt = G.synthesis(ws, c_front, fp,
                             neural_rendering_resolution=args.vis_resolution,
                             cache_backbone=False, use_cached_backbone=True,
                             noise_mode='const', sh_ref_cam=c_front)
        gt_frame = Img.from_normalized_torch(out_gt['image'][0]).to_numpy().img[..., :3]

        # MLP+SVD approx
        flame_55 = torch.tensor(
            np.concatenate([exp_row, jaw_row, eyelid_row]),
            dtype=torch.float32, device=device).unsqueeze(0)   # [1, 55]
        phys = reconstruct_physical_attrs(mlp, w_code_id, flame_55, svd_data,
                                          base_gaus, base_col, device, K)
        phys = apply_geometric_constraints(phys, ov)
        approx_frame = render_from_physical_attrs(
            G, z_id, c_front, fp, phys, zero_deltas, args.vis_resolution)

        return gt_frame, approx_frame

    def build_and_save_grid(pairs_gt, pairs_approx, path, label):
        """Assemble 10-row × 3-col grid of [GT | approx] pairs and save."""
        n_pairs = len(pairs_gt)
        H, W = pairs_gt[0].shape[:2]
        n_rows, n_cols = 10, 3
        grid = np.zeros((n_rows * H, n_cols * 2 * W, 3), dtype=np.uint8)
        for vi in range(min(n_pairs, n_rows * n_cols)):
            r = vi // n_cols
            c = vi % n_cols
            grid[r*H:(r+1)*H, (c*2)*W  :(c*2+1)*W] = pairs_gt[vi]
            grid[r*H:(r+1)*H, (c*2+1)*W:(c*2+2)*W] = pairs_approx[vi]
        Image.fromarray(grid).save(path)
        print(f"  {label} → {path}")

    rng_np_vis = np.random.RandomState(42)

    # ── Training identities grid ───────────────────────────────────────────────
    print(f"\nRendering training identities grid "
          f"({args.n_vis_train} ids × {args.n_vis_samples} exprs)...")
    gt_frames_train    = []
    approx_frames_train = []
    mse_train = []

    with torch.no_grad():
        for id_idx in range(min(args.n_vis_train, args.n_identities)):
            seed = args.id_seed_start + id_idx
            z_id, neutral_fl, w_code_id, base_gaus, base_col = setup_identity_vis(seed)
            idx = rng_np_vis.randint(0, len(ffhq_data), size=args.n_vis_samples)
            print(f"  Train identity seed={seed}...")
            for i in tqdm(range(args.n_vis_samples), desc=f"seed={seed}", leave=False):
                gt_f, approx_f = render_pair(
                    z_id, neutral_fl, w_code_id, base_gaus, base_col,
                    ffhq_exp[idx[i]], ffhq_jaw[idx[i]], ffhq_eyelid[idx[i]])
                gt_frames_train.append(gt_f)
                approx_frames_train.append(approx_f)
                mse_train.append(np.mean((gt_f.astype(np.float32)
                                          - approx_f.astype(np.float32)) ** 2))
            print(f"    MSE mean={np.mean(mse_train[-args.n_vis_samples:]):.4f}")

    print(f"Train grid overall MSE: mean={np.mean(mse_train):.4f}  "
          f"min={np.min(mse_train):.4f}  max={np.max(mse_train):.4f}")
    build_and_save_grid(gt_frames_train, approx_frames_train,
                        os.path.join(vis_dir, 'mlp_reconstruction_grid_train.png'),
                        'Train grid')

    # ── OOD identities grid ────────────────────────────────────────────────────
    ood_seed_start = args.id_seed_start + args.n_identities
    print(f"\nRendering OOD identities grid ({args.n_ood_vis} ids, 1 expr each)...")
    gt_frames_ood    = []
    approx_frames_ood = []
    mse_ood = []

    with torch.no_grad():
        for ood_idx in tqdm(range(args.n_ood_vis), desc="OOD identities"):
            seed = ood_seed_start + ood_idx
            z_id, neutral_fl, w_code_id, base_gaus, base_col = setup_identity_vis(seed)
            exp_idx = rng_np_vis.randint(0, len(ffhq_data))
            gt_f, approx_f = render_pair(
                z_id, neutral_fl, w_code_id, base_gaus, base_col,
                ffhq_exp[exp_idx], ffhq_jaw[exp_idx], ffhq_eyelid[exp_idx])
            gt_frames_ood.append(gt_f)
            approx_frames_ood.append(approx_f)
            mse_ood.append(np.mean((gt_f.astype(np.float32)
                                     - approx_f.astype(np.float32)) ** 2))

    print(f"OOD grid MSE: mean={np.mean(mse_ood):.4f}  "
          f"min={np.min(mse_ood):.4f}  max={np.max(mse_ood):.4f}")
    build_and_save_grid(gt_frames_ood, approx_frames_ood,
                        os.path.join(vis_dir, 'mlp_reconstruction_grid_ood.png'),
                        'OOD grid')

    print(f"\nDone. MLP: {mlp_path}")


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
