"""
Step 1: Data-Driven SVD Basis Extraction — identity-independent version (v2).

Samples N_per_id plausible FLAME expressions for each of n_identities identities,
runs each through the full deformation network, extracts post-activation residuals,
and computes truncated SVD bases (top K PCs) separately for xyz, scale, rotation,
and opacity.

The resulting shared basis works across all identities.
W codes (512-dim identity latents) are saved for MLP training.

Collected residuals are cached to cached_residuals.pt after the GPU collection
loop. If this file exists, collection is skipped and only SVD is recomputed.
If svd_basis.pt also exists, SVD is skipped and only visualizations are regenerated.

Saves:
  data/{tag}/svd_basis.pt  — bases, projected coefficients, FLAME inputs, W codes
Visualizations:
  vis/{tag}/pca_reconstruction_grid_seed{X}.png   — GT vs PCA for each train vis identity
  vis/{tag}/ood_reconstruction_grid.png            — 10x3 grid, 30 OOD identities
  vis/{tag}/principal_components_{attr}.png         — top-16 PCs as avatars (first identity)
"""

from dataclasses import dataclass
from typing import Optional
import tyro
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

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


@dataclass
class Args:
    DEVICE: str = 'cuda:0'
    run_name: str = 'DGGHEAD-158'
    checkpoint: int = 20500
    n_identities: int = 50      # number of training identities
    id_seed_start: int = 0      # seeds go from id_seed_start to id_seed_start + n_identities - 1
    N_per_id: int = 100         # expressions per training identity
    n_vis_identities: int = 3   # training identities to show in per-id reconstruction grids
    n_ood_vis: int = 30         # OOD identities for the combined grid
    cam_scale: float = 8.0
    K: int = 64                 # number of SVD/PCA components to keep
    collect_resolution: int = 256   # resolution for N forward passes (fast)
    vis_resolution: int = 512       # resolution for visualization renders
    n_vis_samples: int = 10         # samples per training identity for reconstruction grid
    n_pc_vis: int = 16              # number of principal components to visualize


# ─────────────────────────────────────────────────── helpers ──────────────────

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
    """Forward pass → (rendered_frame_uint8, physical_gaussian_attrs dict)."""
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
    img = output['image'][0]
    frame = Img.from_normalized_torch(img).to_numpy().img[..., :3]

    ga_raw = output.returned_gaussian_attributes
    ga_phys = {
        GaussianAttribute.POSITION: ga_raw[GaussianAttribute.POSITION],
        GaussianAttribute.SCALE:    G._gaussian_model.scaling_activation(
                                        ga_raw[GaussianAttribute.SCALE]),
        GaussianAttribute.ROTATION: G._gaussian_model.rotation_activation(
                                        ga_raw[GaussianAttribute.ROTATION]),
        GaussianAttribute.OPACITY:  G._gaussian_model.opacity_activation(
                                        ga_raw[GaussianAttribute.OPACITY]),
        GaussianAttribute.COLOR:    ga_raw[GaussianAttribute.COLOR],
    }
    return frame, ga_phys


def render_from_physical_attrs(G, z, c_front, flame_params,
                                phys_attrs_dict, zero_deltas, resolution):
    """
    Render using precomputed physical attrs by injecting them as post_act_blendshapes
    with zero deltas (trick: base = reconstructed attrs, deltas = 0).
    """
    post_act_bs = {
        'base_xyz':      phys_attrs_dict['xyz'],
        'base_scale':    phys_attrs_dict['scale'],
        'base_rotation': phys_attrs_dict['rotation'],
        'base_opacity':  phys_attrs_dict['opacity'],
        'base_color':    phys_attrs_dict['color'],
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


def pca_project_and_reconstruct(gt_attrs, base_gaussians_id, U_dev, device):
    """Out-of-sample PCA projection + reconstruction. Returns recon dict."""
    recon = {}
    for key_str, attr_enum in ATTR_KEYS:
        Gn = base_gaussians_id[key_str].shape[1]
        Cn = base_gaussians_id[key_str].shape[2]
        res = (gt_attrs[attr_enum].detach()
               - base_gaussians_id[key_str].to(device))       # [1, G, C]
        res_flat = res.reshape(1, Gn * Cn).float()
        c_v   = res_flat @ U_dev[key_str].T                   # [1, K]
        delta = (c_v @ U_dev[key_str]).view(1, Gn, Cn)        # [1, G, C]
        recon[key_str] = base_gaussians_id[key_str].to(device) + delta
    return recon


# ──────────────────────────────────────────────────── main ────────────────────

def main(args: Args):
    device = torch.device(args.DEVICE)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    N_total = args.n_identities * args.N_per_id

    # ── output dirs ───────────────────────────────────────────────────────────
    tag = f"{args.run_name}_{args.checkpoint}_M{args.n_identities}_Nper{args.N_per_id}_K{args.K}"
    data_dir = os.path.join(script_dir, 'data', tag)
    vis_dir  = os.path.join(script_dir, 'vis', tag)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir,  exist_ok=True)

    save_path  = os.path.join(data_dir, 'svd_basis.pt')
    cache_path = os.path.join(data_dir, 'cached_residuals.pt')
    skip_svd   = os.path.exists(save_path)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.run_name} checkpoint {args.checkpoint}...")
    model_manager = find_model_manager(args.run_name)
    ckpt = model_manager._resolve_checkpoint_id(args.checkpoint)
    G = model_manager.load_checkpoint(ckpt, load_ema=True).to(device)
    G.eval()
    G._config.use_flame_rasterization = 0

    # ── FFHQ distribution ─────────────────────────────────────────────────────
    ffhq_data   = np.load(FFHQ_FUSED_PARAMS)           # [N_ffhq, 364]
    ffhq_cams   = ffhq_data[:, -6:]
    ffhq_shapes = ffhq_data[:, :300]
    ffhq_exp    = ffhq_data[:, 300:350]                 # [N_ffhq, 50]
    ffhq_jaw    = ffhq_data[:, 353:356]                 # [N_ffhq, 3]
    ffhq_eyelid = ffhq_data[:, 356:358]                 # [N_ffhq, 2]

    mean_cam = np.mean(ffhq_cams, axis=0)
    c_front = torch.tensor(mean_cam, dtype=torch.float32).unsqueeze(0).to(device)
    c_front[0, 3] = args.cam_scale

    rng = torch.Generator(device)

    # shared state populated differently depending on skip_svd
    svd_bases   = {}
    svd_S       = {}
    K_actual    = None
    G_num       = None
    C_color     = None
    zero_deltas = None
    identity_info = []   # list of dicts for training vis identities

    # ══════════════════════════════════════════════════════════════════════════
    if skip_svd:
        # ── Load existing SVD ─────────────────────────────────────────────────
        print(f"Found existing SVD data at {save_path}  →  skipping computation.")
        saved = torch.load(save_path, map_location='cpu')
        for key_str, _ in ATTR_KEYS:
            svd_bases[key_str] = saved[f'U_{key_str}']
            svd_S[key_str]     = saved[f'S_{key_str}']
        K_actual = int(saved['K'])
        G_num    = int(saved['G'])
        for k, v in saved.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)}")

        # Re-run minimal identity setups for vis identities
        print(f"\nRe-computing base gaussians for {args.n_vis_identities} training vis identities...")
        rng_np_vis = np.random.RandomState(42)
        with torch.no_grad():
            for id_idx in range(args.n_vis_identities):
                seed = args.id_seed_start + id_idx
                rng.manual_seed(seed)
                z, fixed_shape, neutral_flame = setup_identity(
                    G, rng, ffhq_shapes, c_front, device)

                _, base_phys = get_physical_attrs(
                    G, z, c_front, neutral_flame, args.collect_resolution,
                    cache_backbone=True, use_cached_backbone=False)

                base_gaussians_id = {k: base_phys[ae].detach().cpu() for k, ae in ATTR_KEYS}
                base_color_id = base_phys[GaussianAttribute.COLOR].detach().cpu()

                if C_color is None:
                    C_color = base_color_id.shape[-1]
                    zero_deltas = {
                        'xyz':      torch.zeros(59, G_num, 3,       device=device),
                        'scale':    torch.zeros(59, G_num, 3,       device=device),
                        'rotation': torch.zeros(59, G_num, 4,       device=device),
                        'opacity':  torch.zeros(59, G_num, 1,       device=device),
                        'color':    torch.zeros(59, G_num, C_color, device=device),
                    }
                    print(f"  Gaussians: {G_num},  color channels: {C_color}")

                # Sample n_vis_samples expressions for vis
                idx = rng_np_vis.randint(0, len(ffhq_data), size=args.n_vis_samples)
                identity_info.append({
                    'seed':              seed,
                    'z':                 z.cpu(),
                    'fixed_shape':       fixed_shape.cpu(),
                    'base_gaussians_id': {k: v.clone() for k, v in base_gaussians_id.items()},
                    'base_color_id':     base_color_id.clone(),
                    'exp':               ffhq_exp[idx],
                    'jaw':               ffhq_jaw[idx],
                    'eyelid':            ffhq_eyelid[idx],
                })

    else:
        # ══════════════════════════════════════════════════════════════════════
        # Residual collection — from cache or full GPU forward passes
        # ══════════════════════════════════════════════════════════════════════
        _cache_valid = False
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location='cpu')
                _cache_valid = True
            except Exception as e:
                print(f"[WARN] Failed to load cache at {cache_path} ({e})  →  re-collecting.")
        if _cache_valid:
            # ── Load cached residuals (skip expensive GPU collection) ──────────
            print(f"Found cached residuals at {cache_path}  →  skipping collection.")
            all_residuals_stacked = cached['all_residuals']   # {key_str: [N_total, G, C]}
            flame_inputs_all      = cached['flame_inputs']    # [N_total, 55]
            w_codes_all           = cached['w_codes']         # [N_total, 512]
            identity_info         = cached['identity_info']
            G_num                 = int(cached['G_num'])
            C_color               = int(cached['C_color'])
            zero_deltas = {
                'xyz':      torch.zeros(59, G_num, 3,       device=device),
                'scale':    torch.zeros(59, G_num, 3,       device=device),
                'rotation': torch.zeros(59, G_num, 4,       device=device),
                'opacity':  torch.zeros(59, G_num, 1,       device=device),
                'color':    torch.zeros(59, G_num, C_color, device=device),
            }
            print(f"  Gaussians: {G_num},  color channels: {C_color}")
            for k, v in all_residuals_stacked.items():
                print(f"  residuals[{k}]: {tuple(v.shape)}")

        else:
            # ── Full collection loop ───────────────────────────────────────────
            # Step 1: Multi-identity residual collection
            rng_np = np.random.RandomState(42)

            all_residuals    = {k: [] for k, _ in ATTR_KEYS}
            all_flame_inputs = []   # will be [N_total, 55]
            all_w_codes      = []   # will be [N_total, 512]

            print(f"Collecting residuals from {args.n_identities} identities × {args.N_per_id} expressions "
                  f"= {N_total} total samples")

            with torch.no_grad():
                for id_idx in tqdm(range(args.n_identities), desc="Identities"):
                    seed = args.id_seed_start + id_idx
                    rng.manual_seed(seed)
                    z, fixed_shape, neutral_flame = setup_identity(
                        G, rng, ffhq_shapes, c_front, device)

                    # Extract W code (identity-only; does not depend on expression)
                    ws_neutral = G.mapping(z, c_front, truncation_psi=0.7, flame_params=neutral_flame)
                    backbone_ws, _ = ws_neutral
                    w_code = backbone_ws[:, 0, :]   # [1, 512]

                    # Base gaussians + cache backbone (MUST re-cache for each new identity)
                    _, base_phys = get_physical_attrs(
                        G, z, c_front, neutral_flame, args.collect_resolution,
                        cache_backbone=True, use_cached_backbone=False)

                    base_gaussians_id = {k: base_phys[ae].detach().cpu() for k, ae in ATTR_KEYS}
                    base_color_id = base_phys[GaussianAttribute.COLOR].detach().cpu()

                    if G_num is None:
                        G_num   = base_gaussians_id['xyz'].shape[1]
                        C_color = base_color_id.shape[-1]
                        zero_deltas = {
                            'xyz':      torch.zeros(59, G_num, 3,       device=device),
                            'scale':    torch.zeros(59, G_num, 3,       device=device),
                            'rotation': torch.zeros(59, G_num, 4,       device=device),
                            'opacity':  torch.zeros(59, G_num, 1,       device=device),
                            'color':    torch.zeros(59, G_num, C_color, device=device),
                        }
                        print(f"  Gaussians: {G_num},  color channels: {C_color}")

                    # Sample N_per_id expressions for this identity
                    idx = rng_np.randint(0, len(ffhq_data), size=args.N_per_id)

                    exp_id    = ffhq_exp[idx]       # [N_per_id, 50]
                    jaw_id    = ffhq_jaw[idx]       # [N_per_id, 3]
                    eyelid_id = ffhq_eyelid[idx]   # [N_per_id, 2]
                    flame_inputs_id = np.concatenate(
                        [exp_id, jaw_id, eyelid_id], axis=1).astype(np.float32)  # [N_per_id, 55]

                    for i in range(args.N_per_id):
                        fp = neutral_flame.clone()
                        fp[0, 300:350] = torch.tensor(exp_id[i],    dtype=torch.float32, device=device)
                        fp[0, 353:356] = torch.tensor(jaw_id[i],    dtype=torch.float32, device=device)
                        fp[0, 356:358] = torch.tensor(eyelid_id[i], dtype=torch.float32, device=device)

                        _, attrs_i = get_physical_attrs(
                            G, z, c_front, fp, args.collect_resolution,
                            cache_backbone=False, use_cached_backbone=True)

                        for key_str, attr_enum in ATTR_KEYS:
                            delta = (attrs_i[attr_enum].detach().cpu()
                                     - base_gaussians_id[key_str])          # [1, G, C]
                            all_residuals[key_str].append(delta)

                    all_flame_inputs.append(torch.tensor(flame_inputs_id))
                    all_w_codes.append(w_code.cpu().expand(args.N_per_id, -1).clone())

                    if id_idx < args.n_vis_identities:
                        identity_info.append({
                            'seed':              seed,
                            'z':                 z.cpu(),
                            'fixed_shape':       fixed_shape.cpu(),
                            'base_gaussians_id': {k: v.clone() for k, v in base_gaussians_id.items()},
                            'base_color_id':     base_color_id.clone(),
                            'exp':               exp_id[:args.n_vis_samples],
                            'jaw':               jaw_id[:args.n_vis_samples],
                            'eyelid':            eyelid_id[:args.n_vis_samples],
                        })

            # Stack residuals and inputs
            all_residuals_stacked = {
                k: torch.cat(v, dim=0) for k, v in all_residuals.items()
            }
            flame_inputs_all = torch.cat(all_flame_inputs, dim=0)   # [N_total, 55]
            w_codes_all      = torch.cat(all_w_codes, dim=0)         # [N_total, 512]

            # Save cache so next run can skip collection
            if False:
                print(f"Saving residual cache → {cache_path}")
                torch.save({
                    'all_residuals': all_residuals_stacked,
                    'flame_inputs':  flame_inputs_all,
                    'w_codes':       w_codes_all,
                    'identity_info': identity_info,
                    'G_num':         G_num,
                    'C_color':       C_color,
                    'n_identities':  args.n_identities,
                    'N_per_id':      args.N_per_id,
                }, cache_path)
                print(f"  Cached {N_total} samples.")

        # ══════════════════════════════════════════════════════════════════════
        # Step 2: Flatten residuals and compute truncated SVD per attribute
        # ══════════════════════════════════════════════════════════════════════
        print("Computing truncated SVD per attribute...")

        for key_str, _ in ATTR_KEYS:
            X = all_residuals_stacked[key_str]              # [N_total, G, C]
            Gn, Cn = X.shape[1], X.shape[2]
            X_flat = X.view(N_total, Gn * Cn).float().to(device)

            print(f"  {key_str}: SVD on [{N_total} x {Gn*Cn}]", end='', flush=True)

            U, S, V = torch.svd_lowrank(X_flat, q=args.K, niter=2)
            Vh  = V.T  # [K, G*C]

            var_total = (torch.linalg.vector_norm(X_flat).item()) ** 2
            var_top_K = (S ** 2).sum().item()
            print(f"  →  var explained: {100*var_top_K/var_total:.1f}%  (K={S.shape[0]})")

            svd_bases[key_str] = Vh.cpu()
            svd_S[key_str]     = S.cpu()
            del X_flat, U, V
            torch.cuda.empty_cache()

        K_actual = next(iter(svd_bases.values())).shape[0]

        # ══════════════════════════════════════════════════════════════════════
        # Step 3: Save dataset
        # ══════════════════════════════════════════════════════════════════════
        # recompute pca_coeffs from (U * S) that we already computed
        # rebuild from stacked residuals — easier: just re-project once
        pca_coeffs = {}
        with torch.no_grad():
            for key_str, _ in ATTR_KEYS:
                X = all_residuals_stacked[key_str]
                Gn, Cn = X.shape[1], X.shape[2]
                X_flat = X.view(N_total, Gn * Cn).float().to(device)
                U_k = svd_bases[key_str].to(device)
                pca_coeffs[key_str] = (X_flat @ U_k.T).cpu()  # [N_total, K]
                del X_flat
                torch.cuda.empty_cache()

        save_dict = {
            'run_name':      args.run_name,
            'checkpoint':    args.checkpoint,
            'id_seed_start': args.id_seed_start,
            'n_identities':  args.n_identities,
            'N_per_id':      args.N_per_id,
            'N_total':       N_total,
            'K':             K_actual,
            'G':             G_num,
            'flame_inputs':  flame_inputs_all,
            'w_codes':       w_codes_all,
        }
        for key_str, _ in ATTR_KEYS:
            save_dict[f'U_{key_str}'] = svd_bases[key_str]
            save_dict[f'S_{key_str}'] = svd_S[key_str]
            save_dict[f'c_{key_str}'] = pca_coeffs[key_str]

        torch.save(save_dict, save_path)
        print(f"\nSaved SVD data → {save_path}")
        for k, v in save_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4: PCA reconstruction quality — per training vis identity
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nVisualizing PCA reconstruction for {len(identity_info)} training identities...")
    U_dev = {k: svd_bases[k].to(device) for k, _ in ATTR_KEYS}

    with torch.no_grad():
        for info in identity_info:
            id_seed   = info['seed']
            z_id      = info['z'].to(device)
            fixed_shp = info['fixed_shape'].to(device)
            base_gaus = info['base_gaussians_id']
            base_col  = info['base_color_id']

            neutral_fl = torch.zeros(1, 358, device=device, dtype=torch.float32)
            neutral_fl[:, :300] = fixed_shp

            # Re-cache backbone at vis resolution
            ws_id = G.mapping(z_id, c_front, truncation_psi=0.7, flame_params=neutral_fl)
            G.synthesis(ws_id, c_front, neutral_fl,
                        neural_rendering_resolution=args.vis_resolution,
                        cache_backbone=True, use_cached_backbone=False,
                        noise_mode='const', sh_ref_cam=c_front)

            n_vis = len(info['exp'])
            gt_frames  = []
            pca_frames = []
            mse_vals   = []

            for vi in tqdm(range(n_vis), desc=f"PCA vis seed={id_seed}"):
                fp = neutral_fl.clone()
                fp[0, 300:350] = torch.tensor(info['exp'][vi],    dtype=torch.float32, device=device)
                fp[0, 353:356] = torch.tensor(info['jaw'][vi],    dtype=torch.float32, device=device)
                fp[0, 356:358] = torch.tensor(info['eyelid'][vi], dtype=torch.float32, device=device)

                gt_frame, gt_attrs = get_physical_attrs(
                    G, z_id, c_front, fp, args.vis_resolution,
                    cache_backbone=False, use_cached_backbone=True)

                recon = pca_project_and_reconstruct(gt_attrs, base_gaus, U_dev, device)
                recon['color'] = base_col.to(device)

                pca_frame = render_from_physical_attrs(
                    G, z_id, c_front, fp, recon, zero_deltas, args.vis_resolution)

                gt_frames.append(gt_frame)
                pca_frames.append(pca_frame)
                mse_vals.append(np.mean((gt_frame.astype(np.float32)
                                         - pca_frame.astype(np.float32)) ** 2))

            print(f"  seed={id_seed}  PCA MSE — mean: {np.mean(mse_vals):.4f} "
                  f" min: {np.min(mse_vals):.4f}  max: {np.max(mse_vals):.4f}")

            H, W = gt_frames[0].shape[:2]
            grid = np.zeros((n_vis * H, 2 * W, 3), dtype=np.uint8)
            for vi in range(n_vis):
                grid[vi*H:(vi+1)*H, :W] = gt_frames[vi]
                grid[vi*H:(vi+1)*H, W:] = pca_frames[vi]

            grid_path = os.path.join(vis_dir, f'pca_reconstruction_grid_seed{id_seed}.png')
            Image.fromarray(grid).save(grid_path)
            print(f"  Grid → {grid_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5: OOD identity reconstruction grid
    #   30 identities NOT seen during training, one random expression each.
    #   Layout: 10 rows × 3 cols, each col = [GT | PCA] pair → 10*H × 6*W
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nVisualizing {args.n_ood_vis} OOD identities (not in training set)...")
    ood_seed_start = args.id_seed_start + args.n_identities
    rng_np_ood = np.random.RandomState(77)

    gt_frames_ood  = []
    pca_frames_ood = []
    mse_vals_ood   = []

    with torch.no_grad():
        for ood_idx in tqdm(range(args.n_ood_vis), desc="OOD identities"):
            ood_seed = ood_seed_start + ood_idx
            rng.manual_seed(ood_seed)
            z_ood, fixed_shp_ood, neutral_fl_ood = setup_identity(
                G, rng, ffhq_shapes, c_front, device)

            # Base gaussians + cache backbone for this OOD identity
            _, base_phys_ood = get_physical_attrs(
                G, z_ood, c_front, neutral_fl_ood, args.vis_resolution,
                cache_backbone=True, use_cached_backbone=False)

            base_gaus_ood = {k: base_phys_ood[ae].detach() for k, ae in ATTR_KEYS}
            base_col_ood  = base_phys_ood[GaussianAttribute.COLOR].detach()

            # Sample one random expression
            exp_idx = rng_np_ood.randint(0, len(ffhq_data))
            fp = neutral_fl_ood.clone()
            fp[0, 300:350] = torch.tensor(ffhq_exp[exp_idx],    dtype=torch.float32, device=device)
            fp[0, 353:356] = torch.tensor(ffhq_jaw[exp_idx],    dtype=torch.float32, device=device)
            fp[0, 356:358] = torch.tensor(ffhq_eyelid[exp_idx], dtype=torch.float32, device=device)

            gt_frame, gt_attrs = get_physical_attrs(
                G, z_ood, c_front, fp, args.vis_resolution,
                cache_backbone=False, use_cached_backbone=True)

            recon = pca_project_and_reconstruct(gt_attrs, base_gaus_ood, U_dev, device)
            recon['color'] = base_col_ood

            pca_frame = render_from_physical_attrs(
                G, z_ood, c_front, fp, recon, zero_deltas, args.vis_resolution)

            gt_frames_ood.append(gt_frame)
            pca_frames_ood.append(pca_frame)
            mse_vals_ood.append(np.mean((gt_frame.astype(np.float32)
                                          - pca_frame.astype(np.float32)) ** 2))

    print(f"OOD PCA MSE — mean: {np.mean(mse_vals_ood):.4f} "
          f" min: {np.min(mse_vals_ood):.4f}  max: {np.max(mse_vals_ood):.4f}")

    # Build grid: 10 rows × 3 cols, each = [GT | PCA] → 10*H × 6*W  (same as v1 val grid)
    H, W  = gt_frames_ood[0].shape[:2]
    n_rows, n_cols = 10, 3
    grid_ood = np.zeros((n_rows * H, n_cols * 2 * W, 3), dtype=np.uint8)
    for vi in range(min(args.n_ood_vis, n_rows * n_cols)):
        r = vi // n_cols
        c = vi % n_cols
        grid_ood[r*H:(r+1)*H, (c*2)*W  :(c*2+1)*W] = gt_frames_ood[vi]
        grid_ood[r*H:(r+1)*H, (c*2+1)*W:(c*2+2)*W] = pca_frames_ood[vi]

    ood_grid_path = os.path.join(vis_dir, 'ood_reconstruction_grid.png')
    Image.fromarray(grid_ood).save(ood_grid_path)
    print(f"OOD grid → {ood_grid_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Step 6: Visualize principal components (using first vis identity)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nVisualizing top {args.n_pc_vis} principal components per attribute...")
    grid_side = int(np.ceil(np.sqrt(args.n_pc_vis)))

    info_pc  = identity_info[0]
    z_pc     = info_pc['z'].to(device)
    shp_pc   = info_pc['fixed_shape'].to(device)
    base_gpc = info_pc['base_gaussians_id']
    base_cpc = info_pc['base_color_id']

    neutral_fl_pc = torch.zeros(1, 358, device=device, dtype=torch.float32)
    neutral_fl_pc[:, :300] = shp_pc

    with torch.no_grad():
        ws_pc = G.mapping(z_pc, c_front, truncation_psi=0.7, flame_params=neutral_fl_pc)
        G.synthesis(ws_pc, c_front, neutral_fl_pc,
                    neural_rendering_resolution=args.vis_resolution,
                    cache_backbone=True, use_cached_backbone=False,
                    noise_mode='const', sh_ref_cam=c_front)

        for key_str, attr_enum in ATTR_KEYS:
            Gn  = base_gpc[key_str].shape[1]
            Cn  = base_gpc[key_str].shape[2]
            U_k = U_dev[key_str]
            S_k = svd_S[key_str].to(device)

            pc_frames = []
            for k in range(min(args.n_pc_vis, K_actual)):
                c_unit = torch.zeros(1, K_actual, device=device)
                c_unit[0, k] = S_k[k]
                delta = (c_unit @ U_k).view(1, Gn, Cn)

                recon = {}
                for ks, ae in ATTR_KEYS:
                    recon[ks] = (base_gpc[ks].to(device)
                                 + (delta if ks == key_str else torch.zeros_like(
                                     base_gpc[ks], device=device)))
                recon['color'] = base_cpc.to(device)

                pc_frame = render_from_physical_attrs(
                    G, z_pc, c_front, neutral_fl_pc, recon, zero_deltas, args.vis_resolution)
                pc_frames.append(pc_frame)

            H, W = pc_frames[0].shape[:2]
            pc_grid = np.zeros((grid_side * H, grid_side * W, 3), dtype=np.uint8)
            for pi, frame in enumerate(pc_frames):
                r = pi // grid_side
                c = pi % grid_side
                pc_grid[r*H:(r+1)*H, c*W:(c+1)*W] = frame

            pc_path = os.path.join(vis_dir, f'principal_components_{key_str}.png')
            Image.fromarray(pc_grid).save(pc_path)
            print(f"  {key_str} PCs → {pc_path}")

    print(f"\nDone. Data: {data_dir} | Vis: {vis_dir}")


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
