"""
export_model_for_web.py — Export SVD+MLP assets for browser-based 3DGS inference (v2).

Reads svd_basis.pt and mlp_regressor.pt, produces in --output-dir/:
  base.splat            — neutral Gaussians for worker depth sorting (32 bytes/Gaussian)
  base_attributes.bin   — RGBA32F texture: 4 texels/Gaussian, high-precision attributes
  pca_basis_K16.bin     — RGBA16F basis texture, K=16 components  (~88 MB)
  pca_basis_K32.bin     — RGBA16F basis texture, K=32 components  (~176 MB)
  pca_basis_K64.bin     — RGBA16F basis texture, K=64 components  (~351 MB)
  mlp_weights.json      — MLP weights + w_code as JSON for browser consumption
  camera_params.json    — initial camera for the web viewer

Usage:
  python export_model_for_web.py \\
    --svd-data-path scripts/deform_blendshapes_pca_v2/data/DGGHEAD-158_20500_M5000_Nper1_K64/svd_basis.pt \\
    --mlp-path scripts/deform_blendshapes_pca_v2/data/DGGHEAD-158_20500_M10000_Nper6_K64_h256_512/mlp_regressor.pt \\
    --id-seed 10 \\
    --output-dir scripts/deform_blendshapes_pca_v2/web_inference/splat/data/
"""

import os
import sys
import json
import gzip
import argparse

import numpy as np
import torch

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
os.environ.setdefault('GGHEAD_MODELS_PATH', '/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/')

from src.gghead.model_manager.finder import find_model_manager
from src.gghead.config.gaussian_attribute import GaussianAttribute
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR

FFHQ_FUSED_PARAMS = f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy'
CAM_SCALE = 8.0

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199

# Camera position used for baking view-dependent SH color (front-view of the face)
_SH_CAM_POS = np.array([0.0, 0.0, 0.8])


def eval_sh_at_front(base_xyz, base_color):
    """Evaluate SH degree-1 color at the front-view direction (camera at _SH_CAM_POS).

    Returns float32 array of shape [G, 3] with RGB values in [0, 1].
    """
    dir_pp = base_xyz - _SH_CAM_POS[None, :]
    dir_norm = dir_pp / np.linalg.norm(dir_pp, axis=-1, keepdims=True)
    x, y, z = dir_norm[:, 0], dir_norm[:, 1], dir_norm[:, 2]

    dc  = base_color[:, 0:3]   # [G, 3] SH DC
    sh1 = base_color[:, 3:6]   # [G, 3] Y1^-1
    sh2 = base_color[:, 6:9]   # [G, 3] Y1^0
    sh3 = base_color[:, 9:12]  # [G, 3] Y1^1

    rgb = (SH_C0 * dc
           + SH_C1 * (-y[:, None] * sh1 + z[:, None] * sh2 + (-x[:, None]) * sh3)
           + 0.5)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def export_base_splat(base_xyz, base_scale, base_rot, base_opa, base_color, output_path):
    """
    Export neutral Gaussians as .splat file for web worker depth sorting.

    Format: 32 bytes/Gaussian
      - pos_xyz (float32 × 3)  = 12 bytes   physical position
      - scale   (float32 × 3)  = 12 bytes   physical scale (already exp'd)
      - rgba    (uint8  × 4)   =  4 bytes   RGBA color
      - quat    (uint8  × 4)   =  4 bytes   wxyz quaternion encoded as (q*128+128)

    NOTE: NO importance sorting — same Gaussian order as the model (index 0 to G-1).
    The worker does view-dependent depth sorting; expression perturbations are small.
    """
    G = base_xyz.shape[0]

    # Float32 part: [pos_x, pos_y, pos_z, scale_x, scale_y, scale_z] per Gaussian
    float_part = np.concatenate([
        base_xyz.astype(np.float32),
        base_scale.astype(np.float32),
    ], axis=1)  # [G, 6]

    # Force C-contiguous memory layout
    float_part = np.ascontiguousarray(float_part)

    # Uint8 part: [r, g, b, a, qw, qx, qy, qz] per Gaussian
    rgb = (eval_sh_at_front(base_xyz, base_color) * 255).astype(np.uint8)  # [G, 3]
    alpha = np.clip(base_opa * 255, 0, 255)[:, None].astype(np.uint8)                 # [G, 1]
    quat_enc = np.clip(base_rot * 128 + 128, 0, 255).astype(np.uint8)                 # [G, 4] wxyz
    uint8_part = np.concatenate([rgb, alpha, quat_enc], axis=1)                        # [G, 8]

    # Pack into G × 32 byte records
    splat_buf = np.zeros((G, 32), dtype=np.uint8)
    splat_buf[:, :24] = float_part.view(np.uint8).reshape(G, 24)
    splat_buf[:, 24:] = uint8_part

    with open(output_path, 'wb') as f:
        f.write(splat_buf.tobytes())
    size_mb = G * 32 / 1e6
    print(f"  base.splat:           {G} Gaussians × 32 bytes = {size_mb:.1f} MB")


def export_base_attributes(base_xyz, base_scale, base_rot, base_opa, base_color, output_path):
    """
    Export high-precision base attributes as an RGBA32F binary blob (GPU texture data).

    Binary format:
      Header: 16 bytes
        uint32 num_gaussians
        uint32 tex_width  (= 2048)
        uint32 tex_height (= ceil(G*4 / tex_width))
        uint32 reserved   (= 0)
      Data: G × 4 RGBA32F texels (= G × 4 × 4 float32 values)
        Texel 0: [pos_x,   pos_y,   pos_z,   scale_x]
        Texel 1: [scale_y, scale_z, rot_w,   rot_x  ]
        Texel 2: [rot_y,   rot_z,   opacity, color_R]
        Texel 3: [color_G, color_B, 0.0,     0.0    ]

    Color channels are SH degree-1 evaluated at the front-view direction, in [0, 1].
    Quaternion order: wxyz  (rot_w = b1.z, rot_x = b1.w, rot_y = b2.x, rot_z = b2.y).
    """
    G = base_xyz.shape[0]
    tex_width = 2048
    total_texels = G * 4
    tex_height = int(np.ceil(total_texels / tex_width))

    # Evaluate SH degree-1 color at front-view direction
    rgb = eval_sh_at_front(base_xyz, base_color)  # [G, 3] float32 in [0, 1]

    # Build [G, 4, 4] float32 texel array
    texels = np.zeros((G, 4, 4), dtype=np.float32)
    # Texel 0: [pos_x, pos_y, pos_z, scale_x]
    texels[:, 0, 0] = base_xyz[:, 0]
    texels[:, 0, 1] = base_xyz[:, 1]
    texels[:, 0, 2] = base_xyz[:, 2]
    texels[:, 0, 3] = base_scale[:, 0]
    # Texel 1: [scale_y, scale_z, rot_w, rot_x]
    texels[:, 1, 0] = base_scale[:, 1]
    texels[:, 1, 1] = base_scale[:, 2]
    texels[:, 1, 2] = base_rot[:, 0]  # w
    texels[:, 1, 3] = base_rot[:, 1]  # x
    # Texel 2: [rot_y, rot_z, opacity, color_R]
    texels[:, 2, 0] = base_rot[:, 2]  # y
    texels[:, 2, 1] = base_rot[:, 3]  # z
    texels[:, 2, 2] = base_opa
    texels[:, 2, 3] = rgb[:, 0]       # R in [0, 1]
    # Texel 3: [color_G, color_B, 0.0, 0.0]
    texels[:, 3, 0] = rgb[:, 1]       # G in [0, 1]
    texels[:, 3, 1] = rgb[:, 2]       # B in [0, 1]
    # texels[:, 3, 2:] = 0.0  (already zero-initialized)

    # Flatten to [G*4, 4] and pad to fill the texture rectangle
    flat_texels = texels.reshape(-1, 4)
    pad_size = tex_width * tex_height - total_texels
    if pad_size > 0:
        flat_texels = np.vstack([flat_texels, np.zeros((pad_size, 4), dtype=np.float32)])

    header = np.array([G, tex_width, tex_height, 0], dtype=np.uint32)
    with open(output_path, 'wb') as f:
        f.write(header.tobytes())
        f.write(flat_texels.astype(np.float32).tobytes())
    nbytes = 16 + tex_width * tex_height * 16
    print(f"  base_attributes.bin:  {tex_width}×{tex_height} RGBA32F = {nbytes / 1e6:.1f} MB")


def export_pca_basis(U_xyz, U_scale, U_rot, U_opa, G, K_web, output_path):
    """
    Export truncated PCA basis as an RGBA16F binary blob (GPU texture data).

    Binary format:
      Header: 16 bytes
        uint32 K_web          (number of PCA components in this file)
        uint32 num_gaussians
        uint32 tex_width  (= 4096)
        uint32 tex_height (= ceil(K_web*G*3 / tex_width))
      Data: K_web × G × 3 RGBA16F texels
        Layout: k (outer) → g (middle) → sub-texel s∈{0,1,2} (inner)
        For texel at linear index  idx = k*G*3 + g*3 + s:
          texture coords = (idx % tex_width, idx / tex_width)
        Texel 0: [d_pos_x,   d_pos_y,   d_pos_z,  d_scale_x]
        Texel 1: [d_scale_y, d_scale_z, d_rot_w,  d_rot_x  ]
        Texel 2: [d_rot_y,   d_rot_z,   d_opacity, 0.0      ]
    """
    # Truncate to K_web most important components (sorted by variance)
    U_xyz_k   = U_xyz[:K_web].reshape(K_web, G, 3)    # [K_web, G, 3]
    U_scale_k = U_scale[:K_web].reshape(K_web, G, 3)  # [K_web, G, 3]
    U_rot_k   = U_rot[:K_web].reshape(K_web, G, 4)    # [K_web, G, 4]  wxyz order
    U_opa_k   = U_opa[:K_web].reshape(K_web, G)        # [K_web, G]

    # Build [K_web, G, 3, 4] float16 tensor
    basis = np.zeros((K_web, G, 3, 4), dtype=np.float16)
    # Texel 0: [d_pos_x, d_pos_y, d_pos_z, d_scale_x]
    basis[:, :, 0, :3] = U_xyz_k.astype(np.float16)
    basis[:, :, 0,  3] = U_scale_k[:, :, 0].astype(np.float16)
    # Texel 1: [d_scale_y, d_scale_z, d_rot_w, d_rot_x]
    basis[:, :, 1, 0]  = U_scale_k[:, :, 1].astype(np.float16)
    basis[:, :, 1, 1]  = U_scale_k[:, :, 2].astype(np.float16)
    basis[:, :, 1, 2]  = U_rot_k[:, :, 0].astype(np.float16)   # w
    basis[:, :, 1, 3]  = U_rot_k[:, :, 1].astype(np.float16)   # x
    # Texel 2: [d_rot_y, d_rot_z, d_opacity, 0.0]
    basis[:, :, 2, 0]  = U_rot_k[:, :, 2].astype(np.float16)   # y
    basis[:, :, 2, 1]  = U_rot_k[:, :, 3].astype(np.float16)   # z
    basis[:, :, 2, 2]  = U_opa_k.astype(np.float16)
    # basis[:, :, 2, 3] = 0.0  (already zero-initialized)

    # Flatten and pack for texture upload
    total_texels  = K_web * G * 3
    basis_tex_w   = 8192
    basis_tex_h   = int(np.ceil(total_texels / basis_tex_w))

    flat = basis.reshape(-1, 4)  # [K_web*G*3, 4]  dtype=float16
    pad  = basis_tex_w * basis_tex_h - total_texels
    if pad > 0:
        flat = np.vstack([flat, np.zeros((pad, 4), dtype=np.float16)])

    # Store bit patterns as uint16 (WebGL reads HALF_FLOAT as raw uint16 bit patterns)
    flat_u16 = flat.view(np.uint16)

    header = np.array([K_web, G, basis_tex_w, basis_tex_h], dtype=np.uint32)
    raw_data = header.tobytes() + flat_u16.tobytes()
    with open(output_path, 'wb') as f:
        f.write(raw_data)
    nbytes = len(raw_data)
    print(f"  pca_basis_K{K_web:2d}.bin:   {basis_tex_w}×{basis_tex_h} RGBA16F = {nbytes / 1e6:.0f} MB")

    # Also save gzip-compressed version for web delivery
    gz_path = output_path + '.gz'
    with gzip.open(gz_path, 'wb', compresslevel=6) as f:
        f.write(raw_data)
    gz_size = os.path.getsize(gz_path)
    print(f"  pca_basis_K{K_web:2d}.bin.gz: {gz_size / 1e6:.0f} MB  ({gz_size / nbytes * 100:.0f}% of raw)")


def export_camera_params(base_xyz, output_path):
    """Export initial camera parameters for the web viewer as JSON.

    Uses a front-facing camera at z=0.8 looking toward the Gaussian cloud origin.
    The rotation [[1,0,0],[0,-1,0],[0,0,-1]] encodes a camera looking along -Z with Y flipped
    (standard 3DGS web viewer convention, compatible with getViewMatrix() in main.js).
    """
    centroid = base_xyz.mean(axis=0).tolist()
    camera_params = {
        "position": list(_SH_CAM_POS.tolist()),
        "rotation": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "fx": 1160.0,
        "fy": 1160.0,
        "width": 1920,
        "height": 1080,
        "centroid": centroid,
    }
    with open(output_path, 'w') as f:
        json.dump(camera_params, f, indent=2)
    print(f"  camera_params.json:   pos={_SH_CAM_POS.tolist()}, centroid={[f'{v:.3f}' for v in centroid]}")


def export_mlp_weights(mlp_path, w_code, output_path):
    """
    Export MLP weights and w_code as a JSON file for browser-side inference.

    JSON format:
      {
        "K": <int>,              MLP output K (number of PCA components in training)
        "activation": "leaky_relu",
        "alpha": 0.2,
        "w_code": [<float>, ...],  512-dim identity latent
        "layers": [
          {"in": <int>, "out": <int>, "weight": [[...]], "bias": [...]},
          ...
        ]
      }

    Layer weights are row-major [out, in] matching PyTorch storage.
    """
    ckpt = torch.load(mlp_path, map_location='cpu')
    sd   = ckpt['state_dict']
    K    = int(ckpt['K'])

    layers = [
        {
            'in':     int(sd['net.0.weight'].shape[1]),
            'out':    int(sd['net.0.weight'].shape[0]),
            'weight': sd['net.0.weight'].numpy().tolist(),
            'bias':   sd['net.0.bias'].numpy().tolist(),
        },
        {
            'in':     int(sd['net.2.weight'].shape[1]),
            'out':    int(sd['net.2.weight'].shape[0]),
            'weight': sd['net.2.weight'].numpy().tolist(),
            'bias':   sd['net.2.bias'].numpy().tolist(),
        },
        {
            'in':     int(sd['net.4.weight'].shape[1]),
            'out':    int(sd['net.4.weight'].shape[0]),
            'weight': sd['net.4.weight'].numpy().tolist(),
            'bias':   sd['net.4.bias'].numpy().tolist(),
        },
    ]
    in_dim = layers[0]['in']
    mlp_json = {
        'K':          K,
        'activation': 'leaky_relu',
        'alpha':      0.2,
        'w_code':     w_code.tolist(),
        'layers':     layers,
    }
    with open(output_path, 'w') as f:
        json.dump(mlp_json, f)
    dims = '→'.join(f"{l['out']}" for l in layers)
    print(f"  mlp_weights.json:     K={K}, arch={in_dim}→{dims}, w_code=[512]")


def compute_base_gaussians(run_name, checkpoint, id_seed, device):
    """Load the model and compute base Gaussian attributes + W code at neutral expression.

    Returns:
        base_xyz   [G, 3]  float32  physical position
        base_scale [G, 3]  float32  physical scale (exp'd)
        base_rot   [G, 4]  float32  physical rotation (normalized quat, wxyz)
        base_opa   [G]     float32  physical opacity (sigmoid'd)
        base_color [G, C]  float32  SH color coefficients
        w_code     [512]   float32  identity latent from backbone_ws[:, 0, :]
    """
    print(f"Loading model {run_name} checkpoint {checkpoint} ...")
    model_manager = find_model_manager(run_name)
    ckpt_id = model_manager._resolve_checkpoint_id(checkpoint)
    G_model = model_manager.load_checkpoint(ckpt_id, load_ema=True).to(device)
    G_model.eval()
    G_model._config.use_flame_rasterization = 0

    ffhq_data   = np.load(FFHQ_FUSED_PARAMS)
    ffhq_cams   = ffhq_data[:, -6:]
    ffhq_shapes = ffhq_data[:, :300]

    mean_cam = np.mean(ffhq_cams, axis=0)
    c_front  = torch.tensor(mean_cam, dtype=torch.float32).unsqueeze(0).to(device)
    c_front[0, 3] = CAM_SCALE

    rng = torch.Generator(device)
    rng.manual_seed(id_seed)
    z = torch.randn((1, G_model._config.z_dim), device=device, generator=rng)
    flame_shape_id = torch.randint(0, ffhq_shapes.shape[0], (1,), generator=rng,
                                   device=device).cpu().item()
    fixed_shape = torch.tensor(ffhq_shapes[flame_shape_id, :300],
                                dtype=torch.float32).to(device)

    neutral_flame = torch.zeros(1, 358, device=device, dtype=torch.float32)
    neutral_flame[:, :300] = fixed_shape

    print(f"  id_seed={id_seed}, flame_shape_id={flame_shape_id}")

    with torch.no_grad():
        ws_init = G_model.mapping(z, c_front, truncation_psi=0.7, flame_params=neutral_flame)
        backbone_ws, _ = ws_init
        w_code = backbone_ws[:, 0, :]  # [1, 512]

        output = G_model.synthesis(
            ws_init, c_front, neutral_flame,
            neural_rendering_resolution=512,
            cache_backbone=False, use_cached_backbone=False,
            noise_mode='const', sh_ref_cam=c_front,
            return_gaussian_attributes=True,
        )

    ga_raw = output.returned_gaussian_attributes
    base_xyz   = ga_raw[GaussianAttribute.POSITION][0].float().cpu().numpy()   # [G, 3]
    base_scale = G_model._gaussian_model.scaling_activation(
        ga_raw[GaussianAttribute.SCALE])[0].float().cpu().numpy()              # [G, 3]
    base_rot   = G_model._gaussian_model.rotation_activation(
        ga_raw[GaussianAttribute.ROTATION])[0].float().cpu().numpy()           # [G, 4]
    base_opa   = G_model._gaussian_model.opacity_activation(
        ga_raw[GaussianAttribute.OPACITY])[0, :, 0].float().cpu().numpy()      # [G]
    base_color = ga_raw[GaussianAttribute.COLOR][0].float().cpu().numpy()      # [G, C]
    w_code_np  = w_code[0].float().cpu().numpy()                               # [512]

    G_num = base_xyz.shape[0]
    print(f"  G={G_num} Gaussians, color_dim={base_color.shape[-1]}")

    return base_xyz, base_scale, base_rot, base_opa, base_color, w_code_np


def export_identity(run_name, checkpoint, id_seed, mlp_path, G_expected, seed_dir, device):
    """Export per-identity files (base.splat, base_attributes.bin, mlp_weights.json, camera_params.json)."""
    os.makedirs(seed_dir, exist_ok=True)
    base_xyz, base_scale, base_rot, base_opa, base_color, w_code = compute_base_gaussians(
        run_name, checkpoint, id_seed, device
    )
    assert G_expected == base_xyz.shape[0], (
        f"Gaussian count mismatch: SVD has G={G_expected}, model produced G={base_xyz.shape[0]}"
    )

    export_base_splat(
        base_xyz, base_scale, base_rot, base_opa, base_color,
        os.path.join(seed_dir, 'base.splat')
    )
    export_base_attributes(
        base_xyz, base_scale, base_rot, base_opa, base_color,
        os.path.join(seed_dir, 'base_attributes.bin')
    )
    export_mlp_weights(
        mlp_path, w_code,
        os.path.join(seed_dir, 'mlp_weights.json')
    )
    export_camera_params(
        base_xyz,
        os.path.join(seed_dir, 'camera_params.json')
    )


def main():
    parser = argparse.ArgumentParser(
        description='Export SVD+MLP assets for browser 3DGS inference (v2).'
    )
    parser.add_argument('--svd-data-path', required=True,
                        help='Path to svd_basis.pt')
    parser.add_argument('--mlp-path', required=True,
                        help='Path to mlp_regressor.pt')
    parser.add_argument('--id-seeds', required=True, type=str,
                        help='Comma-separated identity seeds (e.g. "4,17,25,107,110,118")')
    parser.add_argument('--run-name', default='DGGHEAD-158',
                        help='Model run name (default: DGGHEAD-158)')
    parser.add_argument('--checkpoint', default=20500, type=int,
                        help='Checkpoint step (default: 20500)')
    parser.add_argument('--output-dir', default='splat/data/',
                        help='Output directory (default: splat/data/)')
    parser.add_argument('--device', default='cuda:0',
                        help='Torch device (default: cuda:0)')
    args = parser.parse_args()

    id_seeds = [int(s.strip()) for s in args.id_seeds.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # ── Load SVD basis (shared across all identities) ─────────────────────────
    print(f"\nLoading SVD data from  {args.svd_data_path}")
    data   = torch.load(args.svd_data_path, map_location='cpu')
    G      = int(data['G'])
    K_full = int(data['K'])
    print(f"  G={G} Gaussians,  K={K_full} PCA components")

    U_xyz   = data['U_xyz'].float().numpy()        # [K, G*3]
    U_scale = data['U_scale'].float().numpy()      # [K, G*3]
    U_rot   = data['U_rotation'].float().numpy()   # [K, G*4]
    U_opa   = data['U_opacity'].float().numpy()    # [K, G*1]

    # ── Export shared PCA basis to output-dir (once) ──────────────────────────
    print(f"\nExporting shared PCA basis to {args.output_dir}")
    for K_web in [16, 32, 64]:
        if K_web > K_full:
            print(f"  Skipping K={K_web} (only {K_full} components available in data)")
            continue
        export_pca_basis(
            U_xyz, U_scale, U_rot, U_opa, G, K_web,
            os.path.join(args.output_dir, f'pca_basis_K{K_web}.bin')
        )

    # ── Export per-identity files to seed_{N}/ subdirectories ──────────────────
    print(f"\nLoading MLP from       {args.mlp_path}")
    for id_seed in id_seeds:
        seed_dir = os.path.join(args.output_dir, f'seed_{id_seed}')
        print(f"\n{'='*60}")
        print(f"Exporting identity seed={id_seed} → {seed_dir}")
        print(f"{'='*60}")
        export_identity(
            args.run_name, args.checkpoint, id_seed,
            args.mlp_path, G, seed_dir, device,
        )

    print(f"\nDone. Shared PCA basis in {args.output_dir}, per-identity in seed_*/")
    print("Next: run export_driving_sequence.py, then serve splat/ with python -m http.server 8000")


if __name__ == '__main__':
    main()
