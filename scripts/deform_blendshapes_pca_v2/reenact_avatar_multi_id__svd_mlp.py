"""
Step 3: Reenactment using SVD + MLP deformation bypass (v2, identity-independent).

At each frame:
  1. Extract 55-dim FLAME expression vector [exp(50), jaw(3), eyelid(2)]
  2. Concatenate with 512-dim W code (identity latent): [W, flame_55] → 567-dim
  3. MLP forward: 567 → 4*K PCA coefficients
  4. Reconstruct physical residuals via matrix multiply:
       delta_xyz     = (c_xyz    @ U_xyz).view(G, 3)
       delta_scale   = (c_scale  @ U_scale).view(G, 3)
       delta_rot     = (c_rot    @ U_rot).view(G, 4)
       delta_opacity = (c_opacity@ U_opacity).view(G, 1)
  5. Add to per-identity base_gaussians (computed on-the-fly at identity setup)
  6. Enforce geometric constraints (clamp scale, normalize rot, clamp opacity)
  7. Inject into synthesis via post_act_blendshapes trick (base=attrs, delta=0)
  8. 3DGS rasterize

Deformation network is completely bypassed — only MLP + matrix multiplies at inference.
Per-identity bases and W codes are computed on-the-fly (no base_* keys in svd_basis.pt v2).
"""

from dataclasses import dataclass
from typing import Optional
import tyro
import os
import sys
from glob import glob
from tqdm import tqdm
import json

import mediapy
import numpy as np
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
os.environ.setdefault('GGHEAD_MODELS_PATH', '/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/')

from dreifus.image import Img
from elias.util import ensure_directory_exists
from elias.util.batch import batchify_sliced
from src.gghead.model_manager.finder import find_model_manager
from src.gghead.config.gaussian_attribute import GaussianAttribute
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR

FFHQ_FUSED_PARAMS = f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy'

ATTR_KEYS = [
    ('xyz',      None),
    ('scale',    None),
    ('rotation', None),
    ('opacity',  None),
]


# ─────────────────────────────────────── MLP (must match training v2) ─────────

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


# ────────────────────────────────────────────────── helpers ───────────────────

def parse_id_seed(id_seed_str: str) -> list:
    id_seed_str = id_seed_str.strip()
    if '-' in id_seed_str and ',' not in id_seed_str:
        parts = id_seed_str.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return list(range(int(parts[0]), int(parts[1]) + 1))
    if ',' in id_seed_str:
        return [int(x.strip()) for x in id_seed_str.split(',')]
    return [int(id_seed_str)]


def parse_smirk_processed_video(emica_path, gt_img_path=None, max_len=None):
    sl = slice(0, max_len, 1)
    frame_folders = sorted(glob(os.path.join(emica_path, '*/')),
                           key=lambda x: int(x.split('/')[-2]))

    if gt_img_path is None:
        frame_paths = sorted(glob(emica_path + '/*/*/detections/*_000.png'))
    else:
        frame_paths = sorted(glob(gt_img_path + '/*.png'),
                             key=lambda x: int(os.path.basename(x[:-4])))
    images = [np.array(Image.open(x)) for x in tqdm(frame_paths[sl], desc='Loading images')]

    shapecodes, expcodes, globalposes, jawposes, flameorths, eyelids = [], [], [], [], [], []
    for folder in tqdm(frame_folders[sl], desc='Loading FLAME params'):
        shapecodes.append(np.load(f"{folder}/shape.npy"))
        expcodes.append(np.load(f"{folder}/exp.npy"))
        globalposes.append(np.load(f"{folder}/globalpose.npy"))
        jawposes.append(np.load(f"{folder}/jawpose.npy"))
        flameorths.append(np.load(f"{folder}/cam.npy"))
        eyelids.append(np.load(f"{folder}/eyelid.npy"))

    return (torch.tensor(np.array(shapecodes)),
            torch.tensor(np.array(expcodes)),
            torch.tensor(np.array(globalposes)),
            torch.tensor(np.array(jawposes)),
            torch.tensor(np.array(flameorths)),
            torch.tensor(np.array(eyelids)),
            torch.tensor(np.array(images)))


def reconstruct_physical_attrs_batch(mlp, w_code, flame_55_batch, svd_data,
                                      base_gaussians_id, base_color, device, K):
    """
    mlp: MLPRegressor (v2)
    w_code: [1, 512] identity latent (expanded to batch inside)
    flame_55_batch: [B, 55] tensor on device
    base_gaussians_id: per-identity base dict {key_str: [1, G, C]} on device
    Returns dict: {xyz, scale, rotation, opacity: [B, G, C],  color: [1, G, C_color]}
    """
    B = flame_55_batch.shape[0]
    w_code_batch = w_code.expand(B, -1)                          # [B, 512]
    mlp_input = torch.cat([w_code_batch, flame_55_batch], dim=-1)  # [B, 567]
    c_all = mlp(mlp_input)                                        # [B, 4K]
    c_parts = {
        'xyz':      c_all[:, 0*K:1*K],
        'scale':    c_all[:, 1*K:2*K],
        'rotation': c_all[:, 2*K:3*K],
        'opacity':  c_all[:, 3*K:4*K],
    }
    phys = {}
    for key_str, _ in ATTR_KEYS:
        base = base_gaussians_id[key_str]               # [1, G, C] — per-identity, on device
        U    = svd_data[f'U_{key_str}'].to(device)      # [K, G*C]
        Gn   = base.shape[1]
        Cn   = base.shape[2]
        delta = (c_parts[key_str] @ U).view(-1, Gn, Cn) # [B, G, C]
        phys[key_str] = base + delta                     # broadcast [1,G,C]+[B,G,C]
    phys['color'] = base_color                           # [1, G, C_color] — per-identity, on device
    return phys


def apply_geometric_constraints(phys, ov: float):
    """Enforce valid physical bounds (in-place)."""
    phys['scale']    = torch.clamp(phys['scale'], min=1e-5)
    phys['rotation'] = F.normalize(phys['rotation'], p=2, dim=-1)
    phys['opacity']  = torch.clamp(phys['opacity'],
                                   min=-ov + 1e-6,
                                   max=1.0 + ov - 1e-6)
    return phys


# ─────────────────────────────────────────────────── args ─────────────────────

def get_face_mask(image: np.ndarray, face_mesh) -> Optional[np.ndarray]:
    """Return binary float32 [H, W] mask of face oval region using MediaPipe FaceMesh."""
    import mediapipe as mp
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0]
    h, w = image.shape[:2]
    oval_indices = {p for pair in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for p in pair}
    pts = np.array([(int(landmarks.landmark[i].x * w),
                     int(landmarks.landmark[i].y * h)) for i in oval_indices])
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(np.float32)


def make_error_frame(approx: np.ndarray, gt: np.ndarray,
                     max_err: float = 50.0,
                     mask: Optional[np.ndarray] = None,
                     log_scale: bool = False) -> np.ndarray:
    """
    Perceptually nice pixelwise error visualization.
    approx, gt: uint8 [H, W, 3]
    mask: optional float32 [H, W] binary face mask (zeros out non-face pixels)
    Returns: uint8 [H, W, 3] colormap image where
             dark-blue = 0 error, green/yellow = mid, red/white = max_err+
    """
    diff = np.abs(approx.astype(np.float32) - gt.astype(np.float32))
    err  = diff.mean(axis=2)                          # [H, W] mean over RGB
    if mask is not None:
        err = err * mask
    if log_scale:
        err     = np.log1p(err)
        max_err = np.log1p(max_err)
    err_u8 = np.clip(err / max_err * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(err_u8, cv2.COLORMAP_TURBO)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Burn a small scale-bar strip at the bottom
    bar_h = max(5, colored.shape[0] // 80)
    bar   = np.linspace(0, 255, colored.shape[1], dtype=np.uint8)

    bar_bgr = cv2.applyColorMap(bar[None, :], cv2.COLORMAP_TURBO)
    bar_rgb = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2RGB)

    colored[-bar_h:] = bar_rgb
    return colored


@dataclass
class Args:
    svd_data_path: str = ''           # path to svd_basis.pt (v2 format)
    mlp_path: str = ''                # path to mlp_regressor.pt (v2 format)
    vid_processed_path: Optional[str] = None
    gt_img_path: Optional[str]        = None
    pairs_list: Optional[str]         = None
    FPS: int = 30
    DEVICE: str = 'cuda:0'
    run_name: str = 'DGGHEAD-158'
    checkpoint: int = 20500
    resolution: int = 512
    savgol_win: int = 5
    max_len: Optional[int] = None
    id_seed: str = '10'
    cam_scale: Optional[float] = None
    joint_c_front: int = 0
    render_mode: str = 'RGB'
    batch_size: int = 4
    err_max_pixel: float = 50.0   # pixel value (0-255) that maps to max error color
    truncation_psi: float = 0.7
    face_mask: bool = True        # restrict error to face region only (MediaPipe FaceMesh)
    log_error: bool = False       # apply log-scale to error map


# ─────────────────────────────────────────────────── main ─────────────────────

def main(args: Args):
    device = torch.device(args.DEVICE)
    truncation_psi = args.truncation_psi
    script_dir = os.path.dirname(os.path.abspath(__file__))

    assert args.svd_data_path, "Provide --svd-data-path"
    assert args.mlp_path,      "Provide --mlp-path"

    # ── load SVD data ─────────────────────────────────────────────────────────
    print(f"Loading SVD data from {args.svd_data_path}...")
    svd_data = torch.load(args.svd_data_path, map_location='cpu')
    K     = int(svd_data['U_xyz'].shape[0])   # actual number of PCA components
    G_num = int(svd_data['G'])
    # v2 format: no base_* keys; print U_* shapes only
    for k in ['U_xyz', 'U_scale', 'U_rotation', 'U_opacity']:
        print(f"  {k}: {tuple(svd_data[k].shape)}")

    # ── load MLP ──────────────────────────────────────────────────────────────
    print(f"Loading MLP from {args.mlp_path}...")
    mlp_ckpt = torch.load(args.mlp_path, map_location='cpu')
    mlp = MLPRegressor(
        in_dim=mlp_ckpt['in_dim'],
        hidden1=mlp_ckpt['hidden1'],
        hidden2=mlp_ckpt['hidden2'],
        out_dim=mlp_ckpt['out_dim']).to(device)
    mlp.load_state_dict(mlp_ckpt['state_dict'])
    mlp.eval()
    print(f"  MLP: {mlp_ckpt['in_dim']} → {mlp_ckpt['out_dim']}  (K={K})")

    # ── load model ────────────────────────────────────────────────────────────
    model_manager = find_model_manager(args.run_name)
    ckpt = model_manager._resolve_checkpoint_id(args.checkpoint)
    G = model_manager.load_checkpoint(ckpt, load_ema=True).to(device)
    G.eval()
    G._config.use_flame_rasterization = 0
    G._config.render_mode = args.render_mode

    ov = getattr(G._config, 'opacity_overshoot', 0.0)

    # ── FFHQ params for identity setup ────────────────────────────────────────
    ffhq_data   = np.load(FFHQ_FUSED_PARAMS)
    ffhq_cams   = ffhq_data[:, -6:]
    ffhq_shapes = ffhq_data[:, :300]

    # ── pairs ─────────────────────────────────────────────────────────────────
    if args.pairs_list:
        file_pairs = []
        with open(args.pairs_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    gp = line.strip()
                    vp = os.path.join(gp, 'smirk')
                    file_pairs.append((vp, gp))
                else:
                    file_pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        assert args.vid_processed_path, "Provide --vid-processed-path or --pairs-list"
        file_pairs = [(args.vid_processed_path, args.gt_img_path)]

    id_seeds = parse_id_seed(args.id_seed)
    if len(id_seeds) == len(file_pairs):
        print(f"1:1 mode — {len(file_pairs)} video(s) paired with {len(id_seeds)} ID(s)")
        video_id_pairs = [(pair, [seed]) for pair, seed in zip(file_pairs, id_seeds)]
    else:
        print(f"Processing {len(file_pairs)} video(s) × {len(id_seeds)} ID(s)")
        video_id_pairs = [(pair, id_seeds) for pair in file_pairs]

    face_mesh = None
    if args.face_mask:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop over videos and identities
    # ══════════════════════════════════════════════════════════════════════════
    for (vid_processed_path, gt_img_path), current_id_seeds in tqdm(video_id_pairs, desc='Videos'):
        vid_name = os.path.basename(vid_processed_path.rstrip(os.sep))
        if vid_name == 'smirk':
            vid_name = vid_processed_path.rstrip(os.sep).split(os.sep)[-2]

        (shapecodes, expcodes, globalposes, jawposes,
         flameorths, eyelids, driving_images) = parse_smirk_processed_video(
            vid_processed_path, gt_img_path, args.max_len)

        cams_new     = torch.cat([globalposes, flameorths], dim=-1)   # [T, 6]
        flame_params = torch.cat([shapecodes, expcodes, globalposes,
                                  jawposes, eyelids], dim=-1)          # [T, 358]
        flame_params = flame_params.to(device)

        mean_ffhq_cam = np.mean(ffhq_cams, axis=0)
        c_front = torch.tensor(mean_ffhq_cam, dtype=torch.float32).unsqueeze(0).to(device)
        # Match create_svd_basis_and_dataset.py: apply cam_scale to c_front so that
        # W codes and base gaussians are computed with the same camera as during training.
        if args.cam_scale is not None:
            c_front[0, 3] = args.cam_scale

        c_render = cams_new.clone().to(device)
        c_smooth = savgol_filter(c_render.cpu().numpy(),
                                 window_length=7, polyorder=3, axis=0)
        c_smooth = torch.tensor(c_smooth).to(device)
        c_smooth[:, 3:] = c_smooth[:, 3:].mean(dim=0, keepdim=True)

        sh_ref_cam = c_front

        if args.cam_scale is not None:
            c_smooth[:, 3:]  = c_front[:, 3:]
            c_smooth[:, 3]   = args.cam_scale

        savgol_win = args.savgol_win
        fp_smooth = torch.tensor(
            savgol_filter(flame_params.cpu().numpy(),
                          window_length=savgol_win, polyorder=3, axis=0),
            device=device)

        for current_id_seed in tqdm(current_id_seeds, desc=f'IDs for {vid_name}', leave=False):
            rng = torch.Generator(device)
            rng.manual_seed(current_id_seed)
            z = torch.randn((1, G._config.z_dim), device=device, generator=rng)

            flame_shape_id = torch.randint(0, ffhq_shapes.shape[0], (1,),
                                           generator=rng, device=device).cpu().item()
            fixed_shape = torch.tensor(ffhq_shapes[flame_shape_id, :300],
                                       dtype=torch.float32, device=device)

            if args.joint_c_front:
                c_front    = torch.tensor(ffhq_cams[flame_shape_id, :],
                                          dtype=torch.float32).unsqueeze(0).to(device)
                sh_ref_cam = c_front

            fp_smooth_id = fp_smooth.clone()
            fp_smooth_id[:, :300] = fixed_shape.unsqueeze(0).expand(len(fp_smooth), -1)

            # ── Per-identity setup: extract W code + compute base gaussians ──
            neutral_flame = torch.zeros(1, 358, device=device, dtype=torch.float32)
            neutral_flame[:, :300] = fixed_shape

            c_mapping_single = c_front.clone()
            # Do NOT apply use_concat zeroing here — W codes in svd_basis.pt were
            # computed without it (create_svd_basis_and_dataset.py line 286).

            with torch.no_grad():
                ws_init = G.mapping(z, c_mapping_single, truncation_psi=truncation_psi,
                                    flame_params=neutral_flame)
                backbone_ws, _ = ws_init
                w_code = backbone_ws[:, 0, :]   # [1, 512]

                # Compute base gaussians (neutral expression).
                # Do NOT cache backbone here — cache is set later by the first loop batch
                # at the correct batch size to avoid batch-size mismatch in GT synthesis.
                output_base = G.synthesis(
                    ws_init, c_front, neutral_flame,
                    neural_rendering_resolution=args.resolution,
                    cache_backbone=False, use_cached_backbone=False,
                    noise_mode='const', sh_ref_cam=c_front,
                    return_gaussian_attributes=True,
                )
                ga_raw = output_base.returned_gaussian_attributes
                base_gaussians_id = {
                    'xyz':      ga_raw[GaussianAttribute.POSITION].detach(),
                    'scale':    G._gaussian_model.scaling_activation(ga_raw[GaussianAttribute.SCALE]).detach(),
                    'rotation': G._gaussian_model.rotation_activation(ga_raw[GaussianAttribute.ROTATION]).detach(),
                    'opacity':  G._gaussian_model.opacity_activation(ga_raw[GaussianAttribute.OPACITY]).detach(),
                }
                base_color = ga_raw[GaussianAttribute.COLOR].detach()   # [1, G, C_color]

            C_color = base_color.shape[-1]

            # Precompute 55-dim MLP inputs for all frames
            flame_55_all = torch.cat([
                fp_smooth_id[:, 300:350],   # exp  [T, 50]
                fp_smooth_id[:, 353:356],   # jaw  [T, 3]
                fp_smooth_id[:, 356:358],   # eyelid [T, 2]
            ], dim=-1).float()              # [T, 55]

            # Zero deltas (precomputed once per identity)
            zero_deltas = {
                'xyz':      torch.zeros(59, G_num, 3,       device=device),
                'scale':    torch.zeros(59, G_num, 3,       device=device),
                'rotation': torch.zeros(59, G_num, 4,       device=device),
                'opacity':  torch.zeros(59, G_num, 1,       device=device),
                'color':    torch.zeros(59, G_num, C_color, device=device),
            }

            c_mapping = c_front.repeat(len(fp_smooth_id), 1)
            if G._config.use_concat:
                c_mapping[:, 3:] = 0

            all_frames            = []
            all_generated_frames  = []
            all_driving_frames    = []
            all_gt_frames         = []
            all_comparison_frames = []   # [approx | gt | error | driver]

            # First loop batch caches backbone at args.batch_size (avoids batch=1 mismatch)
            cache_backbone_loop      = True
            use_cached_backbone_loop = False

            with torch.no_grad():
                for c_map_batch, c_ren_batch, fp_batch, fl55_batch, drv_batch in tqdm(
                    zip(
                        batchify_sliced(c_mapping,      batch_size=args.batch_size),
                        batchify_sliced(c_smooth,       batch_size=args.batch_size),
                        batchify_sliced(fp_smooth_id,   batch_size=args.batch_size),
                        batchify_sliced(flame_55_all,   batch_size=args.batch_size),
                        batchify_sliced(driving_images, batch_size=args.batch_size),
                    ),
                    total=(len(fp_smooth_id) // args.batch_size
                           + int(len(fp_smooth_id) % args.batch_size != 0)),
                    leave=False,
                ):
                    B = fp_batch.shape[0]
                    # Skip incomplete tail to avoid caching backbone at a smaller batch size
                    if B != args.batch_size:
                        continue

                    c_map_batch = c_map_batch.clone()
                    c_map_batch[:, :3] = 0

                    # ── MLP: [w_code, flame_55] → physical attrs ─────────────
                    fl55_dev = fl55_batch.to(device).float()
                    phys = reconstruct_physical_attrs_batch(
                        mlp, w_code, fl55_dev, svd_data,
                        base_gaussians_id, base_color, device, K)
                    phys = apply_geometric_constraints(phys, ov)

                    # ── inject via post_act_blendshapes trick ─────────────────
                    post_act_bs = {
                        'base_xyz':       phys['xyz'],
                        'base_scale':     phys['scale'],
                        'base_rotation':  phys['rotation'],
                        'base_opacity':   phys['opacity'],
                        'base_color':     phys['color'],
                        'delta_xyz':      zero_deltas['xyz'],
                        'delta_scale':    zero_deltas['scale'],
                        'delta_rotation': zero_deltas['rotation'],
                        'delta_opacity':  zero_deltas['opacity'],
                        'delta_color':    zero_deltas['color'],
                    }

                    # ── backbone mapping (identity-only w) ─────────────────
                    z_batch  = z.repeat(B, 1)
                    w_batch  = G.mapping(z_batch, c_map_batch,
                                         truncation_psi=truncation_psi,
                                         flame_params=fp_batch)

                    # First batch caches backbone at the correct batch size
                    output = G.synthesis(
                        w_batch, c_ren_batch, fp_batch,
                        sh_ref_cam=sh_ref_cam,
                        return_masks=True,
                        noise_mode='const',
                        neural_rendering_resolution=args.resolution,
                        cache_backbone=cache_backbone_loop,
                        use_cached_backbone=use_cached_backbone_loop,
                        post_act_blendshapes=post_act_bs,
                    )

                    if cache_backbone_loop:
                        cache_backbone_loop      = False
                        use_cached_backbone_loop = True

                    # ── GT inference: full deformation network, reuse cached backbone ──
                    output_gt = G.synthesis(
                        w_batch, c_ren_batch, fp_batch,
                        sh_ref_cam=sh_ref_cam,
                        noise_mode='const',
                        neural_rendering_resolution=args.resolution,
                        cache_backbone=False,
                        use_cached_backbone=True,
                        post_act_blendshapes=None,   # full deformation network path
                    )

                    for image, image_gt, drv_img in zip(
                            output['image'], output_gt['image'], drv_batch):
                        drv_resized = cv2.resize(drv_img.cpu().numpy(),
                                                 (args.resolution, args.resolution))
                        gen_frame = Img.from_normalized_torch(image).to_numpy().img[..., :3]
                        gt_frame  = Img.from_normalized_torch(image_gt).to_numpy().img[..., :3]
                        face_mask_frame = None
                        if args.face_mask:
                            face_mask_frame = get_face_mask(gt_frame, face_mesh)
                        err_frame = make_error_frame(gen_frame, gt_frame, args.err_max_pixel,
                                                     mask=face_mask_frame, log_scale=args.log_error)

                        all_frames.append(np.hstack([gen_frame, drv_resized]))
                        all_generated_frames.append(gen_frame)
                        all_driving_frames.append(drv_resized)
                        all_gt_frames.append(gt_frame)
                        all_comparison_frames.append(
                            np.hstack([gen_frame, gt_frame, err_frame, drv_resized]))

            # ── save video ────────────────────────────────────────────────────
            output_folder = os.path.join(
                script_dir, 'results', 'dynamic_view',
                f"{args.run_name}_{args.checkpoint}__svd_mlp_v2"
                f"__intrinsics_s_{args.cam_scale}"
                f"{'_joint_c_front' if args.joint_c_front else ''}",
                f"seed{current_id_seed}",
            )
            ensure_directory_exists(output_folder)

            mediapy.write_video(f"{output_folder}/{vid_name}.mp4",
                                all_frames, fps=args.FPS)
            mediapy.write_video(f"{output_folder}/left_{vid_name}.mp4",
                                all_generated_frames, fps=args.FPS)
            mediapy.write_video(f"{output_folder}/gt_{vid_name}.mp4",
                                all_gt_frames, fps=args.FPS)
            # 4-panel: [approx | gt (full network) | pixelwise error | driver]
            mediapy.write_video(f"{output_folder}/comparison_{vid_name}.mp4",
                                all_comparison_frames, fps=args.FPS)

            parent_folder = os.path.dirname(output_folder)
            driving_path  = f"{parent_folder}/driving_{vid_name}.mp4"
            if not os.path.exists(driving_path):
                ensure_directory_exists(parent_folder)
                mediapy.write_video(driving_path, all_driving_frames, fps=args.FPS)

            print(f"  Saved → {output_folder}/{vid_name}.mp4")
            print(f"  Saved → {output_folder}/comparison_{vid_name}.mp4  [approx|gt|error|driver]")


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
