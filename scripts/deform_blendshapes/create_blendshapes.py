"""
Extract 59 linear deformation blendshape planes via finite differences.

50 expression planes + 9 jaw rotation residual planes.
Saves base_residual_plane and delta_planes to disk, plus visualizations.
"""

from dataclasses import dataclass
from typing import Optional
import tyro
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from dreifus.image import Img
from dreifus.matrix import Pose

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
os.environ.setdefault('GGHEAD_MODELS_PATH', '/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/')

from src.gghead.model_manager.finder import find_model_manager
from src.gghead.util.flame_rasterizer import batch_rodrigues
from src.gghead.config.gaussian_attribute import GaussianAttribute
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR


@dataclass
class Args:
    DEVICE: str = 'cuda:0'
    run_name: str = 'DGGHEAD-158'
    checkpoint: int = 20500
    resolution: int = 512
    id_seed: int = 10
    epsilon: float = 1.0
    cam_scale: float = 8.0


def rotation_matrix_to_axis_angle_np(R_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to axis-angle (3,) using scipy."""
    return R.from_matrix(R_mat).as_rotvec()


def project_to_SO3(M: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to nearest rotation matrix via SVD."""
    U, _, Vh = np.linalg.svd(M)
    R_proj = U @ Vh
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vh
    return R_proj


def normalize_for_vis(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] for visualization."""
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def save_plane_vis(plane: torch.Tensor, save_path: str,
                   pos_start: int, opacity_start: int):
    """
    Save a plane [C, H, W] as a horizontally concatenated image: xyz | opacity.
    """
    plane_np = plane.cpu().float().numpy()
    H, W = plane_np.shape[1], plane_np.shape[2]

    xyz = plane_np[pos_start:pos_start + 3]  # [3, H, W]
    xyz_vis = normalize_for_vis(xyz)
    xyz_vis = (xyz_vis.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]

    opacity = plane_np[opacity_start:opacity_start + 1]  # [1, H, W]
    opacity_vis = normalize_for_vis(opacity)
    opacity_vis = (opacity_vis[0] * 255).astype(np.uint8)  # [H, W]
    opacity_vis = np.stack([opacity_vis] * 3, axis=-1)  # [H, W, 3]

    combined = np.concatenate([xyz_vis, opacity_vis], axis=1)  # [H, 2*W, 3]
    Image.fromarray(combined).save(save_path)


def render_single(G, z, c_front, flame_params, resolution, device,
                  use_cached_backbone=True, cache_backbone=False):
    """Run full synthesis and return the rendered image as uint8 numpy [H, W, 3]."""
    truncation_psi = 0.7
    ws = G.mapping(z, c_front, truncation_psi=truncation_psi, flame_params=flame_params)
    output = G.synthesis(
        ws, c_front, flame_params,
        neural_rendering_resolution=resolution,
        use_cached_backbone=use_cached_backbone,
        cache_backbone=cache_backbone,
        noise_mode='const',
        sh_ref_cam=c_front,
    )
    img = output['image'][0]
    frame = Img.from_normalized_torch(img).to_numpy().img[..., :3]
    return frame


def main(args: Args) -> None:
    device = torch.device(args.DEVICE)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    n_exp = 50
    n_jaw_res = 9
    n_total = n_exp + n_jaw_res  # 59

    # ------------------------------------------------------------------ model
    model_manager = find_model_manager(args.run_name)
    checkpoint = model_manager._resolve_checkpoint_id(args.checkpoint)
    G = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
    G.eval()
    G._config.use_flame_rasterization = 0

    deform_c2_dim = G.deformation_generator.c2_dim  # 53 or 55
    has_eyelid = (deform_c2_dim == 55)

    pos_start = G._uv_attribute_start_channel[GaussianAttribute.POSITION]
    opacity_start = G._uv_attribute_start_channel[GaussianAttribute.OPACITY]

    # ---------------------------------------------------------- identity setup
    ffhq_data = np.load(f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy')
    ffhq_cams = ffhq_data[:, -6:]
    ffhq_shapes = ffhq_data[:, :300]

    rng = torch.Generator(device)
    rng.manual_seed(args.id_seed)
    z = torch.randn((1, G._config.z_dim), device=device, generator=rng)

    flame_shape_id = torch.randint(0, ffhq_shapes.shape[0], (1,), generator=rng, device=device).cpu().item()
    fixed_shape = torch.tensor(ffhq_shapes[flame_shape_id, :300], dtype=torch.float32).to(device)

    mean_ffhq_cam = np.mean(ffhq_cams, axis=0)
    c_front = torch.tensor(mean_ffhq_cam, dtype=torch.float32).unsqueeze(0).to(device)

    # ---------------------------------------------- neutral FLAME params
    # [shape(300), exp(50), globalpose(3), jawpose(3), eyelid(2)] = 358
    neutral_flame = torch.zeros(1, 358, device=device, dtype=torch.float32)
    neutral_flame[:, :300] = fixed_shape

    # ------------------------------------------------------------ output dirs
    tag = f"{args.run_name}_{args.checkpoint}_seed{args.id_seed}_eps{args.epsilon}"
    data_dir = os.path.join(script_dir, "data", tag)
    vis_planes_dir = os.path.join(script_dir, "vis", tag, "planes")
    vis_planes_exp_dir = os.path.join(vis_planes_dir, "exp")
    vis_planes_jaw_dir = os.path.join(vis_planes_dir, "jaw")
    vis_renders_dir = os.path.join(script_dir, "vis", tag, "renders")
    vis_renders_exp_dir = os.path.join(vis_renders_dir, "exp")
    vis_renders_jaw_dir = os.path.join(vis_renders_dir, "jaw")
    for d in [data_dir, vis_planes_exp_dir, vis_planes_jaw_dir,
              vis_renders_exp_dir, vis_renders_jaw_dir]:
        os.makedirs(d, exist_ok=True)

    truncation_psi = 0.7

    # ======================================================================
    #  Step 1: cache backbone with a neutral pass
    # ======================================================================
    print("Caching backbone features with neutral FLAME params ...")
    with torch.no_grad():
        ws_neutral = G.mapping(z, c_front, truncation_psi=truncation_psi,
                               flame_params=neutral_flame)
        main_ws, _ = ws_neutral

        _ = G.synthesis(
            ws_neutral, c_front, neutral_flame,
            neural_rendering_resolution=args.resolution,
            cache_backbone=True,
            use_cached_backbone=False,
            noise_mode='const',
            sh_ref_cam=c_front,
        )

    features_at_res = G._last_features_at_res.clone()
    img_at_res = G._last_img_at_res.clone()

    # ======================================================================
    #  Step 2: base residual plane (neutral expression/jaw)
    # ======================================================================
    print("Computing base residual plane (neutral) ...")
    with torch.no_grad():
        neutral_deform_ws = torch.zeros(1, deform_c2_dim, device=device)
        base_residual_plane = G.deformation_generator(
            neutral_deform_ws,
            feature_maps=features_at_res,
            img_rgb=img_at_res,
            update_emas=False,
            ws_main=main_ws,
            noise_mode='const',
            condition_map=None,
        )  # [1, C, H, W]

    save_plane_vis(base_residual_plane[0], os.path.join(vis_planes_dir, "base_residual.png"),
                   pos_start, opacity_start)

    # ======================================================================
    #  Step 3: 50 expression delta planes
    # ======================================================================
    print("Computing 50 expression delta planes ...")
    exp_delta_planes = []
    with torch.no_grad():
        for i in tqdm(range(n_exp), desc="Expression planes"):
            perturbed_ws = torch.zeros(1, deform_c2_dim, device=device)
            perturbed_ws[0, i] = args.epsilon

            perturbed_plane = G.deformation_generator(
                perturbed_ws,
                feature_maps=features_at_res,
                img_rgb=img_at_res,
                update_emas=False,
                ws_main=main_ws,
                noise_mode='const',
                condition_map=None,
            )
            delta = (perturbed_plane - base_residual_plane) / args.epsilon
            exp_delta_planes.append(delta)

            save_plane_vis(delta[0],
                           os.path.join(vis_planes_exp_dir, f"{i:02d}.png"),
                           pos_start, opacity_start)

    exp_delta_planes = torch.cat(exp_delta_planes, dim=0)  # [50, C, H, W]

    # ======================================================================
    #  Step 4: 9 jaw rotation residual delta planes
    # ======================================================================
    print("Computing 9 jaw rotation residual delta planes ...")
    jaw_delta_planes = []
    I3 = np.eye(3)
    with torch.no_grad():
        for j in tqdm(range(n_jaw_res), desc="Jaw residual planes"):
            residual_9 = np.zeros(9)
            residual_9[j] = args.epsilon
            R_target = I3 + residual_9.reshape(3, 3)

            R_valid = project_to_SO3(R_target)
            jaw_3d_np = rotation_matrix_to_axis_angle_np(R_valid)
            jaw_3d = torch.tensor(jaw_3d_np, dtype=torch.float32, device=device)

            perturbed_ws = torch.zeros(1, deform_c2_dim, device=device)
            perturbed_ws[0, n_exp:n_exp + 3] = jaw_3d

            perturbed_plane = G.deformation_generator(
                perturbed_ws,
                feature_maps=features_at_res,
                img_rgb=img_at_res,
                update_emas=False,
                ws_main=main_ws,
                noise_mode='const',
                condition_map=None,
            )
            delta = (perturbed_plane - base_residual_plane) / args.epsilon
            jaw_delta_planes.append(delta)

            save_plane_vis(delta[0],
                           os.path.join(vis_planes_jaw_dir, f"{j:02d}.png"),
                           pos_start, opacity_start)

    jaw_delta_planes = torch.cat(jaw_delta_planes, dim=0)  # [9, C, H, W]

    # ======================================================================
    #  Step 5: concatenate and save
    # ======================================================================
    delta_planes = torch.cat([exp_delta_planes, jaw_delta_planes], dim=0)  # [59, C, H, W]
    print(f"delta_planes shape: {delta_planes.shape}")
    print(f"base_residual_plane shape: {base_residual_plane.shape}")

    save_path = os.path.join(data_dir, "blendshape_planes.pt")
    save_dict = {
        'base_residual_plane': base_residual_plane.cpu(),
        'delta_planes': delta_planes.cpu(),
        'epsilon': args.epsilon,
        'id_seed': args.id_seed,
        'run_name': args.run_name,
        'checkpoint': args.checkpoint,
        'deform_c2_dim': deform_c2_dim,
        'n_exp': n_exp,
        'n_jaw_res': n_jaw_res,
    }
    torch.save(save_dict, save_path)
    print(f"Saved blendshape planes to {save_path}")

    # ======================================================================
    #  Step 6: render ground-truth avatars for all 60 states
    # ======================================================================
    print("Rendering ground-truth avatars for neutral + 59 perturbations ...")

    with torch.no_grad():
        # --- neutral render (backbone already cached)
        frame_neutral = render_single(
            G, z, c_front, neutral_flame, args.resolution, device,
            use_cached_backbone=True, cache_backbone=False)
        Image.fromarray(frame_neutral).save(
            os.path.join(vis_renders_dir, "neutral.png"))

        # --- 50 expression renders
        for i in tqdm(range(n_exp), desc="Render expression"):
            fp = neutral_flame.clone()
            fp[0, 300 + i] = args.epsilon
            frame = render_single(
                G, z, c_front, fp, args.resolution, device,
                use_cached_backbone=True, cache_backbone=False)
            Image.fromarray(frame).save(
                os.path.join(vis_renders_exp_dir, f"{i:02d}.png"))

        # --- 9 jaw renders
        for j in tqdm(range(n_jaw_res), desc="Render jaw"):
            residual_9 = np.zeros(9)
            residual_9[j] = args.epsilon
            R_target = I3 + residual_9.reshape(3, 3)
            R_valid = project_to_SO3(R_target)
            jaw_3d_np = rotation_matrix_to_axis_angle_np(R_valid)

            fp = neutral_flame.clone()
            fp[0, 353:356] = torch.tensor(jaw_3d_np, dtype=torch.float32, device=device)
            frame = render_single(
                G, z, c_front, fp, args.resolution, device,
                use_cached_backbone=True, cache_backbone=False)
            Image.fromarray(frame).save(
                os.path.join(vis_renders_jaw_dir, f"{j:02d}.png"))

    print("Done! All outputs saved under:")
    print(f"  Data:   {data_dir}")
    print(f"  Vis:    {os.path.join(script_dir, 'vis', tag)}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
