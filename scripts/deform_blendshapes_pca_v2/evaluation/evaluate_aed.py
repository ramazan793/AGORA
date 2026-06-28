"""
evaluate_aed.py — AED metric using SVD+MLP (PCA v2) deformation bypass.

Based on scripts/evaluate_aed.py. Additional args: --svd-path, --mlp-path.
Output folder defaults to evaluations/apd_aed_1_pca_v2/.

Generation loop:
  1. Load SVD basis + MLP before the identity loop.
  2. Per identity: extract fixed_shape from dataset, setup base gaussians at neutral.
  3. Per expression batch: MLP(w_code, flame_55) → PCA coefficients → physical attrs
     → inject via post_act_blendshapes zero-delta trick.

Metric computation is identical to the baseline (SMIRK + RMSE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import os
import sys
import json
import subprocess

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import tyro

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pca_inference_utils import load_svd_mlp, setup_identity, generate_with_pca
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR


@dataclass
class Args:
    # SVD+MLP paths
    svd_path: str = ""
    mlp_path: str = ""

    # Model and generation settings
    run_name: str = "DGGHEAD-132"
    checkpoint: int = 23500
    resolution: int = 512
    num_sampled_z: int = 500
    num_sampled_cam_exp: int = 20
    cam_expcode_seed: int = 0
    truncation_psi: float = 0.7
    batch_size: int = 20
    device: str = "cuda:5"

    # Generation behavior toggles
    cache_backbone: bool = True
    new_cameras: bool = True

    # Paths
    models_path: str = "/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/"
    ffhq_fused_params_path: str = \
        f"{REPO_ROOT_DIR}/assets/fused_params_dataset.npy"
    dataset_path: str = \
        "/data2/ramazan.fazylov/media/datasets/FFHQ_png_512/FFHQ_png_512.zip"
    output_root: str = \
        "/data3/ramazan.fazylov/media/dyn_gghead_stuff/evaluations/apd_aed_1_pca_v2/"
    output_folder: Optional[str] = None

    # Metrics
    aed_num_components: int = 50

    # Flow control
    skip_generation: bool = False
    skip_smirk: bool = False

    # SMIRK configuration
    smirk_repo_path: str = f"{GGHEAD_DEPENDENCIES_PATH}/smirk/"
    smirk_python_bin: str = \
        "/data2/ramazan.fazylov/micromamba/envs/gghead/bin/python"
    smirk_script_path: str = f"{GGHEAD_DEPENDENCIES_PATH}/smirk/demo_image_folder.py"
    smirk_checkpoint: str = \
        f"{GGHEAD_DEPENDENCIES_PATH}/smirk/pretrained_models/SMIRK_em1.pt"
    smirk_render_orig: bool = True
    smirk_output_subdir: str = "smirk"


def _ensure_output_folder(args: Args) -> str:
    if args.output_folder is None:
        output_folder = os.path.join(
            args.output_root.rstrip("/"),
            f"{args.run_name}_{args.checkpoint}_seed{args.cam_expcode_seed}",
        )
    else:
        output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def _extract_cuda_visible_devices(device_str: str) -> str:
    if device_str.startswith("cuda:"):
        return device_str.split(":", 1)[1]
    return device_str


def generate_samples(args: Args, output_folder: str) -> None:
    assert args.svd_path, "Provide --svd-path"
    assert args.mlp_path, "Provide --mlp-path"

    from elias.util.batch import batchify_sliced
    from dreifus.image import Img
    os.environ["GGHEAD_MODELS_PATH"] = args.models_path
    from src.gghead.model_manager.finder import find_model_manager
    from src.gghead.dataset.image_folder_dataset import DGGHeadMaskImageFolderDataset

    device = torch.device(args.device)

    model_manager = find_model_manager(args.run_name)
    resolved_ckpt = model_manager._resolve_checkpoint_id(args.checkpoint)
    G = model_manager.load_checkpoint(resolved_ckpt, load_ema=True).to(device)
    G.eval()
    G._config.use_flame_rasterization = 0
    ov = getattr(G._config, 'opacity_overshoot', 0.0)

    # Load SVD basis + MLP (done once, shared across all identities)
    svd_data, mlp, K, G_num = load_svd_mlp(args.svd_path, args.mlp_path, device)

    # Dataset
    dataset_config = model_manager.load_dataset_config()
    dataset_config.precomputed_flame_renderings = 0
    dataset_config.path = args.dataset_path
    eval_set = DGGHeadMaskImageFolderDataset(dataset_config.eval())

    # Reference camera from FFHQ stats
    ffhq_fused_params = np.load(args.ffhq_fused_params_path)
    mean_cam = np.median(ffhq_fused_params[:, -6:], axis=0)
    c_front = torch.tensor(mean_cam, device=device).unsqueeze(0)

    rng = np.random.default_rng(args.cam_expcode_seed)
    seeds = list(range(args.num_sampled_z))

    all_cams: List[np.ndarray] = []
    all_flame_params: List[np.ndarray] = []
    all_fnames: List[str] = []

    reals_root = os.path.join(args.output_root, f"reals_seed{args.cam_expcode_seed}")
    save_reals = (not os.path.exists(reals_root)) or (len(os.listdir(reals_root)) == 0)
    if save_reals:
        os.makedirs(reals_root, exist_ok=True)

    for cur_seed in tqdm(seeds, desc="Generating samples (PCA v2 AED)"):
        sampled_indices = rng.integers(0, len(eval_set), size=args.num_sampled_cam_exp)
        cams_np: List[np.ndarray] = []
        flame_params_np: List[np.ndarray] = []
        img_names: List[str] = []
        real_imgs_np = []

        for idx in sampled_indices:
            c = eval_set.get_camera_parameters(int(idx))
            m = eval_set.get_flame_parameters(int(idx))
            img_name = eval_set._mesh_fnames[int(idx)].split(os.sep)[-2]
            cams_np.append(c)
            flame_params_np.append(m)
            img_names.append(img_name)

            if save_reals:
                prev_res = eval_set._dataset_images._resolution
                eval_set._dataset_images._resolution = 512
                eval_set._dataset_images._raw_shape = (
                    eval_set._dataset_images._raw_shape[0], 3, 512, 512)
                real_img, _ = eval_set._dataset_images[int(idx)]
                eval_set._dataset_images._resolution = prev_res
                eval_set._dataset_images._raw_shape = (
                    eval_set._dataset_images._raw_shape[0], 3, prev_res, prev_res)
                real_imgs_np.append(np.transpose(real_img, (1, 2, 0)))

        cams = torch.tensor(np.array(cams_np), device=device)
        flame_params = torch.from_numpy(np.stack(flame_params_np)).to(device)

        # Fix identity: use shape from the first sampled frame
        flame_params[:, :300] = flame_params[0:1, :300]

        all_cams.append(cams.detach().cpu().numpy())
        all_flame_params.append(flame_params.detach().cpu().numpy())
        all_fnames.extend(img_names)

        if args.cache_backbone:
            use_cached_backbone = False
            cache_backbone = True
        else:
            use_cached_backbone = False
            cache_backbone = False

        with torch.no_grad():
            t_rng = torch.Generator(device=device)
            t_rng.manual_seed(int(cur_seed))
            z = torch.randn((1, G._config.z_dim), device=device, generator=t_rng)

            # Per-identity setup: compute w_code and base gaussians at neutral expression
            fixed_shape = flame_params[0, :300].clone()
            w_code, base_gaussians, base_color, zero_deltas, _ = setup_identity(
                G, z, fixed_shape, c_front, svd_data, G_num, device,
                resolution=args.resolution,
                truncation_psi=args.truncation_psi,
            )

            c_render = cams
            c_mapping = c_front.repeat(len(flame_params), 1)

            j = 0
            real_j = 0
            for c_render_batch, c_mapping_batch, fp_batch in zip(
                batchify_sliced(c_render, batch_size=args.batch_size),
                batchify_sliced(c_mapping, batch_size=args.batch_size),
                batchify_sliced(flame_params, batch_size=args.batch_size),
            ):
                B = fp_batch.shape[0]
                z_batch = z.repeat(B, 1)
                w_batch = G.mapping(
                    z_batch, c_mapping_batch,
                    truncation_psi=args.truncation_psi,
                    flame_params=fp_batch,
                    c2=fp_batch,
                )

                output = generate_with_pca(
                    G, w_batch, c_render_batch, fp_batch,
                    w_code, base_gaussians, base_color, zero_deltas,
                    svd_data, mlp, device, K, ov,
                    sh_ref_cam=c_front,
                    return_masks=True,
                    noise_mode="const",
                    neural_rendering_resolution=args.resolution,
                    use_cached_backbone=use_cached_backbone,
                    cache_backbone=cache_backbone,
                )

                if args.cache_backbone:
                    use_cached_backbone = True
                    cache_backbone = False

                frames = [
                    Img.from_normalized_torch(image).to_numpy().img[..., :3]
                    for image in output["image"]
                ]

                for frame in frames:
                    Image.fromarray(frame).save(
                        f"{output_folder}/{int(cur_seed):04d}_{int(j):04d}.png"
                    )
                    if save_reals and real_j < len(real_imgs_np):
                        real_path = os.path.join(
                            reals_root, f"{int(cur_seed):04d}_{int(real_j):04d}.png"
                        )
                        Image.fromarray(real_imgs_np[real_j]).save(real_path)
                        real_j += 1
                    j += 1

    all_cams_np = np.concatenate(all_cams, axis=0)
    all_flame_params_np = np.concatenate(all_flame_params, axis=0)
    np.save(f"{output_folder}/gt_cams.npy", all_cams_np)
    np.save(f"{output_folder}/gt_flame_params.npy", all_flame_params_np)
    with open(f"{output_folder}/meta_fnames.json", "w") as f:
        json.dump(all_fnames, f, indent=4)


def run_smirk(args: Args, output_folder: str) -> None:
    smirk_out = os.path.join(output_folder, args.smirk_output_subdir)
    os.makedirs(smirk_out, exist_ok=True)

    cuda_visible = _extract_cuda_visible_devices(args.device)

    cmd: List[str] = [
        args.smirk_python_bin,
        args.smirk_script_path,
        "--input_path", output_folder,
        "--out_path", smirk_out,
        "--checkpoint", args.smirk_checkpoint,
    ]
    if args.smirk_render_orig:
        cmd.append("--render_orig")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible

    subprocess.run(cmd, check=True, cwd=args.smirk_repo_path, env=env)


def compute_metrics(args: Args, output_folder: str) -> Dict[str, float]:
    all_cams = np.load(f"{output_folder}/gt_cams.npy")
    all_flame_params = np.load(f"{output_folder}/gt_flame_params.npy")
    with open(f"{output_folder}/meta_fnames.json", "r") as f:
        all_fnames = json.load(f)

    smirk_result_folder = os.path.join(output_folder, args.smirk_output_subdir)

    total_samples = int(args.num_sampled_z * args.num_sampled_cam_exp)
    smirk_shapecodes  = np.ones((total_samples, 300), dtype=np.float32) * np.nan
    smirk_expcodes    = np.ones((total_samples, 50),  dtype=np.float32) * np.nan
    smirk_globalposes = np.ones((total_samples, 3),   dtype=np.float32) * np.nan
    smirk_jawposes    = np.ones((total_samples, 3),   dtype=np.float32) * np.nan
    smirk_eyelids     = np.ones((total_samples, 2),   dtype=np.float32) * np.nan

    gen_img_names: List[Optional[str]] = [None] * total_samples

    for folder in tqdm(os.listdir(smirk_result_folder), desc="Reading SMIRK outputs"):
        folder_path = os.path.join(smirk_result_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        if len(os.listdir(folder_path)) == 0:
            continue

        img_name = folder
        parts = img_name.split("_")
        try:
            img_seed = int(parts[-2])
            img_j = int(parts[-1])
        except Exception:
            continue

        img_idx = img_seed * args.num_sampled_cam_exp + img_j
        if img_idx < 0 or img_idx >= total_samples:
            continue

        gen_img_names[img_idx] = img_name

        shapecode  = np.load(os.path.join(folder_path, "shape.npy"))
        expcode    = np.load(os.path.join(folder_path, "exp.npy"))
        globalpose = np.load(os.path.join(folder_path, "globalpose.npy"))
        jawpose    = np.load(os.path.join(folder_path, "jawpose.npy"))
        eyelids    = np.load(os.path.join(folder_path, "eyelid.npy"))

        smirk_shapecodes[img_idx]  = shapecode
        smirk_expcodes[img_idx]    = expcode
        smirk_globalposes[img_idx] = globalpose
        smirk_jawposes[img_idx]    = jawpose
        smirk_eyelids[img_idx]     = eyelids

    nan_mask = np.isnan(smirk_shapecodes).any(axis=1)
    print("Number of bad samples for SMIRK: ", int(np.sum(nan_mask)))

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(((a - b) ** 2).mean()))

    gt_shapecodes  = all_flame_params[:, :300]
    gt_expcodes    = all_flame_params[:, 300:350]
    gt_globalposes = all_flame_params[:, 350:353]
    gt_jawposes    = all_flame_params[:, 353:356]
    gt_eyelids     = all_flame_params[:, 356:358]

    if nan_mask.any() and smirk_expcodes.shape[0] == total_samples:
        keep = ~nan_mask
        smirk_shapecodes  = smirk_shapecodes[keep]
        smirk_expcodes    = smirk_expcodes[keep]
        smirk_globalposes = smirk_globalposes[keep]
        smirk_jawposes    = smirk_jawposes[keep]
        smirk_eyelids     = smirk_eyelids[keep]
        gt_shapecodes     = gt_shapecodes[keep]
        gt_expcodes       = gt_expcodes[keep]
        gt_globalposes    = gt_globalposes[keep]
        gt_jawposes       = gt_jawposes[keep]
        gt_eyelids        = gt_eyelids[keep]

    asd         = rmse(smirk_shapecodes, gt_shapecodes)
    ajd         = rmse(smirk_jawposes, gt_jawposes)
    agpd        = rmse(smirk_globalposes, gt_globalposes)
    aed         = rmse(smirk_expcodes[:, :args.aed_num_components],
                       gt_expcodes[:, :args.aed_num_components])
    aed_eyelids = rmse(smirk_eyelids, gt_eyelids)

    metrics = {
        "AED": aed,
        "AJD": ajd,
        "AGPD": agpd,
        "ASD": asd,
        "AED_eyelids": aed_eyelids,
    }

    print(
        f"AED: {aed:.3f}, AJD: {ajd:.3f}, AGPD: {agpd:.3f}, "
        f"ASD: {asd:.3f}, AED_eyelids: {aed_eyelids:.3f}"
    )

    with open(os.path.join(output_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(output_folder)
    return metrics


def main() -> None:
    args = tyro.cli(Args)
    output_folder = _ensure_output_folder(args)

    if not args.skip_generation:
        generate_samples(args, output_folder)
    else:
        for fname in ["gt_cams.npy", "gt_flame_params.npy", "meta_fnames.json"]:
            fpath = os.path.join(output_folder, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Missing precomputed file (skip_generation=True): {fpath}"
                )

    if not args.skip_smirk:
        run_smirk(args, output_folder)
    else:
        smirk_out = os.path.join(output_folder, args.smirk_output_subdir)
        if not os.path.isdir(smirk_out):
            raise FileNotFoundError(
                f"SMIRK output directory not found (skip_smirk=True): {smirk_out}"
            )

    compute_metrics(args, output_folder)


if __name__ == "__main__":
    main()
