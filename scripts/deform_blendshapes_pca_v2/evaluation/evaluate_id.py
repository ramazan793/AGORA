"""
evaluate_id.py — ArcFace identity similarity using SVD+MLP (PCA v2) deformation bypass.

Based on scripts/evaluate_id.py. Additional args: --svd-path, --mlp-path.
Output folder defaults to evaluations/id_metric_pca_v2/.

Generation loop:
  1. Load SVD basis + MLP before the identity loop.
  2. Per identity: extract fixed_shape from dataset, setup base gaussians at neutral.
  3. Generate num_images_per_identity images with different expressions via PCA inference.

Metric computation: ArcFace cosine similarity (unchanged from baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import os
import sys
import json

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
    num_identities: int = 1000
    num_images_per_identity: int = 2
    cam_expcode_seed: int = 0
    truncation_psi: float = 0.7
    batch_size: int = 2
    id_batch_size: int = 8
    device: str = "cuda"

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
        "/data3/ramazan.fazylov/media/dyn_gghead_stuff/evaluations/id_metric_pca_v2/"
    output_folder: Optional[str] = None

    # ArcFace/InsightFace
    arcface_model_pack: str = "buffalo_l"
    arcface_providers: Tuple[str, ...] = ("CPUExecutionProvider",)
    arcface_ctx_id: int = 0
    arcface_det_size: Tuple[int, int] = (320, 320)

    # Saving / flow control
    save_scores: int = 1
    skip_generation: int = 0


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


def _init_arcface(args: Args):
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=args.arcface_model_pack,
                       providers=list(args.arcface_providers))
    app.prepare(ctx_id=args.arcface_ctx_id, det_size=args.arcface_det_size)
    return app


def _get_embeddings_arcface(app, bgr_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
    embeddings: List[Optional[np.ndarray]] = []
    for img in bgr_images:
        faces = app.get(img)
        if len(faces) == 0:
            embeddings.append(None)
        else:
            embeddings.append(faces[0].normed_embedding)
    return embeddings


def generate_pair_images(args: Args, output_folder: str) -> List[Tuple[str, str]]:
    assert args.svd_path, "Provide --svd-path"
    assert args.mlp_path, "Provide --mlp-path"

    from elias.util.batch import batchify_sliced
    from dreifus.image import Img
    from src.gghead.config.gaussian_attribute import GaussianAttribute
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

    # Load SVD basis + MLP (shared across all identities)
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

    saved_pairs: List[Tuple[str, str]] = []
    reals_root = os.path.join(args.output_root, f"reals_seed{args.cam_expcode_seed}")
    save_reals = (not os.path.exists(reals_root)) or (len(os.listdir(reals_root)) == 0)
    os.makedirs(reals_root, exist_ok=True)

    # zero_deltas is model-constant; initialize lazily from the first synthesis output
    zero_deltas = None

    pbar = tqdm(total=args.num_identities, desc="Generating ID pairs (PCA v2)")
    for group_start in range(0, args.num_identities, args.id_batch_size):
        group_size = min(args.id_batch_size, args.num_identities - group_start)

        # ── Phase 1: collect data for all identities in this group ──
        group_data = []
        for local_i in range(group_size):
            cur_id = group_start + local_i
            sampled_indices = rng.integers(0, len(eval_set), size=args.num_images_per_identity)

            cams_np: List[np.ndarray] = []
            flame_params_np: List[np.ndarray] = []
            real_imgs_np: List[np.ndarray] = []

            for idx in sampled_indices:
                c = eval_set.get_camera_parameters(int(idx))
                m = eval_set.get_flame_parameters(int(idx))
                cams_np.append(c)
                flame_params_np.append(m)

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

            t_rng = torch.Generator(device=device)
            t_rng.manual_seed(int(cur_id))
            z = torch.randn((1, G._config.z_dim), device=device, generator=t_rng)

            group_data.append({
                'cur_id': cur_id,
                'z': z,
                'cams': cams,
                'flame_params': flame_params,
                'real_imgs_np': real_imgs_np,
            })

        # ── Phase 2: batched neutral setup for the whole group ──
        N = len(group_data)
        with torch.no_grad():
            z_all = torch.cat([d['z'] for d in group_data])            # [N, z_dim]
            neutral_flame_all = torch.zeros(N, 358, device=device, dtype=torch.float32)
            for local_i, d in enumerate(group_data):
                neutral_flame_all[local_i, :300] = d['flame_params'][0, :300]

            ws_neutral = G.mapping(
                z_all, c_front.expand(N, -1),
                truncation_psi=args.truncation_psi,
                flame_params=neutral_flame_all,
            )
            backbone_ws_neutral, _ = ws_neutral
            w_codes_all = backbone_ws_neutral[:, 0, :]              # [N, 512]

            output_base = G.synthesis(
                ws_neutral, c_front.expand(N, -1), neutral_flame_all,
                neural_rendering_resolution=args.resolution,
                cache_backbone=False,
                use_cached_backbone=False,
                noise_mode='const',
                sh_ref_cam=c_front,
                return_gaussian_attributes=True,
            )
            ga_raw = output_base.returned_gaussian_attributes
            base_gaussians_all = {
                'xyz':      ga_raw[GaussianAttribute.POSITION].detach(),
                'scale':    G._gaussian_model.scaling_activation(
                                ga_raw[GaussianAttribute.SCALE]).detach(),
                'rotation': G._gaussian_model.rotation_activation(
                                ga_raw[GaussianAttribute.ROTATION]).detach(),
                'opacity':  G._gaussian_model.opacity_activation(
                                ga_raw[GaussianAttribute.OPACITY]).detach(),
            }
            base_color_all = ga_raw[GaussianAttribute.COLOR].detach()

            if zero_deltas is None:
                G_num_val = base_gaussians_all['xyz'].shape[1]
                C_color = base_color_all.shape[-1]
                zero_deltas = {
                    'xyz':      torch.zeros(59, G_num_val, 3,       device=device),
                    'scale':    torch.zeros(59, G_num_val, 3,       device=device),
                    'rotation': torch.zeros(59, G_num_val, 4,       device=device),
                    'opacity':  torch.zeros(59, G_num_val, 1,       device=device),
                    'color':    torch.zeros(59, G_num_val, C_color, device=device),
                }

            # ── Phase 3: generate images for each identity in the group ──
            for local_i, data in enumerate(group_data):
                cur_id = data['cur_id']
                z_i = data['z']
                cams = data['cams']
                flame_params = data['flame_params']
                real_imgs_np = data['real_imgs_np']

                # Slice this identity's base gaussians [1, G, C]
                w_code_i = w_codes_all[local_i:local_i + 1]
                base_gaussians_i = {
                    k: v[local_i:local_i + 1] for k, v in base_gaussians_all.items()
                }
                base_color_i = base_color_all[local_i:local_i + 1]

                c_render = cams
                c_mapping = c_front.repeat(len(flame_params), 1)

                frames_rgb: List[np.ndarray] = []
                for c_render_batch, c_mapping_batch, fp_batch in zip(
                    batchify_sliced(c_render, batch_size=args.batch_size),
                    batchify_sliced(c_mapping, batch_size=args.batch_size),
                    batchify_sliced(flame_params, batch_size=args.batch_size),
                ):
                    B = fp_batch.shape[0]
                    z_batch = z_i.repeat(B, 1)
                    w_batch = G.mapping(
                        z_batch, c_mapping_batch,
                        truncation_psi=args.truncation_psi,
                        flame_params=fp_batch,
                        c2=fp_batch,
                    )

                    output = generate_with_pca(
                        G, w_batch, c_render_batch, fp_batch,
                        w_code_i, base_gaussians_i, base_color_i, zero_deltas,
                        svd_data, mlp, device, K, ov,
                        sh_ref_cam=c_front,
                        return_masks=True,
                        noise_mode="const",
                        neural_rendering_resolution=args.resolution,
                        use_cached_backbone=False,
                        cache_backbone=False,
                    )

                    frames_rgb.extend([
                        Img.from_normalized_torch(image).to_numpy().img[..., :3]
                        for image in output["image"]
                    ])

                # Save both generated images into the output folder
                paths = []
                for j, frame in enumerate(frames_rgb[:args.num_images_per_identity]):
                    out_path = os.path.join(output_folder, f"{int(cur_id):04d}_{int(j):04d}.png")
                    Image.fromarray(frame).save(out_path)
                    paths.append(out_path)

                if len(paths) == 2:
                    saved_pairs.append((paths[0], paths[1]))

                # Save corresponding real images
                if save_reals:
                    for j in range(min(args.num_images_per_identity, len(real_imgs_np))):
                        real_out_path = os.path.join(
                            reals_root, f"{int(cur_id):04d}_{int(j):04d}.png")
                        Image.fromarray(real_imgs_np[j]).save(real_out_path)

        pbar.update(group_size)
    pbar.close()

    with open(os.path.join(output_folder, "pairs.json"), "w") as f:
        json.dump(saved_pairs, f, indent=2)

    return saved_pairs


def _load_pairs_from_json(output_folder: str) -> Optional[List[Tuple[str, str]]]:
    pairs_path = os.path.join(output_folder, "pairs.json")
    if not os.path.exists(pairs_path):
        return None
    try:
        with open(pairs_path, "r") as f:
            pairs = json.load(f)
        return [(str(a), str(b)) for a, b in pairs]
    except Exception:
        return None


def _discover_pairs_from_folders(output_folder: str) -> List[Tuple[str, str]]:
    from collections import defaultdict
    group: dict[str, list[str]] = defaultdict(list)
    try:
        for name in sorted(os.listdir(output_folder)):
            if not name.lower().endswith(".png"):
                continue
            parts = name.split("_")
            if len(parts) < 2:
                continue
            prefix = parts[0]
            group[prefix].append(name)
        pairs: List[Tuple[str, str]] = []
        for prefix, files in group.items():
            files.sort()
            if len(files) >= 2:
                p0 = os.path.join(output_folder, files[0])
                p1 = os.path.join(output_folder, files[1])
                pairs.append((p0, p1))
        return pairs
    except Exception:
        return []


def compute_average_id(args: Args, pairs: List[Tuple[str, str]]) -> float:
    app = _init_arcface(args)

    sims: List[float] = []
    valid = 0
    for p0, p1 in tqdm(pairs, desc="ArcFace similarity"):
        import cv2

        img1 = cv2.imread(p0)
        img2 = cv2.imread(p1)
        if img1 is None or img2 is None:
            continue
        embeddings = _get_embeddings_arcface(app, [img1, img2])
        if embeddings[0] is None or embeddings[1] is None:
            continue
        sim = float(np.dot(embeddings[0], embeddings[1]))
        sims.append(sim)
        valid += 1
        if args.save_scores:
            id_dir = os.path.dirname(p0)
            base_prefix = os.path.basename(p0).split("_")[0]
            score_path = os.path.join(
                id_dir,
                f"{base_prefix}_id_score_{args.arcface_model_pack}"
                f"_{args.arcface_det_size[0]}_{sim:.4f}.txt",
            )
            try:
                with open(score_path, "w") as f:
                    f.write(f"{sim:.6f}\n")
            except Exception:
                pass

    mean_sim = float(np.mean(sims)) if len(sims) > 0 else float("nan")
    # Reported ID is the square root of the mean ArcFace cosine similarity.
    id_score = float(np.sqrt(mean_sim)) if (len(sims) > 0 and mean_sim >= 0) else float("nan")
    print(f"Valid pairs: {valid} / {len(pairs)}")
    print(f"Mean ArcFace cosine: {mean_sim:.4f}")
    print(f"Average ArcFace similarity (ID) [sqrt of mean cosine]: {id_score:.4f}")
    return id_score


def main() -> None:
    args = tyro.cli(Args)
    output_folder = _ensure_output_folder(args)

    if args.skip_generation:
        pairs = _load_pairs_from_json(output_folder) or _discover_pairs_from_folders(output_folder)
        if not pairs:
            raise FileNotFoundError(
                "No pairs found. Run generation first or provide existing output folder."
            )
    else:
        pairs = generate_pair_images(args, output_folder)

    compute_average_id(args, pairs)


if __name__ == "__main__":
    main()
