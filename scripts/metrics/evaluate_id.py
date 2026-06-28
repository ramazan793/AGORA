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


# Ensure project is importable
_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)

from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR


@dataclass
class Args:
    # Model and generation settings
    run_name: str = "DGGHEAD-132"
    checkpoint: int = 23500
    resolution: int = 512
    num_identities: int = 1000
    num_images_per_identity: int = 2
    cam_expcode_seed: int = 0
    truncation_psi: float = 0.7
    batch_size: int = 2
    device: str = "cuda"

    # Generation behavior toggles
    synthesis_flame_cond: bool = True
    cache_backbone: bool = True
    fix_identity: bool = True
    new_cameras: bool = True

    # Paths
    models_path: str = "/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/"
    ffhq_fused_params_path: str = \
        f"{REPO_ROOT_DIR}/assets/fused_params_dataset.npy"
    dataset_path: str = \
        "/data2/ramazan.fazylov/media/datasets/FFHQ_png_512/FFHQ_png_512.zip"
    output_root: str = \
        "/data3/ramazan.fazylov/media/dyn_gghead_stuff/evaluations/id_metric/"
    output_folder: Optional[str] = None

    # ArcFace/InsightFace
    arcface_model_pack: str = "buffalo_l"
    arcface_providers: Tuple[str, ...] = ("CPUExecutionProvider",)
    arcface_ctx_id: int = 0
    arcface_det_size: Tuple[int, int] = (320, 320)

    # Saving
    save_scores: int = 1
    # Flow control
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
    # returns list of (path_img0, path_img1)
    from elias.util.batch import batchify_sliced
    from dreifus.image import Img
    from eg3d.datamanager.nersemble import decode_camera_params
    os.environ["GGHEAD_MODELS_PATH"] = args.models_path
    from src.gghead.model_manager.finder import find_model_manager
    from src.gghead.dataset.image_folder_dataset import DGGHeadMaskImageFolderDataset

    device = torch.device(args.device)

    model_manager = find_model_manager(args.run_name)
    resolved_ckpt = model_manager._resolve_checkpoint_id(args.checkpoint)
    G = model_manager.load_checkpoint(resolved_ckpt, load_ema=True).to(device)
    G._config.use_flame_rasterization = 0

    mapping_takes_flame_params = (
        G._config.use_mouth_branch or G._config.use_extended_uv_generation or True
    )

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

    for cur_id in tqdm(range(args.num_identities), desc="Generating ID pairs"):
        # sample 2 images per identity
        sampled_indices = rng.integers(0, len(eval_set), size=args.num_images_per_identity)

        cams_np: List[np.ndarray] = []
        flame_params_np: List[np.ndarray] = []
        real_imgs_np: List[np.ndarray] = []

        for idx in sampled_indices:
            if args.new_cameras:
                c = eval_set.get_camera_parameters(int(idx))
                m = eval_set.get_flame_parameters(int(idx))
            else:
                c = eval_set._dataset_images.get_label(int(idx))
                m = np.array(eval_set.get_flame_parameters(int(idx)))
            cams_np.append(c)
            flame_params_np.append(m)

            if save_reals:
                prev_res = eval_set._dataset_images._resolution
                eval_set._dataset_images._resolution = 512 
                eval_set._dataset_images._raw_shape = (eval_set._dataset_images._raw_shape[0], 3, 512, 512)
                real_img, _ = eval_set._dataset_images[int(idx)]
                eval_set._dataset_images._resolution = prev_res
                eval_set._dataset_images._raw_shape = (eval_set._dataset_images._raw_shape[0], 3, prev_res, prev_res)

                real_img = np.transpose(real_img, (1, 2, 0))
                real_imgs_np.append(real_img)

        cams = torch.tensor(np.array(cams_np), device=args.device)
        flame_params = torch.from_numpy(np.stack(flame_params_np)).to(args.device)

        if args.fix_identity:
            flame_params[:, :300] = flame_params[0:1, :300]

        if args.cache_backbone:
            use_cached_backbone = False
            cache_backbone = True
        else:
            use_cached_backbone = False
            cache_backbone = False

        with torch.no_grad():
            t_rng = torch.Generator(device=args.device)
            t_rng.manual_seed(int(cur_id))
            z = torch.randn((1, G._config.z_dim), device=device, generator=t_rng)

            if args.new_cameras:
                sh_ref_cam = c_front
            else:
                sh_ref_cam, _ = decode_camera_params(c_front[0].detach().cpu())

            if not mapping_takes_flame_params:
                w = G.mapping(z, c_front, truncation_psi=args.truncation_psi)
                w = w.repeat(len(flame_params), 1, 1)
            else:
                w = torch.empty(len(flame_params), 1, 1, device=device)

            c_render = cams
            c_mapping = c_front.repeat(len(flame_params), 1)

            frames_rgb: List[np.ndarray] = []
            for w_batch, c_render_batch, c_mapping_batch, fp_batch in zip(
                batchify_sliced(w, batch_size=args.batch_size),
                batchify_sliced(c_render, batch_size=args.batch_size),
                batchify_sliced(c_mapping, batch_size=args.batch_size),
                batchify_sliced(flame_params, batch_size=args.batch_size),
            ):
                if mapping_takes_flame_params:
                    z_batch = z.repeat(len(w_batch), 1)
                    w_batch = G.mapping(
                        z_batch, c_mapping_batch, truncation_psi=args.truncation_psi, flame_params=fp_batch, c2=fp_batch
                    )

                if args.synthesis_flame_cond:
                    output = G.synthesis(
                        w_batch,
                        c_render_batch,
                        fp_batch,
                        sh_ref_cam=sh_ref_cam,
                        return_masks=True,
                        noise_mode="const",
                        neural_rendering_resolution=args.resolution,
                        use_cached_backbone=use_cached_backbone,
                        cache_backbone=cache_backbone,
                    )
                else:
                    output = G.synthesis(
                        w_batch,
                        c_render_batch,
                        sh_ref_cam=sh_ref_cam,
                        return_masks=True,
                        noise_mode="const",
                        neural_rendering_resolution=args.resolution,
                        use_cached_backbone=use_cached_backbone,
                        cache_backbone=cache_backbone,
                    )

                if args.cache_backbone:
                    use_cached_backbone = True
                    cache_backbone = False

                frames_rgb.extend([
                    Img.from_normalized_torch(image).to_numpy().img[..., :3]
                    for image in output["image"]
                ])

        # Save both generated images flat into the run output folder
        paths = []
        for j, frame in enumerate(frames_rgb[: args.num_images_per_identity]):
            out_path = os.path.join(output_folder, f"{int(cur_id):04d}_{int(j):04d}.png")
            Image.fromarray(frame).save(out_path)
            paths.append(out_path)

        if len(paths) == 2:
            saved_pairs.append((paths[0], paths[1]))

        # Save corresponding real images flat under reals_seed{seed}
        if save_reals:
            for j in range(min(args.num_images_per_identity, len(real_imgs_np))):
                real_img = real_imgs_np[j]
                if real_img is None:
                    continue
                real_out_path = os.path.join(reals_root, f"{int(cur_id):04d}_{int(j):04d}.png")
                Image.fromarray(real_img).save(real_out_path)

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
        # ensure tuples
        return [(str(a), str(b)) for a, b in pairs]
    except Exception:
        return None


def _discover_pairs_from_folders(output_folder: str) -> List[Tuple[str, str]]:
    # With flat storage, infer pairs by grouping on the identity prefix XXXX in XXXX_YY.png
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
    # Initialize ArcFace
    app = _init_arcface(args)

    sims: List[float] = []
    valid = 0
    for p0, p1 in tqdm(pairs, desc="ArcFace similarity"):
        import cv2

        img1 = cv2.imread(p0)
        img2 = cv2.imread(p1)
        if img1 is None or img2 is None:
            continue
        # ArcFace expects BGR; cv2 already returns BGR
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
                f"{base_prefix}_id_score_{args.arcface_model_pack}_{args.arcface_det_size[0]}_{sim:.4f}.txt",
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
                "No pairs found. Either run generation first or provide existing id_* folders or pairs.json."
            )
    else:
        pairs = generate_pair_images(args, output_folder)
    compute_average_id(args, pairs)


if __name__ == "__main__":
    main()


