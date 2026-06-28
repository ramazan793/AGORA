# AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars

<p align="center">
  <a href="https://ramazan793.github.io/AGORA"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2512.06438"><img src="https://img.shields.io/badge/arXiv-2512.06438-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <img src="https://ramazan793.github.io/AGORA/static/images/method_overview.png" width="100%">
</p>

> **AGORA** generates high-fidelity, animatable 3D Gaussian head avatars that render at **250+ FPS** on GPU and **~9 FPS on CPU-only inference**.

This repository contains the official implementation. It covers four capabilities:

1. **Inference** — generate an avatar from a random latent and drive (reenact) it with a video.
2. **AGORA-M** — distill the per-frame deformation network into a compact, identity-independent
   PCA + MLP module that bypasses the deformation branch at inference (mobile/real-time variant).
3. **Metrics** — FID, FPS, ID (ArcFace identity consistency), and expression/pose errors
   (AED / AED-jaw / APD / ASD) re-estimated with SMIRK.
4. **Training** — the two-stage GAN training recipe (stage 1 at 256², stage 2 progressively grown to 512²).

> PTI / test-time inversion is **not** part of this release.

---

## Table of contents

- [Installation](#installation)
- [Data & checkpoints](#data--checkpoints)
- [Quickstart](#quickstart)
- [Inference (generate + reenact)](#1-inference--generate-and-reenact-an-avatar)
- [AGORA-M (PCA + MLP distillation)](#2-agora-m--pca--mlp-distillation)
- [Metrics](#3-metrics)
- [Training](#4-training)
- [Repository layout](#repository-layout)
- [Troubleshooting / gotchas](#troubleshooting--gotchas)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Installation

AGORA is tested with **Python 3.10.14, PyTorch 2.4.1 (CUDA 11.8), NumPy 1.24.3** on NVIDIA RTX 6000 Ada
GPUs. We use [`uv`](https://docs.astral.sh/uv/) for environment management.

### 1. Python environment

```bash
git clone git@github.com:ramazan793/AGORA.git
cd AGORA

# Create the venv (Python 3.10) and install the pure-Python / wheel dependencies.
uv venv
uv sync --inexact          # see the WARNING below before re-running uv sync
```

> ⚠️ **Never run a bare `uv sync` after the CUDA extensions are installed.** A plain `uv sync`
> (exact mode) *prunes* packages that are not declared in `pyproject.toml` — and the compiled CUDA
> forks below are installed out-of-band, so they would be deleted. Always use **`uv run --no-sync`**
> to run scripts, and `uv sync --inexact` (or `uv pip install`) when you must add a dependency.

### 2. CUDA extensions (compiled forks)

AGORA depends on several CUDA-compiled extensions that are **not** resolved by `uv` and must be built
against your CUDA toolkit (we used **CUDA 11.8**). Install the following into the venv:

| Package | Source | Notes |
|---|---|---|
| `gsplat` | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | Primary rasterizer used for **training** (`--raster_backend gsplat`). |
| `diff_gaussian_rasterization` (+ `_features`, `_radegs`, `_distwar`, `_distwar_features`) | 3DGS / RaDe-GS / DISTWAR forks | Alternative rasterizers used at inference. |
| `simple_knn` | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | KNN for Gaussian init. |
| `gaussian_splatting` | 3DGS scene/render utilities | `.ply` scene IO + rendering. |
| `pytorch3d` | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | Mesh ops / FLAME rasterization. |
| `nvdiffrast` | [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) | Differentiable rasterization. |
| `eg3d` JIT ops (`bias_act`, `upfirdn2d`) | [NVlabs/eg3d](https://github.com/NVlabs/eg3d) | Compiled lazily by `torch.utils.cpp_extension` on first use (cached in `~/.cache/torch_extensions`). Requires `ninja` + `setuptools` (already in `pyproject.toml`). |

**Fast path (reuse an already-compiled environment).** If you have an existing GGHead/EG3D Conda/venv
with these built for the same Python/Torch/CUDA ABI, copy them into the AGORA venv:

```bash
# Point SRC at an existing site-packages that already has the compiled forks.
SRC=/path/to/existing/site-packages bash scripts/setup_cuda_deps.sh
```

> 🔒 **NumPy ABI:** these extensions are compiled against **NumPy 1.x**. NumPy 2.x breaks all of them
> (and the chumpy/FLAME monkeypatch). `numpy==1.24.3` is pinned in `pyproject.toml`. **Re-check
> `python -c "import numpy; print(numpy.__version__)"` after *any* `uv pip install`** — some wheels
> (e.g. unpinned `albumentations`) silently pull NumPy 2.x.

### 3. External dependencies (`dependencies/`)

The small data assets are now bundled **in-repo under `assets/`** (see [Repository layout](#repository-layout)).
The `dependencies/` directory holds only the two **external code packages** that the model imports at
runtime; both are assembled locally (git-ignored, not committed):

- `smirk/` — [SMIRK](https://github.com/georgeretsi/smirk). Clone it here. Needed for the **AED metric** and
  for **processing your own driving videos** (`pretrained_models/SMIRK_em1.pt`, `demo_image_folder.py`). The
  model also loads `smirk/assets/landmark_embedding.npy` at init, so SMIRK must be present even for plain
  inference. Its Python deps (`timm`, `iopath`, `chumpy`, `albumentations==1.3.1`) are already in
  `pyproject.toml`, so SMIRK runs **inside this same venv** — no separate environment.
- `threedim_utils/` — a small FLAME-LBS + mesh/UV utility package, published as a companion repo:
  <https://github.com/ramazan793/threedim_utils> (clone into `dependencies/`). The canonical "with-mouth"
  model loads its `assets/flame_with_mouth_no_backhead_v3/` files (UV mesh, masks, vertex mapping) and
  imports its `flame2020_lbs` code; the license-restricted `FLAME.pkl` must be added separately (see that
  repo's README).

> **FLAME 2020 (download separately).** The FLAME 2020 model is **license-restricted** and is **not**
> redistributed in this repo or in its dependency clones. Download it from the official
> [FLAME website](https://flame.is.tue.mpg.de/) and place the model files where each component expects them:
> SMIRK under `dependencies/smirk/assets/FLAME2020/`, and the with-mouth FLAME pickle under
> `dependencies/threedim_utils/assets/flame_with_mouth_no_backhead_v3/FLAME.pkl` (see that repo's README for
> the exact file list). Raw textures are not needed by the release pipeline.

These resolve under `<repo>/dependencies`; override the location with `GGHEAD_DEPENDENCIES_PATH`. (The in-repo
`assets/` files are addressed relative to the repo root and are not affected by that variable.)

### 4. FFmpeg (for writing MP4s)

The video scripts use `mediapy`, which needs an `ffmpeg` binary on `PATH`. The `imageio-ffmpeg` wheel
(already a dependency) bundles one; expose it to the venv once:

```bash
ln -sf "$PWD/.venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/"ffmpeg-linux-* \
       "$PWD/.venv/bin/ffmpeg"
```

### 5. Configure paths

Set these env vars (or put them in `~/.config/gghead/.env` as `GGHEAD_DATA_PATH=...` etc.):

| Variable | Meaning |
|---|---|
| `GGHEAD_MODELS_PATH` | Root containing trained run folders (checkpoints). |
| `GGHEAD_DATA_PATH` | Parent dir of the dataset zip (FFHQ). Used by FID + training. |
| `GGHEAD_RENDERINGS_PATH` | Output dir for FLAME renderings during training. |
| `GGHEAD_DEPENDENCIES_PATH` | Defaults to `<repo>/dependencies`. |

---

## Data & checkpoints

These large assets live outside the repo. The concrete locations used for the paper:

| Asset | Path | Notes |
|---|---|---|
| Trained checkpoints | `$GGHEAD_MODELS_PATH/dgghead/DGGHEAD-158_...res512.../checkpoints/` | Stage-2 final model. Default checkpoint `20500`. |
| Stage-1 checkpoint | `$GGHEAD_MODELS_PATH/dgghead/DGGHEAD-148_...res256.../checkpoints/checkpoint-6500.pkl` | Stage-1 final; the resume point for stage-2 training. |
| FFHQ dataset | `$GGHEAD_DATA_PATH/FFHQ_png_512/FFHQ_png_512.zip` | 512² FFHQ. |
| FLAME renderings | `.../FFHQ_png_512/FFHQ_png_512__flame_renderings__v4.zip` | **Must sit next to** the dataset zip. |
| Modnet masks | `.../FFHQ_png_512/FFHQ_png_512_masks_modnet.zip` | Foreground masks. |
| SMIRK FLAME meshes | `.../FFHQ_png_512/smirk/crop/img*/{shape,exp,globalpose,jawpose,cam,eyelid}.npy` | Per-image FLAME params. |
| Driving videos | `.../reenact_test_videos/<name>/` | SMIRK-processed clips (e.g. `obama_next3d`, 1786 frames). |

> **Bundled release weights (download from Google Drive).** A few large binaries are referenced from
> `assets/` but are **git-ignored**, hosted on Google Drive →
> <https://drive.google.com/drive/folders/1ZZoKtZsOodnYUfTHrH1P1QxHciIuXsoY?usp=sharing>
> Download them and place under `assets/`, preserving the relative paths below:
> - `assets/checkpoints/DGGHEAD-148_stage1_res256/checkpoint-6500.pkl` (stage-1) and
>   `assets/checkpoints/DGGHEAD-158_stage2_res512/checkpoint-20500.pkl` (stage-2) — the small config JSONs
>   beside them **are** committed.
> - `assets/agora_m/{svd_basis.pt, mlp_regressor.pt}` (AGORA-M, distilled from 158@20500).
> - `assets/fused_params_dataset.npy` (~200 MB, FFHQ FLAME/camera priors).
>
> Together ~1.7 GB. Note the inference/metric scripts also load checkpoints **by run name from
> `$GGHEAD_MODELS_PATH`** (the `assets/checkpoints/` copies are for convenient bundling/reference).

**Driving-video format** (per clip `<name>/`): `<name>/smirk/<frame>/{shape,exp,globalpose,jawpose,cam,eyelid}.npy`
plus `<name>/<frame>.png` (the RGB driving frames, 1:1 with the smirk folders). To process your own video,
run `dependencies/smirk/demo_image_folder.py` on extracted frames.

> ⚠️ **Keep the dataset on a path that does *not* contain the substring `data3`.** The dataset loader
> rewrites the hardcoded `smirk/crop` mesh path `/data2 → /data3` when it sees `data3` in the dataset
> path, which points at a non-existent directory and yields *0 meshes* (FID/training fail silently).

---

## Quickstart

Every command follows the same shape — set the env vars, then `uv run --no-sync python <script>`:

```bash
export GGHEAD_MODELS_PATH=/path/to/models
export GGHEAD_DATA_PATH=/path/to/datasets/FFHQ_png_512   # parent dir of the .zip
export CUDA_VISIBLE_DEVICES=0
```

---

## 1. Inference — generate and reenact an avatar

### Generate an avatar (extract blendshapes)

```bash
uv run --no-sync python scripts/deform_blendshapes/create_blendshapes.py \
    --run-name DGGHEAD-158 --checkpoint 20500 --id-seed 10
```

Loads the model, samples identity `seed 10`, renders a neutral 512² avatar plus expression/jaw blendshape
sweeps (to inspect the generated avatar), and extracts the blendshape planes used by the *alternative*
blendshape-based reenactor. Outputs:
- `scripts/deform_blendshapes/data/DGGHEAD-158_20500_seed10_eps1.0/blendshape_planes.pt`
- `scripts/deform_blendshapes/vis/.../renders/{neutral.png, exp/*.png, jaw/*.png}`

### Reenact (drive the avatar with a video)

```bash
uv run --no-sync python scripts/deform_blendshapes/reenact_avatar_multi_id.py \
    --run-name DGGHEAD-158 --checkpoint 20500 --id-seed 10 \
    --cam-scale 8 --pairs-list scripts/reenact_list.txt
```

This is the canonical AGORA reenactment: it generates the avatar from `--id-seed` and drives its
deformation branch **directly** from each driving frame's SMIRK FLAME parameters — no pre-extracted
blendshape planes needed. `--cam-scale 8` is required for correct framing; the remaining paper settings
are already the script defaults (`--synthesis-flame-cond 1 --cache-backbone 1 --use-narrow-mask 0
--savgol-win 5 --resolution 512`).

`reenact_list.txt` is one driving clip per line. The simplest valid line is a single folder token, which
auto-derives the smirk/RGB sub-paths:

```
/path/to/reenact_test_videos/obama_next3d
```

Outputs (in `scripts/deform_blendshapes/results/dynamic_view/DGGHEAD-158_20500__intrinsics_s_8.0/seed10/`):
`<vid>.mp4` (avatar | driver, side-by-side) and `left_<vid>.mp4` (avatar only).

> **Alternative (blendshape-based) reenactor.** `reenact_avatar_multi_id__deform_blendshapes.py` drives the
> avatar from the precomputed `blendshape_planes.pt` produced above (pass `--blendshape-planes-path … --cam-scale 8`).
> It approximates the direct reenactor; prefer `reenact_avatar_multi_id.py` for the headline result, and
> **always pass `--cam-scale 8`** (omitting it produces a seam/scale artifact on the face).

---

## 2. AGORA-M — PCA + MLP distillation

AGORA-M replaces the per-frame deformation network with a shared, identity-independent SVD basis plus a
small MLP conditioned on `[W (512-dim identity), FLAME expression (55-dim)]`. This bypasses the
deformation branch entirely at inference (the mobile/real-time variant).

> **The release ships the paper-trained distilled module** in `assets/agora_m/` (`svd_basis.pt` +
> `mlp_regressor.pt`, distilled from `DGGHEAD-158 @ 20500`). To just run AGORA-M, skip to
> [Step 3](#step-3--reenact-with-the-distilled-module) and point at those files. Steps 1–2 are only needed
> to re-distill from scratch.

### Step 1 — build the shared SVD basis + training set

```bash
uv run --no-sync python scripts/deform_blendshapes_pca_v2/create_svd_basis_and_dataset.py \
    --run-name DGGHEAD-158 --checkpoint 20500 \
    --n-identities 50 --N-per-id 100 --K 64
```

→ `scripts/deform_blendshapes_pca_v2/data/DGGHEAD-158_20500_M50_Nper100_K64/svd_basis.pt`

### Step 2 — train the MLP regressor

```bash
uv run --no-sync python scripts/deform_blendshapes_pca_v2/train_mlp_regressor.py \
    --svd-data-path scripts/deform_blendshapes_pca_v2/data/DGGHEAD-158_20500_M50_Nper100_K64/svd_basis.pt \
    --epochs 3000 --lr 1e-3
```

→ `.../DGGHEAD-158_20500_M50_Nper100_K64/mlp_regressor.pt`

### Step 3 — reenact with the distilled module

```bash
uv run --no-sync python scripts/deform_blendshapes_pca_v2/reenact_avatar_multi_id__svd_mlp.py \
    --run-name DGGHEAD-158 --checkpoint 20500 \
    --svd-data-path assets/agora_m/svd_basis.pt --mlp-path assets/agora_m/mlp_regressor.pt \
    --pairs-list scripts/reenact_list.txt --id-seed 10 --cam-scale 8
```

(Replace the two paths with your own `data/<tag>/{svd_basis,mlp_regressor}.pt` if you re-distilled in
Steps 1–2.) `--cam-scale 8` is **required** — the SVD basis and W-codes are built with `c_front[0,3] = 8.0`,
so inference must match. Writes a 4-panel comparison MP4 (`comparison_<vid>.mp4`: distilled approx | full
neural | pixel error | driver) plus the individual panels.

> Paper-quality hyperparameters are `--n-identities 50 --N-per-id 100 --K 64` (step 1) and `--epochs 3000`
> (step 2). For a quick smoke test use `--n-identities 3 --N-per-id 10 --K 8` and `--epochs 200`.
> Pass `--no-face-mask` to step 3 if your `mediapipe` build lacks `.solutions` (see
> [gotchas](#troubleshooting--gotchas)).

---

## 3. Metrics

Run on the stage-2 model (`DGGHEAD-158 @ 20500`).

### FPS (render throughput)

```bash
uv run --no-sync python scripts/metrics/measure_fps.py
```
Reports render-only FPS (excludes model load). Paper: **~330 FPS** at 512², batch 8.

### FID

```bash
uv run --no-sync python scripts/metrics/evaluate_fid.py DGGHEAD-158 --fid 50000 --local
```
`--local` repoints the saved dataset path to `$GGHEAD_DATA_PATH`. Paper: **FID-50k ≈ 3.21**.

### ID (identity consistency)

```bash
uv run --no-sync python scripts/metrics/evaluate_id.py \
    --run-name DGGHEAD-158 --checkpoint 20500 --num-identities 1000
```
ArcFace (`buffalo_l`) cosine similarity between two renders of the same latent under different
camera/expression. Downloads the ArcFace weights on first run.

### AED / AED-jaw / APD / ASD (expression & pose fidelity)

```bash
uv run --no-sync python scripts/metrics/evaluate_aed.py \
    --run-name DGGHEAD-158 --checkpoint 20500 \
    --num-sampled-z 500 --num-sampled-cam-exp 20 --device cuda:0
```

Runs SMIRK as a subprocess to re-estimate FLAME on the generated frames and reports
**AED** (expression RMSE), **AED-jaw**, **APD** (global pose), **ASD**, and **AED_eyelids**. By default
SMIRK runs under the *current* interpreter (`sys.executable`), so no separate env is needed; override with
`--smirk-python-bin` only if you keep SMIRK elsewhere.

---

## 4. Training

Two-stage recipe. Set the env block first:

```bash
export GGHEAD_DATA_PATH=/path/to/datasets/FFHQ_png_512   # contains the dataset zip + renderings + masks
export GGHEAD_MODELS_PATH=/path/to/models
export GGHEAD_RENDERINGS_PATH=/path/to/renderings
export BW_IMPLEMENTATION=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Stage 1 — 256²

```bash
uv run --no-sync python scripts/train_gghead.py ffhq FFHQ_png_512.zip 4 32 \
  --kimg 6500 --raster_backend gsplat --use_flame_template_with_mouth True \
  --flame_dual_discrimination 1 --use_flame_cameras 1 --use_flame_to_bfm_registration 0 \
  --use_concat 1 --use_deformation_branch 1 --deformation_start_resolution 64 \
  --deformation_plane_resolution 256 --zero_out_color_residuals 1 --use_deform_mask 1 \
  --scale_offset -3.5 --lambda_raw_gaussian_position 0.25 --lambda_raw_gaussian_scale 0.5 \
  --lambda_raw_deform_gaussian_position 0.5 --lambda_raw_deform_gaussian_scale 1.0 \
  --lambda_raw_deform_gaussian_opacity 0.5 --start_dd_kimg 3200 --use_uv_reg_weights 1 \
  --use_shift_augmentation 1 --shift_aug_value 0.1
```

Produces run `DGGHEAD-148`; its final checkpoint is `checkpoint-6500.pkl`.

### Stage 2 — progressive grow to 512² (resume from stage 1)

```bash
uv run --no-sync python scripts/train_gghead.py ffhq FFHQ_png_512.zip 4 32 \
  --kimg 6500 --image_snapshot_ticks 50 --flame_dual_discrimination 1 \
  --flame_rasterization_light_type ambient --use_mouth_branch 0 \
  --lambda_raw_gaussian_position 0.25 --lambda_raw_gaussian_scale 0.5 \
  --use_uv_reg_weights 1 --use_shift_augmentation 1 --shift_aug_value 0.1 \
  --use_extended_uv_generation 0 --gen_flame_conditioning 0 --use_deformation_branch 1 \
  --use_concat 1 --double_condition 0 --use_deform_mask 1 --use_flame_to_bfm_registration 0 \
  --use_flame_cameras 1 --zero_out_color_residuals 1 --dd_shape_n_components 0 \
  --raster_backend gsplat --use_flame_template_with_mouth --deformation_start_resolution 64 \
  --deformation_plane_resolution 256 --lambda_raw_deform_gaussian_position 0.5 \
  --lambda_raw_deform_gaussian_scale 1.0 --lambda_raw_deform_gaussian_opacity 0.5 \
  --start_dd_kimg 3200 --scale_offset -3.5 \
  --resume_run DGGHEAD-148 --resume_checkpoint 6500 \
  --overwrite_resolution 512 --overwrite_n_uniform_flame_vertices 512 \
  --overwrite_lambda_tv_uv_rendering 1 --overwrite_lambda_beta_loss 1 \
  --kimg 25000 --overwrite_lambda_raw_deform_gaussian_position 1.5 \
  --overwrite_lambda_raw_deform_gaussian_scale 1.5
```

Notes:
- `--overwrite_resolution 512` is the progressive-growth trigger (256 → 512); it also sets the
  discriminator's new-layer start to `--resume_checkpoint`.
- `--kimg` is given twice on purpose — the parser keeps the last (`25000`).
- The FLAME renderings zip (`..._flame_renderings__v4.zip`) and the modnet masks zip must sit next to
  the dataset zip; `precomputed_flame_renderings` reads them from there.

---

## Repository layout

```
src/gghead/                       # the model package (import as `src.gghead`)
scripts/
  train_gghead.py                 # GAN training entry point (stage 1 & 2)
  deform_blendshapes/             # Inference: create_blendshapes + reenact_avatar_multi_id (canonical, direct)
  deform_blendshapes_pca_v2/      # AGORA-M: SVD basis → MLP regressor → distilled reenact
  metrics/                        # FID, FPS, ID, AED
assets/
  gghead/                         # GGHEAD template meshes (.obj/.mtl) + deform masks / uv-position weights
  flame/                          # small FLAME2020 template (uv->3d vertex mapping)
  checkpoints/                    # DGGHEAD-148 / -158 config JSONs (committed); *.pkl git-ignored -> host externally
  agora_m/                        # distilled svd_basis.pt + mlp_regressor.pt (git-ignored -> host externally)
  fused_params_dataset.npy        # FFHQ FLAME/camera priors (git-ignored -> host externally)
dependencies/                     # external clones only: smirk/ + threedim_utils/ (assembled locally, git-ignored)
pyproject.toml                    # uv-managed Python deps (CUDA forks installed separately)
```

---

## Troubleshooting / gotchas

- **`ModuleNotFoundError: No module named 'src'`** — run scripts from the repo root. Every script inserts
  the repo root on `sys.path` via a depth-robust finder; all imports use the `src.gghead` root.
- **`uv sync` deleted my CUDA extensions** — a bare `uv sync` prunes out-of-band packages. Re-run
  `scripts/setup_cuda_deps.sh` (or use `uv sync --inexact`), and use `uv run --no-sync` thereafter.
- **NumPy got upgraded to 2.x and everything broke** — pin it back: `uv pip install "numpy==1.24.3"`. Keep
  `albumentations==1.3.1` (≥2.0 drags NumPy ≥2).
- **`mediapy` can't find ffmpeg** (video scripts run all GPU work, then crash at the MP4-encode step) —
  create the `.venv/bin/ffmpeg` symlink (install step 4). `mediapy` finds `ffmpeg` only when `.venv/bin` is
  on `PATH`: `uv run` puts it there automatically, but if you invoke `.venv/bin/python` directly, prepend it
  yourself (`PATH=$PWD/.venv/bin:$PATH`).
- **FID reports "Found 0 meshes / no image files"** — your dataset path contains `data3`; move it to a path
  without that substring (see the loader rewrite warning above).
- **`module 'mediapipe' has no attribute 'solutions'`** — the installed `mediapipe` build lacks the classic
  `solutions` API; this only affects the optional face-region error overlay in AGORA-M reenactment. Pass
  `--no-face-mask`, or pin a `mediapipe` version that exposes `mediapipe.solutions`.

---

## Acknowledgments

This work builds on a number of excellent open-source projects:

- [EG3D](https://github.com/NVlabs/eg3d) and [GGHead](https://github.com/tobias-kirschstein/gghead) —
  generator architecture and training framework.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
  [gsplat](https://github.com/nerfstudio-project/gsplat), and the RaDe-GS / DISTWAR rasterizer forks —
  real-time Gaussian rendering.
- [FLAME](https://flame.is.tue.mpg.de/) — the 3D head/face model.
- [SMIRK](https://github.com/georgeretsi/smirk) — expression estimation used by the AED metric.
- [Gaussian Déjà-vu](https://github.com/PeizhiYan/gaussian-dejavu) — the UV position weights our deformation
  branch builds on (`assets/gghead/uv_position_weights_dejavu_adapted*.npy`).
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and
  [nvdiffrast](https://github.com/NVlabs/nvdiffrast) — differentiable mesh/rasterization ops.

We thank the authors of these projects. Please consult and cite their original works and respect their
licenses when using this code.

## Citation

If you find this work useful, please cite:

```bibtex
@article{fazylov2025agora,
    author = {Fazylov, Ramazan and Zagoruyko, Sergey and Parkin, Aleksandr and Lefkimmiatis, Stamatis and Laptev, Ivan},
    title = {{AGORA: Adversarial Generation Of Real-time Animatable 3D Gaussian Head Avatars}},
    journal = {arXiv preprint arXiv:2512.06438},
    year = {2025}
}
```
