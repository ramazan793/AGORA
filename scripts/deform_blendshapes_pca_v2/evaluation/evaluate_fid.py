"""
evaluate_fid.py — FID metric using SVD+MLP (PCA v2) deformation bypass.

Based on scripts/evaluate_fid.py. Additional args: --svd-path, --mlp-path.

Approach: PCAGeneratorWrapper wraps G and overrides __call__/forward so the FID
pipeline (calc_metric) transparently uses PCA inference instead of the full
deformation network. The wrapper exposes the same interface as the original G
(z_dim, c_dim, forward signature).

Per call G(z, c, flame_params, c2):
  1. Build neutral FLAME params from flame_params (keep shape, zero exp/jaw/eyelid).
  2. mapping(z, c_front=c, neutral) → w_code (backbone_ws[:, 0, :]).
  3. synthesis(neutral) → base gaussians in physical space (no backbone cache set).
  4. MLP(w_code, flame_55) → PCA coefficients → reconstruct physical attrs.
  5. mapping(z, c, actual flame_params) → ws for backbone.
  6. synthesis(ws, c, flame_params, post_act_blendshapes=...) → rendered image.

NOTE: Each call processes the batch sample-by-sample for correctness
(different z → different base gaussians). This is slower than the original FID
but acceptable for offline evaluation.
"""

from __future__ import annotations

import copy
import sys
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import tyro

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pca_inference_utils import load_svd_mlp, setup_identity, generate_with_pca


class PCAGeneratorWrapper(nn.Module):
    """Wraps DynGGHead generator G to use SVD+MLP deformation bypass.

    Exposes the same interface as G so calc_metric / compute_feature_stats_for_generator
    can use it transparently. The FID pipeline accesses:
      - G.z_dim, G.c_dim  (set as attributes from G)
      - G(z=z, c=c, flame_params=m, c2=m, **G_kwargs)['image']
      - copy.deepcopy(G)  (all fields must be deepcopy-able)
    """

    def __init__(self, G, svd_data: dict, mlp: nn.Module, K: int, G_num: int,
                 truncation_psi: float = 0.7,
                 save_samples_dir: str = "", save_samples_limit: int = 0,
                 gt_also: bool = False):
        super().__init__()
        self.G = G
        self.svd_data = svd_data
        self.mlp = mlp
        self.K = K
        self.G_num = G_num
        self.truncation_psi = truncation_psi
        self.gt_also = gt_also

        # Expose attributes that the FID pipeline reads directly
        self.z_dim = G.z_dim
        self.c_dim = G.c_dim

        # Sample saving state (plain attributes, not nn.Parameters)
        self._save_dir = save_samples_dir
        self._save_limit = save_samples_limit
        self._saved_count = 0

    def forward(self, z: torch.Tensor, c: torch.Tensor,
                flame_params: torch.Tensor, c2=None, **kwargs):
        """Process a batch using PCA inference.

        All B samples are processed together: one batched neutral mapping+synthesis
        to obtain per-identity base gaussians [B,G,C], then one batched actual
        mapping+synthesis for the final render.

        Args:
            z: [B, z_dim]
            c: [B, 6] camera params
            flame_params: [B, 358] FLAME params
            c2: ignored (present for API compatibility)
            **kwargs: forwarded to G.synthesis (noise_mode, neural_rendering_resolution, ...)

        Returns:
            dict with 'image': [B, 3, H, W]
        """
        from src.gghead.config.gaussian_attribute import GaussianAttribute

        device = z.device
        B = z.shape[0]
        ov = getattr(self.G._config, 'opacity_overshoot', 0.0)

        resolution = kwargs.get('neural_rendering_resolution', 512)
        noise_mode = kwargs.get('noise_mode', 'const')

        with torch.no_grad():
            # Build neutral FLAME params [B, 358]: per-sample shape, zero exp/jaw/eyelid
            neutral_flame = torch.zeros(B, 358, device=device, dtype=torch.float32)
            neutral_flame[:, :300] = flame_params[:, :300]

            # Batched mapping at neutral expression → w_codes [B, 512]
            ws_neutral = self.G.mapping(
                z, c,
                truncation_psi=self.truncation_psi,
                flame_params=neutral_flame,
            )
            backbone_ws_neutral, _ = ws_neutral
            w_codes = backbone_ws_neutral[:, 0, :]  # [B, 512]

            # Batched synthesis at neutral → base gaussians [B, G, C]
            output_base = self.G.synthesis(
                ws_neutral, c, neutral_flame,
                neural_rendering_resolution=resolution,
                cache_backbone=False,
                use_cached_backbone=False,
                noise_mode='const',
                sh_ref_cam=c,
                return_gaussian_attributes=True,
            )
            ga_raw = output_base.returned_gaussian_attributes
            base_gaussians = {
                'xyz':      ga_raw[GaussianAttribute.POSITION].detach(),
                'scale':    self.G._gaussian_model.scaling_activation(
                                ga_raw[GaussianAttribute.SCALE]).detach(),
                'rotation': self.G._gaussian_model.rotation_activation(
                                ga_raw[GaussianAttribute.ROTATION]).detach(),
                'opacity':  self.G._gaussian_model.opacity_activation(
                                ga_raw[GaussianAttribute.OPACITY]).detach(),
            }
            base_color = ga_raw[GaussianAttribute.COLOR].detach()

            G_num_val = base_gaussians['xyz'].shape[1]
            C_color = base_color.shape[-1]
            zero_deltas = {
                'xyz':      torch.zeros(59, G_num_val, 3,       device=device),
                'scale':    torch.zeros(59, G_num_val, 3,       device=device),
                'rotation': torch.zeros(59, G_num_val, 4,       device=device),
                'opacity':  torch.zeros(59, G_num_val, 1,       device=device),
                'color':    torch.zeros(59, G_num_val, C_color, device=device),
            }

            # Batched mapping with actual flame_params
            ws_actual = self.G.mapping(
                z, c,
                truncation_psi=self.truncation_psi,
                flame_params=flame_params,
                c2=flame_params,
            )

            output = generate_with_pca(
                self.G, ws_actual, c, flame_params,
                w_codes, base_gaussians, base_color, zero_deltas,
                self.svd_data, self.mlp, device, self.K, ov,
                sh_ref_cam=c,
                noise_mode=noise_mode,
                neural_rendering_resolution=resolution,
                cache_backbone=False,
                use_cached_backbone=False,
            )

            # Full deform path for paired saving
            if self.gt_also:
                output_gt = self.G.synthesis(
                    ws_actual, c, flame_params,
                    neural_rendering_resolution=resolution,
                    cache_backbone=False,
                    use_cached_backbone=False,
                    noise_mode=noise_mode,
                    sh_ref_cam=c,
                )
                gt_images = output_gt['image']
            else:
                gt_images = None

        images = output['image']

        if self._save_dir and self._saved_count < self._save_limit:
            for i, img_t in enumerate(images):
                if self._saved_count >= self._save_limit:
                    break
                if gt_images is not None:
                    # Paired: PCA (left) | full deform (right) → 1024×512
                    gt_img_t = gt_images[i]
                    paired = torch.cat([img_t, gt_img_t], dim=2)  # concat along width
                    img_np = ((paired.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = ((img_t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(img_np).save(
                    os.path.join(self._save_dir, f"{self._saved_count:05d}.png")
                )
                self._saved_count += 1

        return {'image': images}


class FullDeformWrapper(nn.Module):
    """Wraps G with full deformation network inference (no PCA bypass).

    Exposes the same interface as PCAGeneratorWrapper so calc_metric can use it.
    """

    def __init__(self, G, truncation_psi: float = 1.0):
        super().__init__()
        self.G = G
        self.truncation_psi = truncation_psi
        self.z_dim = G.z_dim
        self.c_dim = G.c_dim

    def forward(self, z: torch.Tensor, c: torch.Tensor,
                flame_params: torch.Tensor, c2=None, **kwargs):
        with torch.no_grad():
            out = self.G(z=z, c=c, flame_params=flame_params, c2=flame_params,
                         truncation_psi=self.truncation_psi, **kwargs)
        return {'image': out['image']}


def main(run_names: str,
         /,
         svd_path: str = "",                    # Path to svd_basis.pt (PCA v2)
         mlp_path: str = "",                    # Path to mlp_regressor.pt (PCA v2)
         fid: int = 50000,                      # How many samples to generate for FID
         load_ema: bool = True,                 # Use EMA weights
         checkpoint: Union[int, str] = -1,      # Checkpoint to evaluate (-1 = latest)
         local: bool = False,                   # Use local dataset path
         truncation_psi: float = 1.0,           # Truncation for mapping (1.0 = no truncation, matches vanilla FID)
         batch_gen: int = 16,                   # Batch size fed to PCAGeneratorWrapper
         save_samples: int = 0,                 # Save first N generated samples as PNGs (0 = off)
         gt_also: int = 0,                      # Also compute full deform FID and save paired images
         gpus: int = 1):
    assert svd_path, "Provide --svd-path pointing to svd_basis.pt"
    assert mlp_path, "Provide --mlp-path pointing to mlp_regressor.pt"

    torch.multiprocessing.set_start_method('spawn')

    from eg3d.metrics.metric_main import calc_metric, register_metric
    from src.gghead.env import GGHEAD_DATA_PATH
    from src.gghead.model_manager.finder import find_model_manager
    from src.gghead.util.metrics import fid100, fid1k, fid50k_full, fid5k, fid10k

    for run_name in run_names.split(','):
        try:
            model_manager = find_model_manager(run_name)

            if isinstance(checkpoint, int):
                checkpoint_ids = [model_manager._resolve_checkpoint_id(checkpoint)]
            else:
                checkpoint_ids = [
                    model_manager._resolve_checkpoint_id(int(ckpt))
                    for ckpt in str(checkpoint).split(',')
                ]

            for checkpoint_id in checkpoint_ids:
                print(f"Loading {run_name} checkpoint {checkpoint_id}...")
                model = model_manager.load_checkpoint(checkpoint_id, load_ema=load_ema).cuda()
                model.eval()
                model._config.use_flame_rasterization = 0

                dataset_config = model_manager.load_dataset_config()
                if local:
                    dataset_config.path = f"{GGHEAD_DATA_PATH}/{Path(dataset_config.path).name}"

                # Load SVD+MLP
                device = torch.device('cuda')
                svd_data, mlp, K, G_num = load_svd_mlp(svd_path, mlp_path, device)

                # Sample saving directory (auto-derived when save_samples > 0)
                samples_dir = ""
                if save_samples > 0:
                    dir_name = (
                        f"fid_samples_paired_{run_name}_{checkpoint_id}"
                        if gt_also else
                        f"fid_samples_{run_name}_{checkpoint_id}"
                    )
                    samples_dir = os.path.join(os.path.dirname(svd_path), dir_name)
                    os.makedirs(samples_dir, exist_ok=True)
                    print(f"Saving first {save_samples} samples to {samples_dir}/")

                # Wrap G with PCA bypass
                G_wrapped = PCAGeneratorWrapper(
                    model, svd_data, mlp, K, G_num,
                    truncation_psi=truncation_psi,
                    save_samples_dir=samples_dir,
                    save_samples_limit=save_samples,
                    gt_also=bool(gt_also),
                )

                register_metric(fid100)
                register_metric(fid1k)
                register_metric(fid50k_full)

                if fid == 100:
                    fid_metric = fid100
                elif fid == 1000:
                    fid_metric = fid1k
                elif fid == 5000:
                    fid_metric = fid5k
                elif fid == 10000:
                    fid_metric = fid10k
                elif fid == 50000:
                    fid_metric = fid50k_full
                else:
                    raise ValueError(f"Unsupported FID count {fid}")

                # Monkey-patch compute_feature_stats_for_generator to use custom batch_gen
                import src.gghead.eg3d.metrics.metric_utils_mesh as _mu
                _orig_cfsfg = _mu.compute_feature_stats_for_generator
                _bgv = batch_gen
                def _patched_cfsfg(*a, **kw):
                    kw['batch_gen'] = _bgv
                    return _orig_cfsfg(*a, **kw)
                _mu.compute_feature_stats_for_generator = _patched_cfsfg

                result_dict = calc_metric(
                    metric=fid_metric.__name__,
                    G=G_wrapped,
                    dataset_kwargs=dataset_config.get_eval_dict(),
                    num_gpus=gpus,
                    rank=0,
                    device='cuda',
                )

                print("===========================")
                print(f"FID (PCA v2) for {run_name} - checkpoint {checkpoint_id}")
                print("===========================")
                print(result_dict)

                if gt_also:
                    G_full = FullDeformWrapper(model, truncation_psi=truncation_psi)
                    result_dict_full = calc_metric(
                        metric=fid_metric.__name__,
                        G=G_full,
                        dataset_kwargs=dataset_config.get_eval_dict(),
                        num_gpus=gpus,
                        rank=0,
                        device='cuda',
                    )
                    print("===========================")
                    print(f"FID (Full Deform) for {run_name} - checkpoint {checkpoint_id}")
                    print("===========================")
                    print(result_dict_full)

        except Exception as e:
            print(f"Skipping {run_name} due to {e}")
            raise e


if __name__ == '__main__':
    tyro.cli(main)
