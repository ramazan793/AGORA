"""evaluate_fid_with_vis.py — FID metric with optional sample visualization.

Identical to scripts/evaluate_fid.py but adds --save-samples N to save the
first N generated images as PNGs for visual inspection.
"""
from pathlib import Path
from typing import Union

import os
import torch
import torch.nn as nn
import tyro
from PIL import Image
from eg3d.metrics.metric_main import calc_metric, register_metric

import os
import sys
_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)

from src.gghead.env import GGHEAD_DATA_PATH
from src.gghead.model_manager.base_model_manager import GGHeadEvaluationConfig, GGHeadEvaluationResult
from src.gghead.model_manager.finder import find_model_manager
from src.gghead.util.metrics import fid100, fid1k, fid50k_full, fid5k, fid10k


class SampleSavingWrapper(nn.Module):
    """Thin wrapper around G that saves the first N generated images as PNGs.

    Exposes the same interface as G (z_dim, c_dim, forward signature) so the
    FID pipeline (calc_metric / compute_feature_stats_for_generator) can use
    it transparently.
    """

    def __init__(self, G, save_dir: str, save_limit: int):
        super().__init__()
        self.G = G
        self.z_dim = G.z_dim
        self.c_dim = G.c_dim
        self._save_dir = save_dir
        self._save_limit = save_limit
        self._saved_count = 0

    def forward(self, z, c, flame_params=None, c2=None, **kwargs):
        output = self.G(z=z, c=c, flame_params=flame_params, c2=c2, **kwargs)
        images = output['image']

        if self._saved_count < self._save_limit:
            for img_t in images:
                if self._saved_count >= self._save_limit:
                    break
                img_np = (
                    (img_t.clamp(-1, 1) + 1) / 2 * 255
                ).byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(img_np).save(
                    os.path.join(self._save_dir, f"{self._saved_count:05d}.png")
                )
                self._saved_count += 1

        return output


def main(run_names: str,
         /,
         fid: int = 50000,                  # How many samples to generate for FID
         load_ema: bool = True,             # Use EMA weights
         checkpoint: Union[int, str] = -1,  # Checkpoint to evaluate
         local: bool = False,               # Use local dataset path
         save_samples: int = 0,             # Save first N generated samples as PNGs (0 = off)
         gpus: int = 1):
    torch.multiprocessing.set_start_method('spawn')

    for run_name in run_names.split(','):
        try:
            model_manager = find_model_manager(run_name)
            if checkpoint == 'all':
                checkpoint_ids = model_manager.list_checkpoint_ids()
            elif checkpoint == 'remaining':
                candidate_checkpoint_ids = model_manager.list_checkpoint_ids()
                checkpoint_ids = []
                for checkpoint_id in candidate_checkpoint_ids:
                    evaluation_config = GGHeadEvaluationConfig(checkpoint=checkpoint_id, load_ema=load_ema)
                    if not model_manager.has_evaluation_result(evaluation_config):
                        checkpoint_ids.append(checkpoint_id)
                    else:
                        evaluation_result = model_manager.load_evaluation_result(evaluation_config)
                        if evaluation_result.get_fid(fid) is None:
                            checkpoint_ids.append(checkpoint_id)
            else:
                if isinstance(checkpoint, int):
                    checkpoint_ids = [model_manager._resolve_checkpoint_id(checkpoint)]
                else:
                    checkpoint_ids = [model_manager._resolve_checkpoint_id(int(ckpt)) for ckpt in checkpoint.split(',')]

            for checkpoint_id in checkpoint_ids:
                model = model_manager.load_checkpoint(checkpoint_id, load_ema=load_ema).cuda()
                dataset_config = model_manager.load_dataset_config()

                if local:
                    dataset_config.path = f"{GGHEAD_DATA_PATH}/{Path(dataset_config.path).name}"

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
                    raise ValueError(f"Wrong FID count {fid}")

                G = model
                if save_samples > 0:
                    samples_dir = os.path.join(
                        'evaluations', 'fid_samples', f"{run_name}_{checkpoint_id}"
                    )
                    os.makedirs(samples_dir, exist_ok=True)
                    print(f"Saving first {save_samples} samples to {samples_dir}/")
                    G = SampleSavingWrapper(model, save_dir=samples_dir, save_limit=save_samples)

                result_dict = calc_metric(metric=fid_metric.__name__, G=G,
                                          dataset_kwargs=dataset_config.get_eval_dict(), num_gpus=gpus, rank=0, device='cuda')

                evaluation_config = GGHeadEvaluationConfig(checkpoint=checkpoint_id, load_ema=load_ema)
                evaluation_result = GGHeadEvaluationResult(**result_dict['results'])

                print("===========================")
                print(f"Evaluation for {run_name} - checkpoint {checkpoint_id}")
                print("===========================")
                print(evaluation_result)
                model_manager.store_evaluation_result(evaluation_config, evaluation_result, overwrite=False)
        except Exception as e:
            print(f"Skipping {run_name} due to {e}")
            raise e


if __name__ == '__main__':
    tyro.cli(main)
