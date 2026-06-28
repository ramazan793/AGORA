from dataclasses import dataclass
from typing import Optional, Literal, Dict, List

import numpy as np
import torch
from eg3d.torch_utils import training_stats
from eg3d.torch_utils.ops import conv2d_gradfix
from eg3d.torch_utils.ops import upfirdn2d
from eg3d.training.dual_discriminator import filtered_resizing
from eg3d.training.loss import StyleGAN2Loss
from elias.config import Config, implicit

from src.gghead.config.gaussian_attribute import GaussianAttribute
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH
from src.gghead.models.dyn_gghead_model import GGHeadModel
from src.gghead.util.logging import LoggerBundle

import wandb
import torchvision

from src.gghead.util.flame_rasterizer import batch_rodrigues

@dataclass
class GGHeadStyleGAN2LossConfig(Config):
    r1_gamma: float = 10
    style_mixing_prob: float = 0
    pl_weight: float = 0
    pl_batch_shrink: float = 2
    pl_decay: float = 0.01
    pl_no_weight_grad: bool = False
    blur_init_sigma: int = 0
    blur_fade_kimg: int = 0
    r1_gamma_init: float = 0
    r1_gamma_fade_kimg: int = 0
    neural_rendering_resolution_initial: int = 64
    neural_rendering_resolution_final: Optional[int] = None
    neural_rendering_resolution_fade_kimg: int = 0
    gpc_reg_fade_kimg: int = 1000
    gpc_reg_prob: Optional[float] = None
    dual_discrimination: bool = False
    filter_mode: Literal['antialiased', 'classic', 'none'] = 'antialiased'
    aug: Optional[Literal['noaug', 'ada', 'fixed']] = 'noaug'
    ada_target: Optional[float] = 0.6

    # Mesh conditioning
    gmc_reg_fade_kimg: int = 1000
    gmc_reg_prob: Optional[float] = None

    # Discriminator resizing
    effective_res_disc: float = 1  # If < 1, images will be downscaled and upscaled again before fed to the discriminator

    # Progressive Discriminator growing
    new_layers_disc_start_kimg: Optional[int] = None
    new_layers_disc_blend_kimg: int = 1000
    new_layers_gen_start_kimg: Optional[int] = None
    new_layers_gen_blend_kimg: int = 1000
    plane_resolution_start_kimg: Optional[int] = None
    plane_resolution_blend_kimg: int = 1000
    effective_res_disc_start_kimg: Optional[int] = None
    effective_res_disc_blend_kimg: int = 1000

    # Gaussian regularization
    lambda_gaussian_position: float = 0
    lambda_gaussian_scale: float = 0
    reg_gaussian_position_above: float = 0
    reg_gaussian_position_below: float = 0
    reg_gaussian_scale_above: float = 0
    reg_gaussian_scale_below: float = 0
    reg_raw_gaussian_scale_above: float = 0
    reg_raw_gaussian_scale_below: float = 0
    use_l1_scale_reg: bool = False
    lambda_raw_gaussian_position: float = 0
    lambda_raw_mouth_gaussian_position: Optional[float] = None
    lambda_raw_mouth_gaussian_scale: Optional[float] = None
    lambda_raw_gaussian_scale: float = 0
    lambda_raw_scale_std: float = 0
    lambda_raw_gaussian_rotation: float = 0
    lambda_raw_gaussian_color: float = 0
    lambda_raw_gaussian_opacity: float = 0
    lambda_learnable_template_offsets: float = 0
    lambda_tv_learnable_template_offsets: float = 0
    lambda_tv_uv_rendering: float = 0
    tv_uv_include_transparent_gaussians: bool = False  # Whether, to apply the UV TV loss on a UV rendering that also includes transparent gaussians
    lambda_beta_loss: float = 0
    lambda_raw_xyz_entropy_masked: float = 0
    lambda_raw_deform_gaussian_position: float = 0
    lambda_raw_deform_gaussian_scale: float = 0
    lambda_raw_deform_gaussian_opacity: float = 0

    # Dual-discrimination start schedule (kimg). If set, zero-out FLAME images
    # when feeding D until the given kimg threshold is reached.
    start_dd_kimg: Optional[int] = None

    # Deformation linearity regularization
    lambda_deform_linearity_reg: float = 0

    # ID consistency
    lambda_id_consistency: float = 0
    id_feature_resolutions: Optional[List[int]] = None  # e.g., [64, 32, 16, 8]

    pretrained_resolution: Optional[int] = implicit()

    # UV Regularization weights
    use_uv_reg_weights: bool = False

    # Shift Augmentation
    use_shift_augmentation: bool = False
    shift_aug_value: float = 0.05

    # Weighted Loss
    use_weighted_loss: bool = False

    # Masks
    blur_masks: bool = True
    mask_swap_prob: float = 0
    r1_gamma_mask: Optional[float] = None

    decode_first: str = 'all'
    reg_weight: float = 0.1
    opacity_reg: float = 1
    l1_loss_reg: bool = True
    clamp: bool = False
    ref_scale: float = -5
    progressive_scale_reg_kimg: int = 0
    progressive_scale_reg_end: float = 0.01

    def requires_raw_gaussian_attributes(self) -> bool:
        return (self.lambda_raw_gaussian_position > 0
                or self.lambda_raw_gaussian_scale > 0
                or self.lambda_raw_gaussian_rotation > 0
                or self.lambda_raw_gaussian_color > 0
                or self.lambda_raw_gaussian_opacity > 0
                or self.lambda_raw_scale_std > 0)


class GGHeadStyleGAN2Loss(StyleGAN2Loss):
    def __init__(self,
                 device,
                 G,
                 D,
                 augment_pipe=None,
                 config: GGHeadStyleGAN2LossConfig = GGHeadStyleGAN2LossConfig(),
                 logger_bundle: LoggerBundle = LoggerBundle()) -> None:
        self._config = config
        self._logger_bundle = logger_bundle
        self.gmc_reg_fade_kimg = config.gmc_reg_fade_kimg
        self.gmc_reg_prob = config.gmc_reg_prob
        self.prev_nimg = 0

        if self._config.use_uv_reg_weights:
            # if G._config.mask_type == 'dejavu':
            #     uv_reg_weights = torch.from_numpy(np.load(f'{REPO_ROOT_DIR}/assets/gghead/uv_position_weights_dejavu_adapted_normalized.npy')).to(device) # [256, 256]
            # elif G._config.mask_type == 'narrow':
            #     uv_reg_weights = torch.from_numpy(np.load(f'{REPO_ROOT_DIR}/assets/gghead/narrow_facial_flame_mask.npy')).to(device) # [256, 256]
            uv_reg_weights = torch.from_numpy(np.load(f'{GGHEAD_DEPENDENCIES_PATH}/threedim_utils/assets/flame_with_mouth_no_backhead_v3/flame_facial_uv_mask__wo_mouth.npy')).to(device).float()
            uv_reg_weights = torch.nn.functional.interpolate(uv_reg_weights.unsqueeze(0).unsqueeze(0), size=G._config.plane_resolution, mode='bilinear', antialias=False).squeeze() # [res, res]
            
            # uv_reg_weights = G._deform_mask.squeeze()

            uv_grid_indices = (G._uv_grid.squeeze() + 1) / 2 * G._config.plane_resolution # from [-1, 1] to [0, res]
            uv_grid_indices = uv_grid_indices.round().long()
            self.per_gaussian_reg_weights = uv_reg_weights[uv_grid_indices[:, 1], uv_grid_indices[:, 0]].unsqueeze(0) # [1, G]
            self.uv_reg_mask = (self.per_gaussian_reg_weights > 0).float()

            mouth_uv_reg_weights = torch.from_numpy(np.load(f'{GGHEAD_DEPENDENCIES_PATH}/threedim_utils/assets/flame_with_mouth_no_backhead_v3/flame_just_mouth_uv_mask.npy')).to(device).float()
            mouth_uv_reg_weights = torch.nn.functional.interpolate(mouth_uv_reg_weights.unsqueeze(0).unsqueeze(0), size=G._config.plane_resolution, mode='bilinear', antialias=False).squeeze() # [res, res]
            self.per_gaussian_reg_weights_mouth = mouth_uv_reg_weights[uv_grid_indices[:, 1], uv_grid_indices[:, 0]].unsqueeze(0) # [1, G]
            self.uv_reg_mask_mouth = (self.per_gaussian_reg_weights_mouth > 0).float()
            # Multiplicative mask: 0.5 for mouth region, 1.0 elsewhere (for deform regularization)
            self.uv_deform_mouth_weight_mask = torch.where(self.uv_reg_mask_mouth == 1, 0.5, 1.0)

        if G._config.use_extended_uv_generation:
            if self._config.lambda_raw_mouth_gaussian_position is None:
                self._config.lambda_raw_mouth_gaussian_position = self._config.lambda_raw_gaussian_position 
            if self._config.lambda_raw_mouth_gaussian_scale is None:
                self._config.lambda_raw_mouth_gaussian_scale = self._config.lambda_raw_gaussian_scale

        if self._config.use_shift_augmentation:
            import torchvision.transforms as transforms
            from torchvision.transforms import InterpolationMode
            self.translation_augment = transforms.Compose([
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(self._config.shift_aug_value, self._config.shift_aug_value), 
                    scale=(0.85, 1.15), 
                    shear=None, 
                    interpolation=InterpolationMode.BILINEAR, 
                    fill=1 
                )
            ])

        super().__init__(device, G, D, augment_pipe=augment_pipe,
                         r1_gamma=config.r1_gamma,
                         style_mixing_prob=config.style_mixing_prob,
                         pl_weight=config.pl_weight,
                         pl_batch_shrink=config.pl_batch_shrink,
                         pl_decay=config.pl_decay,
                         pl_no_weight_grad=config.pl_no_weight_grad,
                         blur_init_sigma=config.blur_init_sigma,
                         blur_fade_kimg=config.blur_fade_kimg,
                         r1_gamma_init=config.r1_gamma_init,
                         r1_gamma_fade_kimg=config.r1_gamma_fade_kimg,
                         neural_rendering_resolution_initial=config.neural_rendering_resolution_initial,
                         neural_rendering_resolution_final=config.neural_rendering_resolution_final,
                         neural_rendering_resolution_fade_kimg=config.neural_rendering_resolution_fade_kimg,
                         gpc_reg_fade_kimg=config.gpc_reg_fade_kimg,
                         gpc_reg_prob=config.gpc_reg_prob,
                         dual_discrimination=config.dual_discrimination)

    def run_G(self, z, c, gen_mesh, swapping_prob, neural_rendering_resolution, update_emas=False, swapping_prob_mesh=None, shape_condition_mult=None, precomputed_img_flame: Optional[torch.Tensor] = None, **synthesis_kwargs):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        if c.shape[1] == 6:
            # c_gen_conditioning = c_gen_conditioning[:, :3] # pass just a headpose


            R_c_G = batch_rodrigues(c_gen_conditioning[:, :3]).reshape(c_gen_conditioning.shape[0], 9)
            # concat c[:, 3:]
            if self.G._config.use_concat:
                orth_c = 0 * c_gen_conditioning[:, 3:]
            else:
                orth_c = c_gen_conditioning[:, 3:]
            c_gen_conditioning = torch.cat([R_c_G, orth_c], dim=1)

        if swapping_prob_mesh is not None:
            gen_mesh_swapped = torch.roll(gen_mesh.clone(), 1, 0)
            gen_mesh_gen_conditioning = torch.where(torch.rand((gen_mesh.shape[0], 1), device=c.device) < swapping_prob, gen_mesh_swapped, gen_mesh)
        else:
            gen_mesh_gen_conditioning = gen_mesh

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas, c2=gen_mesh_gen_conditioning, flame_params=gen_mesh_gen_conditioning, shape_condition_mult=shape_condition_mult)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False, c2=gen_mesh)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, gen_mesh, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, **synthesis_kwargs)

        # If precomputed flame images are provided, override generator-produced flames
        if precomputed_img_flame is not None:
            gen_output['image_flame'] = precomputed_img_flame
        return gen_output, ws

    def run_D(self, img, c, mesh, blur_sigma=0, blur_sigma_raw=0, alpha_new_layers_disc: Optional[float] = None, update_emas=False,
              effective_res_disc: Optional[int] = None, other_img: Optional[Dict[str, torch.Tensor]] = None, shape_condition_mult=None):
        blur_size = np.floor(blur_sigma * 3)

        if self._config.mask_swap_prob > 0 and other_img is not None:
            idx_self_other = torch.rand((img['image'].shape[0], 1, 1), device=img['image'].device) > self._config.mask_swap_prob
            img['image'][:, 3] = torch.where(idx_self_other, img['image'][:, 3], other_img['image'][:, 3].detach())

        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                if self._config.blur_masks:
                    img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
                else:
                    img['image'] = torch.cat([upfirdn2d.filter2d(img['image'][:, :3], f / f.sum()), img['image'][:, 3:]], dim=1)

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                          torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear',
                                                                                          antialias=True)],
                                                         dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear',
                                                               antialias=True)
        
        if effective_res_disc is not None:
            image = img['image']
            original_size = image.shape[-1]
            image_low = filtered_resizing(image, size=effective_res_disc, f=self.resample_filter, filter_mode=self.filter_mode)
            image_high = filtered_resizing(image_low, size=original_size, f=self.resample_filter)
            img['image'] = image_high

        if self._config.use_shift_augmentation:
            # Check spatial dimensions match
            assert img['image'].shape[2:] == img['image_raw'].shape[2:], \
                f"Spatial dimensions of img['image'] ({img['image'].shape[2:]}) and " \
                f"img['image_raw'] ({img['image_raw'].shape[2:]}) must match for shared translation augmentation."

            # Store channel counts for splitting later
            c_img = img['image'].shape[1]
            c_raw = img['image_raw'].shape[1]

            # Concatenate along the channel dimension
            if self.dual_discrimination:
                combined_img = torch.cat([img['image'], img['image_raw'], img['image_flame']], dim=1)
            else:
                combined_img = torch.cat([img['image'], img['image_raw']], dim=1)

            # Apply augmentation once to the combined tensor
            augmented_combined = self.translation_augment(combined_img)

            # Split back into original tensors
            img['image'] = augmented_combined[:, :c_img]
            img['image_raw'] = augmented_combined[:, c_img:c_img + c_raw]
            if self.dual_discrimination:
                img['image_flame'] = augmented_combined[:, c_img + c_raw:]

        if c.shape[1] == 6:
            # take just a headpose to prevent (headpose, s, tx, ty) memorization
            c_D = c[:, :3]

            # add noise
            # magnitude = 0.025
            # c_noise = torch.randn_like(c_D) * magnitude
            # R_c_D = batch_rodrigues(c_D)
            # R_c_noise = batch_rodrigues(c_noise)
            # R_c_D_noisy = R_c_noise @ R_c_D
            # c_D_noisy = rot_mat_to_axisangle(R_c_D_noisy)
            # c_D = c_D_noisy

            R_c_D = batch_rodrigues(c_D)
            t_c_D = c[:, 3:]

            # # add noise to R and t
            # magnitude = 0.10
            # R_c_noise = batch_rodrigues(torch.randn_like(c_D) * magnitude)
            # R_c_D = R_c_noise @ R_c_D
            # t_c_noise = torch.randn_like(t_c_D) * magnitude
            # t_c_D = t_c_D + t_c_noise

            R_c_D = R_c_D.reshape(c_D.shape[0], 9)

            if self.G._config.use_concat:
                orth_c = 0 * t_c_D
            else:
                orth_c = t_c_D
            
            c_D = torch.cat([R_c_D, orth_c], dim=1)
        elif c.shape[1] == 25:
            # intrinsics are fixed anyway
            c_D = c
        else:
            raise Exception("Sanity check for D's c_dim!")

        if getattr(self.D, 'use_double_mapping', False):
            c2 = mesh[:, :self.D._config.c2_dim] * shape_condition_mult
        else:
            c2 = None

        if alpha_new_layers_disc is None:
            logits = self.D(img, c_D, update_emas=update_emas, c2=c2)
        else:
            logits = self.D(img, c_D, update_emas=update_emas, alpha_new_layers=alpha_new_layers_disc, c2=c2)
        return logits

    def run_D_features(self, img, c, mesh, alpha_new_layers_disc: Optional[float] = None, feature_resolutions: Optional[List[int]] = None, shape_condition_mult=None):
        # No blur, no augment, no mbstd; extract intermediate features only
        if c.shape[1] == 6:
            c_D = c[:, :3]
            R_c_D = batch_rodrigues(c_D)
            t_c_D = c[:, 3:]
            R_c_D = R_c_D.reshape(c_D.shape[0], 9)
            c_D = torch.cat([R_c_D, t_c_D], dim=1)
        elif c.shape[1] == 25:
            c_D = c
        else:
            raise Exception("Sanity check for D's c_dim!")

        if getattr(self.D, 'use_double_mapping', False):
            c2 = mesh[:, :self.D._config.c2_dim] * shape_condition_mult if shape_condition_mult is not None else mesh[:, :self.D._config.c2_dim]
        else:
            c2 = None

        if alpha_new_layers_disc is None:
            feats = self.D(img, c_D, update_emas=False, c2=c2, return_features=True, feature_resolutions=feature_resolutions)
        else:
            feats = self.D(img, c_D, update_emas=False, alpha_new_layers=alpha_new_layers_disc, c2=c2, return_features=True, feature_resolutions=feature_resolutions)
        return feats

    def loss_clamp_l2(self, source, target, mask=None, clamp=True):
        """
        Args:
            source: (bs, sh, h, w, c)
            target: float value
            mask: (bs, 1, h, w)
        Returns:
            float
        """
        if clamp:
            loss_map = torch.clamp((source - target), min=0) ** 2
        else:
            loss_map = (source - target) ** 2
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(
                source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[..., None]) / torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def loss_clamp_l1(self, source, target_value, mask=None, clamp=False):
        """
        Args:
            source: (bs, sh, h, w, c)
            target_value: float value
            mask: (1, h, w)
        Returns:
            loss: float
        """
        if clamp:
            loss_map = torch.abs(torch.clamp(source - target_value, min=0))
        else:
            loss_map = torch.abs(source - target_value)
        if mask is not None:
            mask = mask.to(source.device)
            texture_mask = filtered_resizing(mask.unsqueeze(0), size=source.shape[-2], f=self.resample_filter, filter_mode=self.filter_mode).repeat(
                source.shape[0], source.shape[1], 1, 1)
            return torch.sum(loss_map * texture_mask[..., None]) / torch.sum(texture_mask)
        else:
            return torch.mean(loss_map)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gen_mesh, gain, cur_nimg, real_img_flame, gen_weight=None, real_mesh=None, gen_img_flame: Optional[torch.Tensor] = None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'G_deform_linearity_reg', 'G_id_consistency']
        if not hasattr(self.G, 'rendering_kwargs') or not isinstance(self.G.rendering_kwargs, dict) or self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        alpha_new_layers_disc = min((cur_nimg - self._config.new_layers_disc_start_kimg * 1e3) / (self._config.new_layers_disc_blend_kimg * 1e3),
                                    1) if self._config.new_layers_disc_start_kimg is not None else None
        alpha_new_layers_gen = min((cur_nimg - self._config.new_layers_gen_start_kimg * 1e3) / (self._config.new_layers_gen_blend_kimg * 1e3),
                                    1) if self._config.new_layers_gen_start_kimg is not None else None
        alpha_plane_resolution = min((cur_nimg - self._config.plane_resolution_start_kimg * 1e3) / (self._config.plane_resolution_blend_kimg * 1e3),
                                   1) if self._config.plane_resolution_start_kimg is not None else None
        effective_res_disc = None
        if self._config.effective_res_disc_start_kimg is not None:
            alpha_effective_res_disc = min((cur_nimg - self._config.effective_res_disc_start_kimg * 1e3) / (self._config.effective_res_disc_blend_kimg * 1e3),
                                        1)
            pretrained_resolution = self._config.pretrained_resolution
            new_resolution = real_img.shape[2]
            effective_res_disc = int(alpha_effective_res_disc * new_resolution + (1 - alpha_effective_res_disc) * pretrained_resolution)

        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        alpha_mesh = min(cur_nimg / (self.gmc_reg_fade_kimg * 1e3), 1) if self.gmc_reg_fade_kimg > 0 else 1
        swapping_prob_mesh = (1 - alpha_mesh) * 1 + alpha_mesh * self.gmc_reg_prob if self.gmc_reg_prob is not None else None

        # schedule shape_condition_mult based on shape_condition_start_kimg, shape_condition_fade_kimg.
        shape_condition_start_kimg = 0
        shape_condition_fade_kimg = 0
        if cur_nimg < shape_condition_start_kimg * 1e3:
            shape_condition_mult = 0
        elif cur_nimg >= shape_condition_start_kimg * 1e3 + shape_condition_fade_kimg * 1e3:
            shape_condition_mult = 1
        else:
            shape_condition_mult = (cur_nimg - shape_condition_start_kimg * 1e3) / (shape_condition_fade_kimg * 1e3)

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        if self._config.progressive_scale_reg_kimg > 0:
            reg_weight_cur = self._config.reg_weight - min(cur_nimg / (self._config.progressive_scale_reg_kimg * 1e3), 1) * (
                    self._config.reg_weight - self._config.progressive_scale_reg_end)
        else:
            reg_weight_cur = self._config.reg_weight

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        gen_img = None

        if self.dual_discrimination and (self._config.start_dd_kimg is not None) and (cur_nimg < self._config.start_dd_kimg * 1000):
            zero_out_img_flame = True
        else:
            zero_out_img_flame = False

        # ID consistency phase: compare D features of paired images differing only in expression/jawpose/camera.
        if phase == 'G_id_consistency' and self._config.lambda_id_consistency > 0:
            bs = gen_z.shape[0]
            assert bs % 2 == 0, 'Batch size for G_id_consistency must be even.'

            with torch.autograd.profiler.record_function('G_id_consistency_forward'):
                # Same z and shapecode for both halves
                # z_first_half = gen_z[:bs//2]
                # z_pair = z_first_half

                # c_first_half = gen_c[:bs//2]
                # c_second_half = gen_c[bs//2:]
                mesh_first_half = gen_mesh[:bs//2].clone()
                mesh_second_half = gen_mesh[bs//2:].clone()
                mesh_second_half[:, :self.G.n_shape] = mesh_first_half[:, :self.G.n_shape].clone()

                # same z for both halves
                z_pair = torch.cat([gen_z[:bs//2], gen_z[:bs//2]], dim=0)
                # same camera for both halves. take same mapping cameras but different render camera
                c_pair = torch.cat([gen_c[:bs//2], gen_c[:bs//2]], dim=0)
                # same mesh for both halves, except expcode, jawcode, eylids
                mesh_pair = torch.cat([mesh_first_half, mesh_second_half], dim=0)

                gen_pair, _ = self.run_G(z_pair, c_pair, mesh_pair, swapping_prob=None, swapping_prob_mesh=None,
                                      neural_rendering_resolution=neural_rendering_resolution,
                                      return_raw_attributes=False, alpha_new_layers=alpha_new_layers_gen,
                                      alpha_plane_resolution=alpha_plane_resolution, shape_condition_mult=shape_condition_mult)

                # gen2, _ = self.run_G(z_pair, c_second_half, mesh_second_half, swapping_prob=None, swapping_prob_mesh=None,
                #                       neural_rendering_resolution=neural_rendering_resolution,
                #                       return_raw_attributes=False, alpha_new_layers=alpha_new_layers_gen,
                #                       alpha_plane_resolution=alpha_plane_resolution, shape_condition_mult=shape_condition_mult)

                feature_res = self._config.id_feature_resolutions
                feats_pair = self.run_D_features(gen_pair, c_pair, mesh_pair, alpha_new_layers_disc=alpha_new_layers_disc,
                                             feature_resolutions=feature_res, shape_condition_mult=shape_condition_mult)
                # feats2 = self.run_D_features(gen2, c_second_half, mesh_second_half, alpha_new_layers_disc=alpha_new_layers_disc,
                #                              feature_resolutions=feature_res, shape_condition_mult=shape_condition_mult)

                id_loss = 0
                for res in feats_pair.keys():
                    f1 = feats_pair[res][:bs//2]
                    f2 = feats_pair[res][bs//2:]
                    # Normalize spatial dims to same size if needed is not required since same D blocks resolutions.
                    id_loss = id_loss + (f1 - f2).abs().mean(dim=[1,2,3])

                self._logger_bundle.log_metrics({'Loss/G/id_consistency': id_loss}, step=cur_nimg)
                loss_Gmain = self._config.lambda_id_consistency * id_loss

            with torch.autograd.profiler.record_function('G_id_consistency_backward'):
                loss_Gmain.mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.G.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_G_id_consistency NAN GRADIENTS: {gradients_with_nan}")

            return

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Proper adversarial loss
                if isinstance(self.G, GGHeadModel):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_mesh, swapping_prob=swapping_prob, swapping_prob_mesh=swapping_prob_mesh, neural_rendering_resolution=neural_rendering_resolution,
                                                  return_raw_attributes=self._config.requires_raw_gaussian_attributes(),
                                                  alpha_new_layers=alpha_new_layers_gen,
                                                  alpha_plane_resolution=alpha_plane_resolution,
                                                  shape_condition_mult=shape_condition_mult,
                                                  precomputed_img_flame=gen_img_flame)
                else:
                    # gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_mesh, swapping_prob=swapping_prob, swapping_prob_mesh=swapping_prob_mesh, neural_rendering_resolution=neural_rendering_resolution)
                    raise NotImplementedError("No implemented for non-GGHeadModel")

                if zero_out_img_flame:
                    gen_img['image_flame'] = torch.zeros_like(gen_img['image_flame'])

                gen_logits = self.run_D(gen_img, gen_c, gen_mesh, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc, effective_res_disc=effective_res_disc,
                                        other_img=real_img, shape_condition_mult=shape_condition_mult)

                img_log_freq_kimg = 10
                if (cur_nimg // (img_log_freq_kimg * 1000)) > (self.prev_nimg // (img_log_freq_kimg * 1000)):
                    self.prev_nimg = cur_nimg
                    images_grid = torchvision.utils.make_grid(gen_img['image'].detach(), nrow=4)

                    if self.dual_discrimination:
                        flame_grid = torchvision.utils.make_grid(gen_img['image_flame'].detach(), nrow=4)
                        joint_grid = torch.hstack([images_grid, flame_grid])
                    else:
                        joint_grid = images_grid
                    self._logger_bundle.log_image('Images/D/fake_input/Gmain_Gboth', [joint_grid], step=cur_nimg)

                    if self.G._config.log_feature_maps:
                        img_features = torch.hstack(gen_img.images_features).permute(2, 0, 1)
                        self._logger_bundle.log_image('Images/feature_maps/main', [img_features], step=cur_nimg)

                        if self.G._config.use_mouth_branch or self.G._config.use_extended_uv_generation:
                            mouth_features = torch.hstack(gen_img.images_features_mouth).permute(2, 0, 1)
                            self._logger_bundle.log_image('Images/feature_maps/mouth', [mouth_features], step=cur_nimg)

                if self._config.use_weighted_loss:
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits) * gen_weight.unsqueeze(1)
                else:
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)

                if loss_Gmain.isnan().any():
                    print("loss_Gmain IS NAN!")

                self._logger_bundle.log_metrics({
                    'Loss/scores/fake': gen_logits,
                    'Loss/signs/fake': gen_logits.sign(),
                    'Loss/G/loss': loss_Gmain
                }, step=cur_nimg)

                if isinstance(self.G, GGHeadModel):
                    loss_Gmain = loss_Gmain.squeeze(1)  # [B, 1] -> [B]
                    raw_gaussian_attributes = gen_img.gaussian_attribute_output.raw_gaussian_attributes
                    gaussian_attributes = gen_img.gaussian_attribute_output.gaussian_attributes

                    # Raw Gaussian Attributes (Before activation functions and adding offsets)
                    if self._config.lambda_raw_gaussian_position > 0:
                        num_static_gaussians = self.G._uv_grid.squeeze().shape[0]
                        
                        reg_raw_gaussian_position = raw_gaussian_attributes[GaussianAttribute.POSITION][:, :num_static_gaussians].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_position': reg_raw_gaussian_position
                        }, step=cur_nimg)

                        if self._config.use_uv_reg_weights:
                            reg_raw_gaussian_position = raw_gaussian_attributes[GaussianAttribute.POSITION][:, :num_static_gaussians].norm(dim=-1) 

                            # get the mean of the gaussian position in the masked region
                            masked_region_raw_gaussian_position = (reg_raw_gaussian_position * self.uv_reg_mask).sum(dim=1) / self.uv_reg_mask.sum(dim=1)

                            # apply weighted mask to reg_raw_gaussian_position
                            reg_raw_gaussian_position = (self.per_gaussian_reg_weights * reg_raw_gaussian_position).mean(dim=1) 

                            self._logger_bundle.log_metrics({
                                'Loss/G/reg_raw_gaussian_position__masked_region': masked_region_raw_gaussian_position
                            }, step=cur_nimg)
                        
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_position * reg_raw_gaussian_position

                        if (self.G._config.use_mouth_branch or self.G._config.use_extended_uv_generation) and raw_gaussian_attributes[GaussianAttribute.POSITION].shape[1] > num_static_gaussians:
                            reg_raw_mouth_gs_position = raw_gaussian_attributes[GaussianAttribute.POSITION][:, num_static_gaussians:].norm(dim=-1).mean(dim=1)
                            self._logger_bundle.log_metrics({
                                'Loss/G/reg_raw_mouth_gaussian_position': reg_raw_mouth_gs_position
                            }, step=cur_nimg)

                            val = reg_raw_mouth_gs_position.sum()
                            if not torch.isfinite(val):
                                print('Non-finite mouth gaussian position! Value: ', val)

                            loss_Gmain = loss_Gmain + self._config.lambda_raw_mouth_gaussian_position * reg_raw_mouth_gs_position

                    if self._config.lambda_raw_deform_gaussian_position > 0:
                        # here is just facial region (because of uv_deform_mask) + no mouth
                        deform_uv_attributes = gen_img.gaussian_attribute_output.deform_uv_attributes
                        raw_deform_position = deform_uv_attributes[:, :, :3].norm(dim=-1)
                        raw_deform_position = raw_deform_position * self.uv_deform_mouth_weight_mask # reduce the weight of the mouth region
                        raw_deform_position = raw_deform_position.mean(dim=1)

                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_DEFORM_gaussian_position': raw_deform_position
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_deform_gaussian_position * raw_deform_position
                    
                    if self._config.lambda_raw_deform_gaussian_scale > 0:
                        # here is just facial region (because of uv_deform_mask) + no mouth
                        deform_uv_attributes = gen_img.gaussian_attribute_output.deform_uv_attributes
                        raw_deform_scale = deform_uv_attributes[:, :, 3:6].norm(dim=-1)
                        raw_deform_scale = raw_deform_scale * self.uv_deform_mouth_weight_mask # reduce the weight of the mouth region
                        raw_deform_scale = raw_deform_scale.mean(dim=1)
                        
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_DEFORM_gaussian_scale': raw_deform_scale
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_deform_gaussian_scale * raw_deform_scale
                        
                    if self._config.lambda_raw_deform_gaussian_opacity > 0:
                        # here is just facial region (because of uv_deform_mask) + no mouth
                        deform_uv_attributes = gen_img.gaussian_attribute_output.deform_uv_attributes
                        raw_deform_opacity = deform_uv_attributes[:, :, -1].abs().mean(dim=1)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_deform_gaussian_opacity * raw_deform_opacity
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_DEFORM_gaussian_opacity': raw_deform_opacity
                        }, step=cur_nimg)
                        
                    if self._config.lambda_raw_gaussian_scale > 0:
                        if self._config.reg_raw_gaussian_scale_above != 0 or self._config.reg_raw_gaussian_scale_below != 0:
                            raw_gaussian_scales = raw_gaussian_attributes[GaussianAttribute.SCALE]  # [B, G]
                            raw_gaussian_scales_to_regularize = torch.cat(
                                [raw_gaussian_scales[raw_gaussian_scales > self._config.reg_raw_gaussian_scale_above] - self._config.reg_raw_gaussian_scale_above,
                                 self._config.reg_raw_gaussian_scale_below - raw_gaussian_scales[raw_gaussian_scales < self._config.reg_raw_gaussian_scale_below]])
                            if len(raw_gaussian_scales_to_regularize) > 0:
                                if self._config.use_l1_scale_reg:
                                    reg_raw_gaussian_scale = raw_gaussian_scales_to_regularize.abs().mean()
                                else:
                                    reg_raw_gaussian_scale = raw_gaussian_scales_to_regularize.square().mean()

                                self._logger_bundle.log_metrics({
                                    'Loss/G/reg_raw_gaussian_scale': reg_raw_gaussian_scale
                                }, step=cur_nimg)
                                loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_scale * reg_raw_gaussian_scale
                        else:
                            num_static_gaussians = self.G._uv_grid.squeeze().shape[0]

                            if self._config.use_l1_scale_reg:
                                reg_raw_gaussian_scale = raw_gaussian_attributes[GaussianAttribute.SCALE][:, :num_static_gaussians].norm(dim=-1, p=1).mean(dim=1)  # [B]
                            else:
                                reg_raw_gaussian_scale = raw_gaussian_attributes[GaussianAttribute.SCALE][:, :num_static_gaussians].norm(dim=-1).mean(dim=1)  # [B]

                            self._logger_bundle.log_metrics({
                                'Loss/G/reg_raw_gaussian_scale': reg_raw_gaussian_scale
                            }, step=cur_nimg)

                            if self._config.use_uv_reg_weights:
                                assert not self._config.use_l1_scale_reg, "L1 scale reg is not supported with uv reg weights"

                                reg_raw_gaussian_scale = raw_gaussian_attributes[GaussianAttribute.SCALE][:, :num_static_gaussians].norm(dim=-1) # [B, G]

                                # get the mean of the gaussian scale in the masked region
                                masked_region_raw_gaussian_scale = (reg_raw_gaussian_scale * self.uv_reg_mask).sum(dim=1) / self.uv_reg_mask.sum(dim=1)

                                # apply weighted mask to reg_raw_gaussian_scale
                                reg_raw_gaussian_scale = (self.per_gaussian_reg_weights * reg_raw_gaussian_scale).mean(dim=1)

                                self._logger_bundle.log_metrics({
                                    'Loss/G/reg_raw_gaussian_scale__masked_region': masked_region_raw_gaussian_scale
                                }, step=cur_nimg)
                            
                            loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_scale * reg_raw_gaussian_scale

                            if self.G._config.use_mouth_branch or self.G._config.use_extended_uv_generation:
                                reg_raw_mouth_gs_scale = raw_gaussian_attributes[GaussianAttribute.SCALE][:, num_static_gaussians:].norm(dim=-1).mean(dim=1)
                                self._logger_bundle.log_metrics({
                                    'Loss/G/reg_raw_mouth_gaussian_scale': reg_raw_mouth_gs_scale
                                }, step=cur_nimg)

                                val = reg_raw_mouth_gs_scale.sum()
                                if not torch.isfinite(val):
                                    print('Non-finite mouth gaussian scale! Value: ', val)

                                loss_Gmain = loss_Gmain + self._config.lambda_raw_mouth_gaussian_scale * reg_raw_mouth_gs_scale

                    if self._config.lambda_raw_scale_std > 0:
                        reg_raw_scale_std = raw_gaussian_attributes[GaussianAttribute.SCALE].std(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_scale_std': reg_raw_scale_std
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain - self._config.lambda_raw_scale_std * reg_raw_scale_std

                    if self._config.lambda_raw_gaussian_rotation > 0:
                        reg_raw_gaussian_rotation = raw_gaussian_attributes[GaussianAttribute.ROTATION].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_rotation': reg_raw_gaussian_rotation
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_rotation * reg_raw_gaussian_rotation
                    if self._config.lambda_raw_gaussian_color > 0:
                        reg_raw_gaussian_color = raw_gaussian_attributes[GaussianAttribute.COLOR].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_color': reg_raw_gaussian_color
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_color * reg_raw_gaussian_color
                    if self._config.lambda_raw_gaussian_opacity > 0:
                        reg_raw_gaussian_opacity = raw_gaussian_attributes[GaussianAttribute.OPACITY].norm(dim=-1).mean(dim=1)  # [B]
                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_raw_gaussian_opacity': reg_raw_gaussian_opacity
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_gaussian_opacity * reg_raw_gaussian_opacity

                    if self._config.lambda_beta_loss > 0:
                        opacities = self.G._apply_opacity_activation(gen_img.gaussian_attribute_output.gaussian_attributes[GaussianAttribute.OPACITY])
                        beta_loss = ((0.1 + opacities).log() + (1.1 - opacities).log() + 2.20727).mean()
                        self._logger_bundle.log_metrics({
                            'Loss/G/beta_loss': beta_loss
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_beta_loss * beta_loss

                    if self._config.lambda_tv_uv_rendering > 0:
                        uv_renderings = self.G.get_uv_rendering(gen_c, gen_img, include_transparent_gaussians=self._config.tv_uv_include_transparent_gaussians, J_transformed=gen_img['J_transformed'])
                        mask = 1 - (uv_renderings[:, [2]] + 1)/2

                        uv_renderings_no_blend = ((uv_renderings - (1 - mask)) / mask)  # Undo effect of alpha blending. Background pixels will be inf
                        background_mask = mask == 0

                        mask_y = (background_mask[:, :, 1:] | background_mask[:, :, :-1]).repeat(1, 3, 1, 1)
                        mask_x = (background_mask[:, :, :, 1:] | background_mask[:, :, :, :-1]).repeat(1, 3, 1, 1)

                        uv_difference_y = uv_renderings_no_blend[:, :, 1:] - uv_renderings_no_blend[:, :, :-1]
                        uv_difference_x = uv_renderings_no_blend[:, :, :, 1:] - uv_renderings_no_blend[:, :, :, :-1]

                        reg_tv_uv_rendering_y = uv_difference_y[~mask_y].abs().mean()
                        reg_tv_uv_rendering_x = uv_difference_x[~mask_x].abs().mean()
                        reg_tv_uv_rendering = reg_tv_uv_rendering_x + reg_tv_uv_rendering_y

                        self._logger_bundle.log_metrics({
                            'Loss/G/reg_tv_uv_rendering': reg_tv_uv_rendering
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_tv_uv_rendering * reg_tv_uv_rendering

                    if self._config.lambda_raw_xyz_entropy_masked > 0:
                        num_static_gaussians = self.G._uv_grid.squeeze().shape[0]
                        norms_masked = self.uv_reg_mask * (raw_gaussian_attributes[GaussianAttribute.POSITION][:, :num_static_gaussians].norm(dim=-1) + 1e-6) # [B, G]
                        probs_masked = norms_masked / norms_masked.sum(dim=-1, keepdim=True)

                        masked_position_entropy = -1 * probs_masked * probs_masked.clamp(min=1e-6).log()
                        masked_position_entropy = masked_position_entropy.sum(dim=-1) # [B]

                        # use as KL to uniform distribution to not disturb Generator's loss scale
                        n_masked_gaussians = int(self.uv_reg_mask[0].sum().item())
                        log_n = torch.log(torch.tensor(float(n_masked_gaussians), device=masked_position_entropy.device))
                        entropy_loss = log_n - masked_position_entropy

                        self._logger_bundle.log_metrics({
                            'Loss/G/raw_xyz_entropy_masked': entropy_loss
                        }, step=cur_nimg)
                        loss_Gmain = loss_Gmain + self._config.lambda_raw_xyz_entropy_masked * entropy_loss
                else:
                    print('You don\'t use GGHEAD regularizations. Are you sure?')


            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.G.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Gmain NAN GRADIENTS: {gradients_with_nan}")

        if phase == 'G_deform_linearity_reg':
            bs = gen_mesh.shape[0]

            gen_z = gen_z[0, :].repeat(bs, 1)
            gen_c = gen_c[0, :].repeat(bs, 1)
            (_, deformation_planes), _ = self.run_G(gen_z, gen_c, gen_mesh, swapping_prob=swapping_prob, swapping_prob_mesh=swapping_prob_mesh, neural_rendering_resolution=neural_rendering_resolution,
                                                  return_raw_attributes=self._config.requires_raw_gaussian_attributes(),
                                                  alpha_new_layers=alpha_new_layers_gen,
                                                  alpha_plane_resolution=alpha_plane_resolution,
                                                  return_deformation_planes=True,
                                                  shape_condition_mult=shape_condition_mult
                                                  )


            gm_1 = gen_mesh[:bs//2]
            gm_2 = gen_mesh[bs//2:]
            alpha = torch.rand(bs//2, device=gen_mesh.device)[:, None]
            gen_mesh_interpolated = gm_1 * (1 - alpha) + gm_2 * alpha

            (_, deformation_planes_interpolated), _ = self.run_G(gen_z[:bs//2], gen_c[:bs//2], gen_mesh_interpolated, swapping_prob=swapping_prob, swapping_prob_mesh=swapping_prob_mesh, neural_rendering_resolution=neural_rendering_resolution,
                                                return_raw_attributes=self._config.requires_raw_gaussian_attributes(),
                                                alpha_new_layers=alpha_new_layers_gen,
                                                alpha_plane_resolution=alpha_plane_resolution,
                                                return_deformation_planes=True,
                                                shape_condition_mult=shape_condition_mult
                                                )

            dp_1 = deformation_planes[:bs//2]
            dp_2 = deformation_planes[bs//2:]
            dp_alpha = alpha[:, :, None, None]
            linearity_rhs = dp_1 * (1 - dp_alpha) + dp_2 * dp_alpha

            # TODO: exclude scale, rotation, color. just keep colors to be linear.

            num_deform_attributes = deformation_planes.shape[1] 
            if self.G._config.zero_out_color_residuals:
                num_deform_attributes -= self.G._n_color_channels

            if self.G._config.use_deform_mask:
                num_masked_pixels = (self.G._deform_mask > 0).sum()
                valid_pixels = num_deform_attributes * num_masked_pixels
            else:
                valid_pixels = num_deform_attributes * deformation_planes.shape[2] * deformation_planes.shape[3]

            linearity_term = ((deformation_planes_interpolated - linearity_rhs)**2).sum(dim=(-1, -2, -3)) / valid_pixels

            self._logger_bundle.log_metrics({
                'Loss/G/deform_linearity_reg': linearity_term
            }, step=cur_nimg)

            loss_Gmain = self._config.lambda_deform_linearity_reg * linearity_term

            with torch.autograd.profiler.record_function('G_deform_linearity_reg_backward'):
                loss_Gmain.mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.G.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Gmain NAN GRADIENTS in G_deform_linearity_reg: {gradients_with_nan}")

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                if isinstance(self.G, GGHeadModel):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_mesh, swapping_prob=swapping_prob, swapping_prob_mesh=swapping_prob_mesh, neural_rendering_resolution=neural_rendering_resolution,
                                                  update_emas=True, alpha_new_layers=alpha_new_layers_gen, alpha_plane_resolution=alpha_plane_resolution, shape_condition_mult=shape_condition_mult,
                                                  precomputed_img_flame=gen_img_flame)
                else:
                    # gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_mesh, swapping_prob=swapping_prob, swapping_prob_mesh=swapping_prob_mesh, neural_rendering_resolution=neural_rendering_resolution,
                                                #   update_emas=True)
                    raise NotImplementedError("No implemented for non-GGHeadModel")
                
                if zero_out_img_flame:
                    gen_img['image_flame'] = torch.zeros_like(gen_img['image_flame'])
                
                gen_logits = self.run_D(gen_img, gen_c, gen_mesh, blur_sigma=blur_sigma, update_emas=True, alpha_new_layers_disc=alpha_new_layers_disc,
                                        effective_res_disc=effective_res_disc, other_img=real_img, shape_condition_mult=shape_condition_mult)

                if cur_nimg % 102400 == 0 or (cur_nimg < 1_000 and cur_nimg % 16 == 0):
                    images_grid = torchvision.utils.make_grid(gen_img['image'].detach(), nrow=4)
                    if self.dual_discrimination:
                        flame_grid = torchvision.utils.make_grid(gen_img['image_flame'].detach(), nrow=4)
                        joint_grid = torch.hstack([images_grid, flame_grid])
                    else:
                        joint_grid = images_grid
                    self._logger_bundle.log_image('Images/D/fake_input/Dmain_Dboth', [joint_grid], step=cur_nimg)
                                        
                self._logger_bundle.log_metrics({
                    'Loss/scores/fake': gen_logits,
                    'Loss/signs/fake': gen_logits.sign()
                }, step=cur_nimg)
                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                if loss_Dgen.isnan().any():
                    print("loss_Dgen IS NAN !")
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.D.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Dgen NAN GRADIENTS: {gradients_with_nan}")

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                if self.dual_discrimination:
                    real_img_tmp_image_flame = real_img_flame.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                else:
                    real_img_tmp_image_flame = None
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 'image_flame' : real_img_tmp_image_flame}

                if zero_out_img_flame:
                    real_img_tmp['image_flame'] = torch.zeros_like(real_img_tmp['image_flame'])

                real_logits = self.run_D(real_img_tmp, real_c, real_mesh, blur_sigma=blur_sigma, alpha_new_layers_disc=alpha_new_layers_disc, effective_res_disc=effective_res_disc,
                                         other_img=gen_img, shape_condition_mult=shape_condition_mult)

                if cur_nimg % 102400 == 0 or (cur_nimg < 1_000 and cur_nimg % 16 == 0):
                    images_grid = torchvision.utils.make_grid(real_img_tmp['image'].detach(), nrow=4)
                    if self.dual_discrimination:
                        flame_grid = torchvision.utils.make_grid(real_img_tmp['image_flame'].detach(), nrow=4)
                        joint_grid = torch.hstack([images_grid, flame_grid])
                    else:
                        joint_grid = images_grid
                    self._logger_bundle.log_image('Images/D/real_input', [joint_grid], step=cur_nimg)

                self._logger_bundle.log_metrics({
                    'Loss/scores/real': real_logits,
                    'Loss/signs/real': real_logits.sign()
                }, step=cur_nimg)
                # training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    if loss_Dreal.isnan().any():
                        print("loss_Dreal IS NAN !")
                    self._logger_bundle.log_metrics({
                        'Loss/D/loss': loss_Dgen + loss_Dreal,
                    }, step=cur_nimg)
                    # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            # TODO: Is this used for Gaussian Discriminator? maybe dual_discrimination should be set to False
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_flame']],
                                                           create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
                    else:  # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        if self._config.r1_gamma_mask is not None:
                            r1_grads_image[:, 3] *= np.sqrt(self._config.r1_gamma_mask)

                        r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    self._logger_bundle.log_metrics({
                        'Loss/r1_penalty': r1_penalty,
                        'Loss/D/reg': loss_Dr1
                    }, step=cur_nimg)
                    # training_stats.report('Loss/r1_penalty', r1_penalty)
                    # training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
                gradients_with_nan = [n for n, p in self.D.named_parameters() if p.grad is not None and p.grad.isnan().any()]
                if len(gradients_with_nan) > 0:
                    print(f"loss_Dreal NAN GRADIENTS: {gradients_with_nan}")

# ----------------------------------------------------------------------------
