import os
from copy import copy
from dataclasses import dataclass, field, asdict
from typing import Literal, List, Union, Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from dreifus.camera import PoseType, CameraCoordinateConvention
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector.vector_torch import to_homogeneous
from eg3d.datamanager.nersemble import decode_camera_params

from elias.config import Config, implicit

from torch import nn
from torch.nn import init
from torch.nn.functional import grid_sample

from gaussian_splatting.arguments import PipelineParams2
from gaussian_splatting.gaussian_renderer import render_distwar as render
from gaussian_splatting.scene import GaussianModel
from gaussian_splatting.scene.cameras import pose_to_rendercam
from gaussian_splatting.utils.sh_utils import C0, eval_sh
from gsplat import rasterization

from src.gghead.constants import DEFAULT_INTRINSICS
from src.gghead.config.gaussian_attribute import GaussianAttribute, GaussianAttributeConfig
from src.gghead.env import REPO_ROOT_DIR, GGHEAD_DEPENDENCIES_PATH
from src.gghead.models.stylegan2 import GGHGenerator as GGHStyleGAN2Backbone, GGHSynthesisNetwork, GGHSynthesisBlock, MeshGGHGenerator, MouthGGHGenerator
from src.gghead.util.activation import mip_tanh, mip_sigmoid, mip_tanh2
from src.gghead.util.logging import LoggerBundle
from src.gghead.util.mesh import gaussians_to_mesh
from src.gghead.util.rotation import axis_angle_to_quaternion, quat_mult
from src.gghead.util.uv import gen_tritex
from src.gghead.util.flame_rasterizer import get_gradient_flame_texture, FLAME_rasterizer, parse_flame_deca_cameras
from src.gghead.util.mouth_interior_torch import generate_mouth_interior_uv
from src.gghead.models.stylegan2 import GGHGenerator_FLAME

import sys
if GGHEAD_DEPENDENCIES_PATH not in sys.path:
    sys.path.insert(0, GGHEAD_DEPENDENCIES_PATH)
from threedim_utils.flame2020_lbs.FLAME import FLAME
from threedim_utils.flame2020_lbs import DEFAULT_CONFIG
from threedim_utils.flame import query_flame_by_mask, load_flame_masks

from smirk.src.FLAME.FLAME_adapted import FLAME as FLAME_smirk

from eg3d.torch_utils.misc import params_and_buffers

from argparse import Namespace

@dataclass
class MappingNetworkConfig(Config):
    num_layers: int = 8  # Number of mapping layers.
    embed_features: Optional[int] = None  # Label embedding dimensionality, None = same as w_dim.
    layer_features: Optional[int] = None  # Number of intermediate features in the mapping layers, None = same as w_dim.
    activation: Literal['lrelu', 'linear', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.
    lr_multiplier: float = 0.01  # Learning rate multiplier for the mapping layers.
    # w_avg_beta: float = 0.998  # Decay for tracking the moving average of W during training, None = do not track.


@dataclass
class SynthesisNetworkConfig(Config):
    channel_base: int = 32768  # Overall multiplier for the number of channels.
    channel_max: int = 512  # Maximum number of channels in any layer.
    num_fp16_res: int = 4  # Use FP16 for the N highest resolutions.

    # Block Config
    architecture: Literal['orig', 'skip', 'resnet'] = 'skip'  # Architecture: 'orig', 'skip', 'resnet'.
    resample_filter: List[int] = field(
        default_factory=lambda: [1, 3, 3, 1])  # Low-pass filter to apply when resampling activations.
    conv_clamp: Optional[int] = 256  # Clamp the output of convolution layers to +-X, None = disable clamping.
    fp16_channels_last: bool = False  # Use channels-last memory format with FP16?
    fused_modconv_default: Union[
        bool, str] = True  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.

    # Layer config
    kernel_size: int = 3  # Convolution kernel size.
    use_noise: bool = True  # Enable noise input?
    activation: Literal['lrelu', 'linear', 'relu'] = 'lrelu'  # Activation function: 'relu', 'lrelu', etc.

    def get_block_kwargs(self) -> dict:
        block_kwargs = {k: v for k, v in asdict(self).items() if
                        k not in ['channel_base', 'channel_max', 'num_fp16_res']}
        return block_kwargs

@dataclass
class GsplatRasterSettings(Config):
    absgrad: bool = False
    sparse_grad: bool = False
    rasterize_mode: str = 'antialiased' # or classic
    camera_model: str = 'pinhole'
    packed: bool = False

@dataclass
class RenderingConfig(Config):
    c_gen_conditioning_zero: bool = True  # if true, fill generator pose conditioning label with dummy zero vector
    c_scale: float = 1  # Scale factor for generator pose conditioning
    box_warp: float = 1  # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
    raster_backend: str = '3dgs'
    gsplat_raster_settings: GsplatRasterSettings = GsplatRasterSettings()


@dataclass
class SuperResolutionConfig(Config):
    use_superresolution: bool = False
    superresolution_version: int = 1
    n_channels: int = 3
    n_downsampling_layers: int = 1
    use_skip: bool = True
    cbase: int = 32768  # Capacity multiplier
    cmax: int = 512  # Max. feature maps
    fused_modconv_default: str = 'inference_only'
    sr_num_fp16_res: int = 4  # Number of fp16 layers in superresolution
    sr_antialias: bool = True
    noise_mode: Literal['random', 'none'] = 'none'


@dataclass
class GGHeadConfig(Config):
    z_dim: int = 512
    w_dim: int = 512
    c_dim: int = implicit(default=25)
    # img_resolution: int
    mapping_network_config: MappingNetworkConfig = MappingNetworkConfig()
    synthesis_network_config: SynthesisNetworkConfig = SynthesisNetworkConfig()
    rendering_config: RenderingConfig = RenderingConfig()
    super_resolution_config: SuperResolutionConfig = SuperResolutionConfig()

    uv_attributes: List[GaussianAttribute] = field(default_factory=lambda: [
        GaussianAttribute.POSITION])  # Which attributes should be predicted in UV space
    n_triplane_channels: int = 16  # number of channels for each TriPlane
    disable_position_offsets: bool = False  # If set, no position offsets will be predicted and Gaussians will always be fixed to template vertices
    use_align_corners: bool = False  # For grid_sample()
    interpolation_mode: str = 'bilinear'

    # FLAME template
    n_flame_subdivisions: int = 0  # How often the FLAME template mesh should be subdivided (increases number of predicted Gaussians)
    use_uniform_flame_vertices: bool = False  # If true, will not use predefined FLAME vertices, but instead uniformly distribute points on mesh surface using UV
    n_uniform_flame_vertices: int = 64  # How many points (squared) should be sampled in FLAME's UV space. Final number of Gaussians will be slightly smaller due to holes in UV map
    n_shells: int = 1
    shell_distance: float = 0.05
    use_learnable_template_offsets: bool = False  # If true, position of flame vertices can be adapted during training
    use_learnable_template_offset_plane: bool = False
    learnable_template_offset_plane_size: int = 64
    use_gsm_flame_template: bool = False  # Use template with back removed and more efficient UV layout
    use_flame_template_v2: bool = False
    use_sphere_template: bool = False
    use_plane_template: bool = False
    use_flame2020_template: bool = False
    use_flame_template_with_mouth: bool = False

    # DGGHEAD settings
    use_flame_rasterization: bool = False
    dd_shape_n_components: int = 0
    flame_rasterization_light_type: str = 'ambient'
    gen_flame_conditioning: bool = False
    gen_cond_zero_shape: bool = False
    use_flame_specific_double_mapping: bool = False
    gfc_double_modulation: bool = False
    gfc_additive_condition: bool = False

    c2_dim: int = 406 # used only in case of MAIN generator flame conditioning 
    log_feature_maps: bool = True

    # Mouth Branch settings
    use_mouth_branch: bool = False
    mouth_start_resolution: int = 16  # Starting resolution for mouth branch
    mouth_plane_resolution: int = 64  # output resolution for mouth branch
    mouth_n_vertices: int = 64
    mouth_center: tuple = (106/256, 128/256)
    mouth_radius: float = 16/256 #16/256 – just mouth region 
    mouth_gs_clipping: bool = False
    cut_mouth_gs_from_main_planes: bool = False

    # Extended UV generation (for mouth interior)
    use_extended_uv_generation: bool = False
    # Condition maps
    use_spade: bool = False
    use_concat: bool = False
    condition_nc: int = 3

    # Deformation branch
    use_deformation_branch: bool = False
    deformation_start_resolution: int = 80
    deformation_plane_resolution: int = 320
    zero_out_color_residuals: bool = False
    zero_out_opacity_residuals: bool = False
    use_deform_mask: bool = False
    mask_type: str = 'dejavu'
    double_condition: bool = False
    disable_deformation_branch: bool = False

    use_gaussian_blendshape: bool = False
    k_blendshapes: Optional[int] = None

    use_auxiliary_sphere: bool = False  # Predict additional set of Gaussians in front of face to models microphones, hands, other stuff that occludes the face
    auxiliary_sphere_radius: float = 0.1
    auxiliary_sphere_position: Tuple[float, float, float] = (0, -0.1, 0.4)
    uv_grid_threshold: Optional[
        float] = None  # If set, template positions with uv coordinates closer to the boundary than threshold will be dropped

    plane_resolution: int = 256
    effective_plane_resolution: Optional[int] = None
    pretrained_plane_resolution: Optional[int] = implicit()
    pretrained_resolution: Optional[int] = implicit()
    # Gaussian Attribute decoding
    use_position_activation: bool = True
    use_color_activation: bool = True
    use_scale_activation: bool = False
    center_scale_activation: bool = False  # If true, the max_scale option will be properly applied inside the softplus
    use_initial_scales: bool = False
    use_rotation_activation: bool = False
    use_periodic_rotation_activation: bool = False  # If true, will use sine() activation instead of tanh()
    normalize_quaternions: bool = True
    position_attenuation: float = 1
    position_range: float = 1  # Maximum range that predicted positions can have. 1 means [-1, 1]
    color_attenuation: float = 1
    scale_attenuation: float = 1
    rotation_attenuation: float = 1
    scale_offset: float = -5
    additional_scale_offset: float = 0
    max_scale: float = 1
    use_softplus_scale_activation: bool = False
    no_exp_scale_activation: bool = False  # Disable 3DGS default exp() scale activation
    scale_overshoot: float = 0.001
    color_overshoot: float = 0  # Allows prediction of colors slightly outside of the range to prevent tanh saturation. EG3D uses 0.001
    opacity_overshoot: float = 0  # Avoid having to predict ridiculously large opacities to saturate sigmoid
    clamp_opacity: bool = False
    use_optimizable_gaussian_attributes: bool = False  # For debugging: Gaussians are directly learnable instead of building them from predicted UV / TriPlanes
    gaussian_attribute_config: GaussianAttributeConfig = GaussianAttributeConfig()
    use_zero_conv_position: bool = False
    use_zero_conv_scale: bool = False
    use_density_map: bool = False

    # Gaussian Attribute MLP
    mlp_layers: int = 1
    mlp_hidden_dim: int = 256

    # Gaussian Hierarchy MLP
    use_gaussian_hierarchy: bool = False
    exclude_position_from_hierarchy: bool = False  # If true, positions will be directly sampled in uv map while all other attributes will be decoded with MLP
    use_uv_position_and_hierarchy: bool = False  # If true, positions will be directly sampled in uv map in addition to decoded offset
    n_gaussians_per_texel: int = 1
    gaussian_hierarchy_feature_dim: int = 16  # number of features in uv map that will be decoded into actual Gaussian UV attributes
    use_separate_hierarchy_mlps: bool = False  # If true, use one MLP per attribute

    # Gradient Multipliers
    grad_multiplier_position: Optional[float] = None
    grad_multiplier_scale: Optional[float] = None
    grad_multiplier_rotation: Optional[float] = None
    grad_multiplier_color: Optional[float] = None
    grad_multiplier_opacity: Optional[float] = None

    # Background modeling
    use_background_plane: bool = False  # If True, will additionally generate Gaussians behind the FLAME template
    curve_background_plane: bool = False
    background_cylinder_angle: float = torch.pi / 2  # Angle of the cylinder patch if curve_background_plane=True. Larger angle = larger background plane
    background_plane_distance: float = 1  # Distance of background plane to FLAME template
    background_plane_width: float = 1
    background_plane_height: float = 1
    n_background_gaussians: int = 64  # Number of background gaussians PER DIMENSION that will be distributed on background plane. E.g., 128 -> 128x128
    use_background_cnn: bool = False  # If True, will use 3 additional RGB channels from StyleGAN2 to models background
    use_background_upsampler: bool = False  # If use_background_cnn=True and the rendering resolution is larger than the backbone synthesis resolution
    use_separate_background_cnn: bool = False  # If True, will use an additional StyleGAN network to models background
    n_background_channels: int = 3  # Relevant if bg upsampler is used. Will be number of channels for intermediate upsampling layers
    use_masks: bool = False
    fix_alpha_blending: bool = False
    use_cnn_adaptor: bool = False

    # Maintenance
    maintenance_interval: Optional[int] = None  # How often Gaussians should be densified / pruned
    maintenance_grad_threshold: float = 0.01
    use_pruning: bool = False
    use_densification: bool = True
    use_template_update: bool = False
    template_update_attributes: List[GaussianAttribute] = field(default_factory=list)
    position_map_update_factor: float = 1  # How much of the average position map should be baked into the template at each maintenance step
    prune_opacity_threshold: float = 0.005

    use_autodecoder: bool = False  # Whether to assign one learnable latent code to each person
    use_flame_to_bfm_registration: bool = False
    load_average_offset_map: bool = False
    img_resolution: int = 512
    neural_rendering_resolution: int = 512

    n_persons: Optional[int] = implicit()
    random_background: Optional[bool] = implicit(default=False)
    return_background: Optional[bool] = implicit(default=False)
    background_color: Tuple[int, int, int] = implicit(
        default=(255, 255,
                 255))  # Background color to use during training. Should match the background color used in the dataset

    @staticmethod
    def from_eg3d_config(z_dim,  # Input latent (Z) dimensionality.
                         c_dim,  # Conditioning label (C) dimensionality.
                         w_dim,  # Intermediate latent (W) dimensionality.
                         img_resolution,  # Output resolution.
                         img_channels,  # Number of output color channels.
                         sr_num_fp16_res=0,
                         mapping_kwargs={},  # Arguments for MappingNetwork.
                         rendering_kwargs={},
                         sr_kwargs={},
                         **synthesis_kwargs,  # Arguments for SynthesisNetwork
                         ) -> 'GGHeadConfig':
        config = GGHeadConfig(z_dim, w_dim,
                                                 mapping_network_config=MappingNetworkConfig(**mapping_kwargs),
                                                 synthesis_network_config=SynthesisNetworkConfig(**synthesis_kwargs),
                                                 rendering_config=RenderingConfig(**rendering_kwargs),
                                                 use_flame_to_bfm_registration=True,
                                                 img_resolution=img_resolution)
        config.c_dim = c_dim
        return config

    @staticmethod
    def default() -> 'GGHeadConfig':
        config = GGHeadConfig(512, 512,
                                                 mapping_network_config=MappingNetworkConfig(),
                                                 synthesis_network_config=SynthesisNetworkConfig(),
                                                 rendering_config=RenderingConfig(),
                                                 use_flame_to_bfm_registration=True)
        config.c_dim = 25
        return config


@dataclass
class GaussianAttributeOutput:
    gaussian_attributes: Dict[GaussianAttribute, torch.Tensor]

    # Needed for regularization
    raw_gaussian_attributes: Optional[Dict[GaussianAttribute, torch.Tensor]] = None

    # Diagnostics
    uv_map: Optional[torch.Tensor] = None  # [B, S, UV, H_f, W_f]
    background_uv_map: Optional[torch.Tensor] = None
    auxiliary_gaussian_attributes: Optional[Dict[GaussianAttribute, torch.Tensor]] = None
    raw_auxiliary_gaussian_attributes: Optional[Dict[GaussianAttribute, torch.Tensor]] = None
    deform_uv_attributes: Optional[torch.Tensor] = None


@dataclass
class GGHeadOutput(dict):
    images: torch.Tensor  # [B, 3, H, W] in [-1, 1]
    images_raw: torch.Tensor  # [B, 3, H_raw, W_raw] in [-1, 1]
    images_depth: torch.Tensor  # [B, H, W]
    gaussian_attribute_output: GaussianAttributeOutput
    masks: Optional[torch.Tensor] = None  # [B, 1, H, W] in [-1, 1]
    backgrounds: Optional[torch.Tensor] = None  # [B, 3, H, W] in [-1, 1]
    images_features: Optional[torch.Tensor] = None

    # DGGHEAD
    images_flame: Optional[torch.tensor] = None # [B, 3, H, W]
    images_mouth: Optional[torch.tensor] = None
    images_wo_mouth: Optional[torch.tensor] = None
    images_features_mouth: Optional[torch.tensor] = None
    J_transformed: Optional[torch.tensor] = None

    # # Used for Gaussian maintenance
    # viewspace_points: Optional[List[torch.Tensor]] = None
    # visibility_filters: Optional[torch.Tensor] = None
    # radii: Optional[torch.Tensor] = None

    def __getitem__(self, key) -> torch.Tensor:
        # Legacy support for EG3D code. Behave like a dictionary
        if key == 'image':
            images = self.images
            if self.masks is not None:
                images = torch.cat([images, self.masks], dim=1)
            if self.backgrounds is not None:
                images = torch.cat([images, self.backgrounds], dim=1)
            return images
        elif key == 'image_raw':
            return self.images_raw
        elif key == 'image_depth':
            return self.images_depth
        elif key == 'image_flame':
            return self.images_flame
        elif key == 'image_mouth':
            return self.images_mouth
        elif key == 'image_wo_mouth':
            return self.images_wo_mouth
        elif key == 'J_transformed':
            return self.J_transformed
        else:
            raise ValueError(f"Unknown key: {key}")

    def __setitem__(self, key, value):
        # Legacy support for EG3D code. Behave like a dictionary
        if key == 'image':
            if self.masks is None:
                if self.backgrounds is None:
                    self.images = value
                else:
                    self.images = value[:, :-3]
                    self.backgrounds = value[:, -3:]
            else:
                if self.backgrounds is None:
                    self.images = value[:, :-1]
                    self.masks = value[:, [-1]]
                else:
                    self.images = value[:, :-4]
                    self.masks = value[:, [3]]
                    self.backgrounds = value[:, -3:]
        elif key == 'image_raw':
            self.images_raw = value
        elif key == 'image_depth':
            self.images_depth = value
        elif key == 'image_flame':
            self.images_flame = value
        else:
            raise ValueError(f"Unknown key: {key}")

    def keys(self):
        # Legacy support for EG3D code. Behave like a dictionary
        return iter(['image', 'image_raw', 'image_depth', 'image_flame', 'image_mouth', 'image_wo_mouth'])

    def values(self):
        # Legacy support for EG3D code. Behave like a dictionary
        return iter([self[key] for key in self.keys()])

    def items(self):
        # Legacy support for EG3D code. Behave like a dictionary
        return zip(self.keys(), self.values())

    def __len__(self) -> int:
        return self.images.shape[0]


def process_template(uv_coords, uv_faces, plane_resolution, n_uniform_flame_vertices, align_corners, interpolation, 
                     for_mouth=False, mouth_center=None, mouth_radius=None):
    idxim, _, barim = gen_tritex(uv_coords, uv_faces, uv_faces, plane_resolution)

    if for_mouth:
        # sample on a square around mouth
        xs = torch.linspace(mouth_center[0]-mouth_radius, mouth_center[0]+mouth_radius, steps=n_uniform_flame_vertices)
        ys = torch.linspace(mouth_center[1]-mouth_radius, mouth_center[1]+mouth_radius, steps=n_uniform_flame_vertices)
    else:
        xs = torch.linspace(-1, 1, steps=n_uniform_flame_vertices)
        ys = torch.linspace(-1, 1, steps=n_uniform_flame_vertices)

    xs, ys = torch.meshgrid(xs, ys, indexing='ij')
    sampled_uv_coords = torch.stack([ys, xs], dim=-1) # stack in ys, xs order since grid_sample uses coordinates in reverse order ([k]ji). 

    torch_face_index_map = torch.from_numpy(idxim).permute(2, 0, 1)
    valid_uv_map = (torch_face_index_map > 0).any(dim=0).float()[None]  # [1, H_map, W_map]

    valid_samples = torch.nn.functional.grid_sample(valid_uv_map.unsqueeze(0), sampled_uv_coords.unsqueeze(0),
                                                    align_corners=align_corners,
                                                    mode=interpolation)[0].permute(1, 2, 0)
    valid_samples = valid_samples[:, :, 0] > 0.99
    valid_uv_coords = sampled_uv_coords[valid_samples]  # [G, 2]
    uv_grid = valid_uv_coords.unsqueeze(0).unsqueeze(2)  # [1, G, 1, 2]

    return uv_grid, idxim, barim


# TODO: move to uv.py
def uv_to_3d(
    uv_grid: torch.Tensor,       # [B, G, 2]  or  [G, 2] if you want the same for all B
    vertices: torch.Tensor,      # [B, N, 3]
    idxim: torch.Tensor,         # [R, R, 3]  or [B, R, R, 3] 
    barim: torch.Tensor,         # [R, R, 3]  or [B, R, R, 3]
    align_corners: bool = False,
    interpolation: str = 'bilinear'
):
    """
    Projects 2D UV samples into 3D space for a batch of meshes.

    Parameters
    ----------
    uv_grid : [1, G, 1, 2]
        UV sample positions in normalized coordinates (grid_sample style).
    vertices : [B, N, 3]
        3D vertices for each of the B meshes.
    idxim : [R, R, 3] or [B, R, R, 3]
        The vertex indices for each pixel in some UV image. If shape is [R,R,3],
        we will broadcast it across B. If [B, R, R, 3], each mesh has its own
        indexing image.
    barim : [R, R, 3] or [B, R, R, 3]
        Barycentric coordinates at each pixel. Similarly broadcastable if shape is [R,R,3].
    align_corners : bool
        Passed to grid_sample.
    interpolation : str
        'bilinear' or 'nearest' or any valid grid_sample mode.

    Returns
    -------
    sampled_positions : [B, G, 3]
        The 3D coordinates corresponding to each UV sample for each mesh.
    """
    if idxim.dim() == 3:  # shape [R, R, 3]
        # Expand to [B, R, R, 3]
        idxim = idxim.unsqueeze(0).expand(vertices.shape[0], -1, -1, -1)
    if barim.dim() == 3:  # shape [R, R, 3]
        barim = barim.unsqueeze(0).expand(vertices.shape[0], -1, -1, -1)

    B, R, R_, _ = idxim.shape

    # 2) Gather vertex positions for each pixel in the UV map
    # idxim[..., 0], idxim[..., 1], idxim[..., 2] are each [B, R, R]
    # We can use advanced/broadcast indexing:
    batch_indices = torch.arange(B, device=idxim.device).view(B, 1, 1)
    # print(vertices.shape, idxim.min(), idxim.max())
    # torch.Size([8, 5023, 3]) tensor(0, device='cuda:2', dtype=torch.int32) tensor(5113, device='cuda:2', dtype=torch.int32)
    # TODO: we can actually do this without batch_indices. just : instead

    
    v0_map = vertices[batch_indices, idxim[..., 0], :]  # [B, R, R, 3]
    v1_map = vertices[batch_indices, idxim[..., 1], :]  # [B, R, R, 3]
    v2_map = vertices[batch_indices, idxim[..., 2], :]  # [B, R, R, 3]

    # 3) Interpolate via barycentric coords => [B, R, R, 3]
    flame_position_map = (
        barim[..., [0]] * v0_map
      + barim[..., [1]] * v1_map
      + barim[..., [2]] * v2_map
    )

    # 4) Format for grid_sample => [B, 3, R, R]
    flame_position_map = flame_position_map.permute(0, 3, 1, 2)

    if uv_grid.shape[0] != B:
        uv_grid = uv_grid.expand(B, -1, -1, -1)  # => [B, G, 1, 2]

    sampled_positions = torch.nn.functional.grid_sample(
        flame_position_map,        # [B, 3, R, R]
        uv_grid,                   # [B, G, 1, 2]
        mode=interpolation,
        align_corners=align_corners,
        padding_mode='border'
    )
    # => [B, 3, G, 1]

    # 7) Reshape => [B, G, 3]
    sampled_positions = sampled_positions.permute(0, 2, 3, 1)  # [B, G, 1, 3]
    sampled_positions = sampled_positions.squeeze(2)          # [B, G, 3]

    return sampled_positions

class GGHeadModel(nn.Module):
    z_dim: int

    def __init__(self, config: GGHeadConfig, logger_bundle: Optional[LoggerBundle] = None,
                 post_init: bool = True):
        super().__init__()
        self._config = config

        # Gaussians have the following attributes:
        #  - Position: 3
        #  - Scale: 3
        #  - Rotation: 3 (cause we use rodriguez rotation)
        #  - Opacity: 1
        #  - Color: 12 = 3 * 4 = 3 * (1 + sh_degree)**2 = 3 * 4 [sh_degree = 1]

        # Acctually takes order from config.uv_attributes
        # "uv_attributes": [
        #     "POSITION",
        #     "SCALE",
        #     "ROTATION",
        #     "COLOR",
        #     "OPACITY"
        # ],

        self._all_gaussian_attribute_names = [GaussianAttribute.POSITION, GaussianAttribute.SCALE,
                                              GaussianAttribute.ROTATION, GaussianAttribute.OPACITY,
                                              GaussianAttribute.COLOR]
        self._uv_attribute_names = [attribute_name for attribute_name in config.uv_attributes
                                    if (
                                            attribute_name != GaussianAttribute.POSITION or not config.disable_position_offsets)]

        self._n_uv_channels = sum(
            [gaussian_attribute.get_n_channels(config.gaussian_attribute_config) for gaussian_attribute in
             self._uv_attribute_names])
        n_gaussian_attributes = self._n_uv_channels

        # Setup StyleGAN2
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.w_dim = config.w_dim

        n_backbone_channels = self._n_uv_channels
        if self._config.use_background_cnn:
            n_backbone_channels += self._config.n_background_channels

        if self._config.use_gaussian_blendshape:
            self.k_blendshapes = self._config.k_blendshapes

            self._blendshape_attribute_names = [attribute_name for attribute_name in config.uv_attributes
                                    if (
                                            attribute_name != GaussianAttribute.COLOR and attribute_name != GaussianAttribute.OPACITY)]
            self.per_blendshape_channels = sum(
                [gaussian_attribute.get_n_channels(config.gaussian_attribute_config) for gaussian_attribute in
                 self._blendshape_attribute_names])
            self.total_blendshape_channels = self.per_blendshape_channels * self.k_blendshapes

            n_backbone_channels += self.total_blendshape_channels

            self.expcode_blendshape_size = self.k_blendshapes - 3 # 3 for jawpose
            
        if self._config.use_extended_uv_generation:
            self.extended_uv_resolution = self._config.plane_resolution + self._config.mouth_plane_resolution

        if self._config.gen_flame_conditioning or self._config.use_extended_uv_generation:
            self.c2_dim = config.c2_dim

            if self._config.gfc_double_modulation:
                from src.gghead.models.stylegan2_double_modulation import MeshGGHGenerator as MeshGGHGenerator_double_modulation
                self.backbone = MeshGGHGenerator_double_modulation(self.z_dim, self.c_dim, self.c2_dim, self.w_dim,
                                                img_resolution=self._config.plane_resolution,
                                                pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                                                img_channels=n_backbone_channels,
                                                mapping_kwargs=asdict(config.mapping_network_config),
                                                flame_double_mapping=self._config.use_flame_specific_double_mapping,
                                                **asdict(config.synthesis_network_config))
            elif self._config.gfc_additive_condition:
                from src.gghead.models.stylegan2_additive_modulation import MeshGGHGenerator as MeshGGHGenerator_additive_condition
                self.backbone = MeshGGHGenerator_additive_condition(self.z_dim, self.c_dim, self.c2_dim, self.w_dim,
                                                img_resolution=self._config.plane_resolution,
                                                pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                                                img_channels=n_backbone_channels,
                                                mapping_kwargs=asdict(config.mapping_network_config),
                                                flame_double_mapping=self._config.use_flame_specific_double_mapping,
                                                **asdict(config.synthesis_network_config))
            elif self._config.use_extended_uv_generation:
                deform_c2_dim = 50 + 3 + 2 # expression + jawpose + eyelid
                self.backbone = GGHGenerator_FLAME(self.z_dim, self.c_dim, deform_c2_dim if self.c2_dim == 406 else self.c2_dim, self.w_dim,
                                             img_resolution=self.extended_uv_resolution,
                                             pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                                             img_channels=n_backbone_channels,
                                             mapping_kwargs=asdict(config.mapping_network_config),
                                             start_res=self.extended_uv_resolution // 64,
                                             use_spade=self._config.use_spade and (not self._config.use_deformation_branch or self._config.double_condition or True),
                                             use_concat=self._config.use_concat and (not self._config.use_deformation_branch or self._config.double_condition or True),
                                             condition_nc=self._config.condition_nc,
                                             double_mapping_for_flame=self._config.gen_flame_conditioning,
                                             **asdict(config.synthesis_network_config))
            else: # double mapping
                deform_c2_dim = 50 + 3 + 2 # expression + jawpose + eyelid
                self.backbone = GGHGenerator_FLAME(self.z_dim, self.c_dim, deform_c2_dim if self.c2_dim == 406 else self.c2_dim, self.w_dim,
                                                img_resolution=self._config.plane_resolution,
                                                pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                                                img_channels=n_backbone_channels,
                                                mapping_kwargs=asdict(config.mapping_network_config),
                                                use_spade=self._config.use_spade and (not self._config.use_deformation_branch or self._config.double_condition or True),
                                                use_concat=self._config.use_concat and (not self._config.use_deformation_branch or self._config.double_condition or True),
                                                condition_nc=self._config.condition_nc,
                                                double_mapping_for_flame=self._config.gen_flame_conditioning,
                                                **asdict(config.synthesis_network_config))
                # Deprecated:
                # self.backbone = MeshGGHGenerator(self.z_dim, self.c_dim, self.c2_dim, self.w_dim,
                #                                 img_resolution=self._config.plane_resolution,
                #                                 pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                #                                 img_channels=n_backbone_channels,
                #                                 mapping_kwargs=asdict(config.mapping_network_config),
                #                                 flame_double_mapping=self._config.use_flame_specific_double_mapping,
                #                                 **asdict(config.synthesis_network_config))
        else:
            self.c2_dim = config.c2_dim
            deform_c2_dim = 50 + 3 + 2 # expression + jawpose + eyelid
            self.backbone = GGHGenerator_FLAME(self.z_dim, self.c_dim, deform_c2_dim if self.c2_dim == 406 else self.c2_dim, self.w_dim,
                                            img_resolution=self._config.plane_resolution,
                                            pretrained_plane_resolution=self._config.pretrained_plane_resolution,
                                            img_channels=n_backbone_channels,
                                            mapping_kwargs=asdict(config.mapping_network_config),
                                            use_spade=self._config.use_spade and (not self._config.use_deformation_branch or self._config.double_condition or True),
                                            use_concat=self._config.use_concat and (not self._config.use_deformation_branch or self._config.double_condition or True),
                                            condition_nc=self._config.condition_nc,
                                            double_mapping_for_flame=self._config.gen_flame_conditioning,
                                            **asdict(config.synthesis_network_config))
            # Deprecated
            # self.backbone = GGHStyleGAN2Backbone(self.z_dim, self.c_dim, self.w_dim,
            #                                  img_resolution=self._config.plane_resolution,
            #                                  pretrained_plane_resolution=self._config.pretrained_plane_resolution,
            #                                  img_channels=n_backbone_channels,
            #                                  mapping_kwargs=asdict(config.mapping_network_config),
            #                                  **asdict(config.synthesis_network_config))

        if self._config.use_background_cnn and self._config.use_background_upsampler and self._config.img_resolution > self._config.plane_resolution:
            img_resolution_log2 = int(np.log2(self._config.img_resolution))
            plane_resolution_log2 = int(np.log2(self._config.plane_resolution))
            n_upsampling_layers = img_resolution_log2 - plane_resolution_log2
            background_channels = [self._config.n_background_channels] * n_upsampling_layers + [3]
            self._background_upsampling_blocks = []
            for i in range(n_upsampling_layers):
                in_channels = background_channels[i]
                out_channels = background_channels[i + 1]
                use_fp16 = False
                is_last = i == (n_upsampling_layers - 1)

                block = GGHSynthesisBlock(in_channels, out_channels, w_dim=self.w_dim,
                                          resolution=2 ** (plane_resolution_log2 + i + 1),
                                          img_channels=3, is_last=is_last, use_fp16=use_fp16,
                                          **config.synthesis_network_config.get_block_kwargs())
                self._background_upsampling_blocks.append(block)  # TODO: Set torgb() to zeros?

            self._background_upsampling_blocks = nn.ModuleList(self._background_upsampling_blocks)
        
        if self._config.use_mouth_branch:
            
            self.mouth_generator = MouthGGHGenerator(
                self.z_dim, self.c_dim, 103, self.w_dim,
                img_resolution=self._config.mouth_plane_resolution,
                img_channels=22,  # Keeping the same channels as before
                start_resolution=self._config.mouth_start_resolution,
                **asdict(config.synthesis_network_config)
            )

        if self._config.use_deformation_branch:
            self.deformation_generator = MouthGGHGenerator(
                self.z_dim, self.c_dim, deform_c2_dim, self.w_dim,
                img_resolution=self._config.deformation_plane_resolution,
                img_channels=22,  # Keeping the same channels as before
                start_resolution=self._config.deformation_start_resolution,
                start_res=self._config.deformation_plane_resolution // 64,
                use_spade=self._config.use_spade and self._config.double_condition,
                use_concat=self._config.use_concat and self._config.double_condition,
                condition_nc=self._config.condition_nc,
                **asdict(config.synthesis_network_config)
            )


        self._uv_attribute_start_channel = dict()
        self._uv_attribute_n_channels = dict()
        c = 0
        for attribute_name in self._uv_attribute_names:
            n_channels = attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            self._uv_attribute_start_channel[attribute_name] = c
            self._uv_attribute_n_channels[attribute_name] = n_channels
            c += n_channels

        if config.use_zero_conv_position:
            n_channels_position = GaussianAttribute.POSITION.get_n_channels(config.gaussian_attribute_config)

            c = 0
            for attribute_name in self._uv_attribute_names:
                if attribute_name == GaussianAttribute.POSITION:
                    break
                c += attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            self._position_start_channel = c
            self._n_position_channels = n_channels_position

            zero_conv_position = nn.Conv2d(n_channels_position, n_channels_position, 1)
            init.zeros_(zero_conv_position.weight)
            init.zeros_(zero_conv_position.bias)
            self._zero_conv_position = zero_conv_position

            if self._config.use_mouth_branch:
                self._mouth_zero_conv_position = nn.Conv2d(n_channels_position, n_channels_position, 1)
                init.zeros_(self._mouth_zero_conv_position.weight)
                init.zeros_(self._mouth_zero_conv_position.bias)
            elif self._config.use_extended_uv_generation:
                # since they came from the same conv
                self._mouth_zero_conv_position = self._zero_conv_position

            if self._config.use_deformation_branch:
                self._deformation_zero_conv_position = nn.Conv2d(n_channels_position, n_channels_position, 1)
                init.zeros_(self._deformation_zero_conv_position.weight)
                init.zeros_(self._deformation_zero_conv_position.bias)

        if self._config.use_deformation_branch:
            c = 0
            for attribute_name in self._uv_attribute_names:
                if attribute_name == GaussianAttribute.COLOR:
                    break
                c += attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            self._color_start_channel = c
            self._n_color_channels = GaussianAttribute.COLOR.get_n_channels(config.gaussian_attribute_config)

        self.neural_rendering_resolution = self._config.neural_rendering_resolution
        self.rendering_config = config.rendering_config

        if config.use_gsm_flame_template:
            flame_template_mesh = trimesh.load(
                f"{REPO_ROOT_DIR}/assets/gghead/flame_uv_no_back_close_mouth_no_subdivision.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_flame_template_v2:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/gghead/flame_template_v2.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_sphere_template:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/gghead/sphere_template.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_plane_template:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/gghead/plane_template.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces
        elif config.use_flame2020_template:
            flame_template_mesh = trimesh.load(f"{REPO_ROOT_DIR}/assets/gghead/head_template_mesh.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces # by default trimesh treats uv_faces as faces
            self.register_buffer("_tidx_to_idx", torch.load(f'{REPO_ROOT_DIR}/assets/flame/uv_to_3d__vert_idx_mapping.pt'))

            _idx_to_tidx = torch.full((5023,), -1, dtype=torch.long)
            for tidx, idx in enumerate(self._tidx_to_idx.tolist()):
                if _idx_to_tidx[idx] == -1:
                    _idx_to_tidx[idx] = tidx
            self.register_buffer("_idx_to_tidx", _idx_to_tidx)
            flame_pickle_path = f'{GGHEAD_DEPENDENCIES_PATH}/smirk/assets/FLAME2020/generic_model.pkl'
            self.deform_mask_path = f'{REPO_ROOT_DIR}/assets/gghead/uv_position_weights_dejavu_adapted.npy'
        elif config.use_flame_template_with_mouth:
            flame_template_mesh = trimesh.load(f"{GGHEAD_DEPENDENCIES_PATH}/threedim_utils/assets/flame_with_mouth_no_backhead_v3/head_template_mesh_stretched_uv.obj")
            uvs_per_flame_vertex = flame_template_mesh.visual.uv
            uv_coords = uvs_per_flame_vertex
            uv_faces = flame_template_mesh.faces # by default trimesh treats uv_faces as faces
            self.register_buffer("_tidx_to_idx", torch.load(f'{GGHEAD_DEPENDENCIES_PATH}/threedim_utils/assets/flame_with_mouth_no_backhead_v3/uv_to_3d__vert_idx_mapping.pt'))

            _idx_to_tidx = torch.full((self._tidx_to_idx.max()+1,), -1, dtype=torch.long)
            for tidx, idx in enumerate(self._tidx_to_idx.tolist()):
                if _idx_to_tidx[idx] == -1:
                    _idx_to_tidx[idx] = tidx
            self.register_buffer("_idx_to_tidx", _idx_to_tidx)
            flame_pickle_path = f'{GGHEAD_DEPENDENCIES_PATH}/threedim_utils/assets/flame_with_mouth_no_backhead_v3/FLAME.pkl'
            self.deform_mask_path = f'{GGHEAD_DEPENDENCIES_PATH}/threedim_utils/assets/flame_with_mouth_no_backhead_v3/flame_facial_uv_mask.npy'
        else:
            raise ValueError("No mesh template specified!")
        faces = torch.tensor([self._tidx_to_idx[x] for face in uv_faces for x in face]).reshape(-1, 3)

        self.register_buffer("template_uv_coords", torch.tensor(np.array(uv_coords)).contiguous())
        self.register_buffer("template_faces", torch.tensor(np.array(faces)).contiguous())
        self.register_buffer("template_uv_faces", torch.tensor(np.array(uv_faces)).contiguous())

        uv_grid, idxim, barim = process_template(self.template_uv_coords, self.template_uv_faces, 
                                                   self._config.plane_resolution, self._config.n_uniform_flame_vertices, 
                                                   self._config.use_align_corners, self._config.interpolation_mode, 
                                                   )    
            
        # self.flame_lbs_model = FLAME(DEFAULT_CONFIG)
        # self.n_shape = DEFAULT_CONFIG.n_shape
        # self.n_exp = DEFAULT_CONFIG.n_exp

        self.flame_lbs_model = FLAME_smirk(flame_pickle_path, f'{GGHEAD_DEPENDENCIES_PATH}/smirk/assets/landmark_embedding.npy')
        self.n_shape = 300
        self.n_exp = 50

        for param in params_and_buffers(self.flame_lbs_model):
            if param.numel() > 0:
                param.data = param.data.contiguous()



        self.register_buffer("_uv_grid", uv_grid.contiguous())
        self.register_buffer("_idxim", torch.tensor(idxim).contiguous())
        self.register_buffer("_barim", torch.tensor(barim).contiguous())

        if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
            uv_grid_template = self.template_uv_coords.clone().unsqueeze(0).unsqueeze(2).float()
            uv_grid_template = uv_grid_template * 2 - 1 # scale to [-1, 1]
            self.register_buffer("_uv_grid_template", uv_grid_template.contiguous())

        # Sample mouth gaussians
        if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
            xs = torch.linspace(-1, 1, self._config.mouth_n_vertices)
            ys = torch.linspace(-1, 1, self._config.mouth_n_vertices)
            ys, xs = torch.meshgrid(ys, xs, indexing='ij')
            grid_points = torch.stack([xs, ys], dim=-1)  # [n_vertices, n_vertices, 2]

            # Reshape to [G, 2] format
            grid_flat = grid_points.reshape(-1, 2)  # [G, 2]

            # Format for grid_sample: [1, G, 1, 2]
            mouth_uv_grid = grid_flat.unsqueeze(0).unsqueeze(2)

            self.register_buffer("_mouth_uv_grid", mouth_uv_grid.contiguous())

        if self._config.use_extended_uv_generation:
            self.mouth_res = self.extended_uv_resolution - self._config.plane_resolution
            assert self.mouth_res == self._config.mouth_n_vertices

        if self._config.gfc_additive_condition or self._config.use_spade or self._config.use_concat:
            vertices = self.flame_lbs_model.v_template.float()
            if self._config.use_flame_to_bfm_registration:
                vertices = self.flame_to_bfm_registration(vertices)
            dynamic_template_vertices = vertices[self._tidx_to_idx]
            v0_map = dynamic_template_vertices[self._idxim[..., 0]]
            v1_map = dynamic_template_vertices[self._idxim[..., 1]]
            v2_map = dynamic_template_vertices[self._idxim[..., 2]]

            template_uv_positions_map = self._barim[..., [0]] * v0_map + self._barim[..., [1]] * v1_map + self._barim[
                ..., [2]] * v2_map 
            template_uv_positions_map = template_uv_positions_map.float().permute(2, 0, 1).unsqueeze(0)
            self.register_buffer("_template_uv_positions_map", template_uv_positions_map.contiguous())

            if self._config.use_extended_uv_generation:
                mouth_vertices = generate_mouth_interior_uv(vertices.unsqueeze(0), self.mouth_res, self.mouth_res // 2).permute(0, 3, 1, 2)
                self.register_buffer("_template_uv_positions_map_mouth_interior", mouth_vertices.contiguous())




        if config.use_flame_rasterization:
            barim_mask = self._barim.sum(axis=-1) != 0
            barim_mask = barim_mask.unsqueeze(-1).expand(-1, -1, 3)
            grad_texture = get_gradient_flame_texture(barim_mask, resolution=self._config.plane_resolution)
            self.flame_rasterizer = FLAME_rasterizer(
                self.template_uv_faces, 
                self.template_uv_coords, 
                self.template_faces, 
                grad_texture, 
                self._config.plane_resolution, 
                config.flame_rasterization_light_type)


        if self._config.use_deform_mask: 
            uv_reg_weights = torch.from_numpy(np.load(self.deform_mask_path)) # [256, 256]
            
            uv_reg_weights = torch.nn.functional.interpolate(uv_reg_weights.unsqueeze(0).unsqueeze(0), size=self._config.plane_resolution, mode='bilinear', antialias=False).squeeze() # [res, res]
            uv_reg_mask = (uv_reg_weights >= 0.5).float().unsqueeze(0).unsqueeze(0)
            
            if self._config.use_extended_uv_generation:
                uv_reg_mask_padded = torch.zeros((1, 1, self.extended_uv_resolution, self.extended_uv_resolution))
                pad_size = (self.extended_uv_resolution - self._config.plane_resolution) // 2
                uv_reg_mask_padded[:, :, :self._config.plane_resolution, pad_size:pad_size + self._config.plane_resolution] = uv_reg_mask
                pad_size_mouth = (self.extended_uv_resolution - self.mouth_res) // 2
                uv_reg_mask_padded[:, :, self._config.plane_resolution:self._config.plane_resolution + self.mouth_res, pad_size_mouth:pad_size_mouth + self.mouth_res] = 1.0
                deform_mask = uv_reg_mask_padded
            else:
                deform_mask = uv_reg_mask

            self.register_buffer("_deform_mask", deform_mask.contiguous())


        self.use_shape_clusters = False
        self.n_shape_clusters = 128
        if self.use_shape_clusters:
            self.shape_lookup_table = torch.nn.Embedding(self.n_shape_clusters, self.c2_dim)

        # Setup Gaussian Model for rendering
        self._gaussian_model = GaussianModel(sh_degree=self._config.gaussian_attribute_config.sh_degree)
        self._gaussian_model.active_sh_degree = self._config.gaussian_attribute_config.sh_degree
        self._gaussian_model.opacity_activation = self._apply_opacity_activation
        if config.no_exp_scale_activation: # default is False in GGHEAD
            # Note: inverse_scaling_activation is not changed
            self._gaussian_model.scaling_activation = self._apply_scale_activation

        gaussian_bg = torch.Tensor([1 for _ in range(config.gaussian_attribute_config.n_color_channels)])
        gaussian_bg_train = torch.Tensor(self._config.background_color) / 255
        self.register_buffer("_gaussian_bg", gaussian_bg, persistent=False)
        self.register_buffer("_gaussian_bg_train", gaussian_bg_train, persistent=False)

        self._last_planes = None

        # Needed for EG3D visualizer
        self.img_resolution = config.img_resolution
        self.rendering_kwargs = {
            'depth_resolution': 48,
            'depth_resolution_importance': 48,
        }

        # Logging
        self._logger_bundle = logger_bundle

    def sample_z(self, person_ids: torch.Tensor) -> torch.Tensor:
        if self._config.use_autodecoder:
            z = self._identity_codes(person_ids)
        else:
            z = torch.randn((len(person_ids), self._config.z_dim)).cuda()

        return z

    # ==========================================================
    # Forward Helpers
    # ==========================================================

    def _sh_to_rgb(self, gaussian_positions, gaussian_colors, sh_ref_camera_center):
        '''
        Args:
            gaussian_positions: [B, G, 3]
            gaussian_colors: [B, G, SH-1, 3] – gaussian features (dc + rest)
            sh_ref_camera_center: [B, 3]
        '''
        B, G, _ = gaussian_positions.shape

        sh_degree = self._config.gaussian_attribute_config.sh_degree
        n_feature_channels = self._config.gaussian_attribute_config.n_color_channels
        shs_view = gaussian_colors.view(B, G, (sh_degree + 1) ** 2, n_feature_channels).permute(0, 1, 3, 2) # [B, G, 3, sh_bases_size]
        dir_pp = (gaussian_positions - sh_ref_camera_center.unsqueeze(1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
        sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors

    def _render_gsplat(self, 
                    gaussian_positions: torch.Tensor, 
                    gaussian_colors: torch.Tensor, 
                    gaussian_scales: torch.Tensor, 
                    gaussian_rotations: torch.Tensor, 
                    gaussian_opacities: torch.Tensor, 
                    world2cam_matrix: torch.Tensor, 
                    intrinsics_matrix: torch.Tensor, 
                    neural_rendering_resolution: int, 
                    override_color: torch.Tensor = None,
                    backgrounds: torch.Tensor = None,
                    ):
        '''
            Accepts batched gaussian attributes and cameras: [B, ...]
            Attributes before activation!
        '''
        # 1) activate attributes
        means = gaussian_positions
        quats = self._gaussian_model.rotation_activation(gaussian_rotations)
        scales = self._gaussian_model.scaling_activation(gaussian_scales)
        opacities = self._gaussian_model.opacity_activation(gaussian_opacities)

        if override_color is None: 
            # gaussian_colors are B, G, K, 3
            features_dc = gaussian_colors[:, :, [0]] 
            features_rest = gaussian_colors[:, :, 1:]
            colors = torch.cat([features_dc, features_rest], dim=2)
        else:
            colors = override_color

        render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities.squeeze(-1),
                colors=colors,
                sh_degree=self._gaussian_model.active_sh_degree if override_color is None else None,
                viewmats=world2cam_matrix.unsqueeze(1),  # [B, C=1, 4, 4]
                Ks=intrinsics_matrix.unsqueeze(1),  # [B, C=1, 3, 3]
                width=neural_rendering_resolution,
                height=neural_rendering_resolution,
                packed=self.rendering_config.gsplat_raster_settings.packed,
                absgrad=self.rendering_config.gsplat_raster_settings.absgrad,
                sparse_grad=self.rendering_config.gsplat_raster_settings.sparse_grad,
                rasterize_mode=self.rendering_config.gsplat_raster_settings.rasterize_mode,
                camera_model=self.rendering_config.gsplat_raster_settings.camera_model,
                backgrounds=backgrounds
            )
        # [B, C=1, H, W, D] –> [B, D, H, W]
        render_colors = render_colors.squeeze(1).permute(0, 3, 1, 2)
        render_colors = render_colors * 2 - 1 # [0, 1] -> [-1, 1]

        render_alphas = render_alphas.squeeze(1).permute(0, 3, 1, 2)
        render_alphas = render_alphas * 2 - 1 # [0, 1] -> [-1, 1]

        return render_colors, render_alphas, info

    def render_gs_batch(
                    self, 
                    gaussian_positions: torch.Tensor, 
                    gaussian_colors: torch.Tensor, 
                    gaussian_scales: torch.Tensor, 
                    gaussian_rotations: torch.Tensor, 
                    gaussian_opacities: torch.Tensor, 
                    sh_ref_cam: torch.Tensor, 
                    J_transformed: torch.Tensor, 
                    cam2world_matrix: torch.Tensor = None, 
                    world2cam_matrix: torch.Tensor = None, 
                    intrinsics_matrix: torch.Tensor = None, 
                    neural_rendering_resolution: int = None, 
                    return_masks: bool = False,
                    also_render_mouth: bool = False,
                    background_rgb: torch.Tensor = None,
                    raster_backend: str = '3dgs',
                    override_color: torch.Tensor = None,
                    ):
        
        if raster_backend == 'gsplat':
            scaled_intrinsics_matrix = intrinsics_matrix.clone()
            scaled_intrinsics_matrix[:, :2, :] *= neural_rendering_resolution

            if override_color is None and sh_ref_cam is not None:
                sh_ref_w2c, _ = parse_flame_deca_cameras(sh_ref_cam, J_transformed[0:1]) # [R|t]
                sh_ref_cam_center = -sh_ref_w2c[:, :3, :3].permute(0, 2, 1) @ sh_ref_w2c[:, :3, 3].unsqueeze(-1) # -R^T @ t
                sh_ref_cam_center = sh_ref_cam_center.squeeze(-1)
                override_color = self._sh_to_rgb(gaussian_positions, gaussian_colors, sh_ref_cam_center)
            
            backgrounds = self._gaussian_bg_train.unsqueeze(0).unsqueeze(0).repeat(gaussian_positions.shape[0], 1, 1) # [B, C, 3]

            render_colors, render_alphas, _ = self._render_gsplat(
                gaussian_positions,
                gaussian_colors,
                gaussian_scales,
                gaussian_rotations,
                gaussian_opacities,
                world2cam_matrix,
                scaled_intrinsics_matrix,
                neural_rendering_resolution,
                override_color = override_color,
                backgrounds = backgrounds
                )

            if not return_masks:
                render_alphas = None
            
            render_colors_mouth = None
            render_colors_wo_mouth = None
            if (self._config.use_mouth_branch or self._config.use_extended_uv_generation) and also_render_mouth:
                n_mouth_gs = self._mouth_uv_grid.shape[1]
                render_colors_mouth, _, _ = self._render_gsplat(
                    gaussian_positions[:, -n_mouth_gs:],
                    gaussian_colors[:, -n_mouth_gs:],
                    gaussian_scales[:, -n_mouth_gs:],
                    gaussian_rotations[:, -n_mouth_gs:],
                    gaussian_opacities[:, -n_mouth_gs:],
                    world2cam_matrix,
                    scaled_intrinsics_matrix,
                    neural_rendering_resolution,
                    override_color = override_color[:, -n_mouth_gs:] if override_color is not None else None,
                    backgrounds = backgrounds
                )

                n_head_gs = gaussian_positions.shape[1] - n_mouth_gs
                render_colors_wo_mouth, _, _ = self._render_gsplat(
                    gaussian_positions[:, :n_head_gs],
                    gaussian_colors[:, :n_head_gs],
                    gaussian_scales[:, :n_head_gs],
                    gaussian_rotations[:, :n_head_gs],
                    gaussian_opacities[:, :n_head_gs],
                    world2cam_matrix,
                    scaled_intrinsics_matrix,
                    neural_rendering_resolution,
                    override_color = override_color[:, :n_head_gs] if override_color is not None else None,
                    backgrounds = backgrounds
                )
            return render_colors, render_alphas, render_colors_mouth, render_colors_wo_mouth

        elif raster_backend == '3dgs':
            if (self._config.use_mouth_branch or self._config.use_extended_uv_generation) and also_render_mouth:
                rgb_images_mouth = []
                rgb_images_wo_mouth = []
        
            device = gaussian_positions.device
            rgb_images = []
            masks = []

            B = len(gaussian_positions)
            for i in range(B):
                if cam2world_matrix is not None:
                    cam_2_world_pose = Pose(cam2world_matrix[i].cpu().numpy(), pose_type=PoseType.CAM_2_WORLD,
                                            disable_rotation_check=True)
                elif world2cam_matrix is not None:
                    cam_2_world_pose = Pose(world2cam_matrix[i].cpu().numpy(), pose_type=PoseType.WORLD_2_CAM,
                                            disable_rotation_check=True, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                
                intrinsics = Intrinsics(intrinsics_matrix[i].cpu().numpy())
                intrinsics = intrinsics.rescale(neural_rendering_resolution,
                                                inplace=False)  # EG3D intrinsics are given in normalized format wrt to [0-1] image
                gaussian_camera = pose_to_rendercam(cam_2_world_pose, intrinsics, neural_rendering_resolution,
                                                    neural_rendering_resolution, device=device)

                self._gaussian_model._xyz = gaussian_positions[i]
                self._gaussian_model._features_dc = gaussian_colors[i][:, [0]]
                self._gaussian_model._features_rest = gaussian_colors[i][:, 1:]  # [G, SH-1, 3]
                self._gaussian_model._scaling = gaussian_scales[i]
                self._gaussian_model._rotation = gaussian_rotations[i].contiguous()  # Rotation needs to be contiguous!
                self._gaussian_model._opacity = gaussian_opacities[i]

                if sh_ref_cam is not None and override_color is None:
                    if sh_ref_cam.shape[1] == 6:
                        sh_ref_w2c, _ = parse_flame_deca_cameras(sh_ref_cam, J_transformed[0:1])
                        sh_ref_cam = Pose(sh_ref_w2c[0].cpu().numpy(), pose_type=PoseType.WORLD_2_CAM,
                                            disable_rotation_check=True, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)

                    gaussian_sh_ref_cam = pose_to_rendercam(sh_ref_cam, intrinsics, neural_rendering_resolution, neural_rendering_resolution, device=device)

                    sh_degree = self._config.gaussian_attribute_config.sh_degree
                    n_feature_channels = self._config.gaussian_attribute_config.n_color_channels
                    shs_view = self._gaussian_model.get_features.view(-1, (sh_degree + 1) ** 2, n_feature_channels).permute(0, 2, 1)
                    dir_pp = (self._gaussian_model.get_xyz - gaussian_sh_ref_cam.camera_center.repeat(1, 1))
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
                    sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
                    colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
                    override_color = colors

                gaussian_bg = self._gaussian_bg_train

                # The with statement is necessary, since otherwise the rasterizer internally may move something to the wrong GPU
                with torch.cuda.device(device):
                    rendered_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(), gaussian_bg,
                                            override_color=override_color)
                
                rendered_image = rendered_image['render']  # [3, H, W]
                rendered_image = rendered_image * 2 - 1  # [0, 1] -> [-1, 1]

                if self._config.use_background_cnn or return_masks:
                    # Obtain alpha image by rendering a second time with all Gaussians set to black
                    black_colors = torch.ones_like(gaussian_colors[i][:, 0]) * 0

                    with torch.cuda.device(device):
                        rendered_alpha_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(),
                                                    self._gaussian_bg, override_color=black_colors)
                    rendered_alpha_image = 1 - rendered_alpha_image['render']  # 0 is background, 1 is foreground

                    if self._config.use_background_cnn:
                        rendered_image = (rendered_image + 1) / 2  # Blending has to be done in [0, 1] range
                        bg_img = (background_rgb[i] + 1) / 2
                        # Alpha blending of Gaussian rendering with CNN background
                        if self._config.fix_alpha_blending:
                            # rendered_image contains blended white colors. Remove them here
                            rendered_image = rendered_image - (1 - rendered_alpha_image) * self._gaussian_bg[:, None, None]
                            rendered_image = (1 - rendered_alpha_image) * bg_img + rendered_image
                        else:
                            rendered_image = (1 - rendered_alpha_image) * bg_img + rendered_alpha_image * rendered_image
                        rendered_image = rendered_image * 2 - 1  # [0, 1] -> [-1, 1]

                    if return_masks:
                        # Rendered alpha image has 3 channels, but they are all the same
                        rendered_alpha_image = rendered_alpha_image * 2 - 1  # [0, 1] -> [-1, 1]
                        masks.append(rendered_alpha_image[[0]])  # [1, H, W]

                if (self._config.use_mouth_branch or self._config.use_extended_uv_generation) and also_render_mouth:
                    n_mouth_gs = self._mouth_uv_grid.shape[1]
                    self._gaussian_model._xyz = gaussian_positions[i][-n_mouth_gs:]
                    self._gaussian_model._features_dc = gaussian_colors[i][:, [0]][-n_mouth_gs:]
                    self._gaussian_model._features_rest = gaussian_colors[i][:, 1:][-n_mouth_gs:]  # [G, SH-1, 3]
                    self._gaussian_model._scaling = gaussian_scales[i][-n_mouth_gs:]
                    self._gaussian_model._rotation = gaussian_rotations[i][-n_mouth_gs:].contiguous()  # Rotation needs to be contiguous!
                    self._gaussian_model._opacity = gaussian_opacities[i][-n_mouth_gs:]

                    with torch.cuda.device(device):
                        rendered_mouth_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(), gaussian_bg,
                                                override_color=override_color)

                    rendered_mouth_image = rendered_mouth_image['render']  # [3, H, W]
                    rendered_mouth_image = rendered_mouth_image * 2 - 1  # [0, 1] -> [-1, 1]

                    rgb_images_mouth.append(rendered_mouth_image)

                    # Render without mouth Gaussians
                    n_head_gs = gaussian_positions.shape[1] - n_mouth_gs
                    self._gaussian_model._xyz = gaussian_positions[i][:n_head_gs]
                    self._gaussian_model._features_dc = gaussian_colors[i][:, [0]][:n_head_gs]
                    self._gaussian_model._features_rest = gaussian_colors[i][:, 1:][:n_head_gs]
                    self._gaussian_model._scaling = gaussian_scales[i][:n_head_gs]
                    self._gaussian_model._rotation = gaussian_rotations[i][:n_head_gs].contiguous()
                    self._gaussian_model._opacity = gaussian_opacities[i][:n_head_gs]

                    with torch.cuda.device(device):
                        rendered_wo_mouth_image = render(gaussian_camera, self._gaussian_model, PipelineParams2(), gaussian_bg,
                                                    override_color=override_color[:n_head_gs] if override_color is not None else None) # TODO: check override color slicing

                    rendered_wo_mouth_image = rendered_wo_mouth_image['render'] # [3, H, W]
                    rendered_wo_mouth_image = rendered_wo_mouth_image * 2 - 1 # [0, 1] -> [-1, 1]
                    rgb_images_wo_mouth.append(rendered_wo_mouth_image)


                    # restore gaussian model
                    self._gaussian_model._xyz = gaussian_positions[i]
                    self._gaussian_model._features_dc = gaussian_colors[i][:, [0]]
                    self._gaussian_model._features_rest = gaussian_colors[i][:, 1:]  # [G, SH-1, 3]
                    self._gaussian_model._scaling = gaussian_scales[i]
                    self._gaussian_model._rotation = gaussian_rotations[i].contiguous()  # Rotation needs to be contiguous!
                    self._gaussian_model._opacity = gaussian_opacities[i]
                
                rgb_images.append(rendered_image)
            
            rgb_images_direct = torch.stack(rgb_images)
            masks = torch.stack(masks) if self._config.use_masks or return_masks else None

            rgb_images = rgb_images_direct

            if (self._config.use_mouth_branch or self._config.use_extended_uv_generation) and also_render_mouth:
                rgb_images_mouth = torch.stack(rgb_images_mouth)
                rgb_images_wo_mouth = torch.stack(rgb_images_wo_mouth)
            else:
                rgb_images_mouth = None
                rgb_images_wo_mouth = None

            return rgb_images, masks, rgb_images_mouth, rgb_images_wo_mouth
        else:
            raise ValueError(f"Invalid raster backend: {raster_backend}")
        


    def sample_uv_map(self, planes: torch.Tensor, uv_grid: torch.Tensor, n_shells: int = 1, zero_conv_position_layer: nn.Conv2d = None):
        B, C, H_f, W_f = planes.shape
        S = n_shells
        C_uv = self._n_uv_channels

        uv_map = planes.clone()
        if self._config.use_zero_conv_position and GaussianAttribute.POSITION in self._uv_attribute_names and zero_conv_position_layer is not None:
            position_start_channel = self._position_start_channel

            # Why we need this? We pass positions through w=0, b=0 inited Conv2d in order to have 0 displacement at the beginning
            zeroed_positions = zero_conv_position_layer(
                uv_map[:, position_start_channel: position_start_channel + zero_conv_position_layer.in_channels])
            uv_map = torch.cat([uv_map[:, :position_start_channel],
                                zeroed_positions,
                                uv_map[:, position_start_channel + zero_conv_position_layer.in_channels:]], dim=1)

        uv_map = uv_map.reshape(B * S, C_uv, H_f, W_f)  # [B*S, UV, H_f, W_f]

        uv_attributes = grid_sample(uv_map, uv_grid.repeat(B * S, 1, 1, 1),
                                    align_corners=self._config.use_align_corners,
                                    mode=self._config.interpolation_mode)  # [B*S, C_uv, G, 1]
        uv_attributes = uv_attributes.squeeze(3).permute(0, 2, 1)  # [B*Shells, G, C_uv]
        G = uv_attributes.shape[1]

        uv_attributes = uv_attributes.reshape(B, S * G, C_uv)

        return uv_attributes, uv_map

    def predict_planes(self, ws: torch.Tensor, update_emas=False, cache_backbone=False, use_cached_backbone=False,
                       alpha_plane_resolution: Optional[float] = None, flame_params=None, noise_cond=None,
                       extract_features_at_resolution=None, condition_map=None, **synthesis_kwargs):
        # Predict 2D planes
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes.clone()
            features_at_res = self._last_features_at_res.clone()
            img_at_res = self._last_img_at_res.clone()
        else:
            if extract_features_at_resolution is not None:
                # Extract intermediate features
                if self._config.gfc_double_modulation:
                    flame_params = flame_params.clone()
                    expcode = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
                    posecode = flame_params[:, (self.n_shape + self.n_exp):(self.n_shape + self.n_exp + 8)]
                    jawpose = posecode[:, 3:6]
                    w2 = torch.cat([expcode, jawpose], dim=1)
                    planes, features_at_res, img_at_res = self.backbone.synthesis.forward_with_intermediate_features(
                        ws, target_resolution=extract_features_at_resolution, 
                        update_emas=update_emas, w2=w2, **synthesis_kwargs)
                elif self._config.gfc_additive_condition:
                    planes, features_at_res, img_at_res = self.backbone.synthesis.forward_with_intermediate_features(
                        ws, target_resolution=extract_features_at_resolution,
                        update_emas=update_emas, w2=noise_cond, **synthesis_kwargs)
                elif (self._config.use_spade or self._config.use_concat) and (not self._config.use_deformation_branch or self._config.double_condition or True):
                    planes, features_at_res, img_at_res = self.backbone.synthesis.forward_with_intermediate_features(
                        ws, target_resolution=extract_features_at_resolution,
                        update_emas=update_emas, condition_map=condition_map, **synthesis_kwargs)
                else:
                    planes, features_at_res, img_at_res = self.backbone.synthesis.forward_with_intermediate_features(
                        ws, target_resolution=extract_features_at_resolution,
                        update_emas=update_emas, **synthesis_kwargs)
            else:
                # Standard forward pass
                if self._config.gfc_double_modulation:
                    flame_params = flame_params.clone()
                    expcode = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
                    posecode = flame_params[:, (self.n_shape + self.n_exp):(self.n_shape + self.n_exp + 8)]
                    jawpose = posecode[:, 3:6]
                    w2 = torch.cat([expcode, jawpose], dim=1)
                    planes = self.backbone.synthesis(ws, update_emas=update_emas, w2=w2, **synthesis_kwargs)
                elif self._config.gfc_additive_condition:
                    planes = self.backbone.synthesis(ws, update_emas=update_emas, w2=noise_cond, **synthesis_kwargs)
                elif (self._config.use_spade or self._config.use_concat) and (not self._config.use_deformation_branch or self._config.double_condition):
                    planes = self.backbone.synthesis(ws, update_emas=update_emas, condition_map=condition_map, **synthesis_kwargs)
                else:
                    planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
                features_at_res = None
                img_at_res = None
                
        if cache_backbone:
            self._last_planes = planes.clone()
            self._last_features_at_res = features_at_res.clone()
            self._last_img_at_res = img_at_res.clone()
            
        if extract_features_at_resolution is not None:
            return planes, features_at_res, img_at_res
        else:
            return planes

    def predict_gaussian_attributes(self,
                                    planes: torch.Tensor,
                                    vertices : torch.Tensor,
                                    return_raw_attributes: bool = False,
                                    return_uv_map: bool = False,
                                    mouth_planes: torch.Tensor = None,
                                    mouth_3d_lmk: torch.Tensor = None,
                                    inference_options: Optional[Dict[str, float]] = None,
                                    deform_planes: torch.Tensor = None
                                    ) -> GaussianAttributeOutput:

        gaussian_attributes = dict()
        raw_gaussian_attributes = dict()

        # Predict UV textures and collect gaussian attributes
        C_uv = self._n_uv_channels
        planes_uv = planes[:, -self._n_uv_channels:]
        planes_main = planes_uv[:, : C_uv]
        uv_attributes, uv_map_main = self.sample_uv_map(planes_main, self._uv_grid, zero_conv_position_layer=self._zero_conv_position)
        uv_map = uv_map_main

        # sample displacements for FLAME vertices (deprecated)
        # if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
        #     template_uv_attributes, _ = self.sample_uv_map(planes_main, self._uv_grid_template, zero_conv_position_layer=self._zero_conv_position)
        #     template_pos_displacement = template_uv_attributes[:, :, self._position_start_channel:self._position_start_channel + self._zero_conv_position.in_channels]
        #     # [B, 5118, 3]
        template_pos_displacement = None
        
        if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
            # ТУТ БЫЛ БАГ. Он делал тот же самый zero_conv для губ, что и для головы.
            mouth_uv_attributes, _ = self.sample_uv_map(mouth_planes, self._mouth_uv_grid, zero_conv_position_layer=self._mouth_zero_conv_position)
            uv_attributes = torch.cat([uv_attributes, mouth_uv_attributes], dim=1)

        if deform_planes is not None:
            deform_uv_attributes, _ = self.sample_uv_map(deform_planes, self._uv_grid, zero_conv_position_layer=self._deformation_zero_conv_position if hasattr(self, '_deformation_zero_conv_position') else None)
        else:
            # Fallback: no deformation branch → zeros (no residuals)
            B = planes.shape[0]
            G_head = self._uv_grid.shape[1]
            deform_uv_attributes = torch.zeros(B, G_head, C_uv, device=planes.device, dtype=planes.dtype)
        # B, G_head, C_uv

        collected_uv_attributes, raw_uv_attributes = self._collect_gaussian_attributes(
            self._uv_attribute_names,
            uv_attributes,
            vertices,
            return_raw_attributes=return_raw_attributes,
            mouth_3d_lmk=mouth_3d_lmk,
            inference_options=inference_options,
            template_pos_displacement=template_pos_displacement,
            deform_predictions=deform_uv_attributes
            )

        gaussian_attributes.update(collected_uv_attributes)
        raw_gaussian_attributes.update(raw_uv_attributes)

        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION]
        gaussian_scales = gaussian_attributes[GaussianAttribute.SCALE]
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION]
        gaussian_opacities = gaussian_attributes[GaussianAttribute.OPACITY]
        gaussian_colors = gaussian_attributes[GaussianAttribute.COLOR]

        if self.training and self._logger_bundle is not None:
            self._logger_bundle.log_metrics({
                "Analyze/norm_gaussian_positions": gaussian_positions.norm(dim=-1).mean(),
                "Analyze/gaussian_scales": gaussian_scales.mean(),
                "Analyze/angle_gaussian_rotations": 2 * torch.acos(gaussian_rotations[..., 0]),
                "Analyze/gaussian_opacities": gaussian_opacities.mean(),
                "Analyze/gaussian_colors": gaussian_colors.norm(dim=-1).mean()
            })

        gaussian_attribute_output = GaussianAttributeOutput(gaussian_attributes,
                                                            raw_gaussian_attributes,
                                                            uv_map=uv_map if return_uv_map else None,
                                                            deform_uv_attributes=deform_uv_attributes)
        return gaussian_attribute_output

    def _log_gradients(self, name: str, tensor: torch.Tensor, num_mouth_vertices = None):
        def log_grad_callback(grad, n=name, num_mouth_vertices=num_mouth_vertices):
            log_dict = {f"Analyze/Gradients/grad_gaussian_{n}": grad.norm(dim=1).mean()}

            if num_mouth_vertices is not None and num_mouth_vertices > 0:
                log_dict[f"Analyze/Gradients/Mouth/grad_gaussian_{n}"] = grad[-num_mouth_vertices:].norm(dim=1).mean()

            self._logger_bundle.log_metrics(
                log_dict
            )

        
        if tensor.requires_grad and self._logger_bundle is not None:
            tensor.register_hook(log_grad_callback)

        # if tensor.requires_grad and self._logger_bundle is not None:
        #     tensor.register_hook(lambda grad, n=name: self._logger_bundle.log_metrics(
        #         {f"Analyze/Gradients/grad_gaussian_{n}": grad.norm(dim=1).mean()}))
    
    # TODO: Do we need this?
    def get_uv_rendering(self, c: torch.Tensor, output: GGHeadOutput,
                         include_transparent_gaussians: bool = False, J_transformed: torch.Tensor = None) -> torch.Tensor:
        '''
            Renders gaussian splats with colors from uv_grid positions.
        '''
        B = len(c)
        device = c.device
        resolution = output['image'].shape[2]
        debug_gaussian_attributes = copy(output.gaussian_attribute_output.gaussian_attributes)
        uv_colors = self._uv_grid.squeeze(2)  # [1, G, 2]
        if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
            uv_colors = torch.cat([uv_colors, self._mouth_uv_grid.squeeze(2)], dim=1)

        uv_colors = torch.concatenate([uv_colors, -torch.ones((1, uv_colors.shape[1], 1), device=device)],
                                      dim=-1)  # [1, G, 3]
        debug_gaussian_attributes[GaussianAttribute.COLOR] = uv_colors.repeat(B, 1, 1)
        if include_transparent_gaussians:
            debug_gaussian_attributes[GaussianAttribute.OPACITY] = torch.ones_like(
                debug_gaussian_attributes[GaussianAttribute.OPACITY])

        if self.rendering_config.raster_backend == '3dgs':
            all_uv_renders = []
            for i, single_c in enumerate(c.cpu()):
                gaussian_model = self._setup_gaussian_model(debug_gaussian_attributes, i)

                if single_c.shape[0] == 6:
                    world2cam_matrix, intrinsics_matrix = parse_flame_deca_cameras(c[i:i+1, :], J_transformed[i:i+1, :])
                    pose = Pose(world2cam_matrix[0].cpu().numpy(), pose_type=PoseType.WORLD_2_CAM,
                                            disable_rotation_check=True, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV)
                    intrinsics = Intrinsics(intrinsics_matrix[0].cpu().numpy())
                else:
                    pose, intrinsics = decode_camera_params(single_c)

                intrinsics = intrinsics.rescale(resolution)
                gs_cam = pose_to_rendercam(pose, intrinsics, resolution, resolution, device=device)

                with torch.cuda.device(device):
                    uv_render = render(gs_cam, gaussian_model, PipelineParams2(), torch.tensor([1., 1., 1.], device=device),
                                    override_color=uv_colors[0])
                all_uv_renders.append(uv_render['render'])

            all_uv_renders = torch.stack(all_uv_renders)
            return all_uv_renders
        elif self.rendering_config.raster_backend == 'gsplat':
            world2cam_matrix, intrinsics_matrix = parse_flame_deca_cameras(c, J_transformed)
            scaled_intrinsics_matrix = intrinsics_matrix.clone()
            scaled_intrinsics_matrix[:, :2, :] *= resolution

            all_uv_renders, _, _ = self._render_gsplat(
                debug_gaussian_attributes[GaussianAttribute.POSITION],
                debug_gaussian_attributes[GaussianAttribute.COLOR],
                debug_gaussian_attributes[GaussianAttribute.SCALE],
                debug_gaussian_attributes[GaussianAttribute.ROTATION],
                debug_gaussian_attributes[GaussianAttribute.OPACITY],
                world2cam_matrix,
                scaled_intrinsics_matrix,
                resolution,
                override_color = uv_colors.repeat(B, 1, 1)
                )
            return all_uv_renders
        else:
            raise ValueError(f"Invalid raster backend: {self.rendering_config.raster_backend}")

    # TODO: Do we need this?
    def get_gaussian_mesh(self,
                          gaussian_attributes: Dict[GaussianAttribute, torch.Tensor],
                          idx: int = 0,
                          use_spheres: bool = True, random_colors: bool = True, ellipsoid_res: int = 5,
                          scale_factor: float = 1.5,
                          overwrite_colors: Optional[torch.Tensor] = None,
                          opacity_threshold: float = 0.01,
                          max_n_gaussians: Optional[int] = None,
                          min_scale: Optional[float] = None,
                          max_scale: Optional[float] = None,
                          include_alphas: bool = False) -> trimesh.Trimesh:
        if overwrite_colors is None:
            device = gaussian_attributes[GaussianAttribute.POSITION].device
            pose_front = Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 2.7), pose_type=PoseType.CAM_2_WORLD,
                              camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)

            gaussian_sh_ref_cam = pose_to_rendercam(pose_front, DEFAULT_INTRINSICS, 512, 512, device=device)
            sh_degree = self._config.gaussian_attribute_config.sh_degree
            shs_view = gaussian_attributes[GaussianAttribute.COLOR][idx].view(-1, (sh_degree + 1) ** 2, 3).permute(0, 2,
                                                                                                                   1)
            dir_pp = (gaussian_attributes[GaussianAttribute.POSITION][idx] - gaussian_sh_ref_cam.camera_center.repeat(1,
                                                                                                                      1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
            sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)

            gaussian_colors = torch.clamp(sh2rgb + 0.5, 0.0, 1.0)
        else:
            gaussian_colors = overwrite_colors

        gaussian_opacities = self._gaussian_model.opacity_activation(
            gaussian_attributes[GaussianAttribute.OPACITY][idx])
        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION][idx]
        gaussian_scales = self._gaussian_model.scaling_activation(gaussian_attributes[GaussianAttribute.SCALE][idx])
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION][idx]

        if min_scale is not None:
            gaussian_scales = gaussian_scales.clamp(min=min_scale)
        if max_scale is not None:
            gaussian_scales = gaussian_scales.clamp(max=max_scale)

        combined_mesh = gaussians_to_mesh(gaussian_positions, gaussian_scales, gaussian_rotations, gaussian_colors,
                                          gaussian_opacities,
                                          use_spheres=use_spheres, random_colors=random_colors,
                                          ellipsoid_res=ellipsoid_res, scale_factor=scale_factor,
                                          opacity_threshold=opacity_threshold, max_n_gaussians=max_n_gaussians,
                                          include_alphas=include_alphas)

        return combined_mesh

    def planes_to_feature_images(self, planes, idx=0):
        planes = planes[idx].detach().clone()

        image_position = planes[:3].permute(1, 2, 0)
        image_scale = planes[3:6].permute(1, 2, 0)

        sh_degree = self._config.gaussian_attribute_config.sh_degree
        num_color_ch = (sh_degree + 1) ** 2 * 3

        image_color = planes[9:9+num_color_ch] # C, H, W
        image_color = mip_tanh(image_color, overshoot=self._config.color_overshoot)
        image_color = image_color * (0.5 / C0)  # Force colors between [-1.78, 1.78]
        image_color = image_color.permute(1, 2, 0) # H, W, C

        # flattent image color
        image_color = image_color.reshape(-1, num_color_ch) # [N, SH*C]

        # obtain SH directional vectors
        pose_front = Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 2.7), pose_type=PoseType.CAM_2_WORLD,
                              camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
        pose_point = Pose(matrix_or_rotation=np.eye(3), translation=(0, 0, 1.35), pose_type=PoseType.CAM_2_WORLD,
                              camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL)
        gaussian_sh_ref_cam = pose_to_rendercam(pose_front, DEFAULT_INTRINSICS, 512, 512, device=planes.device)
        point_cam = pose_to_rendercam(pose_point, DEFAULT_INTRINSICS, 512, 512, device=planes.device)
        sh_degree = self._config.gaussian_attribute_config.sh_degree

        shs_view = image_color.view(-1, (sh_degree + 1) ** 2, 3).permute(0, 2, 1)
        dir_pp = (point_cam.camera_center.repeat(1, 1) - gaussian_sh_ref_cam.camera_center.repeat(1, 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=-1, keepdim=True)
        sh2rgb = eval_sh(sh_degree, shs_view, dir_pp_normalized)
        image_color = torch.clamp(sh2rgb + 0.5, 0.0, 1.0)
        image_color = image_color.view(planes.shape[1], planes.shape[2], 3)

        image_opacity = planes[9+num_color_ch:]
        image_opacity = image_opacity.expand(3, -1, -1).permute(1, 2, 0)

        # normalize for visualiation
        image_position = (image_position - image_position.min()) / (image_position.max() - image_position.min() + 1e-6)
        image_scale    = (image_scale - image_scale.min()) / (image_scale.max() - image_scale.min() + 1e-6)
        # image_color    = (image_color - image_color.min()) / (image_color.max() - image_color.min() + 1e-6)
        image_opacity  = (image_opacity - image_opacity.min()) / (image_opacity.max() - image_opacity.min() + 1e-6)

        return image_position, image_scale, image_color, image_opacity

    # ==========================================================
    # Main forward
    # ==========================================================

    def forward(self, z, c, flame_params, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None,
                update_emas=False, cache_backbone=False,
                use_cached_backbone=False, c2=None, inference_options: Optional[Dict[str, float]] = None, also_render_mouth=False, **synthesis_kwargs):
        # Render a batch of generated images.
        # used by metrics calculations and print_module_summary

        if self._config.gen_flame_conditioning:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          update_emas=update_emas, c2=c2, flame_params=flame_params)
        else:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                            update_emas=update_emas, flame_params=flame_params)
        
        return self.synthesis(ws, c, flame_params, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution,
                              cache_backbone=cache_backbone,
                              use_cached_backbone=use_cached_backbone, 
                              inference_options=inference_options,
                              also_render_mouth=also_render_mouth,
                              **synthesis_kwargs)

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, c2=None, flame_params=None, shape_condition_mult=1.0):
        if self.rendering_config.c_gen_conditioning_zero:
            c = torch.zeros_like(c)
        
        if c.shape[1] == 6:
            if self.c_dim == 3:
                c = c[:, :3] # print_module_summary and FID support
            else:
                from src.gghead.util.flame_rasterizer import batch_rodrigues
                R_c = batch_rodrigues(c[:, :3]).reshape(c.shape[0], 9)
                # concat c[:, 3:]
                if self._config.use_concat:
                    orth_c = 0 * c[:, 3:]
                else:
                    orth_c = c[:, 3:]
                c = torch.cat([R_c, orth_c], dim=1)

        if self._config.gen_flame_conditioning and not self._config.gfc_double_modulation and not self._config.gfc_additive_condition:

            if self.use_shape_clusters:
                if c2.shape[1] == self.c2_dim:
                    cluster_indices = torch.ones(c2.shape[0], 1, device=c2.device).long() # print_module_summary
                    print("Dummy cluster indices! Should only happen in print_module_summary") 
                else:
                    cluster_indices = c2[:, 406:407].long()
                c2 = self.shape_lookup_table(cluster_indices) * shape_condition_mult # B, 1, c2_dim
                c2 = c2.squeeze(1)
            else:
                if self.c2_dim != 406:
                    c2 = c2[:, :self.c2_dim]
                elif self.c2_dim == 53:
                    c2 = torch.cat(c2[:, 300:350], c2[:, 353:356], dim=1)

            res = self.backbone.mapping(z, c * self.rendering_config.c_scale, c2, truncation_psi=truncation_psi,
                                        truncation_cutoff=truncation_cutoff,
                                        update_emas=update_emas)
        else:
            res = self.backbone.mapping(z, c * self.rendering_config.c_scale, truncation_psi=truncation_psi,
                                  truncation_cutoff=truncation_cutoff,
                                  update_emas=update_emas)
        
        if True:
            # Extract expression parameters - no need for mapping network
            flame_params = flame_params.clone()
            expcode = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
            jawpose = flame_params[:, (self.n_shape + self.n_exp + 3):(self.n_shape + self.n_exp + 6)]

            # concat jawpose and expression parameters
            if self.deformation_generator.c2_dim == 50 + 3 + 2:
                eyelid = flame_params[:, (self.n_shape + self.n_exp + 6):(self.n_shape + self.n_exp + 8)]
                expcode = torch.cat([expcode, jawpose, eyelid], dim=1)
            else:
                expcode = torch.cat([expcode, jawpose], dim=1)

            # No need to run mapping, return expression code directly
            return res, expcode
        else:
            # deprecated
            return res
            

    def render_flame_mesh(self, flame_params, return_vertices=False, rasterize=True, 
                          zero_out_campose=True, zero_out_shape=False, zero_out_exp=False, zero_out_jawpose=False, zero_out_eyelid=False):
        flame_params = flame_params.clone()
        shapecode = flame_params[:, :self.n_shape]
        expcode = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
        posecode = flame_params[:, (self.n_shape + self.n_exp):(self.n_shape + self.n_exp + 8)]

        if zero_out_campose:
            posecode[:, :3] = 0 
        
        if zero_out_exp:
            expcode = expcode * 0
        if zero_out_shape:
            shapecode = shapecode * 0
        if zero_out_jawpose:
            posecode[:, 3:6] = 0
        if zero_out_eyelid:
            posecode[:, 6:8] = 0 

        with torch.no_grad():
            vertices, lmk2d, lmk3d, J_transformed = self.flame_lbs_model(
                    shape_params=shapecode, 
                    expression_params=expcode,
                    pose_params=posecode,
                    calc_landmarks=self._config.cut_mouth_gs_from_main_planes,
                    return_J_transformed=True
                    )
            vertices = vertices.contiguous()

            if self._config.cut_mouth_gs_from_main_planes:
                mouth_3d_lmk = lmk3d[:, 48:68]
            else:
                mouth_3d_lmk = None

            if rasterize:
                if hasattr(self._config, 'dd_shape_n_components'):
                    dd_shapecode = shapecode
                    dd_shapecode[:, min(self._config.dd_shape_n_components, shapecode.shape[1] - 1):] = 0
                else:
                    dd_shapecode = shapecode * 0
                
                raster_vertices, _, _ = self.flame_lbs_model(
                    shape_params=dd_shapecode, 
                    expression_params=expcode,
                    pose_params=posecode
                    )
                
                raster_vertices = raster_vertices.contiguous()
                images_flame = self.flame_rasterizer(raster_vertices).contiguous() * 2 - 1

                if self._config.plane_resolution != self.neural_rendering_resolution:
                    images_flame = torch.nn.functional.interpolate(images_flame, (self.neural_rendering_resolution, self.neural_rendering_resolution), mode='bilinear')
            else:
                images_flame = None


        if return_vertices:
            if self._config.use_flame_to_bfm_registration:
                vertices = self.flame_to_bfm_registration(vertices)
                if mouth_3d_lmk is not None:
                    mouth_3d_lmk = self.flame_to_bfm_registration(mouth_3d_lmk)
        
            return images_flame, vertices, mouth_3d_lmk, J_transformed
        else:
            return images_flame

    def flame_to_bfm_registration(self, vertices):
        flame_to_bfm_neutral_37_face_only = torch.tensor([
            [2.682716929557429, 0.010446918125791843, 0.04746649927210553, 0.0030014961233934233],
            [-0.009421598630294275, 2.682515580606053, -0.05790466856909523, 0.04740944787628047],
            [-0.047680604263683285, 0.05772857372357329, 2.682112266656848, -0.0024605516344398115],
            [0.0, 0.0, 0.0, 1.0]
        ], device=vertices.device)

        vertices = (to_homogeneous(vertices) @ flame_to_bfm_neutral_37_face_only.T)[..., :3]
        return vertices

    def get_uv_displacement_map(self, vertices):
        dynamic_vertices = vertices[:, self._tidx_to_idx]
        v0_map = dynamic_vertices[:, self._idxim[..., 0]]
        v1_map = dynamic_vertices[:, self._idxim[..., 1]]
        v2_map = dynamic_vertices[:, self._idxim[..., 2]]

        uv_position_map = self._barim[..., [0]] * v0_map + self._barim[..., [1]] * v1_map + self._barim[
            ..., [2]] * v2_map 
        uv_position_map = uv_position_map.float().permute(0, 3, 1, 2)
        uv_position_displacement_map = uv_position_map - self._template_uv_positions_map

        if self._config.use_extended_uv_generation:
            # pad the mouth region with zeros
            uv_padded = torch.zeros((uv_position_displacement_map.shape[0], 3, self.extended_uv_resolution, self.extended_uv_resolution), device=uv_position_displacement_map.device)
            pad_size = (self.extended_uv_resolution - self._config.plane_resolution) // 2
            uv_padded[:, :, :self._config.plane_resolution, pad_size:pad_size + self._config.plane_resolution] = uv_position_displacement_map
            mouth_interior_uv = generate_mouth_interior_uv(vertices, self.mouth_res, self.mouth_res // 2).permute(0, 3, 1, 2) # [B, 3, 2K, M]
            mouth_interior_uv_displacement = mouth_interior_uv - self._template_uv_positions_map_mouth_interior

            pad_size_mouth = (self.extended_uv_resolution - self.mouth_res) // 2
            uv_padded[:, :, self._config.plane_resolution:self._config.plane_resolution + self.mouth_res, pad_size_mouth:pad_size_mouth + self.mouth_res] = mouth_interior_uv_displacement
            uv_position_displacement_map = uv_padded
        return uv_position_displacement_map

    def synthesis(self, ws, c, flame_params, neural_rendering_resolution=None, update_emas=False, cache_backbone=False,
                  use_cached_backbone=False,
                  return_raw_attributes: bool = False, return_uv_map: bool = False,
                  alpha_plane_resolution: Optional[float] = None,
                  return_masks: bool = False,
                  sh_ref_cam: Optional[Pose] = None,
                  also_render_mouth: bool = False,
                  return_deformation_planes: bool = False,
                  inference_type: Optional[str] = None,
                  inference_options: Optional[Dict[str, float]] = None,
                  blendshape_deform_planes: Optional[Dict[str, torch.Tensor]] = None,
                  post_act_blendshapes: Optional[Dict[str, torch.Tensor]] = None,
                  return_gaussian_attributes: bool = False,
                  **synthesis_kwargs) -> GGHeadOutput:
        if self._config.use_mouth_branch:
            main_ws, mouth_ws = ws  # expcode is the raw expression parameters
            ws = main_ws
        elif self._config.use_deformation_branch:
            main_ws, deformation_ws = ws  # expcode is the raw expression parameters
            # if self._config.use_concat:
            #     deformation_ws *= 0
            flame_ws = deformation_ws
            ws = self.backbone.prepare_ws(flame_ws, main_ws, zero_flame_ws=True)
        elif self._config.use_extended_uv_generation and not self._config.use_deformation_branch:
            main_ws, expcode = ws  # expcode is the raw expression parameters
            zero_flame_ws = not self._config.gen_flame_conditioning
            ws = self.backbone.prepare_ws(expcode, main_ws, zero_flame_ws=zero_flame_ws)


        if c.shape[1] == 25:
            cam2world_matrix = c[:, :16].view(-1, 4, 4)
            intrinsics_matrix = c[:, 16:25].view(-1, 3, 3)
        elif c.shape[1] == 6:
            # vert_scale = c[:, 3:4].unsqueeze(1) # [B, 1, 1]
            vert_scale = None
        else:
            raise Exception("Sanity check c_dim:", c.shape[1])
        
        if inference_type == 'static_flame' or inference_type == 'static_flame_w_main_offsets':
            flame_params = flame_params.clone()
            saved_flame_params = flame_params.clone()
            flame_params *= 0

        images_flame, vertices, mouth_3d_lmk, J_transformed = self.render_flame_mesh(flame_params, return_vertices=True, rasterize=self._config.use_flame_rasterization)

        if inference_type == 'static_flame' or inference_type == 'static_flame_w_main_offsets':
            flame_params = saved_flame_params

        if c.shape[1] == 6:
            world2cam_matrix, intrinsics_matrix = parse_flame_deca_cameras(c, J_transformed)
        
        
        if self._config.gfc_additive_condition or self._config.use_spade or self._config.use_concat:
            if not self._config.double_condition:
                if self._config.use_gaussian_blendshape:
                    if use_cached_backbone:
                        uv_position_displacement_map = self._last_uv_position_displacement_map
                    else:
                        _, vertices_shapecode, _, _ = self.render_flame_mesh(flame_params, return_vertices=True, rasterize=False, zero_out_exp=True, zero_out_jawpose=True, zero_out_eyelid=True)
                        uv_position_displacement_map = self.get_uv_displacement_map(vertices_shapecode)

                    if cache_backbone:
                        self._last_uv_position_displacement_map = uv_position_displacement_map
                elif self._config.use_deformation_branch:
                    if use_cached_backbone:
                        uv_position_displacement_map = self._last_uv_position_displacement_map
                    else:
                        _, vertices_shapecode, _, _ = self.render_flame_mesh(flame_params, return_vertices=True, rasterize=False, zero_out_exp=True, zero_out_jawpose=True, zero_out_eyelid=True)
                        uv_position_displacement_map = self.get_uv_displacement_map(vertices_shapecode)
                    
                    if cache_backbone:
                        self._last_uv_position_displacement_map = uv_position_displacement_map
                else:
                    uv_position_displacement_map = self.get_uv_displacement_map(vertices)
                
                main_uv_position_displacement_map = uv_position_displacement_map
                deform_uv_position_displacement_map = None
            else:
                _, vertices_shapecode, _, _ = self.render_flame_mesh(flame_params, return_vertices=True, rasterize=False, zero_out_exp=True, zero_out_jawpose=True, zero_out_eyelid=True)
                main_uv_position_displacement_map = self.get_uv_displacement_map(vertices_shapecode)

                _, vertices_expcode, _, _ = self.render_flame_mesh(flame_params, return_vertices=True, rasterize=False, zero_out_shape=True)
                deform_uv_position_displacement_map = self.get_uv_displacement_map(vertices_expcode)


        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Get target resolution for feature extraction
        extract_features_resolution = None
        if self._config.use_mouth_branch:
            extract_features_resolution = self._config.mouth_start_resolution
        if self._config.use_deformation_branch:
            extract_features_resolution = self._config.deformation_start_resolution

        # Predict planes with intermediate features if needed
        if extract_features_resolution:
            planes, features_at_res, img_at_res = self.predict_planes(
                ws, update_emas=update_emas, cache_backbone=cache_backbone,
                use_cached_backbone=use_cached_backbone,
                alpha_plane_resolution=alpha_plane_resolution, 
                flame_params=flame_params,
                noise_cond=uv_position_displacement_map if 'uv_position_displacement_map' in locals() else None,
                extract_features_at_resolution=extract_features_resolution, 
                condition_map=main_uv_position_displacement_map if 'main_uv_position_displacement_map' in locals() else None,
                **synthesis_kwargs)
        else:
            planes = self.predict_planes(ws, update_emas=update_emas, cache_backbone=cache_backbone,
                                      use_cached_backbone=use_cached_backbone,
                                      alpha_plane_resolution=alpha_plane_resolution, 
                                      flame_params=flame_params,
                                      noise_cond=uv_position_displacement_map if 'uv_position_displacement_map' in locals() else None,
                                      condition_map=main_uv_position_displacement_map if 'main_uv_position_displacement_map' in locals() else None,
                                      **synthesis_kwargs)

        if self._config.use_background_cnn:
            background_rgb = planes[:, -self._config.n_background_channels:]

            if self._config.use_background_upsampler:
                x = background_rgb
                bg_img = background_rgb[:,
                         :3]  # First 3 channels of background plane tensor have special meaning: they are already the low-res background
                for block in self._background_upsampling_blocks:
                    x, bg_img = block(x, bg_img, ws[:, :block.num_conv + block.num_torgb])
                background_rgb = bg_img

            if neural_rendering_resolution > background_rgb.shape[-1]:
                background_rgb = torch.nn.functional.interpolate(background_rgb, (
                    neural_rendering_resolution, neural_rendering_resolution), mode='bilinear')

            background_rgb = mip_tanh(background_rgb)
            planes = planes[:, :-self._config.n_background_channels]
        else:
            background_rgb = None

        if self._config.use_gaussian_blendshape:
            blendshape_planes = planes[:, -self.total_blendshape_channels:]
            planes = planes[:, :-self.total_blendshape_channels]

            blendshape_planes = blendshape_planes.view(planes.shape[0], self.k_blendshapes, self.per_blendshape_channels, planes.shape[-2], planes.shape[-1])

            # pad zero color channels to match the number of channels in the planes
            # include opacity
            n_color_channels = GaussianAttribute.COLOR.get_n_channels(self._config.gaussian_attribute_config) \
                + GaussianAttribute.OPACITY.get_n_channels(self._config.gaussian_attribute_config)
            
            padding = torch.zeros(planes.shape[0], self.k_blendshapes, n_color_channels, planes.shape[-2], planes.shape[-1], device=planes.device, dtype=planes.dtype)
            blendshape_planes = torch.cat([blendshape_planes, padding], dim=2)

            residuals = blendshape_planes - planes.unsqueeze(1)

            # zero out color residuals
            residuals[:, :, -n_color_channels:] = 0

            expcode = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
            jawpose = flame_params[:, (self.n_shape + self.n_exp + 3):]

            expcode = expcode[:, :self.expcode_blendshape_size]

            coeffs = torch.cat([expcode, jawpose], dim=-1)

            weighted_residuals = residuals * coeffs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            planes = planes + weighted_residuals.sum(dim=1)

        if self._config.use_mouth_branch:
            mouth_planes = self.mouth_generator(
                mouth_ws,                  # Pass expression parameters directly 
                feature_maps=features_at_res,
                img_rgb=img_at_res,
                update_emas=update_emas,
                ws_main=main_ws,
                **synthesis_kwargs
            )
        
        if self._config.use_deformation_branch and not self._config.disable_deformation_branch and post_act_blendshapes is None:
            if blendshape_deform_planes is not None:
                from src.gghead.util.flame_rasterizer import batch_rodrigues as _bs_rodrigues
                _bs_base = blendshape_deform_planes['base_residual_plane']
                _bs_deltas = blendshape_deform_planes['delta_planes']
                _bs_exp = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
                _bs_jaw3 = flame_params[:, (self.n_shape + self.n_exp + 3):(self.n_shape + self.n_exp + 6)]
                _bs_jaw_rot = _bs_rodrigues(_bs_jaw3)
                _bs_eye3 = torch.eye(3, device=_bs_jaw3.device, dtype=_bs_jaw3.dtype).unsqueeze(0)
                _bs_jaw_res = (_bs_jaw_rot - _bs_eye3).reshape(-1, 9)
                _bs_coeffs = torch.cat([_bs_exp, _bs_jaw_res], dim=-1)
                deformation_planes = _bs_base + torch.einsum('bi,ichw->bchw', _bs_coeffs, _bs_deltas)
            else:
                deformation_planes = self.deformation_generator(
                    deformation_ws,
                    feature_maps=features_at_res,
                    img_rgb=img_at_res,
                    update_emas=update_emas,
                    ws_main=main_ws,
                    condition_map=deform_uv_position_displacement_map if 'deform_uv_position_displacement_map' in locals() else None,
                    **synthesis_kwargs
                )

            # zero out color residuals
            if self._config.zero_out_color_residuals:
                deformation_planes[:, self._color_start_channel:self._color_start_channel + self._n_color_channels] = 0

            if self._config.zero_out_opacity_residuals:
                opacity_start_channel = self._color_start_channel + self._n_color_channels
                deformation_planes[:, opacity_start_channel:opacity_start_channel + 1] = 0

            if self._config.use_deform_mask:
                deformation_planes[:, :, :, :] *= self._deform_mask

            if return_deformation_planes:
                main_planes = planes.clone()

            if inference_options is not None and inference_options['override_color'] != None:
                inference_options = dict(inference_options)
                if inference_options['override_color'] == 'deform_offsets':
                    _planes = deformation_planes.clone()
                elif inference_options['override_color'] == 'main_offsets':
                    _planes = planes.clone()
                else:
                    raise Exception(f"Invalid override color: {inference_options['override_color']}")

                if self._config.use_extended_uv_generation:
                    main_padded, mouth_padded = torch.split(_planes, [self._config.plane_resolution, self.mouth_res], dim=2)
                    pad_size = (self.extended_uv_resolution - self._config.plane_resolution) // 2
                    pad_size_mouth = (self.extended_uv_resolution - self.mouth_res) // 2
                    _planes = main_padded[:, :, :, pad_size:pad_size + self._config.plane_resolution]
                    mouth_planes = mouth_padded[:, :, :, pad_size_mouth:pad_size_mouth + self.mouth_res]
                    mouth_uv_attributes, _ = self.sample_uv_map(mouth_planes, self._mouth_uv_grid, zero_conv_position_layer=self._mouth_zero_conv_position)
                    uv_attributes, _ = self.sample_uv_map(_planes, self._uv_grid, zero_conv_position_layer=self._zero_conv_position)
                else:
                    uv_attributes, _ = self.sample_uv_map(_planes, self._uv_grid, zero_conv_position_layer=self._zero_conv_position)
                
                if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
                    uv_attributes = torch.cat([uv_attributes, mouth_uv_attributes], dim=1)
                else:
                    uv_attributes = uv_attributes
                
                xyz_offsets = self._apply_position_activation(uv_attributes[:, :, :3].clone())
                inference_options['override_color'] = (xyz_offsets - xyz_offsets.min()) / (xyz_offsets.max() - xyz_offsets.min()) * 2 - 1
                # xyz_offsets = self._apply_position_activation(uv_attributes[..., :3].float())
                # m = xyz_offsets.norm(dim=-1, keepdim=True)
                # mn, mx = m.min(), m.max()
                # s = (m - mn) / (mx - mn + 1e-8)
                # inference_options['override_color'] = torch.cat([1 - s, s, torch.zeros_like(s)], dim=-1) * 2 - 1

            if self._config.use_extended_uv_generation:
                deform_planes_wo_mouth, _ = torch.split(deformation_planes, [self._config.plane_resolution, self.mouth_res], dim=2)
            else:
                # these are just for deform planes regularization
                deform_planes_wo_mouth = deformation_planes

            if inference_type in ['static_flame', 'static_flame_w_main_offsets', 'dynamic_flame', 'dynamic_flame_w_main_offsets']:
                deform_planes_wo_mouth = None

        if inference_type in ['static_flame', 'dynamic_flame']:
            planes[:, :3] = 0

        if self._config.use_extended_uv_generation:
            main_padded, mouth_padded = torch.split(planes, [self._config.plane_resolution, self.mouth_res], dim=2)
            pad_size = (self.extended_uv_resolution - self._config.plane_resolution) // 2
            pad_size_mouth = (self.extended_uv_resolution - self.mouth_res) // 2

            planes = main_padded[:, :, :, pad_size:pad_size + self._config.plane_resolution]

            mouth_planes = mouth_padded[:, :, :, pad_size_mouth:pad_size_mouth + self.mouth_res]

        if post_act_blendshapes is not None:
            from src.gghead.util.flame_rasterizer import batch_rodrigues as _pa_rodrigues
            _pa_exp = flame_params[:, self.n_shape:(self.n_shape + self.n_exp)]
            _pa_jaw3 = flame_params[:, (self.n_shape + self.n_exp + 3):(self.n_shape + self.n_exp + 6)]
            _pa_jaw_rot = _pa_rodrigues(_pa_jaw3)
            _pa_I3 = torch.eye(3, device=_pa_jaw3.device, dtype=_pa_jaw3.dtype).unsqueeze(0)
            _pa_jaw_res = (_pa_jaw_rot - _pa_I3).reshape(-1, 9)
            _pa_coeffs = torch.cat([_pa_exp, _pa_jaw_res], dim=-1)  # [B, 59]

            _pa_attr_keys = [
                ('xyz', GaussianAttribute.POSITION),
                ('scale', GaussianAttribute.SCALE),
                ('rotation', GaussianAttribute.ROTATION),
                ('opacity', GaussianAttribute.OPACITY),
                ('color', GaussianAttribute.COLOR),
            ]

            # Linear combination in physical (post-all-activations) space
            _pa_phys = {}
            for key_str, attr_enum in _pa_attr_keys:
                _base = post_act_blendshapes[f'base_{key_str}']   # [1, G, C]
                _delta = post_act_blendshapes[f'delta_{key_str}'] # [59, G, C]
                #_delta[1:, :, :3] = 0 # zero out all xyz of >= 1 frames 
                _pa_phys[attr_enum] = _base + torch.einsum('bi,igc->bgc', _pa_coeffs, _delta)

            # Enforce geometric constraints in physical space
            _pa_phys[GaussianAttribute.SCALE] = torch.clamp(
                _pa_phys[GaussianAttribute.SCALE], min=1e-5)
            _pa_phys[GaussianAttribute.ROTATION] = torch.nn.functional.normalize(
                _pa_phys[GaussianAttribute.ROTATION], p=2, dim=-1)
            _ov_clamp = self._config.opacity_overshoot
            _opa_min = -_ov_clamp + 1e-6
            _opa_max = 1.0 + _ov_clamp - 1e-6
            _pa_phys[GaussianAttribute.OPACITY] = torch.clamp(
                _pa_phys[GaussianAttribute.OPACITY], min=_opa_min, max=_opa_max)

            # Convert physical values back to pre-renderer-activation space
            # (the renderer applies exp/sigmoid/normalize internally)
            _ov = self._config.opacity_overshoot
            _opa_pre = torch.logit(
                (_pa_phys[GaussianAttribute.OPACITY] + _ov) / (1.0 + 2.0 * _ov))

            gaussian_attributes = {
                GaussianAttribute.POSITION: _pa_phys[GaussianAttribute.POSITION],
                GaussianAttribute.SCALE: torch.log(_pa_phys[GaussianAttribute.SCALE]),
                GaussianAttribute.ROTATION: _pa_phys[GaussianAttribute.ROTATION],
                GaussianAttribute.OPACITY: _opa_pre,
                GaussianAttribute.COLOR: _pa_phys[GaussianAttribute.COLOR],
            }

            gaussian_attribute_output = GaussianAttributeOutput(
                gaussian_attributes=gaussian_attributes)
        else:
            gaussian_attribute_output = self.predict_gaussian_attributes(
                 planes,
                 vertices,
                 return_raw_attributes=return_raw_attributes,
                 return_uv_map=return_uv_map,
                 mouth_planes=mouth_planes if self._config.use_mouth_branch or self._config.use_extended_uv_generation else None,
                 mouth_3d_lmk=mouth_3d_lmk if 'mouth_3d_lmk' in locals() else None,
                 inference_options=inference_options,
                 deform_planes=deform_planes_wo_mouth if 'deform_planes_wo_mouth' in locals() else None
             )
        
        gaussian_attributes = gaussian_attribute_output.gaussian_attributes

        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION]
        gaussian_scales = gaussian_attributes[GaussianAttribute.SCALE]
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION]
        gaussian_opacities = gaussian_attributes[GaussianAttribute.OPACITY]
        gaussian_colors = gaussian_attributes[GaussianAttribute.COLOR]  # [B, G, SH*3]
        B = len(c)
        G = gaussian_colors.shape[1]
        C = 3
        gaussian_colors = gaussian_colors.view(B, G, -1, C)

        # Gradient logging
        if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
            num_mouth_vertices = self._mouth_uv_grid.shape[1]
        else:
            num_mouth_vertices = None
        
        for attribute_name, attribute_value in gaussian_attributes.items():
            self._log_gradients(attribute_name, attribute_value, num_mouth_vertices=num_mouth_vertices)

        # mouth regularizations
        if self._config.use_mouth_branch and self._config.cut_mouth_gs_from_main_planes:
            num_mouth_vertices = self._mouth_uv_grid.shape[1]

            min_extent = mouth_3d_lmk.min(dim=1).values
            max_extent = mouth_3d_lmk.max(dim=1).values
            mouth_size = max_extent - min_extent

            extend_ratio = 0.1
            min_extent = min_extent - mouth_size * extend_ratio
            max_extent = max_extent + mouth_size * extend_ratio

            # clamp to mouth aabb cube
            min_extent = min_extent.unsqueeze(1)  # Shape becomes [B, 1, 3]
            max_extent = max_extent.unsqueeze(1)  # Shape becomes [B, 1, 3]
            main_mouth_mask = torch.all(torch.logical_and(gaussian_positions > min_extent, gaussian_positions < max_extent), dim=-1) # [B, N]
            
            # don't include gaussians from mouth planes 
            # print('With mouth', main_mouth_mask.float().sum(dim=-1).mean())
            main_mouth_mask[:, -num_mouth_vertices:] = False
            # print('Just main', main_mouth_mask.float().sum(dim=-1).mean())
            
            # zero out mouth gaussians of main planes, we compensate them via mouth planes.
            gaussian_opacities[main_mouth_mask] = 0

        # Rasterization

        if (self._config.use_mouth_branch or self._config.use_extended_uv_generation) and also_render_mouth:
            rgb_images_mouth = []
            rgb_images_wo_mouth = []
        
        rgb_images = []
        masks = []

        # rendering loop signature
        # inputs: ...
        # outputs: rgb_images, [masks, rgb_images_mouth, rgb_images_wo_mouth]

        if inference_options is not None and inference_options['override_color'] is not None:
            override_color = inference_options['override_color']
        else:
            override_color = None

        rgb_images, masks, rgb_images_mouth, rgb_images_wo_mouth = self.render_gs_batch(
            gaussian_positions,
            gaussian_colors,
            gaussian_scales,
            gaussian_rotations,
            gaussian_opacities,
            sh_ref_cam,
            J_transformed,
            cam2world_matrix if 'cam2world_matrix' in locals() else None,
            world2cam_matrix if 'world2cam_matrix' in locals() else None,
            intrinsics_matrix,
            neural_rendering_resolution,
            return_masks,
            also_render_mouth,
            background_rgb,
            raster_backend=self.rendering_config.raster_backend,
            override_color=override_color,
            )
        #### End of rendering loop

        rgb_images_raw = rgb_images
        
        if self._config.log_feature_maps:
            images_features = self.planes_to_feature_images(planes)
            if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
                images_features_mouth = self.planes_to_feature_images(mouth_planes)
            else:
                images_features_mouth = None
        else:
            images_features = None
            images_features_mouth = None

        output = GGHeadOutput(rgb_images, rgb_images_raw, rgb_images,
                              gaussian_attribute_output=gaussian_attribute_output,
                              masks=masks, images_flame=images_flame, images_mouth=rgb_images_mouth,
                              images_wo_mouth=rgb_images_wo_mouth,
                              images_features=images_features, images_features_mouth=images_features_mouth,
                              J_transformed=J_transformed
                              )

        if return_gaussian_attributes:
            output.returned_gaussian_attributes = gaussian_attribute_output.gaussian_attributes

        if return_deformation_planes:
            return output, \
                [torch.hstack(self.planes_to_feature_images(main_planes, idx)) for idx in range(main_planes.shape[0])], \
                [torch.hstack(self.planes_to_feature_images(deformation_planes, idx)) for idx in range(deformation_planes.shape[0])] 
        
        return output

    def _setup_gaussian_model(self, gaussian_attributes: Dict[GaussianAttribute, torch.Tensor],
                              i: int) -> GaussianModel:
        gaussian_positions = gaussian_attributes[GaussianAttribute.POSITION]
        gaussian_scales = gaussian_attributes[GaussianAttribute.SCALE]
        gaussian_rotations = gaussian_attributes[GaussianAttribute.ROTATION]
        gaussian_opacities = gaussian_attributes[GaussianAttribute.OPACITY]
        gaussian_colors = gaussian_attributes[GaussianAttribute.COLOR]  # [B, G, SH*3]
        B, G, _ = gaussian_colors.shape
        gaussian_colors = gaussian_colors.view(B, G, -1, 3)

        self._gaussian_model._xyz = gaussian_positions[i]
        self._gaussian_model._features_dc = gaussian_colors[i][:, [0]]
        self._gaussian_model._features_rest = gaussian_colors[i][:, 1:]  # [G, SH-1, 3]
        self._gaussian_model._scaling = gaussian_scales[i]
        self._gaussian_model._rotation = gaussian_rotations[i].contiguous()  # Important: Rotation needs to be contiguous!
        self._gaussian_model._opacity = gaussian_opacities[i]

        return self._gaussian_model

    def _collect_gaussian_attributes(self, attribute_names: List[GaussianAttribute], predictions: torch.Tensor, vertices : torch.Tensor,
                                     return_raw_attributes: bool = False, mouth_3d_lmk: torch.Tensor = None, inference_options: Optional[Dict[str, float]] = None, template_pos_displacement = None,
                                     deform_predictions: Optional[torch.Tensor] = None) \
            -> Tuple[Dict[GaussianAttribute, torch.Tensor], Dict[GaussianAttribute, torch.Tensor]]:
        gaussian_attributes = dict()
        raw_gaussian_attributes = dict()
        c = 0
        for attribute_name in attribute_names:
            n_channels = attribute_name.get_n_channels(self._config.gaussian_attribute_config)
            attribute_map = predictions[..., c: c + n_channels]  # Slice corresponding channels from sampled plane
            deform_map = None
            if deform_predictions is not None:
                deform_map = deform_predictions[..., c: c + n_channels]
            if return_raw_attributes:
                raw_gaussian_attributes[attribute_name] = attribute_map

            if self.training and self._logger_bundle is not None:
                self._logger_bundle.log_metrics({
                    f"Analyze/norm_raw_{attribute_name}": attribute_map.norm(dim=-1).mean(),
                    f"Analyze/max_raw_{attribute_name}": attribute_map.max(),
                    f"Analyze/min_raw_{attribute_name}": attribute_map.min(),
                })

            activated = self._apply_gaussian_attribute_activation(attribute_name, attribute_map, vertices, mouth_3d_lmk, inference_options=inference_options, template_pos_displacement=template_pos_displacement,
                                                                    deform_value=deform_map)
            gaussian_attributes[attribute_name] = activated
            c += n_channels

        return gaussian_attributes, raw_gaussian_attributes

    # ==========================================================
    # Activations
    # ==========================================================

    def _apply_gaussian_attribute_activation(self, attribute_name: GaussianAttribute, value: torch.Tensor, vertices : torch.Tensor, mouth_3d_lmk : torch.Tensor, inference_options: Optional[Dict[str, float]] = None, template_pos_displacement : torch.Tensor = None, deform_value: Optional[torch.Tensor] = None) -> torch.Tensor:

        B = value.shape[0]

        # POSITION
        if attribute_name == GaussianAttribute.POSITION:
            main_offsets = self._apply_position_activation(value)
            deform_offsets = torch.zeros_like(main_offsets) if deform_value is None else self._apply_position_activation(deform_value)
            deform_offsets = deform_offsets * 0.1
            
            # value = value * 0; print('DEBUGGING')

            # initial vertices as 3d indices, reindex them via uv indices 
            ### TODO: тут точно нет бага? Проверить чз логгинг texture maps, position должны быть гладкими
            dynamic_vertices = vertices[:, self._tidx_to_idx]

            # sample points on displaced mesh

            if self._config.use_mouth_branch or self._config.use_extended_uv_generation:
                # uv_grid = torch.cat([self._uv_grid, self._mouth_uv_grid], dim=1)
                # vertices_3d = uv_to_3d(uv_grid, dynamic_vertices, self._idxim, self._barim, self._config.use_align_corners, self._config.interpolation_mode)

                # sample head gaussians on the FLAME surface
                vertices_3d = uv_to_3d(self._uv_grid, dynamic_vertices, self._idxim, self._barim, self._config.use_align_corners, self._config.interpolation_mode)

                # sample mouth gaussians on the MOUTH surface of TEMPLATE VERTICES
                # it takes static vertices in idx, not tidx
                mouth_interior_uv = generate_mouth_interior_uv(vertices, self._config.mouth_n_vertices, self._config.mouth_n_vertices // 2) # [B, 2K, M, 3]

                # sample mouth_interior_uv wrt to PREDICTED DISPLACEMENTS OF TEMPLATE VERTICES 
                # dynamic_vertices_displaced = dynamic_vertices + self._apply_position_activation(template_pos_displacement)
                # dynamic_vertices_displaced_idx = dynamic_vertices_displaced[:, self._idx_to_tidx] 
                # mouth_interior_uv = generate_mouth_interior_uv(dynamic_vertices_displaced_idx, self._config.mouth_n_vertices, self._config.mouth_n_vertices // 2) # [B, 2K, M, 3]

                mouth_interior_uv = mouth_interior_uv.permute(0, 1, 3, 2).permute(0, 2, 1, 3) # [B, 3, 2K, M]

                # uv_map = uv_map.reshape(B * S, C_uv, H_f, W_f)  # [B*S, UV, H_f, W_f]

                mouth_vertices_3d = grid_sample(mouth_interior_uv, self._mouth_uv_grid.repeat(B, 1, 1, 1),
                                            align_corners=self._config.use_align_corners,
                                            mode=self._config.interpolation_mode, padding_mode='border')  # [B, 3, G, 1]
                mouth_vertices_3d = mouth_vertices_3d.squeeze(3).permute(0, 2, 1)  # [B, G, 3]

                # concatenate head and mouth vertices
                vertices_3d = torch.cat([vertices_3d, mouth_vertices_3d], dim=1)    
            else:
                vertices_3d = uv_to_3d(self._uv_grid, dynamic_vertices, self._idxim, self._barim, self._config.use_align_corners, self._config.interpolation_mode)

            value = vertices_3d + (main_offsets + deform_offsets)

            if self._config.use_mouth_branch and self._config.mouth_gs_clipping:
                num_mouth_vertices = self._mouth_uv_grid.shape[1]
                mouth_vertices = value[:, -num_mouth_vertices:]

                min_extent = mouth_3d_lmk.min(dim=1).values
                max_extent = mouth_3d_lmk.max(dim=1).values
                mouth_size = max_extent - min_extent

                extend_ratio = 0.25
                min_extent = min_extent - mouth_size * extend_ratio
                max_extent = max_extent + mouth_size * extend_ratio

                # clamp to mouth aabb cube
                min_extent = min_extent.unsqueeze(1)  # Shape becomes [B, 1, 3]
                max_extent = max_extent.unsqueeze(1)  # Shape becomes [B, 1, 3]

                # naive. bad, since no gradients for out of boundary points.
                # vertices_3d[:, -num_mouth_vertices:] = torch.min(torch.max(mouth_vertices, min_extent), max_extent)

                # softclamping, gradients could be near zero
                value[:, -num_mouth_vertices:] = min_extent + torch.nn.functional.softplus(mouth_vertices - min_extent, beta=100)
                value[:, -num_mouth_vertices:] = max_extent - torch.nn.functional.softplus(max_extent - value[:, -num_mouth_vertices:], beta=100)

                # Straight Through estimation, tricky one, but saves the gradients
                # vertices_3d[:, -num_mouth_vertices:] = mouth_vertices + torch.min(torch.max(mouth_vertices, min_extent), max_extent).detach() - mouth_vertices.detach()
                    

            # value already holds base + combined offsets

        # SCALE
        elif attribute_name == GaussianAttribute.SCALE:
            # These scale activations are working around 3DGS's default exp() activation for scaling
            def activate_scale(x: torch.Tensor) -> torch.Tensor:
                x = -(x + self._config.scale_offset)
                if self._config.center_scale_activation:
                    return self._config.max_scale - torch.nn.functional.softplus(x + self._config.max_scale)
                else:
                    return self._config.max_scale - torch.nn.functional.softplus(x)

            main_pre = value
            main_act = activate_scale(main_pre)

            if deform_value is not None:    
                deform_act = activate_scale(deform_value)
            else:
                deform_act = torch.zeros_like(main_pre)

            # inference_options override for fixed scales 
            if inference_options is not None and isinstance(inference_options, dict):
                if 'fixed_scale' in inference_options and inference_options['fixed_scale'] is not None:
                    fixed_scale = float(inference_options['fixed_scale'])
                    main_act = torch.ones_like(main_act) * fixed_scale
                    deform_act = torch.zeros_like(deform_act)

            value = main_act + deform_act 

        # ROTATION
        elif attribute_name == GaussianAttribute.ROTATION:
            main_pre = value
            deform_pre = torch.zeros_like(main_pre) if deform_value is None else deform_value

            def activate_rotation(x: torch.Tensor, aa_cap = 2 * torch.pi) -> torch.Tensor:
                if self._config.gaussian_attribute_config.use_rodriguez_rotation:
                    if self._config.use_rotation_activation:
                        x = torch.tanh(self._config.rotation_attenuation * x) * aa_cap
                    x = axis_angle_to_quaternion(x)
                    # x = torch.cat([x[..., [3]], x[..., :3]], dim=-1)  # xyzr -> rxyz ???BUG???
                elif self._config.normalize_quaternions:
                    x = x / x.norm(dim=2).unsqueeze(2)
                return x

            main_act = activate_rotation(main_pre, aa_cap=torch.pi)
            deform_act = activate_rotation(deform_pre, aa_cap=torch.pi)

            value = quat_mult(deform_act, main_act).contiguous()
            # value = (main_act + deform_act).contiguous()  # Important: Rotation needs to be contiguous!

        # COLOR
        elif attribute_name == GaussianAttribute.COLOR:
            main_value = self._apply_color_activation(value)
            if deform_value is None or self._config.zero_out_color_residuals:
                value = main_value
            else:
                deform_value = self._apply_color_activation(deform_value)
                value = main_value + deform_value
        
        elif attribute_name == GaussianAttribute.OPACITY:
            main_value = value
            if deform_value is None or self._config.zero_out_opacity_residuals:
                value = main_value
            else:
                value = main_value + deform_value

            # inference_options override for fixed opacity (post-activation expectation: 1.0 maps to fully opaque pre-activation large value)
            if inference_options is not None and isinstance(inference_options, dict):
                if 'fixed_opacity' in inference_options and inference_options['fixed_opacity'] is not None:
                    fixed_opacity = float(inference_options['fixed_opacity'])
                    value = torch.ones_like(value) * fixed_opacity

        return value

    def _apply_position_activation(self, value: torch.Tensor) -> torch.Tensor:
        # Dividing by some value before tanh() is super important. Otherwise, tanh() saturates super quickly at -1 or 1
        if self._config.use_position_activation:
            value = self._config.position_range * torch.tanh(self._config.position_attenuation * value)
        else:
            value = self._config.position_attenuation * value
        return value

    def _apply_opacity_activation(self, value: torch.Tensor) -> torch.Tensor:
        return mip_sigmoid(value, overshoot=self._config.opacity_overshoot, clamp=self._config.clamp_opacity)

    def _apply_color_activation(self, value: torch.Tensor) -> torch.Tensor:
        if self._config.use_color_activation:
            color_value = value[..., :3]  # First 3 channels are always color values
            color_value = mip_tanh(color_value, overshoot=self._config.color_overshoot)
            color_value = color_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

            # TODO: SH bands have the same scaling as color bands
            sh_value = value[..., 3:]
            sh_value = mip_tanh(sh_value, overshoot=self._config.color_overshoot)
            sh_value = sh_value * (0.5 / C0)  # Force colors between [-1.78, 1.78]

            value = torch.cat([color_value, sh_value], dim=-1)

        return value

    def _apply_cnn_color_activation(self, value: torch.Tensor) -> torch.Tensor:
        return mip_tanh2(value, clamp=True)

    def _apply_scale_activation(self, value: torch.Tensor) -> torch.Tensor:
        # Scales should be between [0, 0.05]
        value = value + np.exp(self._config.scale_offset)
        return mip_sigmoid(value, overshoot=self._config.scale_overshoot) * np.exp(self._config.max_scale)