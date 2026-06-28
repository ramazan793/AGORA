from typing import Optional

import numpy as np
import torch
from eg3d.torch_utils import misc
from eg3d.torch_utils.ops import upfirdn2d
from src.gghead.eg3d.training.networks_stylegan2 import MappingNetwork, SynthesisLayer, ToRGBLayer, Conv2dLayer
from torch import nn


class GGHSynthesisBlock(nn.Module):

    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of output color channels.
                 is_last,  # Is this the last block?
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last memory format with FP16?
                 fused_modconv_default=True,  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
                 use_spade=False,
                 use_concat=False,
                 condition_nc=3,
                 **layer_kwargs,  # Arguments for SynthesisLayer.
                 ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                                        resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, use_spade=use_spade, use_concat=use_concat, condition_nc=condition_nc, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last, use_spade=use_spade, use_concat=use_concat, condition_nc=condition_nc, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                                    conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                                    resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, alpha_new_layers: float = 1, condition_map=None, **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        if condition_map is not None:
            condition_map = condition_map.to(dtype=dtype)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, condition_map=condition_map, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, condition_map=condition_map, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), condition_map=condition_map, **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, condition_map=condition_map, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, condition_map=condition_map, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            if alpha_new_layers is not None:
                y = alpha_new_layers * y  # Potentially lower contribution of output map if it comes from a newly introduced layer after progressive growing
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

class GGHSynthesisBlockNoUp(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution,
                conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, alpha_new_layers: float = 1, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        # if img is not None:
            # misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            # img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            if alpha_new_layers is not None:
                y = alpha_new_layers * y  # Potentially lower contribution of output map if it comes from a newly introduced layer after progressive growing

            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

class GGHSynthesisNetwork(nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 pretrained_plane_resolution: Optional[int] = None,  # For progressive Growing
                 start_res=4,  # Starting resolution for the generator 
                 use_spade=False,
                 use_concat=False,
                 condition_nc=3,
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        if start_res == 4:
            assert img_resolution >= start_res and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.pretrained_plane_resolution = pretrained_plane_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_resolution_log2_pretrained = int(np.log2(pretrained_plane_resolution)) if pretrained_plane_resolution is not None else self.img_resolution_log2
        if num_fp16_res > 0:
            # If new layers are added and the previous last n layers had fp16, those should still have fp16 in addition to the new layers that come after
            num_fp16_res += (self.img_resolution_log2 - self.img_resolution_log2_pretrained)
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [start_res * 2 ** (i - 2) for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            is_new_layer = pretrained_plane_resolution is not None and res > pretrained_plane_resolution
            in_channels = channels_dict[res // 2] if res > start_res else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution) or (pretrained_plane_resolution is not None and res == pretrained_plane_resolution)
            block = GGHSynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                                      img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, use_spade=use_spade, use_concat=use_concat, condition_nc=condition_nc, **block_kwargs)
            if is_new_layer:
                # Initialize new layers with 0 torgb, to not disturb the lower resolution output in the beginning
                block.torgb.weight.data.zero_()

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, alpha_new_layers: float = 1, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            if self.pretrained_plane_resolution is not None and res > self.pretrained_plane_resolution:
                x, img = block(x, img, cur_ws, alpha_new_layers=alpha_new_layers, **block_kwargs)
            else:
                x, img = block(x, img, cur_ws, **block_kwargs)
        return img
    
    def forward_with_intermediate_features(self, ws, target_resolution=None, alpha_new_layers: float = 1, **block_kwargs):
        """Forward pass that returns both the final image and intermediate features at the target resolution.
        
        Args:
            ws: Style vectors
            target_resolution: Resolution at which to extract intermediate features
            alpha_new_layers: Weight for new layers 
            
        Returns:
            tuple: (final_img, features_at_target_resolution, img_at_target_resolution)
        """
        if target_resolution is None:
            return self.forward(ws, alpha_new_layers, **block_kwargs), None, None
            
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        features_at_target = None
        img_at_target = None
        
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            if self.pretrained_plane_resolution is not None and res > self.pretrained_plane_resolution:
                x, img = block(x, img, cur_ws, alpha_new_layers=alpha_new_layers, **block_kwargs)
            else:
                x, img = block(x, img, cur_ws, **block_kwargs)
                
            # Store intermediate features and RGB at target resolution
            if res == target_resolution:
                features_at_target = x
                img_at_target = img
                
        return img, features_at_target, img_at_target

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class GGHGenerator(nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 pretrained_plane_resolution: Optional[int] = None,  # For progressive Growing
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = GGHSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                             pretrained_plane_resolution=pretrained_plane_resolution,
                                             **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img
        
    def forward_with_intermediate_features(self, z, c, target_resolution, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        """Forward pass that also returns intermediate features at the target resolution."""
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img, features, img_at_res = self.synthesis.forward_with_intermediate_features(
            ws, target_resolution=target_resolution, update_emas=update_emas, **synthesis_kwargs)
        return img, features, img_at_res

from src.gghead.eg3d.training.networks_stylegan2 import DoubleMappingNetwork, DoubleMappingNetwork_FLAME

class MeshGGHGenerator(nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 c2_dim,  # Conditioning label (C) dimensionality – 3DMM parameters.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 pretrained_plane_resolution: Optional[int] = None,  # For progressive Growing
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 flame_double_mapping=False, # default or flame specific
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.c2_dim = c2_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = GGHSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                             pretrained_plane_resolution=pretrained_plane_resolution,
                                             **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws

        if not flame_double_mapping:
            self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        else:
            # self.mapping = DoubleMappingNetwork(z_dim=z_dim, c_dim=c_dim, c2_dim=c2_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
            self.mapping = DoubleMappingNetwork_FLAME(z_dim=z_dim, c_dim=c_dim, c2_dim=c2_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, c2, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, c2, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img


class MouthSynthesisNetwork(nn.Module):
    def __init__(self,
                 w_dim,                 # Intermediate latent (W) dimensionality (now equals expression code dimension).
                 img_resolution,        # Output image resolution.
                 img_channels,          # Number of output color channels.
                 start_resolution=16,   # Resolution to start from (using features from main generator)
                 channel_base=32768,    # Overall multiplier for the number of channels.
                 channel_max=512,       # Maximum number of channels in any layer.
                 num_fp16_res=0,        # Use FP16 for the N highest resolutions.
                 start_res=4,
                 use_spade=False,
                 use_concat=False,
                 condition_nc=3,
                 **block_kwargs,        # Arguments for SynthesisBlock.
                 ):
        super().__init__()
        self.w_dim = w_dim  # w_dim now equals expression code dimension
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.start_resolution = start_resolution
        self.block_resolutions = [start_res * 2 ** (i - 2) for i in range(int(np.log2(start_resolution)), int(np.log2(img_resolution))+1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (int(np.log2(img_resolution)) - num_fp16_res), 8)
        
        # Create blocks starting from the second resolution in our sequence
        self.blocks = nn.ModuleList()
        
        # Calculate total number of style inputs needed (for proper expcode expansion)
        self.num_ws = 0
        
        for i, res in enumerate(self.block_resolutions[1:], 1):
            in_channels = channels_dict[self.block_resolutions[i-1]]
            out_channels = channels_dict[res]
            is_last = (res == self.block_resolutions[-1])
            block = GGHSynthesisBlock(
                in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last,
                use_fp16=(res >= fp16_resolution), use_spade=use_spade, use_concat=use_concat, condition_nc=condition_nc, 
                **block_kwargs
            )
            self.blocks.append(block)
            
            # Account for ws used in this block
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
    
    def forward(self, 
                ws,                # Expression parameters used directly as style vectors 
                feature_maps=None, # Features from main generator at the start resolution
                img_rgb=None,      # RGB image from main generator at the start resolution
                update_emas=False, # Should we update exponential moving averages?
                **block_kwargs     # Arguments for block
                ):
        """
        Forward pass for the mouth synthesis network.
        Uses expression parameters directly as style inputs.
        """
        # Start with features and RGB from main generator
        x = feature_maps  # Use features directly 
        img = img_rgb     # Use RGB directly if provided
        
        # Process remaining blocks
        w_idx = 0
        for block in self.blocks:
            block_ws = ws.narrow(1, w_idx, block.num_conv + block.num_torgb) 
            x, img = block(x, img, block_ws, update_emas=update_emas, **block_kwargs)
            w_idx += block.num_conv
            
        return img


class MouthGGHGenerator(nn.Module):
    def __init__(self,
                 z_dim,                  # Input latent (Z) dimensionality (not used).
                 c_dim,                  # Conditioning label (C) dimensionality (not used).
                 c2_dim,                 # FLAME parameters dimensionality.
                 w_dim,                  # Main generator's latent dimensionality (not used).
                 img_resolution,         # Output resolution.
                 img_channels,           # Number of output color channels.
                 start_resolution=16,    # Starting resolution for mouth branch.
                 mapping_kwargs={},      # Arguments for MappingNetwork (not used).
                 use_mlp_for_flame=True,
                 start_res=4,
                 use_spade=False,
                 use_concat=False,
                 condition_nc=3,
                 **synthesis_kwargs,     # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim # Not used but kept for consistency
        self.c_dim = c_dim # Not used but kept for consistency
        self.c2_dim = c2_dim
        self.original_w_dim = w_dim # Main generator's w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.start_resolution = start_resolution
        self.use_mlp_for_flame = use_mlp_for_flame

        if use_concat:
            self.use_mlp_for_flame = False

        # Determine the dimension for the synthesis network's w_dim based on MLP usage
        if self.use_mlp_for_flame:
            # Define MLP to process FLAME params (expcode)
            self.mlp_output_dim = self.original_w_dim # Let MLP output match main w_dim
            self.flame_mlp = nn.Sequential(
                nn.Linear(c2_dim, self.original_w_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.original_w_dim, self.mlp_output_dim)
            )
            synthesis_w_dim = self.original_w_dim + self.mlp_output_dim
        else:
            self.flame_mlp = None
            self.mlp_output_dim = c2_dim # If no MLP, the effective dim is c2_dim
            synthesis_w_dim = self.original_w_dim + self.mlp_output_dim

        # Create synthesis network that uses expression parameters directly
        self.synthesis = MouthSynthesisNetwork(
            w_dim=synthesis_w_dim,  # Pass the combined dimension
            img_resolution=img_resolution,
            img_channels=img_channels,
            start_resolution=start_resolution,
            start_res=start_res,
            use_spade=use_spade,
            use_concat=use_concat,
            condition_nc=condition_nc,
            **synthesis_kwargs
        )
        
        # Store number of style inputs needed for expansion
        self.num_ws = self.synthesis.num_ws

    def forward(self, 
                expcode,               # Expression parameters used directly
                feature_maps=None,     # Feature maps from the main generator
                img_rgb=None,          # RGB output from the main generator
                update_emas=False,     # Should we update exponential moving averages?
                ws_main=None,
                **synthesis_kwargs     # Arguments for synthesis
                ):
        """
        Forward pass using expression parameters directly as style vectors.
        Expands the expression code to the required number of style inputs.
        """
        # Process expcode through MLP if enabled
        if self.use_mlp_for_flame:
            flame_w = self.flame_mlp(expcode) # [batch_size, mlp_output_dim]
        else:
            flame_w = expcode # [batch_size, c2_dim]

        # Expand flame features/code to match required number of style inputs
        batch_size = flame_w.shape[0]
        ws_flame_expanded = flame_w.unsqueeze(1).repeat(1, self.num_ws, 1) # [batch_size, num_ws, mlp_output_dim or c2_dim]

        # we can take only the first num_ws_mouth, since they are all the same cause we don't use truncation_cutoff anywhere
        assert ws_main is not None, "ws_main must be provided for MouthGGHGenerator"
        ws_main_sliced = ws_main[:, :self.num_ws, :] # [batch_size, num_ws, original_w_dim]

        # Combine main ws with flame ws
        ws_combined = torch.cat([ws_main_sliced, ws_flame_expanded], dim=2) # [batch_size, num_ws, synthesis_w_dim]

        # Generate image using features and RGB from main generator
        img = self.synthesis(ws_combined, feature_maps=feature_maps, img_rgb=img_rgb, 
                            update_emas=update_emas, **synthesis_kwargs)
        return img


class GGHGenerator_FLAME(nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 c2_dim,                 # FLAME parameters dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 pretrained_plane_resolution: Optional[int] = None,  # For progressive Growing
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 use_mlp_for_flame=False,
                 double_mapping_for_flame=False,
                 start_res=4,
                 flame_alpha=None, # flame ws scaling factor with respect to main ws norm
                 use_spade=False,
                 use_concat=False,
                 condition_nc=3,
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.use_mlp_for_flame = use_mlp_for_flame
        self.flame_alpha = flame_alpha
        self.double_mapping_for_flame = double_mapping_for_flame
        
        assert not double_mapping_for_flame or (double_mapping_for_flame and not use_mlp_for_flame), "double_mapping_for_flame is only supported while use_mlp_for_flame=False"

        # Determine the dimension for the synthesis network's w_dim based on MLP usage
        if self.use_mlp_for_flame:
            # Define MLP to process FLAME params (expcode)
            self.mlp_output_dim = w_dim # Let MLP output match main w_dim
            self.flame_mlp = nn.Sequential(
                nn.Linear(c2_dim, w_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(w_dim, w_dim)
            )
            nn.init.zeros_(self.flame_mlp[-1].weight)
            nn.init.zeros_(self.flame_mlp[-1].bias)

            synthesis_w_dim = w_dim + self.mlp_output_dim
        elif not double_mapping_for_flame: # Legacy, not supported!
            self.flame_mlp = None 
            synthesis_w_dim = w_dim + c2_dim 
        else:
            self.flame_mlp = None 
            synthesis_w_dim = w_dim
        
        self.synthesis = GGHSynthesisNetwork(w_dim=synthesis_w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                             pretrained_plane_resolution=pretrained_plane_resolution,
                                             start_res=start_res,
                                             use_spade=use_spade,
                                             use_concat=use_concat,
                                             condition_nc=condition_nc,
                                             **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws

        if double_mapping_for_flame:
            self.mapping = DoubleMappingNetwork(z_dim=z_dim, c_dim=c_dim, c2_dim=c2_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        else:
            self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    # def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
    #     ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    #     img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
    #     return img

    def prepare_ws(self, expcode, ws, zero_flame_ws=False):
        if getattr(self, "double_mapping_for_flame", False):
            return ws

        # Process expcode through MLP if enabled
        if self.use_mlp_for_flame:
            flame_w = self.flame_mlp(expcode) # [batch_size, mlp_output_dim]
        else:
            flame_w = expcode # [batch_size, c2_dim]

        # Expand flame features/code to match required number of style inputs
        ws_flame_expanded = flame_w.unsqueeze(1).repeat(1, self.num_ws, 1) # [batch_size, num_ws, mlp_output_dim or c2_dim]

        if self.flame_alpha is not None:
            ws_norm = torch.linalg.norm(ws[:, :, :], dim=-1, keepdim=True)
            ws_flame_norm = torch.linalg.norm(ws_flame_expanded[:, :, :], dim=-1, keepdim=True)
            mult = ws_norm / (ws_flame_norm + 1e-8) * self.flame_alpha
            ws_flame_expanded = mult * ws_flame_expanded

        if zero_flame_ws:
            ws_flame_expanded = torch.zeros_like(ws_flame_expanded)

        # Combine main ws with flame ws
        ws_combined = torch.cat([ws, ws_flame_expanded], dim=2) # [batch_size, num_ws, synthesis_w_dim]

        return ws_combined