"""Shared SVD+MLP (PCA v2) inference utilities used by all evaluation scripts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTR_KEYS = ['xyz', 'scale', 'rotation', 'opacity']


class MLPRegressor(nn.Module):
    """Maps concat(W[512], flame_55[55]) = 567-dim → 4*K PCA coefficients."""

    def __init__(self, in_dim: int = 567, hidden1: int = 256, hidden2: int = 512,
                 out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_svd_mlp(svd_path: str, mlp_path: str, device):
    """Load SVD basis and MLP regressor from disk.

    Returns:
        svd_data: dict with U_xyz, U_scale, U_rotation, U_opacity, G, K, ...
        mlp: MLPRegressor on device
        K: number of PCA components
        G_num: number of Gaussians
    """
    svd_data = torch.load(svd_path, map_location='cpu')
    K = int(svd_data['U_xyz'].shape[0])
    G_num = int(svd_data['G'])

    mlp_ckpt = torch.load(mlp_path, map_location='cpu')
    mlp = MLPRegressor(
        in_dim=mlp_ckpt['in_dim'],
        hidden1=mlp_ckpt['hidden1'],
        hidden2=mlp_ckpt['hidden2'],
        out_dim=mlp_ckpt['out_dim'],
    ).to(device)
    mlp.load_state_dict(mlp_ckpt['state_dict'])
    mlp.eval()

    print(f"SVD basis: K={K}, G={G_num}")
    print(f"MLP: {mlp_ckpt['in_dim']} → {mlp_ckpt['out_dim']} (val_loss={mlp_ckpt.get('val_loss', 'n/a')})")
    return svd_data, mlp, K, G_num


def setup_identity(G, z, fixed_shape, c_front, svd_data, G_num, device,
                   resolution: int = 512, truncation_psi: float = 0.7):
    """Compute per-identity W code and base Gaussian attributes at neutral expression.

    NOTE: Does NOT set/use backbone cache (cache_backbone=False, use_cached_backbone=False).
    The first synthesis call in the generation loop should set cache_backbone=True.

    Args:
        G: DynGGHead generator
        z: [1, z_dim] latent code
        fixed_shape: [300] FLAME shape code
        c_front: [1, 6] front camera params
        svd_data: dict loaded from svd_basis.pt
        G_num: number of Gaussians
        device: torch device
        resolution: neural rendering resolution

    Returns:
        w_code: [1, 512] identity latent (backbone_ws[:, 0, :])
        base_gaussians: dict {xyz, scale, rotation, opacity} → [1, G, C] in physical space
        base_color: [1, G, C_color] raw color attribute
        zero_deltas: dict {xyz, scale, rotation, opacity, color} → [59, G, C] zeros
        neutral_flame: [1, 358] neutral FLAME params with fixed_shape
    """
    from src.gghead.config.gaussian_attribute import GaussianAttribute

    neutral_flame = torch.zeros(1, 358, device=device, dtype=torch.float32)
    neutral_flame[:, :300] = fixed_shape

    ws_init = G.mapping(z, c_front, truncation_psi=truncation_psi,
                        flame_params=neutral_flame)
    backbone_ws, _ = ws_init
    w_code = backbone_ws[:, 0, :]  # [1, 512]

    output_base = G.synthesis(
        ws_init, c_front, neutral_flame,
        neural_rendering_resolution=resolution,
        cache_backbone=False,
        use_cached_backbone=False,
        noise_mode='const',
        sh_ref_cam=c_front,
        return_gaussian_attributes=True,
    )
    ga_raw = output_base.returned_gaussian_attributes
    base_gaussians = {
        'xyz':      ga_raw[GaussianAttribute.POSITION].detach(),
        'scale':    G._gaussian_model.scaling_activation(ga_raw[GaussianAttribute.SCALE]).detach(),
        'rotation': G._gaussian_model.rotation_activation(ga_raw[GaussianAttribute.ROTATION]).detach(),
        'opacity':  G._gaussian_model.opacity_activation(ga_raw[GaussianAttribute.OPACITY]).detach(),
    }
    base_color = ga_raw[GaussianAttribute.COLOR].detach()

    C_color = base_color.shape[-1]
    zero_deltas = {
        'xyz':      torch.zeros(59, G_num, 3,       device=device),
        'scale':    torch.zeros(59, G_num, 3,       device=device),
        'rotation': torch.zeros(59, G_num, 4,       device=device),
        'opacity':  torch.zeros(59, G_num, 1,       device=device),
        'color':    torch.zeros(59, G_num, C_color, device=device),
    }

    return w_code, base_gaussians, base_color, zero_deltas, neutral_flame


def generate_with_pca(G, ws, c_render, flame_params, w_code, base_gaussians,
                      base_color, zero_deltas, svd_data, mlp, device, K,
                      ov: float = 0.0, **synthesis_kwargs):
    """Reconstruct physical Gaussian attributes via SVD+MLP and render.

    Args:
        G: DynGGHead generator
        ws: output of G.mapping [B, ...] (backbone + deform latents)
        c_render: camera for synthesis [B, 6]
        flame_params: [B, 358]
        w_code: [1, 512] per-identity W code
        base_gaussians: {xyz, scale, rotation, opacity} → [1, G, C] physical attrs
        base_color: [1, G, C_color]
        zero_deltas: {xyz, scale, rotation, opacity, color} → [59, G, C] zeros
        svd_data: dict with U_xyz, U_scale, U_rotation, U_opacity
        mlp: MLPRegressor
        device: torch device
        K: number of PCA components
        ov: opacity overshoot (from G._config.opacity_overshoot)
        **synthesis_kwargs: passed to G.synthesis (noise_mode, cache_backbone, etc.)

    Returns:
        output dict from G.synthesis (contains 'image', etc.)
    """
    B = flame_params.shape[0]

    # Extract 55-dim FLAME expression vector [exp(50), jaw(3), eyelid(2)]
    flame_55 = torch.cat([
        flame_params[:, 300:350],  # exp   [B, 50]
        flame_params[:, 353:356],  # jaw   [B, 3]
        flame_params[:, 356:358],  # eyelid [B, 2]
    ], dim=-1).float()  # [B, 55]

    # MLP forward: concat(w_code, flame_55) → 4K PCA coefficients
    w_code_batch = w_code.expand(B, -1)           # [B, 512]
    mlp_input = torch.cat([w_code_batch, flame_55], dim=-1)  # [B, 567]
    c_all = mlp(mlp_input)                        # [B, 4K]

    c_parts = {
        'xyz':      c_all[:, 0 * K:1 * K],
        'scale':    c_all[:, 1 * K:2 * K],
        'rotation': c_all[:, 2 * K:3 * K],
        'opacity':  c_all[:, 3 * K:4 * K],
    }

    # Reconstruct physical attributes: base + (coeffs @ U).view(B, G, C)
    phys = {}
    for key_str in ATTR_KEYS:
        base = base_gaussians[key_str]                      # [1, G, C]
        U = svd_data[f'U_{key_str}'].to(device)            # [K, G*C]
        Gn = base.shape[1]
        Cn = base.shape[2]
        delta = (c_parts[key_str] @ U).view(B, Gn, Cn)    # [B, G, C]
        phys[key_str] = base + delta                        # broadcast [1,G,C]+[B,G,C]
    phys['color'] = base_color                              # [1, G, C_color]

    # Enforce geometric constraints in physical space
    phys['scale']    = torch.clamp(phys['scale'], min=1e-5)
    phys['rotation'] = F.normalize(phys['rotation'], p=2, dim=-1)
    phys['opacity']  = torch.clamp(phys['opacity'], min=-ov + 1e-6, max=1.0 + ov - 1e-6)

    # Inject via zero-delta trick: base = reconstructed physical attrs, deltas = 0
    post_act_bs = {
        'base_xyz':       phys['xyz'],
        'base_scale':     phys['scale'],
        'base_rotation':  phys['rotation'],
        'base_opacity':   phys['opacity'],
        'base_color':     phys['color'],
        'delta_xyz':      zero_deltas['xyz'],
        'delta_scale':    zero_deltas['scale'],
        'delta_rotation': zero_deltas['rotation'],
        'delta_opacity':  zero_deltas['opacity'],
        'delta_color':    zero_deltas['color'],
    }

    return G.synthesis(ws, c_render, flame_params,
                       post_act_blendshapes=post_act_bs,
                       **synthesis_kwargs)
