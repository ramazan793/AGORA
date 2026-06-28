import torch
import torch.nn.functional as F

# Predefined indices of the mouth interior lines on FLAME mesh
mouth_line_upper_indices = [1572, 1594, 1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731, 2730, 2708]
mouth_line_lower_indices = [1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708]
mouth_line_end_indices   = [1572, 2708]

def interpolate_2d_array(data, M, kind='linear'):
    """
    Interpolate a 2D torch tensor with shape [B, N, 3] (or [N, 3]) to shape [B, M, 3] (or [M, 3]).
    Only supports linear interpolation.
    
    Parameters:
        data (torch.Tensor): Original data tensor of shape [B, N, 3] or [N, 3].
        M (int): Number of points in the interpolated array.
        kind (str): Type of interpolation (only 'linear' is supported).
    
    Returns:
        torch.Tensor: Interpolated data tensor of shape [B, M, 3] (or [M, 3] if input was non-batched).
    """
    if kind != 'linear':
        raise NotImplementedError("Only linear interpolation is implemented")
    
    # If data is not batched, add a batch dimension.
    squeeze = False
    if data.ndim == 2:
        data = data.unsqueeze(0)  # shape becomes [1, N, 3]
        squeeze = True
        
    if data.size(2) != 3:
        raise ValueError("Input tensor must have shape [B, N, 3] or [N, 3]")
        
    # data shape: [B, N, 3]
    B, N, _ = data.shape
    # Permute to [B, 3, N] for interpolation along the "time" dimension
    data_perm = data.transpose(1, 2)  # shape [B, 3, N]
    # Use F.interpolate to perform 1D linear interpolation to size M along the last dimension
    data_interp = F.interpolate(data_perm, size=M, mode='linear', align_corners=True)
    data_interp = data_interp.transpose(1, 2)  # shape [B, M, 3]
    if squeeze:
        data_interp = data_interp.squeeze(0)
    return data_interp

def interpolate_between_arrays(A, B, K):
    """
    Interpolate between two batched arrays A and B to create a new tensor with shape [B, K, M, 3].
    
    Parameters:
        A (torch.Tensor): Starting array with shape [B, M, 3].
        B (torch.Tensor): Ending array with shape [B, M, 3].
        K (int): Number of interpolated steps.
    
    Returns:
        torch.Tensor: Interpolated tensor with shape [B, K, M, 3].
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.size(-1) != 3:
        raise ValueError("A and B must have shape [B, M, 3]")
        
    # Create interpolation ratios of shape [1, K, 1, 1] for broadcasting
    ratios = torch.linspace(0, 1, steps=K, device=A.device, dtype=A.dtype).view(1, K, 1, 1)
    # Expand A and B for the interpolation
    A_exp = A.unsqueeze(1)  # shape becomes [B, 1, M, 3]
    B_exp = B.unsqueeze(1)  # shape becomes [B, 1, M, 3]
    return A_exp * (1 - ratios) + B_exp * ratios

def generate_mouth_interior_uv(vertices, M, K):
    """
    Generate the UV position map for the mouth interior.
    
    Parameters:
        vertices (torch.Tensor): Batched vertices tensor with shape [B, V, 3].
        M (int): The width (number of points) of the UV mouth interior patch.
        K (int): The half-height (number of rows) of the UV mouth interior patch.
    
    Returns:
        torch.Tensor: Mouth interior UV patch of shape [B, 2K, M, 3].
    """
    # Extract the mouth line vertices using predefined indices
    upper = vertices[:, mouth_line_upper_indices, :].clone()  # shape [B, len(upper), 3]
    lower = vertices[:, mouth_line_lower_indices, :].clone()  # shape [B, len(lower), 3]
    endline = vertices[:, mouth_line_end_indices, :].clone()  # shape [B, len(endline), 3]
    
    # Move the endline slightly back along the z-axis
    endline[..., 2] = endline[..., 2] - 0.02
    
    # Interpolate each mouth line to have M points
    upper_interpolated = interpolate_2d_array(upper, M)       # shape [B, M, 3]
    lower_interpolated = interpolate_2d_array(lower, M)       # shape [B, M, 3]
    endline_interpolated = interpolate_2d_array(endline, M)     # shape [B, M, 3]
    
    # Interpolate between lines to generate the interior surfaces
    upper_surface_interpolated = interpolate_between_arrays(upper_interpolated, endline_interpolated, K)  # shape [B, K, M, 3]
    lower_surface_interpolated = interpolate_between_arrays(endline_interpolated, lower_interpolated, K)  # shape [B, K, M, 3]
    
    # Concatenate the upper and lower surfaces vertically to form the full mouth interior UV patch
    mouth_interior_uv_patch = torch.cat([upper_surface_interpolated, lower_surface_interpolated], dim=1)  # shape [B, 2K, M, 3]
    
    return mouth_interior_uv_patch
