def interpolate_2d_array(data, M, kind='linear'):
    """
    Interpolate a 2D numpy array with shape [N, 3] to shape [M, 3].
    
    Parameters:
        data (numpy.ndarray): Original data array of shape [N, 3].
        M (int): Number of points in the interpolated array.
        kind (str): Type of interpolation (e.g., 'linear', 'cubic').
    
    Returns:
        numpy.ndarray: Interpolated data array of shape [M, 3].
    """
    if data.shape[1] != 3:
        raise ValueError("Input array must have shape [N, 3]")
    
    N = data.shape[0]
    x = np.linspace(0, N-1, N)  # Original x values
    x_new = np.linspace(0, N-1, M)  # New x values for interpolation
    
    # Interpolating each dimension independently
    f0 = interp1d(x, data[:, 0], kind=kind)
    f1 = interp1d(x, data[:, 1], kind=kind)
    f2 = interp1d(x, data[:, 2], kind=kind)
    
    # Getting the interpolated values
    data_new = np.zeros((M, 3))
    data_new[:, 0] = f0(x_new)
    data_new[:, 1] = f1(x_new)
    data_new[:, 2] = f2(x_new)
    
    return data_new

def interpolate_between_arrays(A, B, K):
    """
    Interpolate between two arrays A and B to create a new array C with shape [K, M, 3].
    
    Parameters:
        A (numpy.ndarray): Starting array with shape [M, 3].
        B (numpy.ndarray): Ending array with shape [M, 3].
        K (int): Number of interpolated steps.
    
    Returns:
        numpy.ndarray: Interpolated array with shape [K, M, 3].
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.shape[1] != 3:
        raise ValueError("A and B must have shape [M, 3]")
    
    M = A.shape[0]
    # Initialize the array C
    C = np.zeros((K, M, 3))
    
    # Create interpolation ratios
    for i in range(K):
        ratio = i / (K - 1)
        C[i] = A * (1 - ratio) + B * ratio
    
    return C


# predefined indices of the mouth interior lines on FLAME mesh
mouth_line_upper_indices = [1572, 1594, 1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731, 2730, 2708]
mouth_line_lower_indices = [1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708]
mouth_line_end_indices = [1572, 2708]
def generate_mouth_interior_uv(vertices : np.array, M : int, K : int):
    """
    Generate the UV position map for the mouth interior

    Parameters:
        vertices: [V, 3], np.array
        M: The width of UV mouth interior patch
        K: The half-height of UV mouth interior patch 

    Returns:
        mouth_interior_uv_patch: [2K, M, 3], np.array
    """
    # retrieve mouth lines vertices
    upper = np.copy(vertices[mouth_line_upper_indices])
    lower = np.copy(vertices[mouth_line_lower_indices])
    endline = np.copy(vertices[mouth_line_end_indices])
    endline[:,2] = endline[:,2] - 0.02 # move the endline to the back
    
    # interpolate mouth lines
    upper_interpolated = interpolate_2d_array(data=upper, M=M) # [M, 3]
    lower_interpolated = interpolate_2d_array(data=lower, M=M) # [M, 3]
    endline_interpolated = interpolate_2d_array(data=endline, M=M) # [M, 3]
        
    # interpolate the mouth interior surfaces vertices
    upper_surface_interpolated = interpolate_between_arrays(A=upper_interpolated, B=endline_interpolated, K=K) # [K, M, 3]
    lower_surface_interpolated = interpolate_between_arrays(A=endline_interpolated, B=lower_interpolated, K=K) # [K, M, 3]
    
    # covnert the mouth interior vertices to UV patch
    mouth_interior_uv_patch = np.zeros([K*2, M, 3], dtype=np.float32)
    mouth_interior_uv_patch[:K, :, :] = upper_surface_interpolated
    mouth_interior_uv_patch[K:, :, :] = lower_surface_interpolated

    return mouth_interior_uv_patch