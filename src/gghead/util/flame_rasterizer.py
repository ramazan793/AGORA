import torch
from pytorch3d.renderer import PerspectiveCameras, TexturesUV, BlendParams, SoftPhongShader, PointLights, RasterizationSettings, MeshRenderer, MeshRasterizer
from pytorch3d.structures import Meshes


def get_gradient_flame_texture(texture_mask, resolution):
    x = torch.linspace(0, 1, resolution).view(1, resolution)
    y = torch.linspace(0, 1, resolution).view(resolution, 1)

    x_grid = x.expand(resolution, resolution)
    y_grid = y.expand(resolution, resolution)

    R = x_grid
    G = y_grid
    B = (1-x_grid + 1-y_grid) / 2

    gradient_image = torch.stack([R, G, B], dim=-1)

    grad_texture = torch.flip(texture_mask, dims=[0]) * gradient_image 
    return grad_texture


class FLAME_rasterizer(torch.nn.Module):
    def __init__(self, uv_faces, uv_verts, template_faces, grad_texture, resolution=256, light_type='ambient'):
        """
        Initialize the FLAME rasterizer with given UV coordinates, faces, texture, and camera settings.

        Args:
            uv_faces (torch.Tensor): UV faces for texture mapping, shape [K, 3]
            uv_verts (torch.Tensor): UV vertices coordinates, shape [N, 2]
            template_faces (torch.Tensor): Template mesh faces, shape [F, 3]
            grad_texture (torch.Tensor): Texture map tensor, shape [H, W, 3]
            resolution (int): Output image resolution (default: 256)
        """
        super().__init__()

        self.register_buffer('uv_faces', uv_faces)
        self.register_buffer('uv_verts', uv_verts)
        self.register_buffer('template_faces', template_faces)
        self.register_buffer('grad_texture', grad_texture)
        self.resolution = resolution

        # Setup camera rotation matrix (flipping z-axis)
        R = torch.eye(3).unsqueeze(0)
        R[:, 2, 2] = -1
        T = torch.tensor([[0, 0.015, 1.6]])

        # Calculate focal length and principal point in absolute coordinates
        fl = 2 * 4.2647
        pp = 0.5
        abs_fl = resolution * fl
        abs_pp = resolution * pp

        # Initialize cameras
        self.cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=torch.ones(1, 2) * abs_fl,
            principal_point=torch.ones(1, 2) * abs_pp,
            in_ndc=False,
            image_size=torch.ones(1, 2) * resolution
        )

        # Setup lighting (front light with ambient component)

        if light_type == 'ambient':
            self.lights = PointLights(
                ambient_color=[[1.0, 1.0, 1.0]],
                diffuse_color=[[0.0, 0.0, 0.0]],
                specular_color=[[0.0, 0.0, 0.0]]
            )
        elif light_type == 'directional':
            self.lights = PointLights(
                location=[[0.0, 0.0, 3.0]]
            )
        else:
            raise ValueError('Wrong light_type')

        # Configure rasterization settings
        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Create the renderer with SoftPhong shader
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                cameras=self.cameras,
                lights=self.lights,
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
            )
        )

    def forward(self, vertices):
        """
        Rasterize the input vertices into images using the configured settings.

        Args:
            vertices (torch.Tensor): Batch of vertex positions, shape [B, V, 3]

        Returns:
            torch.Tensor: Rendered images in RGB format, shape [B, H, W, 3]
        """
        B = vertices.shape[0]  # Batch size

        self.renderer = self.renderer.to(vertices.device)

        # Prepare batched UV texture components
        uv_texture_map_batched = self.grad_texture.unsqueeze(0).expand(B, -1, -1, -1)
        uv_faces_batched = self.uv_faces.unsqueeze(0).expand(B, -1, -1)
        uv_verts_batched = self.uv_verts.unsqueeze(0).expand(B, -1, -1).float()

        # Create UV textures
        texture = TexturesUV(
            maps=uv_texture_map_batched,
            faces_uvs=uv_faces_batched,
            verts_uvs=uv_verts_batched
        )

        # Prepare batched faces
        faces_batched = self.template_faces.unsqueeze(0).expand(B, -1, -1)

        # Create mesh structure
        mesh = Meshes(
            verts=vertices,
            faces=faces_batched,
            textures=texture
        )

        # Render the mesh
        rendered_images = self.renderer(mesh)  # [B, H, W, 4]
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2) # [B, 3, H, W]
        rendered_images = rendered_images.flip(-1)
        return rendered_images

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def parse_flame_deca_cameras(cam_params, J_transformed, Z0=2.28):
    """
    cam_params: (B,6) = [globalpose(3), s, tx, ty]
    J_transformed: (B, K, 3), root joint at index 0, in world units (no global rot baked in)
    Z0 estimated as an average FLAME vertices depth (z_camera) variance multiplied by 15 (so, depth variance ~ 6,6%)
    Returns:
      w2c: (B,4,4) world->camera
      K_norm: (B,3,3) intrinsics in normalized coords (u,v in [0,1]),
              later convert to pixels via: fx*=W, cx*=W, fy*=H, cy*=H
    """
    device = cam_params.device
    dtype  = cam_params.dtype
    B = cam_params.shape[0]

    # unpack
    globalpose   = cam_params[:, :3]          # axis-angle (B,3)
    s   = cam_params[:, 3:4]         # (B,1)
    tx  = cam_params[:, 4:5]         # (B,1)
    ty  = cam_params[:, 5:6]         # (B,1)
    pivot = J_transformed[:, 0]      # (B,3) root joint

    # rotations
    R_head = batch_rodrigues(globalpose)                   # (B,3,3)
    F = torch.diag(torch.tensor([1., -1., -1.], device=device, dtype=dtype)).expand(B,3,3)
    R_cam = F @ R_head                            # world->cam

    R_inv = R_head.transpose(1, 2)                # (B,3,3)
    ez = torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand(B,3)

    # camera center in world coords (rotate about pivot, then push back by Z0)
    C_pivot = pivot - (R_inv @ pivot.unsqueeze(-1)).squeeze(-1)           # (B,3)
    C_depth = Z0 * (R_inv @ ez.unsqueeze(-1)).squeeze(-1)                 # (B,3)
    C = C_pivot + C_depth                                                 # (B,3)

    # world->cam translation
    t = -(R_cam @ C.unsqueeze(-1)).squeeze(-1)                            # (B,3)

    # assemble w2c
    w2c = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)
    w2c[:, :3, :3] = R_cam
    w2c[:, :3,  3] = t

    # intrinsics (normalized to [0,1])
    ax = 0.5 * s
    ay = 0.5 * s
    fx = ax * Z0                      # (B,1)
    fy = ay * Z0
    cx = 0.5 + ax * tx
    cy = 0.5 - ay * ty

    K = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    K[:, 0, 0] = fx.squeeze(-1)
    K[:, 1, 1] = fy.squeeze(-1)
    K[:, 0, 2] = cx.squeeze(-1)
    K[:, 1, 2] = cy.squeeze(-1)
    K[:, 2, 2] = 1.0

    return w2c, K