from typing import Optional
from dataclasses import dataclass
import tyro
import pickle
import sys
import os
from glob import glob
from tqdm import tqdm
import json
import time

import mediapy
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import torch
from PIL import Image
import cv2

from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.image import Img
from dreifus.matrix import Pose
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from eg3d.datamanager.nersemble import encode_camera_params, decode_camera_params
from elias.util import ensure_directory_exists
from elias.util.batch import batchify_sliced

_agora_root = os.path.dirname(os.path.abspath(__file__))
while _agora_root != os.path.dirname(_agora_root) and not os.path.isdir(os.path.join(_agora_root, "src", "gghead")):
    _agora_root = os.path.dirname(_agora_root)
if _agora_root not in sys.path:
    sys.path.insert(0, _agora_root)
os.environ.setdefault('GGHEAD_MODELS_PATH', '/data3/ramazan.fazylov/media/dyn_gghead_stuff/logs/models/')

from src.gghead.model_manager.finder import find_model_manager
from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR

@dataclass
class Args:
    vid_processed_path: Optional[str] = '/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/ID_13/MAU_mild_C_center_crop_512/smirk/'
    gt_img_path: Optional[str] = '/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/ID_13/MAU_mild_C_center_crop_512/crop/'
    pairs_list: Optional[str] = None
    FPS: int = 30
    DEVICE: str = 'cuda:0'
    run_name: str = 'DGGHEAD-116'
    checkpoint: int = 13800
    resolution: int = 512
    CACHE_BACKBONE: int = 1
    SYNTHESIS_FLAME_COND: int = 1
    savgol_win: int = 5
    max_len: int = 10000000
    id_seed: str = "0"  # Changed to str to support ranges and lists
    use_narrow_mask: int = 0
    cam_scale: float = None
    joint_c_front: int = 0
    render_mode: str = 'RGB'

def parse_id_seed(id_seed_str: str) -> list[int]:
    """
    Parse id_seed string to get a list of integers.
    Supports:
    - Single value: "5" -> [5]
    - Range: "0-8" -> [0, 1, 2, 3, 4, 5, 6, 7, 8]
    - Comma-separated: "0,4,2,9,10" -> [0, 4, 2, 9, 10]
    """
    id_seed_str = id_seed_str.strip()
    
    # Check if it's a range (e.g., "0-8")
    if '-' in id_seed_str:
        parts = id_seed_str.split('-')
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
    
    # Check if it's comma-separated (e.g., "0,4,2,9,10")
    if ',' in id_seed_str:
        return [int(x.strip()) for x in id_seed_str.split(',')]
    
    # Single value
    return [int(id_seed_str)]

def parse_smirk_processed_video(emica_path, cam_path=None, gt_img_path=None, max_len=None, take_no_crop_cam=False):
    sl = slice(0, max_len, 1)
    
    frame_folders = sorted(glob(os.path.join(emica_path, f'*/')), key=lambda x: int(x.split('/')[-2]))

    if gt_img_path is None:
        frame_paths = sorted(glob(emica_path + f'/*/*/detections/*_000.png'))
    else:
        frame_paths = sorted(glob(gt_img_path + f'/*.png'), key=lambda x: int(os.path.basename(x[:-4])))
    images = [np.array(Image.open(x)) for x in tqdm(frame_paths[sl], desc='Loading images')]

    if cam_path:
        with open(cam_path, 'r') as f:
            raw_cams = json.load(f)['labels']
    
    shapecodes = []
    expcodes = []
    globalposes = []
    jawposes = []
    flameorths = []
    eyelids = []
    cams = []

    i = 0
    for folder in tqdm(frame_folders[sl], desc='Loading flame parameters'):
        shape = np.load(f"{folder}/shape.npy")
        exp = np.load(f"{folder}/exp.npy")
        globalpose = np.load(f"{folder}/globalpose.npy")
        jawpose = np.load(f"{folder}/jawpose.npy")

        if take_no_crop_cam:
            if '__smirk_estimations' in folder and os.path.exists(f"{folder.replace('__smirk_estimations', '__smirk_estimations__no_crop')}/cam.npy"):
                flameorth = np.load(f"{folder.replace('__smirk_estimations', '__smirk_estimations__no_crop')}/cam.npy") # to align with input video
            elif 'smirk' in folder and os.path.exists(f"{folder.replace('smirk', 'smirk__no_crop')}/cam.npy"):
                flameorth = np.load(f"{folder.replace('smirk', 'smirk__no_crop')}/cam.npy")
            elif os.path.exists(f"{folder}/no_crop_for_cam.txt"):
                flameorth = np.load(f"{folder}/cam.npy")
            else:
                raise ValueError(f"No no_crop cam.npy found for {folder}")
        else:
            flameorth = np.load(f"{folder}/cam.npy")
        
        eyelid = np.load(f"{folder}/eyelid.npy")

        if cam_path:
            cam = np.array(raw_cams[i][1])
            cams.append(cam)
        i += 1

        shapecodes.append(shape)
        expcodes.append(exp)
        globalposes.append(globalpose)
        jawposes.append(jawpose)
        flameorths.append(flameorth)
        eyelids.append(eyelid)

    shapecodes = torch.tensor(np.array(shapecodes))
    expcodes = torch.tensor(np.array(expcodes))
    globalposes = torch.tensor(np.array(globalposes))
    jawposes = torch.tensor(np.array(jawposes))
    flameorths = torch.tensor(np.array(flameorths))
    eyelids = torch.tensor(np.array(eyelids))
    if cam_path:
        cams = torch.tensor(np.array(cams))
    else:
        cams = None
    images = torch.tensor(np.array(images))
    
    return shapecodes, expcodes, globalposes, jawposes, flameorths, eyelids, cams, images

def smooth_extrinsics(flat_44, window=15, poly=3):
    """
    flat_44 : (T, 16) array of row-major 4×4 matrices
    window  : odd int ≥ poly+2   —  length of the Savitzky-Golay window (in frames)
    poly    : int < window       —  polynomial order for Savitzky-Golay
    returns : (T, 16)  smoothed in the same format
    """
    assert window % 2 == 1, "window_length must be odd"
    T = flat_44.shape[0]

    M = flat_44.reshape(T, 4, 4)

    Rmats = M[:, :3, :3]
    quats = R.from_matrix(Rmats).as_quat()

    for t in range(1, T):
        if np.dot(quats[t-1], quats[t]) < 0:
            quats[t] *= -1

    quats_s = savgol_filter(quats, window, poly, axis=0)
    quats_s /= np.linalg.norm(quats_s, axis=1, keepdims=True)

    Rmats_s = R.from_quat(quats_s).as_matrix()

    trans = M[:, :3, 3]
    trans_s = savgol_filter(trans, window, poly, axis=0)

    M_s = np.zeros_like(M)
    M_s[:, :3, :3] = Rmats_s
    M_s[:, :3, 3]  = trans_s
    M_s[:, 3, :]   = np.array([0, 0, 0, 1])

    return M_s.reshape(T, 16)

def convert_depth_to_normal_map(depth_map: np.ndarray, ksize: int = 5, scale: float = 5) -> np.ndarray:
    rows, cols = depth_map.shape

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize, scale=scale)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize, scale=scale)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    return normal

def main(args: Args) -> None:
    cam_path = None

    fixed_view = False
    truncation_psi = 0.7
    batch_size = 4
    script_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device(args.DEVICE)

    model_manager = find_model_manager(args.run_name)
    checkpoint = model_manager._resolve_checkpoint_id(args.checkpoint)
    G = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)

    if args.use_narrow_mask:
        G._config.mask_type = 'narrow'
        uv_reg_weights = torch.from_numpy(np.load(f'{REPO_ROOT_DIR}/assets/gghead/narrow_facial_flame_mask_with_eyeballs_v2.npy')) # [256, 256]
        # uv_reg_weights = torch.from_numpy(np.load('/netapp/r.fazylov/research_workspace/dyn_gghead/assets/uv_position_weights_dejavu_adapted_narrow.npy')) # [256, 256]
        
        uv_reg_weights = torch.nn.functional.interpolate(uv_reg_weights.unsqueeze(0).unsqueeze(0), size=G._config.plane_resolution, mode='bilinear', antialias=False).squeeze() # [res, res]
        uv_reg_mask = (uv_reg_weights >= 0.5).float().unsqueeze(0).unsqueeze(0)
        
        uv_reg_mask_padded = torch.zeros((1, 1, G.extended_uv_resolution, G.extended_uv_resolution))
        pad_size = (G.extended_uv_resolution - G._config.plane_resolution) // 2
        uv_reg_mask_padded[:, :, :G._config.plane_resolution, pad_size:pad_size + G._config.plane_resolution] = uv_reg_mask
        pad_size_mouth = (G.extended_uv_resolution - G.mouth_res) // 2
        uv_reg_mask_padded[:, :, G._config.plane_resolution:G._config.plane_resolution + G.mouth_res, pad_size_mouth:pad_size_mouth + G.mouth_res] = 1.0

        deform_mask = uv_reg_mask_padded.to(device)
        G.register_buffer("_deform_mask", deform_mask.contiguous())

    MAPPING_TAKES_FLAME_PARAMS = G._config.use_mouth_branch or G._config.use_extended_uv_generation or True

    G._config.use_flame_rasterization = 0
    G._config.render_mode = args.render_mode

    if args.pairs_list:
        file_pairs = []
        with open(args.pairs_list, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    gp = line.strip()
                    vp = os.path.join(gp, 'smirk')
                    file_pairs.append((vp, gp))
                else:
                    vp = parts[0].strip()
                    gp = parts[1].strip()
                    file_pairs.append((vp, gp))
    else:
        file_pairs = [(args.vid_processed_path, args.gt_img_path)]

    print("File pairs:", file_pairs)
    
    # Parse id_seed string to get list of IDs
    id_seeds = parse_id_seed(args.id_seed)
    print(f"Processing ID seeds: {id_seeds}")

    ffhq_flame_and_cams = np.load(f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy')
    ffhq_cams = ffhq_flame_and_cams[:, -6:]
    ffhq_shapes = ffhq_flame_and_cams[:, :300]

    # Outer loop: iterate over videos
    for vid_processed_path, gt_img_path in tqdm(file_pairs, desc='Processing videos'):
        vid_name = os.path.basename(vid_processed_path.rstrip(os.sep))
        if vid_name == 'smirk':
            vid_name = vid_processed_path.rstrip(os.sep).split(os.sep)[-2]

        shapecodes, expcodes, globalposes, jawposes, flameorths, eyelids, _cams, driving_images = \
            parse_smirk_processed_video(vid_processed_path, cam_path, gt_img_path, args.max_len, take_no_crop_cam=False)

        cams_new = torch.cat([globalposes, flameorths], dim=-1)
        flame_params = torch.cat([shapecodes, expcodes, globalposes, jawposes, eyelids], dim=-1)

        flame_params = flame_params.to(device)

        mean_ffhq_cam = np.mean(ffhq_cams, axis=0)
        c_front = torch.tensor(mean_ffhq_cam).unsqueeze(0).to(device)

        c_render = cams_new.clone().to(device)
        c_smooth = savgol_filter(c_render.cpu().numpy(), window_length=7, polyorder=3, axis=0)
        c_smooth = torch.tensor(c_smooth).to(device)
        c_smooth[:, 3:] = c_smooth[:, 3:].mean(dim=0, keepdims=True)

        sh_ref_cam = c_front

        # take intrinsics from mean ffhq cam
        if args.cam_scale is not None:
            c_smooth[:, 3:] = c_front[:, 3:]
            c_smooth[:, 3] = args.cam_scale
            print('Intrinsics taken from mean ffhq cam and scaled', c_smooth[0, 3:])

        savgol_win = args.savgol_win
        flame_params_smooth = savgol_filter(flame_params.cpu().numpy(), window_length=savgol_win, polyorder=3, axis=0)
        flame_params_smooth = torch.tensor(flame_params_smooth).to(flame_params.device)

        # Inner loop: iterate over ID seeds
        for current_id_seed in tqdm(id_seeds, desc=f'Processing IDs for {vid_name}', leave=False):
            rng = torch.Generator(device)
            rng.manual_seed(current_id_seed)
            z = torch.randn((1, G._config.z_dim), device=device, generator=rng)

            flame_shape_id = torch.randint(0, ffhq_shapes.shape[0], (1,), generator=rng, device=device).cpu().item()
            fixed_shape = torch.tensor(ffhq_shapes[flame_shape_id, :300]).to(device)

            if args.joint_c_front:
                c_front = torch.tensor(ffhq_cams[flame_shape_id, :]).unsqueeze(0).to(device)
                sh_ref_cam = c_front

            # set fixed shape for this ID
            flame_params_smooth_id = flame_params_smooth.clone()
            flame_params_smooth_id[:, :300] = fixed_shape.repeat(len(flame_params), 1)

            if args.CACHE_BACKBONE:
                use_cached_backbone = False
                cache_backbone = True
            else:
                use_cached_backbone = False
                cache_backbone = False

            with torch.no_grad():
                w = torch.empty(len(flame_params), 1, 1)
                
                if fixed_view:
                    c = c_front.repeat(len(flame_params), 1)
                    view_folder = ''
                    c_render_seq = c
                else:
                    view_folder = 'dynamic_view'
                    c_mapping = c_front.repeat(len(flame_params), 1)
                    if G._config.use_concat:
                        c_mapping[:, 3:] = 0
                    
                all_frames = []
                all_generated_frames = []
                all_driving_frames = []
                
                for w_batch, c_mapping_batch, c_render_batch, fp_batch, driving_image_batch in tqdm(
                zip(
                    batchify_sliced(w, batch_size=batch_size), 
                    batchify_sliced(c_mapping, batch_size=batch_size),
                    batchify_sliced(c_smooth, batch_size=batch_size),
                    batchify_sliced(flame_params_smooth_id, batch_size=batch_size),
                    batchify_sliced(driving_images, batch_size=batch_size),
                ), total=len(flame_params) // batch_size + int(len(flame_params) % batch_size != 0), leave=False):
                    c_mapping_batch[:, :3] = 0
                    if MAPPING_TAKES_FLAME_PARAMS:
                        z_batch = z.repeat(len(w_batch), 1)
                        w_batch = G.mapping(z_batch, c_mapping_batch, truncation_psi=truncation_psi, flame_params=fp_batch)

                    if args.CACHE_BACKBONE and len(c_render_batch) != batch_size:
                        continue

                    if args.SYNTHESIS_FLAME_COND:
                        output = G.synthesis(w_batch, c_render_batch, fp_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const', \
                                             neural_rendering_resolution=args.resolution, \
                                             use_cached_backbone=use_cached_backbone, cache_backbone=cache_backbone)
                    else:
                        output = G.synthesis(w_batch, c_render_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const', \
                                             neural_rendering_resolution=args.resolution, \
                                             use_cached_backbone=use_cached_backbone, cache_backbone=cache_backbone)

                    if args.CACHE_BACKBONE:
                        use_cached_backbone = True
                        cache_backbone = False
                    
                    frames = []
                    generated_frames = []
                    driving_frames = []
                    
                    for image, d_image in zip(output['image'], driving_image_batch):
                        d_image_resized = cv2.resize(d_image.cpu().numpy(), (args.resolution, args.resolution))
                        generated_frame = Img.from_normalized_torch(image).to_numpy().img[..., :3]
                        
                        # Concatenated frame
                        frame = np.hstack([generated_frame, d_image_resized])
                        frames.append(frame)
                        
                        # Separate frames
                        generated_frames.append(generated_frame)
                        driving_frames.append(d_image_resized)
                    
                    all_frames.extend(frames)
                    all_generated_frames.extend(generated_frames)
                    all_driving_frames.extend(driving_frames)
                    
                    # if args.render_mode != 'RGB':
                    #     alpha_mask = output.masks[0].cpu().numpy()
                    #     depth = output.depth[0].cpu().numpy()
                    #     normal_map = convert_depth_to_normal_map(depth)
                        
                output_folder = os.path.join(
                    script_dir, "results",
                    f"{view_folder}",
                    f"{args.render_mode if args.render_mode != 'RGB' else ''}",
                    f"{args.run_name}_{args.checkpoint}"
                    f"__intrinsics_s_{args.cam_scale}"
                    f"{'_joint_c_front' if args.joint_c_front else ''}",
                    f"seed{current_id_seed}",
                )
                if args.use_narrow_mask:
                    output_folder = os.path.join(output_folder, 'narrow_mask')
                
                ensure_directory_exists(output_folder)
                
                # Save concatenated video (generated + driving side by side)
                mediapy.write_video(f"{output_folder}/{vid_name}.mp4", all_frames, fps=args.FPS)
                
                # Save generated avatar video separately with "left_" prefix
                mediapy.write_video(f"{output_folder}/left_{vid_name}.mp4", all_generated_frames, fps=args.FPS)
                
                # Save driving video in parent folder (only once, not per seed)
                parent_folder = os.path.dirname(output_folder)
                driving_video_path = f"{parent_folder}/driving_{vid_name}.mp4"
                if not os.path.exists(driving_video_path):
                    ensure_directory_exists(parent_folder)
                    mediapy.write_video(driving_video_path, all_driving_frames, fps=args.FPS)
                    print(f"Saved driving video: {driving_video_path}")

                print()
                print(f"Saved: seed{current_id_seed}/{vid_name}.mp4")
                print(f"Saved: seed{current_id_seed}/left_{vid_name}.mp4")
                print(f"http://0.0.0.0:8228{os.path.abspath(output_folder)}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)

