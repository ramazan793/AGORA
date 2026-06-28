from typing import Optional
import tyro
import pickle
import sys
import os
from glob import glob
from tqdm.auto import tqdm
import json
import time

import mediapy
import numpy as np
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
from src.gghead.constants import DEFAULT_INTRINSICS
from src.gghead.model_manager.finder import find_model_manager

from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

def smooth_extrinsics(flat_44, window=15, poly=3):
    """
    flat_44 : (T, 16) array of row-major 4×4 matrices
    window  : odd int ≥ poly+2   —  length of the Savitzky-Golay window (in frames)
    poly    : int < window       —  polynomial order for Savitzky-Golay
    returns : (T, 16)  smoothed in the same format
    """
    assert window % 2 == 1, "window_length must be odd"
    T = flat_44.shape[0]

    # --- reshape to (T, 4, 4) ----------------------------------------------
    M = flat_44.reshape(T, 4, 4)

    # --- 1. ROTATIONS -------------------------------------------------------
    Rmats = M[:, :3, :3]                       # (T, 3, 3)
    quats = R.from_matrix(Rmats).as_quat()     # (T, 4)  (x,y,z,w)

    # fix quaternion sign flips (enforce continuity)
    for t in range(1, T):
        if np.dot(quats[t-1], quats[t]) < 0:
            quats[t] *= -1

    # smooth each component, then renormalise
    quats_s = savgol_filter(quats, window, poly, axis=0)
    quats_s /= np.linalg.norm(quats_s, axis=1, keepdims=True)

    Rmats_s = R.from_quat(quats_s).as_matrix()  # back to matrices

    # --- 2. TRANSLATIONS ----------------------------------------------------
    trans = M[:, :3, 3]                        # (T, 3)
    trans_s = savgol_filter(trans, window, poly, axis=0)

    # --- 3. REASSEMBLE ------------------------------------------------------
    M_s = np.zeros_like(M)
    M_s[:, :3, :3] = Rmats_s
    M_s[:, :3, 3]  = trans_s
    M_s[:, 3, :]   = np.array([0, 0, 0, 1])

    return M_s.reshape(T, 16)


def parse_smirk_processed_video(emica_path, cam_path=None, gt_img_path=None, max_len=None):
    sl = slice(0, max_len, 1)
    
    frame_folders = sorted(glob(os.path.join(emica_path, f'*/')))

    if gt_img_path is None:
        frame_paths = sorted(glob(emica_path + f'/*/*/detections/*_000.png'))
    else:
        frame_paths = sorted(glob(gt_img_path + f'*.png'))
    # images = [np.array(Image.open(x)) for x in tqdm(frame_paths[sl], desc='Loading images')]
    images = None
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
        flameorth = np.load(f"{folder.replace('__smirk_estimations', '__smirk_estimations__no_crop')}/cam.npy") # to align with input video
        eyelid = np.load(f"{folder}/eyelid.npy")

        cam = np.array(raw_cams[i][1])
        i += 1

        shapecodes.append(shape)
        expcodes.append(exp)
        globalposes.append(globalpose)
        jawposes.append(jawpose)
        flameorths.append(flameorth)
        eyelids.append(eyelid)
        cams.append(cam)

    shapecodes = torch.tensor(np.array(shapecodes))
    expcodes = torch.tensor(np.array(expcodes))
    globalposes = torch.tensor(np.array(globalposes))
    jawposes = torch.tensor(np.array(jawposes))
    flameorths = torch.tensor(np.array(flameorths))
    eyelids = torch.tensor(np.array(eyelids))
    cams = torch.tensor(np.array(cams))
    images = torch.tensor(np.array(images)) if images is not None else None
    
    return shapecodes, expcodes, globalposes, jawposes, flameorths, eyelids, cams, images


DEVICE = 'cuda'
device = torch.device(DEVICE)

# vid_processed_path = '/home/r.fazylov/research_workspace/reenact_test_videos/obama_yt_emica/'
vid_processed_path = '/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/obama_next3d/smirk/'
cam_path = '/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/obama_next3d/preprocessed_dataset/dataset.json'
gt_img_path = '/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/obama_next3d/'

FPS = 30
vid_name = os.path.basename(vid_processed_path.rstrip(os.sep))
shapecodes, expcodes, globalposes, jawposes, flameorths, eyelids, cams, driving_images = \
    parse_smirk_processed_video(vid_processed_path, cam_path, gt_img_path, 1760)


cams_new = torch.cat([globalposes, flameorths], dim=-1)
flame_params = torch.cat([shapecodes, expcodes, globalposes, jawposes, eyelids], dim=-1)
flame_params = flame_params.to(device)


fixed_view = False
run_name = 'DGGHEAD-158'
checkpoint = 20500
seed = "2-3"
truncation_psi = 0.7
batch_size = 8
resolution = 512
inference_kwargs = {}

# scale_factor = np.sqrt(1/1)
# sample_mode = 'preactivation' # preactivation | final
# inference_kwargs = {
#     'inference_type' : 'resample_uv_gaussians',
#     'inference_options' : {
#         'scale_factor' : scale_factor,
#         'sample_mode' : sample_mode,
#     }                                                                                                   
# }


model_manager = find_model_manager(run_name)
checkpoint = model_manager._resolve_checkpoint_id(checkpoint)
G = model_manager.load_checkpoint(checkpoint, load_ema=True).to(device)
G.rendering_config.raster_backend = 'gsplat'
G.rendering_config.gsplat_raster_settings.rasterize_mode = 'classic'

SYNTHESIS_FLAME_COND = True
MAPPING_TAKES_FLAME_PARAMS_JUST_TO_RETURN_EXPCODE = \
    (G._config.use_mouth_branch or G._config.use_extended_uv_generation or G._config.use_flame_template_with_mouth) and not G._config.gen_flame_conditioning
G._config.use_flame_rasterization = 0
CACHE_BACKBONE = not G._config.gen_flame_conditioning

from src.gghead.util.flame_rasterizer import parse_flame_deca_cameras

c_front = torch.mean(cams_new, dim=0, keepdims=True).to(device)
c_front[:, :3] = 0 # eq. to identity rotation matrix

c_render = cams_new.clone().to(device)
c_smooth = savgol_filter(c_render.cpu().numpy(), window_length=7, polyorder=3, axis=0)
c_smooth = torch.tensor(c_smooth).to(device)

# J_transformed = torch.tensor(ss['J'][0:1], device = c_render.device).unsqueeze(0).float() # J_transformed are always same as J
# sh_ref_cam, intrinsics = parse_flame_deca_cameras(c_render[0:1], J_transformed)

sh_ref_cam = c_front


flame_params_smooth = savgol_filter(flame_params.cpu().numpy(), window_length=7, polyorder=3, axis=0)
flame_params_smooth = torch.tensor(flame_params_smooth).to(flame_params.device)



if isinstance(seed, str):
    seeds = range(*map(int, seed.split('-')))
else:
    seeds = [seed]

global_start_time = time.time()
total_time = 0
total_num_frames = 0

total_perftimes = [0]*8

for cur_seed in seeds:

    if CACHE_BACKBONE:
        use_cached_backbone = False
        cache_backbone = True
    else:
        use_cached_backbone = False
        cache_backbone = False
    
    with torch.no_grad():
        rng = torch.Generator(device)
        rng.manual_seed(cur_seed)
        z = torch.randn((1, G._config.z_dim), device=device, generator=rng)
    
        if not MAPPING_TAKES_FLAME_PARAMS_JUST_TO_RETURN_EXPCODE:
            # w = G.mapping(z, c_front, truncation_psi=truncation_psi)
            # w = w.repeat(len(flame_params), 1, 1)
            w = torch.empty(len(flame_params), 1, 1, device=device)
        else:
            w, _ = G.mapping(z, c_front, truncation_psi=truncation_psi, flame_params=flame_params_smooth[0:1])
            w = w.repeat(len(flame_params), 1, 1)
    
        if fixed_view:
            c = c_front.repeat(len(flame_params), 1)
            view_folder = ''
            c_render = c
        else:
            c = cams_new.to(device)
            view_folder = 'dynamic_view'
            c_render = c
            c_mapping = c_front.repeat(len(flame_params), 1)
            
        
        all_frames = []
        for w_batch, c_render_batch, c_mapping_batch, fp_batch in tqdm(
        zip(
            batchify_sliced(w, batch_size=batch_size), 
            # batchify_sliced(c_render, batch_size=batch_size),
            batchify_sliced(c_smooth, batch_size=batch_size), # smoothed camera
            batchify_sliced(c_mapping, batch_size=batch_size), # default camera
            # batchify_sliced(flame_params, batch_size=batch_size), # default flame params
            # batchify_sliced(flame_params_smooth_fix_shape, batch_size=batch_size), # default flame params
            batchify_sliced(flame_params_smooth, batch_size=batch_size), # smoothed flame params
            # batchify_sliced(flame_params[:1].repeat(flame_params.shape[0],1), batch_size=batch_size), # fixed expression
            # batchify_sliced(driving_images, batch_size=batch_size),
        ), total=len(flame_params) // batch_size + int(len(flame_params) % batch_size != 0)):
            # fp_batch = fp_batch.clone()
            # fp_batch[:, -3] = 0.25 # force open mouth
            
            start_time = time.time()

            if MAPPING_TAKES_FLAME_PARAMS_JUST_TO_RETURN_EXPCODE:
                expcode = fp_batch[:, G.n_shape:(G.n_shape + G.n_exp)]
                jawpose = fp_batch[:, (G.n_shape + G.n_exp + 3):(G.n_shape + G.n_exp + 8)]
                expcode = torch.cat([expcode, jawpose], dim=1)

                w_batch = (w_batch, expcode)
            else:
                z_batch = z.repeat(len(fp_batch), 1)
                w_batch = G.mapping(z_batch, c_mapping_batch, truncation_psi=truncation_psi, flame_params=fp_batch, c2=fp_batch)
            
            if CACHE_BACKBONE and len(fp_batch) != batch_size:
                # drop last unfull batch to respect batched cache
                continue
            
            if SYNTHESIS_FLAME_COND:
                output = G.synthesis(w_batch, c_render_batch, fp_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const', \
                                     neural_rendering_resolution=resolution, \
                                     use_cached_backbone=use_cached_backbone, cache_backbone=cache_backbone, \
                                     return_deformation_planes=False, return_uv_map=True, **inference_kwargs)
            else:
                output = G.synthesis(w_batch, c_render_batch, sh_ref_cam=sh_ref_cam, return_masks=True, noise_mode='const', \
                                     neural_rendering_resolution=resolution, \
                                     use_cached_backbone=use_cached_backbone, cache_backbone=cache_backbone, \
                                     **inference_kwargs)
            if isinstance(output, tuple):
                main_planes, deform_planes = output[1], output[2]
                output = output[0]
            
            end_time = time.time()
            total_time += end_time - start_time
            total_num_frames += len(output['image'])

            # for j, perftime in enumerate(perftimes):
            #     total_perftimes[j] += perftime
            
            if CACHE_BACKBONE:
                use_cached_backbone = True
                cache_backbone = False


            # frames = [Img.from_normalized_torch(image).to_numpy().img[..., :3] for image in output['image']]
    
            # frames = []
            # for image, d_image in zip(output['image'], driving_image_batch):
            #     d_image = cv2.resize(d_image.cpu().numpy(), (resolution, resolution))
            #     frame = np.hstack([Img.from_normalized_torch(image).to_numpy().img[..., :3], d_image])
            #     frames.append(frame)
            # all_frames.extend(frames)
            
        
        # output_folder = f"./dgghead_renderings/video_inference/{view_folder}/{vid_name}/smooth/{run_name}_{checkpoint}/"
        # output_folder = f"./dgghead_renderings/video_inference/{view_folder}/{vid_name}/{run_name}_{checkpoint}/expr_fix/"
        # ensure_directory_exists(output_folder)
        # mediapy.write_video(f"{output_folder}/{cur_seed:04d}.mp4", all_frames, fps=FPS)

        # print(f"Cumulative time: {total_time} seconds")
        # print(f"Cumulative FPS: {total_num_frames / total_time}")
    
    # print(f'http://0.0.0.0:8228{os.path.abspath(output_folder)}')

print(f"Model total time: {total_time} seconds")
print(f"FPS: {total_num_frames / total_time}")
print(f"Global time: {time.time() - global_start_time} seconds")


# print("\nDetailed times:")
# total_time = sum(total_perftimes)
# for j, tp in enumerate(total_perftimes):
#     if j == 7:
#         t = "r"
#     else:
#         t = j+1
#     print(f"Time {t}: {tp}")

# print("\nDetailed times (%):")
# for j, tp in enumerate(total_perftimes):
#     if j == 7:
#         t = "r"
#     else:
#         t = j+1
#     print(f"Time {t}: {tp/total_time * 100}")