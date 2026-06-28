import json
import os
import zipfile
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Optional, Tuple, Literal, Union

import PIL
import PIL.Image
import numpy as np
from eg3d.training.dataset import Dataset, pyspng
from elias.config import Config
from elias.util.io import resize_img, load_json
from glob import glob
import trimesh

from src.gghead.env import GGHEAD_DEPENDENCIES_PATH, REPO_ROOT_DIR

MaskMethod = Literal['deeplabv3', 'modnet']
@dataclass
class GGHeadImageFolderDatasetConfig(Config):
    path: str
    base_path: Optional[str] = None
    resolution: Optional[int] = None
    use_calibration: bool = False
    use_masks: bool = False
    mask_method: MaskMethod = 'modnet'
    apply_masks: bool = False  # If true, masks will not be returned as additional alpha channel but instead RGB will replaced with white for background
    return_masks: bool = False  # Only relevant if apply_masks=True. Will return mask as 4th channel in addition to applying it to image
    return_background: bool = False
    random_background: bool = False  # If true, background will be a random, solid color
    background_color: Tuple[int, int, int] = (255, 255, 255)  # Background color to use when apply_masks=True
    filter_ffhq: bool = False
    sample_weights_path: Optional[str] = None
    use_flame_cameras: int = 0
    preload_params: bool = True  # Preload FLAME and camera parameters into memory for faster iteration
    fused_params_path: Optional[str] = f'{REPO_ROOT_DIR}/assets/fused_params_dataset.npy' # If FLAME params and FLAME cams are saved as a single file

    # If 1, load precomputed FLAME rasterizations alongside images and return them.
    # Expected path convention: <stem>_flame_renderings.zip next to <path> (or a directory with same layout)
    precomputed_flame_renderings: int = 1

    max_size: Optional[int] = None  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
    use_labels: bool = False  # Enable conditioning labels? False = label dimension is zero.
    xflip: bool = False  # Artificially double the size of the dataset via x-flips. Applied after max_size.
    random_seed: int = 0  # Random seed to use when applying max_size.

    def get_eg3d_name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def get_eg3d_dict(self) -> dict:
        return dict(path=self.path, resolution=self.resolution, use_calibration=self.use_calibration, max_size=self.max_size, use_labels=self.use_labels,
                    xflip=self.xflip, random_seed=self.random_seed)

    def eval(self) -> 'GGHeadImageFolderDatasetConfig':
        # During evaluation, do not randomize background and always use white background
        eval_config = replace(self,
                              random_background=False,
                              background_color=(255, 255, 255),
                              sample_weights_path=None,
                              )
        return eval_config

    def get_eval_dict(self) -> dict:
        eval_config = self.eval()
        eval_dict = asdict(eval_config)
        return eval_dict


class GGHeadImageFolderDataset(Dataset):
    def __init__(self, config: GGHeadImageFolderDatasetConfig, mesh_ids = None):
        self._path = config.path
        self._zipfile = None
        self._use_calibration = config.use_calibration
        self._config = config

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)

        if config.filter_ffhq:
            filter_result = load_json(self.get_filtering_result_path())
            image_ids = [int(Path(n).stem[-8:]) for n in self._image_fnames]
            self._image_fnames = [image_fname for image_id, image_fname in zip(image_ids, self._image_fnames)
                                  if not self.is_filtered(filter_result['results'][f"{image_id}"])]

        if mesh_ids is not None:
            image_ids = [int(Path(n).stem[-8:]) for n in self._image_fnames]
            l = len(self._image_fnames)
            self._image_fnames = [image_fname for image_id, image_fname in zip(image_ids, self._image_fnames)
                                  if image_id in mesh_ids]
            print(f'Found {len(self._image_fnames)}/{l} meshes')

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self._resolution = config.resolution
        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if config.resolution is not None and (raw_shape[2] != config.resolution or raw_shape[3] != config.resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape,
                         max_size=config.max_size,
                         use_labels=config.use_labels,
                         xflip=config.xflip,
                         random_seed=config.random_seed)

    def is_filtered(self, filter_result: dict) -> bool:
        filter_mode = 'masked' if self._config.use_masks else 'regular'
        return (filter_result[filter_mode]['prob_mic'] > 0.80
                or filter_result[filter_mode]['prob_hand'] > 0.90
                or filter_result[filter_mode]['prob_multiple_persons'] > 0.70
                or filter_result[filter_mode]['n_detected_hands'] > 1
                or filter_result[filter_mode]['n_detected_isightfaces'] > 1
                )

    def get_filtering_result_path(self) -> str:
        base_path = self._config.base_path if self._config.base_path is not None else self._config.path
        return f"{Path(self._config.path).parent}/{Path(base_path).stem}_filtering_result.json"

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC

        if image.dtype == bool:
            image = image.astype(np.uint8) * 255  # bool -> np.uint8

            if self._resolution is not None:
                image = resize_img(image, self._resolution / image.shape[0], interpolation='nearest', use_opencv=True)[..., None]
        else:
            if self._resolution is not None:
                if image.shape[2] == 1:
                    image = resize_img(image[..., 0], self._resolution / image.shape[0])[..., None]
                else:
                    image = resize_img(image, self._resolution / image.shape[0])
        image = image.transpose(2, 0, 1)  # HWC => CHW

        return image

    def _load_raw_labels(self):
        if self._use_calibration:
            fname = 'dataset_calibration_fitted.json'
        else:
            fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels


class GGHeadMaskImageFolderDataset(Dataset):
    def __init__(self, config: GGHeadImageFolderDatasetConfig):
        self._config = config

        if config.use_masks:
            config_mask = replace(config)
            config_mask.path = f"{Path(config.path).parent}/{Path(config.path).stem}_masks_{config.mask_method}.zip"
            config_mask.base_path = config.path

            self._dataset_images = GGHeadImageFolderDataset(config)
            self._dataset_masks = GGHeadImageFolderDataset(config_mask)

            assert len(self._dataset_images) == len(self._dataset_masks)
        else:
            self._dataset_images = GGHeadImageFolderDataset(config)

        name = os.path.splitext(os.path.basename(self._config.path))[0]
        raw_shape = [len(self)] + list(self[0][0].shape)
        super().__init__(name=name, raw_shape=raw_shape,
                         max_size=config.max_size,
                         use_labels=config.use_labels,
                         xflip=config.xflip,
                         random_seed=config.random_seed)
        self._get_raw_labels()  # Ensure that _raw_labels_std is populated

    def get_filtering_result_path(self) -> str:
        return f"{Path(self._config.path).parent}/{Path(self._config.path).stem}_filtering_result.json"

    def __len__(self) -> int:
        return len(self._dataset_images)

    def __getitem__(self, index: int) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        if self._config.use_masks:
            image_output = self._dataset_images[index]
            mask_output = self._dataset_masks[index]

            if self._config.apply_masks:
                alpha_mask = mask_output[0] / 255.
                background_image = np.ones_like(image_output[0])
                if self._config.random_background:
                    background_color = np.random.rand(3)
                    background_image = background_image * background_color[:, None, None]
                else:
                    background_image = background_image * np.array(self._config.background_color)[:, None, None] / 255
                combined_image = (alpha_mask * image_output[0] / 255 + (1 - alpha_mask) * background_image) * 255
                combined_image = np.clip(np.round(combined_image), 0, 255).astype(np.uint8)

                if self._config.return_masks:
                    combined_image = np.concatenate([combined_image, mask_output[0]], axis=0)

                if self._config.random_background and self._config.return_background:
                    combined_image = np.concatenate([combined_image, background_image], axis=0)
            else:
                # Mask will simply be 4th channel of image
                combined_image = np.concatenate([image_output[0], mask_output[0]], axis=0)

            return combined_image, image_output[1]
        else:
            return self._dataset_images[index]

    def _load_raw_image(self, raw_idx):
        return self._dataset_images._load_raw_image(raw_idx)

    def _load_raw_labels(self):
        return self._dataset_images._load_raw_labels()

import torch

class DGGHeadMaskImageFolderDataset(Dataset):
    def __init__(self, config: GGHeadImageFolderDatasetConfig):
        self._config = config


        if config.use_flame_cameras:
            # mesh_path = '/ssd_new/r.fazylov/media/research/FFHQ_png_512/extracted_emica_ffhqcrop/'
            mesh_path = '/data2/ramazan.fazylov/media/datasets/FFHQ_png_512/smirk/crop/'
            # self._label_dim = 3 # condition D & G_mapping just on globalpose. dont let them memorize s, tx, ty  
            self._label_dim = 12
            img_id_split = slice(-8, None)
        else:
            mesh_path = '/data2/ramazan.fazylov/media/datasets/FFHQ_png_512/extracted_emica/'
            img_id_split = slice(-8, -2)
            self._label_dim = 25

        if 'data3' in self._config.path:
            mesh_path = mesh_path.replace("/data2/", "/data3/")

        self.use_shape_clusters = False
        self.n_shape_clusters = 128

        self.img_id_2_mesh_fname = {}
        for mesh_fname in glob(os.path.join(mesh_path, 'img*/shape.npy')):
            img_id = int(mesh_fname.split(os.sep)[-2][img_id_split])
            self.img_id_2_mesh_fname[img_id] = mesh_fname

        # # filter out extreme YAW (tmp)
        # import pickle
        # with open('/ssd_new/r.fazylov/media/research/FFHQ_png_512/extracted_emica_ffhqcrop/filter_extreme_yaws.pkl', 'rb') as f:
        #     filter_paths = pickle.load(f)
        
        # for img_id, mesh_fname in list(self.img_id_2_mesh_fname.items()):
        #     if mesh_fname in filter_paths:
        #         self.img_id_2_mesh_fname.pop(img_id)


        if config.use_masks:
            config_mask = replace(config)
            config_mask.path = f"{Path(config.path).parent}/{Path(config.path).stem}_masks_{config.mask_method}.zip"
            config_mask.base_path = config.path

            self._dataset_images = GGHeadImageFolderDataset(config, self.img_id_2_mesh_fname.keys())
            self._dataset_masks = GGHeadImageFolderDataset(config_mask, self.img_id_2_mesh_fname.keys())
            assert len(self._dataset_images) == len(self._dataset_masks)
        else:
            self._dataset_images = GGHeadImageFolderDataset(config, self.img_id_2_mesh_fname.keys())

        # Optional precomputed FLAME rasterizations aligned with images
        self._dataset_flame = None
        if config.precomputed_flame_renderings:
            config_flame = replace(config)
            # config_flame.path = f"{Path(config.path).parent}/{Path(config.path).stem}__flame_renderings__ambient_v2.zip"
            config_flame.path = f"{Path(config.path).parent}/{Path(config.path).stem}__flame_renderings__v4.zip"
            config_flame.base_path = config.path
            # try:
            # Keep alignment with filtered image set if applicable
            self._dataset_flame = GGHeadImageFolderDataset(
                config_flame,
                self.img_id_2_mesh_fname.keys()
            )
            # except Exception:
            #     # Fallback to unfiltered loading if mesh-id filtering is not desired
            #     self._dataset_flame = GGHeadImageFolderDataset(config_flame)
            assert len(self._dataset_images) == len(self._dataset_flame), \
                "Precomputed FLAME renderings count must match image count"
            print('Using precomputed FLAME renderings!')

        raw_idx = self._dataset_images._raw_idx
        self.ordered_image_ids = [int(Path(self._dataset_images._image_fnames[raw_idx[idx]]).stem[-8:]) for idx in range(len(self._dataset_images))]
        self._mesh_fnames = [self.img_id_2_mesh_fname[x] for x in self.ordered_image_ids]

        if config.sample_weights_path is not None:
            import pandas as pd
            df = pd.read_csv(config.sample_weights_path).set_index('ffhq_img_id')
            self._sample_weights = [df.loc[img_id]['final_weight'] for img_id in self.ordered_image_ids]

        print('Found meshes: ', len(self._mesh_fnames))

        # Optional preloading of FLAME and camera parameters to eliminate per-iteration small-file I/O
        self._flame_params_all = None
        self._cam_params_all = None
         
        if self._config.preload_params and self._config.use_flame_cameras:
            if not self._config.fused_params_path:
                # Preload FLAME params
                flame_params_list = [self._load_flame_parameters_from_disk(i) for i in range(len(self._mesh_fnames))]
                if len(flame_params_list) > 0:
                    self._flame_params_all = np.stack(flame_params_list).astype(np.float32, copy=False)

                # # Preload camera params if used
                # if self._config.use_flame_cameras:
                cam_params_list = [self._load_camera_parameters_from_disk(i) for i in range(len(self._mesh_fnames))]
                if len(cam_params_list) > 0:
                    self._cam_params_all = np.stack(cam_params_list).astype(np.float32, copy=False)
            else:
                fused_params = np.load(self._config.fused_params_path)
                self._flame_params_all = fused_params[:, :-6]
                self._cam_params_all = fused_params[:, -6:]
                print('Pre-loaded fused FLAME and camera params!')

        name = os.path.splitext(os.path.basename(self._config.path))[0]
        raw_shape = [len(self)] + list(self[0][0].shape)
        super().__init__(name=name, raw_shape=raw_shape,
                         max_size=config.max_size,
                         use_labels=config.use_labels,
                         xflip=config.xflip,
                         random_seed=config.random_seed)
        self._get_raw_labels()  # Ensure that _raw_labels_std is populated

    def get_filtering_result_path(self) -> str:
        return f"{Path(self._config.path).parent}/{Path(self._config.path).stem}_filtering_result.json"

    def __len__(self) -> int:
        return len(self._dataset_images)

    def __getitem__(self, index: int) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        flame_output = self.get_flame_parameters(index, return_weight=False)
        # override camera parameters: take from FLAME estimations
        if self._config.use_flame_cameras:
            camera_parameters = self.get_camera_parameters(index)

        if self._config.use_masks:
            image_output = self._dataset_images[index]
            mask_output = self._dataset_masks[index]

            if self._config.apply_masks:
                alpha_mask = mask_output[0] / 255.
                background_image = np.ones_like(image_output[0])
                if self._config.random_background:
                    background_color = np.random.rand(3)
                    background_image = background_image * background_color[:, None, None]
                else:
                    background_image = background_image * np.array(self._config.background_color)[:, None, None] / 255
                combined_image = (alpha_mask * image_output[0] / 255 + (1 - alpha_mask) * background_image) * 255
                combined_image = np.clip(np.round(combined_image), 0, 255).astype(np.uint8)

                if self._config.return_masks:
                    combined_image = np.concatenate([combined_image, mask_output[0]], axis=0)

                if self._config.random_background and self._config.return_background:
                    combined_image = np.concatenate([combined_image, background_image], axis=0)
            else:
                # Mask will simply be 4th channel of image
                combined_image = np.concatenate([image_output[0], mask_output[0]], axis=0)

            if self._dataset_flame is not None:
                flame_img = self._dataset_flame[index][0]
                if self._config.use_flame_cameras:
                    return combined_image, camera_parameters, flame_output, flame_img
                else:
                    return combined_image, image_output[1], flame_output, flame_img
            else:
                if self._config.use_flame_cameras:
                    return combined_image, camera_parameters, flame_output
                else:
                    return combined_image, image_output[1], flame_output

        else:
            if self._dataset_flame is not None:
                flame_img = self._dataset_flame[index][0]
                if self._config.use_flame_cameras:
                    return self._dataset_images[index][0], camera_parameters, flame_output, flame_img
                else:
                    img, lbl = self._dataset_images[index]
                    return img, lbl, flame_output, flame_img
            else:
                if self._config.use_flame_cameras:
                    return self._dataset_images[index][0], camera_parameters, flame_output
                else:
                    return self._dataset_images[index], flame_output

    def _load_raw_image(self, raw_idx):
        return self._dataset_images._load_raw_image(raw_idx)

    def _load_raw_labels(self):
        return self._dataset_images._load_raw_labels()

    def get_mesh(self, idx):
        mesh_fname = self._mesh_fnames[idx]
        mesh = np.array(trimesh.load(mesh_fname).vertices)
        return mesh

    def get_camera_parameters(self, idx):
        # Serve from cache if available
        if self._cam_params_all is not None:
            return self._cam_params_all[idx]

        return self._load_camera_parameters_from_disk(idx)

    def _load_camera_parameters_from_disk(self, idx):
        mesh_fname = self._mesh_fnames[idx]
        globalpose_fname = mesh_fname.replace('shape.npy', 'globalpose.npy')

        # we take s, tx, ty from the non-cropped (FFHQ) images to match FFHQ
        auxcam_fname = mesh_fname.replace('shape.npy', 'cam.npy').replace('/crop/', '/no_crop/')
        assert 'no_crop' in auxcam_fname

        globalpose = np.load(globalpose_fname)
        auxcam = np.load(auxcam_fname)
        cam = np.concatenate([globalpose, auxcam]).astype(np.float32, copy=False)
        return cam
        
    def get_flame_parameters(self, idx, return_weight: bool = False):
        # Serve from cache if available
        if self._flame_params_all is not None:
            params = self._flame_params_all[idx]
            if return_weight:
                weight = np.float32(self._sample_weights[idx])
                return params, weight
            return params

        params = self._load_flame_parameters_from_disk(idx)
        if return_weight:
            weight = np.float32(self._sample_weights[idx])
            return params, weight
        return params

    def _load_flame_parameters_from_disk(self, idx):
        mesh_fname = self._mesh_fnames[idx]

        if not self.use_shape_clusters:
            shape_fname = mesh_fname.replace('shape.npy', 'shape.npy')
            cluster_idx = None
        else:
            shape_fname = mesh_fname.replace('shape.npy', f'shapecode_cluster_{self.n_shape_clusters}.npy')
            cluster_idx = np.load(mesh_fname.replace('shape.npy', f'shapecode_cluster_id_{self.n_shape_clusters}.npy'))

        exp_fname = mesh_fname.replace('shape.npy', 'exp.npy')
        globalpose_fname = mesh_fname.replace('shape.npy', 'globalpose.npy')
        jawpose_fname = mesh_fname.replace('shape.npy', 'jawpose.npy')
        eyelid_fname = mesh_fname.replace('shape.npy', 'eyelid.npy')

        shape = np.load(shape_fname).astype(np.float32, copy=False)
        exp = np.load(exp_fname).astype(np.float32, copy=False)
        globalpose = np.load(globalpose_fname).astype(np.float32, copy=False)
        jawpose = np.load(jawpose_fname).astype(np.float32, copy=False)

        if os.path.exists(eyelid_fname):
            eyelid = np.load(eyelid_fname).astype(np.float32, copy=False)
            posecode = np.concatenate([globalpose, jawpose, eyelid], axis=-1).astype(np.float32, copy=False)
        else:
            posecode = np.concatenate([globalpose, jawpose], axis=-1).astype(np.float32, copy=False)

        params = np.concatenate([shape, exp, posecode], axis=0).astype(np.float32, copy=False)

        if self.use_shape_clusters:
            params = np.concatenate([params, cluster_idx.astype(np.float32, copy=False)], axis=0)

        return params


    def get_rendered_mesh(self, idx):
        mesh_fname = self._mesh_fnames[idx]
        mesh_rendered = PIL.Image.open(mesh_fname.replace('shape.npy', 'geometry.png')).convert('RGB')
        return np.asarray(mesh_rendered)

    @property
    def label_dim(self):
        return self._label_dim

    # Access a precomputed FLAME rendering for a given index (CHW uint8)
    def get_flame_rendering(self, idx: int) -> np.ndarray:
        if self._dataset_flame is None:
            raise RuntimeError('Precomputed FLAME renderings not enabled in dataset config')
        return self._dataset_flame[idx][0]

# Just to use workers for generator auxiliary sampling
class GeneratorSampleDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds: DGGHeadMaskImageFolderDataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        # Camera or label
        if self.base_ds._config.use_flame_cameras:
            c = self.base_ds.get_camera_parameters(idx)
        else:
            c = self.base_ds.get_label(idx)
        # FLAME params
        mesh = self.base_ds.get_flame_parameters(idx)
        # Precomputed flame image (CHW uint8)
        flame = self.base_ds.get_flame_rendering(idx)
        return c.astype(np.float32, copy=False), mesh.astype(np.float32, copy=False), flame
