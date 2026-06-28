import copy
import json
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, is_dataclass, replace
from datetime import timedelta
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import psutil
import torch
from dreifus.util.visualizer import ImageWindow
from eg3d import legacy, dnnlib
from eg3d.dnnlib import EasyDict
from eg3d.dnnlib.util import format_time, Logger
from eg3d.torch_utils import training_stats, custom_ops
from eg3d.torch_utils.misc import InfiniteSampler, print_module_summary, params_and_buffers, nan_to_num, constant, \
    check_ddp_consistency
from eg3d.torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from eg3d.training.augment import AugmentPipe
from eg3d.training.dual_discriminator import DualDiscriminator
# from eg3d.training.training_loop import setup_snapshot_image_grid, save_image_grid
from src.gghead.eg3d.torch_utils.misc import copy_params_and_buffers

from .metrics.metric_main_mesh import calc_metric, report_metric, register_metric

from eg3d.training.triplane import TriPlaneGenerator
from elias.util import ensure_directory_exists_for_file, ensure_directory_exists
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.gghead.config.gaussian_attribute import GaussianAttribute
from src.gghead.dataset.image_folder_dataset import DGGHeadMaskImageFolderDataset
from src.gghead.eg3d.loss import GGHeadStyleGAN2Loss
from src.gghead.model_manager.base_model_manager import GGHeadEvaluationConfig, GGHeadEvaluationResult
from src.gghead.model_manager.finder import find_model_manager
from src.gghead.model_manager.dgghead_model_manager import GGHeadExperimentConfig, DGGHeadModelFolder
from src.gghead.models.gaussian_discriminator import GaussianDiscriminator
from src.gghead.models.dyn_gaussian_discriminator import GaussianDiscriminator as DGaussianDiscriminator
from src.gghead.models.dyn_gghead_model import GGHeadModel
from src.gghead.util.logging import LoggerBundle
from src.gghead.util.metrics import fid100, fid1k, fid50k_full, fid10k


# ----------------------------------------------------------------------------


def subprocess_fn(rank: int, experiment_config: GGHeadExperimentConfig, c, temp_dir, name: Optional[str] = None):
    num_gpus = experiment_config.train_setup.gpus

    # Init torch.distributed.
    if num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=num_gpus, timeout=timedelta(minutes=30))

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop(experiment_config, rank=rank, name=name, **c)


# ----------------------------------------------------------------------------

def launch_training(experiment_config: GGHeadExperimentConfig, c, dry_run, name: Optional[str] = None):
    Logger(should_flush=True)

    num_gpus = experiment_config.train_setup.gpus

    # Print options.
    print()
    print('Training options:')

    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if is_dataclass(o):
                return asdict(o)
            return super().default(o)

    print(json.dumps(c, indent=2, cls=EnhancedJSONEncoder))
    print()
    # print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {num_gpus}')
    print(f'Batch size:          {experiment_config.optimizer_config.batch_size} images')
    print(f'Training duration:   {experiment_config.train_setup.total_kimg} kimg')
    print(f'Dataset path:        {experiment_config.dataset_config.path}')
    print(f'Dataset size:        {experiment_config.dataset_config.max_size} images')
    print(f'Dataset resolution:  {experiment_config.dataset_config.resolution}')
    print(f'Dataset labels:      {experiment_config.dataset_config.use_labels}')
    print(f'Dataset x-flips:     {experiment_config.dataset_config.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Launch processes.
    print('Launching processes...')
    # set_start_method('spawn') is essential for the combination of DataLoader and zip Dataset to work
    # Otherwise, get random errors on Unix systems where fork is the default start method since parallel reading of zipfiles doesn't work there
    torch.multiprocessing.set_start_method('spawn')
    with TemporaryDirectory() as temp_dir:
        if num_gpus == 1:
            subprocess_fn(experiment_config=experiment_config, rank=0, c=c, temp_dir=temp_dir, name=name)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(experiment_config, c, temp_dir, name), nprocs=num_gpus)


# ----------------------------------------------------------------------------


def training_loop(
        # @formatter:off
        experiment_config: GGHeadExperimentConfig,
        run_dir                 = '.',      # Output directory.
        # training_set_kwargs     = {},       # Options for training set.
        data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
        G_kwargs                = {},       # Options for generator network.
        # D_kwargs                = {},       # Options for discriminator network.
        augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
        # loss_kwargs             = {},       # Options for loss function.
        # metrics                 = [],       # Metrics to evaluate during training.
        random_seed             = 0,        # Global random seed.
        num_gpus                = 1,        # Number of GPUs participating in the training.
        rank                    = 0,        # Rank of the current process in [0, num_gpus[.
        batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
        ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
        augment_p               = 0,        # Initial value of augmentation probability.
        ada_target              = None,     # ADA target value. None = fixed p.
        ada_interval            = 4,        # How often to perform ADA adjustment?
        ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
        kimg_per_tick           = 4,        # Progress snapshot interval.
        image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
        network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
        resume_pkl              = None,     # Network pickle to resume training from.
        resume_kimg             = 0,        # First kimg to report when resuming training.
        cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
        abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
        use_gaussians: bool     = False,
        use_vis_window: bool    = False,
        name: Optional[str]     = None,
        # @formatter:on
):
    dataset_config = experiment_config.dataset_config
    model_config = experiment_config.model_config

    # ----------------------------------------------------------
    # Create Model manager
    # ----------------------------------------------------------
    generator_type = experiment_config.model_config.generator_type
    if rank == 0:
        run_desc = (f'{experiment_config.dataset_config.get_eg3d_name():s}'
                    f'-gpus{experiment_config.train_setup.gpus:d}'
                    f'-batch{experiment_config.optimizer_config.batch_size:d}'
                    f'-gamma{experiment_config.optimizer_config.loss_config.r1_gamma:g}'
                    f'-res{model_config.generator_config.img_resolution}')
        if experiment_config.train_setup.resume_run is not None:
            run_desc += f'-resume{experiment_config.train_setup.resume_run}'
        if name is not None:
            run_desc += "_" + name
        model_manager = DGGHeadModelFolder().new_run(run_desc)

        run_dir = model_manager.get_model_store_path()

        print("===============================================================")
        print(f"Start training {model_manager.get_run_name()}")
        print("===============================================================")

    Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    if rank == 0:
        ensure_directory_exists(model_manager.get_wandb_folder())
        wandb_logger = WandbLogger(
            project=experiment_config.train_setup.project_name,
            group=experiment_config.train_setup.group_name,
            name=model_manager.get_run_name(),
            config=experiment_config.to_json(),
            save_dir=model_manager.get_wandb_folder())
        # wandb API drift: newer wandb exposes the run id as `.id`; older builds used `._run_id`.
        _wandb_run = wandb_logger.experiment
        experiment_config.train_setup.wandb_run_id = getattr(_wandb_run, 'id', None) or getattr(_wandb_run, '_run_id', None)
        logger_bundle = LoggerBundle([wandb_logger], accumulate=experiment_config.train_setup.accumulate_metrics)
    else:
        # Other processes should not log anything
        logger_bundle = LoggerBundle()

    print(logger_bundle._current_step)

    # ----------------------------------------------------------
    # Initialize
    # ----------------------------------------------------------
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    # torch.backends.cuda.matmul.allow_tf32 = False  # Improves numerical accuracy.
    # torch.backends.cudnn.allow_tf32 = False # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_tf32 = True  # Significantly improves training speed.
    torch.backends.cudnn.allow_tf32 = True # Significantly improves training speed.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = False  # Avoids errors with the augmentation pipe.

    # ----------------------------------------------------------
    # Load training set
    # ----------------------------------------------------------
    if rank == 0:
        print('Loading training set...')
    training_set = DGGHeadMaskImageFolderDataset(dataset_config)
    eval_set = DGGHeadMaskImageFolderDataset(dataset_config.eval())
    training_set_sampler = InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # ----------------------------------------------------------
    # Construct Networks
    # ----------------------------------------------------------
    if rank == 0:
        print('Constructing networks...')
    # common_kwargs = dict(img_channels=training_set.num_channels)
    generator_config = model_config.generator_config
    do_resume = (experiment_config.train_setup.resume_run is not None) and (rank == 0)

    if generator_type == 'gaussians':
        G = GGHeadModel(generator_config, logger_bundle, post_init=not do_resume).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    elif generator_type == 'triplanes':
        G = TriPlaneGenerator(
            z_dim=generator_config.z_dim,
            c_dim=generator_config.c_dim,
            w_dim=generator_config.w_dim,
            img_resolution=generator_config.img_resolution,
            neural_rendering_resolution=generator_config.neural_rendering_resolution,
            img_channels=training_set.num_channels,
            mapping_kwargs=asdict(generator_config.mapping_network_config),
            **asdict(generator_config.synthesis_network_config),
            **G_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    else:
        raise ValueError(f"Unkown generator type: {generator_type}")

    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    discriminator_config = model_config.discriminator_config
    if use_gaussians or not discriminator_config.use_dual_discrimination:
        if discriminator_config.use_dual_discrimination:
            D = DGaussianDiscriminator(discriminator_config).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        else:
            D = GaussianDiscriminator(discriminator_config).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    else:
        D = DualDiscriminator(
            c_dim=discriminator_config.c_dim,
            img_resolution=discriminator_config.img_resolution,
            img_channels=discriminator_config.img_channels,
            architecture=discriminator_config.architecture,
            channel_base=discriminator_config.channel_base,
            channel_max=discriminator_config.channel_max,
            num_fp16_res=discriminator_config.num_fp16_res,
            conv_clamp=discriminator_config.conv_clamp,
            cmap_dim=discriminator_config.cmap_dim,
            disc_c_noise=discriminator_config.disc_c_noise,
            block_kwargs=asdict(discriminator_config.block_config),
            mapping_kwargs=asdict(discriminator_config.mapping_network_config),
            epilogue_kwargs=asdict(discriminator_config.epilogue_config)).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # If using precomputed flame renderings, disable generator-side flame rasterization
    if dataset_config.precomputed_flame_renderings:
        if hasattr(G, '_config') and hasattr(G._config, 'use_flame_rasterization'):
            G._config.use_flame_rasterization = False
        if hasattr(G_ema, '_config') and hasattr(G_ema._config, 'use_flame_rasterization'):
            G_ema._config.use_flame_rasterization = False

    # ----------------------------------------------------------
    # Resume from existing pickle
    # ----------------------------------------------------------
    if do_resume:
        resume_run = experiment_config.train_setup.resume_run
        checkpoint = experiment_config.train_setup.resume_checkpoint

        print(f'Resuming from {resume_run} - checkpoint {checkpoint}')
        model_manager_loaded = find_model_manager(resume_run)
        G_loaded = model_manager_loaded.load_checkpoint(checkpoint)
        G_ema_loaded = model_manager_loaded.load_checkpoint(checkpoint, load_ema=True)
        D_loaded = model_manager_loaded.load_discriminator(checkpoint)

        def copy_params(src_module: torch.nn.Module, dst_module: torch.nn.Module, require_all: bool = False):
            important_buffer_names = [  # "_uv_grid", "_flame_vertices",
                "_maintenance_pos_gradients", "_maintenance_gaussian_counts", "_maintenance_max_opacities",
                "_maintenance_position_maps", "_maintenance_average_position_map", "_maintenance_position_map_counts"]
            buffer_names = [k for k in dict(src_module.named_buffers()).keys() if k not in important_buffer_names]
            if buffer_names:
                print(f"source module {type(src_module)} has buffers {buffer_names} which won't be loaded'")

            important_src_buffers = [(k, p) for k, p in src_module.named_buffers() if k in important_buffer_names]
            important_dest_buffers = [(k, p) for k, p in dst_module.named_buffers() if k in important_buffer_names]
            src_tensors = dict(src_module.named_parameters())
            src_tensors.update(important_src_buffers)
            for name, tensor in chain(dst_module.named_parameters(), important_dest_buffers):
                assert (name in src_tensors) or (not require_all)
                if name in src_tensors:
                    try:
                        tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
                    except RuntimeError as e:
                        if not isinstance(src_module, GGHeadModel) and not 'torgb' in name:
                            raise e

                        # Assume, there is a shape mismatched because super-resolution module was added to a pre-trained model
                        target = torch.zeros_like(tensor)
                        cloned_src = src_tensors[name].detach().clone()
                        c_dst = 0
                        c_src = 0
                        n_channels_total = sum([att.get_n_channels(dst_module._config.gaussian_attribute_config) for att in dst_module._config.uv_attributes])
                        n_channels_exclude_color = n_channels_total - GaussianAttribute.COLOR.get_n_channels(dst_module._config.gaussian_attribute_config)
                        for attr in src_module._config.uv_attributes:
                            dim_channel = 0
                            n_channels_src = attr.get_n_channels(src_module._config.gaussian_attribute_config)
                            n_channels_dst = attr.get_n_channels(dst_module._config.gaussian_attribute_config)
                            if attr == GaussianAttribute.COLOR:
                                n_color_channels_src = src_module._config.gaussian_attribute_config.n_color_channels
                                n_color_channels_dst = dst_module._config.gaussian_attribute_config.n_color_channels
                                n_sh_dims = n_channels_src // n_color_channels_src
                                color_tensor_src = cloned_src[c_src: c_src + n_channels_src]
                                src_shape = color_tensor_src.shape
                                color_tensor_src = color_tensor_src.view(n_sh_dims, n_color_channels_src, *src_shape[1:])
                                zeros_tensor_src = torch.zeros((n_sh_dims, n_color_channels_dst - n_color_channels_src, *src_shape[1:]),
                                                               dtype=color_tensor_src.dtype, device=color_tensor_src.device)
                                torch.nn.init.normal_(zeros_tensor_src)
                                color_tensor_src = torch.cat([color_tensor_src, zeros_tensor_src], dim=1)
                                color_tensor_src = color_tensor_src.reshape(n_channels_dst, *src_shape[1:])
                                target[c_dst: c_dst + n_channels_dst] = color_tensor_src
                            else:
                                target[c_dst: c_dst + n_channels_src] = cloned_src[c_src: c_src + n_channels_src]  # TODO: Use dim_channel
                            c_src += n_channels_src
                            c_dst += n_channels_dst

                        print(f'Merging loaded tensor {cloned_src.shape} into model tensor {target.shape} for key {name}')
                        tensor.copy_(target).requires_grad_(tensor.requires_grad)

        copy_params(G_loaded, G, require_all=False)
        copy_params(G_ema_loaded, G_ema, require_all=False)

        if experiment_config.train_setup.resume_load_D:
            copy_params(D_loaded, D, require_all=False)
        
        resume_kimg = checkpoint
        if not experiment_config.train_setup.reset_cur_nimg:
            logger_bundle.set_step(resume_kimg * 1000)

    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            copy_params_and_buffers(resume_data[name], module, require_all=False)
    # ----------------------------------------------------------
    # Print network summary tables
    # ----------------------------------------------------------
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        if use_gaussians:
            if not dataset_config.use_flame_cameras:
                c = torch.tensor([1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0,
                                0, 0, 0, 1,
                                1, 0, 0,
                                0, 1, 0,
                                0, 0, 1], device=device, dtype=torch.float32).unsqueeze(0).repeat((batch_gpu, 1))
            else:
                c = torch.tensor([0, 0, 0, 8, 0, 0], device=device, dtype=torch.float32).unsqueeze(0).repeat((batch_gpu, 1))
        else:
            c = torch.empty([batch_gpu, G.c_dim], device=device)
        mesh = torch.empty([batch_gpu, G.n_shape + G.n_exp + 8], device=device)

        if G._config.gen_flame_conditioning:
            c2 = torch.empty([batch_gpu, G.c2_dim], device=device)
            img = print_module_summary(G, [z, c, mesh], named_inputs={'c2' : c2})
        else:
            img = print_module_summary(G, [z, c, mesh])
        
        if dataset_config.precomputed_flame_renderings:
            img_flame = torch.randn([img['image'].shape[0], 3, G.img_resolution, G.img_resolution], device=device)
            img['image_flame'] = img_flame

        if c.shape[1] == D.c_dim:
            if D._config.c2_dim > 0:
                c2 = torch.randn(c.shape[0], D._config.c2_dim, device=device)
                print_module_summary(D, [img, c], named_inputs={'c2': c2})
            else:
                print_module_summary(D, [img, c])
        else:
            if D._config.c2_dim > 0:
                c2 = torch.randn(c.shape[0], D._config.c2_dim, device=device)
                print_module_summary(D, [img, torch.randn(c.shape[0], D.c_dim, device=device)], named_inputs={'c2': c2})
            else:
                print_module_summary(D, [img, torch.randn(c.shape[0], D.c_dim, device=device)])

    # ----------------------------------------------------------
    # Setup Augmentation
    # ----------------------------------------------------------
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = AugmentPipe(**augment_kwargs).train().requires_grad_(False).to(device)
        # augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # ----------------------------------------------------------
    # Distribute across GPUs
    # ----------------------------------------------------------
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    try:
                        torch.distributed.broadcast(param, src=0)
                    except Exception as e:
                        print(module)
                        print(param.shape)
                        raise e

    if do_resume and hasattr(G, 'post_init'):
        G.post_init()
        G_ema.post_init()

    # ----------------------------------------------------------
    # Setup training phases
    # ----------------------------------------------------------
    if rank == 0:
        print('Setting up training phases...')

    loss = GGHeadStyleGAN2Loss(device=device, G=G, D=D, augment_pipe=augment_pipe,
                               config=experiment_config.optimizer_config.loss_config,
                               logger_bundle=logger_bundle)
    # loss = StyleGAN2Loss(device=device, G=G, D=D, augment_pipe=augment_pipe,
    #                      **asdict(experiment_config.optimizer_config.loss_config))  # subclass of training.loss.Loss
    phases = []
    generator_optimizer_config = experiment_config.optimizer_config.generator_optimizer_config
    discriminator_optimizer_config = experiment_config.optimizer_config.discriminator_optimizer_config

    D_reg_interval = int(16 * (32 / batch_size))
    if batch_size != 32:
        print(f"Batch_gpu is {batch_size} -> adjusting D_reg_interval to {D_reg_interval}")

    for name, module, opt_config, reg_interval in [('G', G, generator_optimizer_config, G_reg_interval),
                                                   ('D', D, discriminator_optimizer_config, D_reg_interval)]:
        if reg_interval is None:
            if name == 'G' and experiment_config.optimizer_config.separate_lr_template_offsets is not None:
                params = [
                    {'params': module.named_parameters(),
                     'lr': opt_config.lr,
                     'betas': (opt_config.beta1, opt_config.beta2),
                     'eps': opt_config.eps}
                ]
                opt = Adam(params=params)
            else:
                opt = Adam(params=module.parameters(), lr=opt_config.lr, betas=(opt_config.beta1, opt_config.beta2),
                           eps=opt_config.eps)  # subclass of torch.optim.Optimizer
            for _ in range(opt_config.n_phases):
                phases += [EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
            
            # deformation linearity regularization
            if name == 'G' and experiment_config.optimizer_config.loss_config.lambda_deform_linearity_reg > 0:
                phases += [EasyDict(name=name + '_deform_linearity_reg', module=module, opt=opt, interval=4)]
            # ID consistency
            if name == 'G' and experiment_config.optimizer_config.loss_config.lambda_id_consistency > 0:
                phases += [EasyDict(name=name + '_id_consistency', module=module, opt=opt, interval=1)]
            
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt = Adam(module.parameters(),
                       lr=opt_config.lr * mb_ratio,
                       betas=(opt_config.beta1 ** mb_ratio, opt_config.beta2 ** mb_ratio),
                       eps=opt_config.eps)
            for _ in range(opt_config.n_phases):
                phases += [EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
    
    print('Final phases setup:', [p.name for p in phases], ' Intervals: ', [p.interval for p in phases])

    # Generator samples DataLoader (for precomputed flames & faster gen sampling)
    gen_sample_iterator = None
    if dataset_config.precomputed_flame_renderings:
        from src.gghead.dataset.image_folder_dataset import GeneratorSampleDataset
        gen_fetch_batch = len(phases) * batch_size
        gen_sample_dataset = GeneratorSampleDataset(training_set)
        gen_sample_sampler = InfiniteSampler(dataset=gen_sample_dataset, rank=rank, num_replicas=num_gpus, seed=random_seed + 1337)
        gen_sample_iterator = iter(DataLoader(dataset=gen_sample_dataset,
                                              sampler=gen_sample_sampler,
                                              batch_size=gen_fetch_batch,
                                              **data_loader_kwargs))

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # ----------------------------------------------------------
    # Export sample images
    # ----------------------------------------------------------
    grid_size = None
    grid_z = None
    grid_c = None
    # Expression grid (new strategy) buffers
    flame_grid_size = None
    flame_grid_z = None
    flame_grid_c = None
    flame_grid = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels, meshes, images_meshes = setup_snapshot_image_grid(training_set=eval_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        # save_image_grid(images_meshes, os.path.join(run_dir, 'reals_meshes.png'), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_meshes = torch.from_numpy(meshes).float().to(device).split(batch_gpu)

        if experiment_config.model_config.generator_config.use_flame_rasterization or dataset_config.precomputed_flame_renderings:
            predefined_expression_image_paths: list[str] = [
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_1_FF_NR_C_head512/000060.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_1_FF_NR_C_head512/000240.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_1_FF_NR_C_head512/000300.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_1_FF_NR_C_head512/000450.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_1_FF_NR_C_head512/000660.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_1_FF_NR_C_head512/000780.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_2_MAU_strong_C_head512/000030.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_2_MAU_strong_C_head512/000240.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_2_MAU_strong_C_head512/000450.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_2_MAU_strong_C_head512/000330.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_2_MAU_strong_C_head512/000120.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_2_MAU_strong_C_head512/000600.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_3_MAU_mild_C_head512/000060.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_3_MAU_mild_C_head512/000150.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_3_MAU_mild_C_head512/000270.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_3_MAU_mild_C_head512/000360.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_3_MAU_mild_C_head512/000500.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/feed_evaluation_videos/head_crops/ID_3_MAU_mild_C_head512/000590.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/drive_images_gaia/1.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/drive_images_gaia/80.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/drive_images_gaia/160.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/drive_images_gaia/240.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/drive_images_gaia/320.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/drive_images_gaia/400.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/HDTF/RD_Radio23_000/000001.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/HDTF/RD_Radio23_000/000300.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/HDTF/RD_Radio23_000/000600.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/HDTF/RD_Radio23_000/000900.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/HDTF/RD_Radio23_000/001200.png",
                "/data2/ramazan.fazylov/media/dgghead_workspace/reenact_test_videos/HDTF/RD_Radio23_000/001500.png",
            ]

            if len(predefined_expression_image_paths) != 30:
                print(f"[warning] Expressions grid disabled: expected 30 image paths, got {len(predefined_expression_image_paths)}")
            else:
                import PIL
                # Save a single-row grid of the 30 real images once
                try:
                    reals_expr_imgs = []
                    for p in predefined_expression_image_paths:
                        img = PIL.Image.open(p).convert('RGB').resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
                        img = np.asarray(img, dtype=np.uint8)
                        img = np.transpose(img, (2, 0, 1))
                        reals_expr_imgs.append(img)
                    reals_expr_imgs = np.stack(reals_expr_imgs)
                    save_image_grid(reals_expr_imgs, os.path.join(run_dir, 'reals_expressions.png'), drange=[0, 255], grid_size=(30, 1))
                except Exception as e:
                    print(f"[warning] Failed to save reals_expressions.png: {e}")

                # 2) Build FLAME grid of size (n_rows x 30)
                n_rows = grid_size[1]
                width = 30
                flame_grid_size = (width, n_rows)

                rng = np.random.default_rng(42)

                # Fixed z per row
                flame_grid_z_full = torch.randn([n_rows, 1, G.z_dim], device=device).repeat(1, width, 1).view(-1, G.z_dim).to(device)
                flame_grid_z = flame_grid_z_full.split(batch_gpu)

                # Fixed camera per row
                if hasattr(training_set, 'get_camera_parameters') and training_set._config.use_flame_cameras:
                    row_c_list = [training_set.get_camera_parameters(i) for i in rng.integers(0, len(training_set), n_rows)]
                else:
                    row_c_list = [training_set.get_label(i) for i in rng.integers(0, len(training_set), n_rows)]
                row_c = torch.from_numpy(np.stack(row_c_list)).to(device)
                flame_grid_c_full = row_c.unsqueeze(1).repeat(1, width, 1).view(-1, row_c.shape[-1])
                flame_grid_c = flame_grid_c_full.split(batch_gpu)

                # Fixed shapecode per row (from training set), expressions/jaw/eyelid from 30 images
                row_params_np = np.stack([training_set.get_flame_parameters(i) for i in rng.integers(0, len(training_set), n_rows)])
                row_params = torch.from_numpy(row_params_np).float().to(device)  # [n_rows, flame_dim]
                flame_dim = row_params.shape[-1]
                flame_grid_shape = row_params.unsqueeze(1).repeat(1, width, 1).clone()  # [n_rows, width, flame_dim]

                exp_start = G.n_shape
                exp_end = G.n_shape + G.n_exp
                pose_start = exp_end
                jaw_start = pose_start + 3  # first 3 are global pose
                eyelid_start = pose_start + 6

                # Load params from the 30 images once
                exp_list = []
                jaw_list = []
                eyelid_list = []
                for p in predefined_expression_image_paths:
                    d = os.path.join(os.path.dirname(p), 'smirk', os.path.splitext(os.path.basename(p))[0])
                    exp = np.load(os.path.join(d, 'exp.npy')).astype(np.float32, copy=False)
                    jaw = np.load(os.path.join(d, 'jawpose.npy')).astype(np.float32, copy=False)
                    eyelid = np.load(os.path.join(d, 'eyelid.npy')).astype(np.float32, copy=False)
                    # Ensure sizes
                    exp = exp[:G.n_exp]
                    if exp.shape[0] < G.n_exp:
                        exp = np.pad(exp, (0, G.n_exp - exp.shape[0]))
                    jaw = jaw[:3]
                    if jaw.shape[0] < 3:
                        jaw = np.pad(jaw, (0, 3 - jaw.shape[0]))
                    eyelid = eyelid[:2]
                    if eyelid.shape[0] < 2:
                        eyelid = np.pad(eyelid, (0, 2 - eyelid.shape[0]))
                    exp_list.append(exp)
                    jaw_list.append(jaw)
                    eyelid_list.append(eyelid)

                exp_mat = torch.from_numpy(np.stack(exp_list)).float().to(device)          # [30, n_exp]
                jaw_mat = torch.from_numpy(np.stack(jaw_list)).float().to(device)          # [30, 3]
                eyelid_mat = torch.from_numpy(np.stack(eyelid_list)).float().to(device)    # [30, 2]

                # Broadcast over rows: per column j, use the j-th exp/jaw/eyelid for all rows
                for j in range(width):
                    flame_grid_shape[:, j, exp_start:exp_end] = exp_mat[j].unsqueeze(0).repeat(n_rows, 1)
                    flame_grid_shape[:, j, jaw_start:jaw_start + 3] = jaw_mat[j].unsqueeze(0).repeat(n_rows, 1)
                    flame_grid_shape[:, j, eyelid_start:eyelid_start + 2] = eyelid_mat[j].unsqueeze(0).repeat(n_rows, 1)

                flame_grid_full = flame_grid_shape.view(-1, flame_dim)
                flame_grid = flame_grid_full.split(batch_gpu)
                    


    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

    # ----------------------------------------------------------
    # Setup Viewer
    # ----------------------------------------------------------
    if use_vis_window:
        vis_img_buffer = np.zeros((generator_config.img_resolution, generator_config.img_resolution, 3), dtype=np.float32)
        image_window = ImageWindow(vis_img_buffer)
        z_valid = torch.randn((1, model_config.generator_config.z_dim), device=device)

    # ----------------------------------------------------------
    # Register evaluation metrics
    # ----------------------------------------------------------
    register_metric(fid100)
    register_metric(fid1k)
    register_metric(fid10k)
    register_metric(fid50k_full)

    # ----------------------------------------------------------
    # Store configs
    # ----------------------------------------------------------
    if rank == 0:
        model_manager.store_model_config(experiment_config.model_config)
        model_manager.store_dataset_config(dataset_config)
        model_manager.store_optimization_config(experiment_config.optimizer_config)
        model_manager.store_train_setup(experiment_config.train_setup)

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if experiment_config.train_setup.reset_cur_nimg:
        cur_nimg = 0
    else:
        cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    profiler = torch.autograd.profiler.profile(with_stack=True, profile_memory=True)
    profile_batch_idx = 10

    while True:
        if batch_idx == profile_batch_idx:
            profiler.__enter__()

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            if dataset_config.precomputed_flame_renderings:
                # Dataset returns an extra precomputed flame image per sample
                phase_real_img, phase_real_c, phase_real_mesh, phase_real_img_flame = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device, non_blocking=True).split(batch_gpu)
                phase_real_mesh = phase_real_mesh.to(device, non_blocking=True).float().split(batch_gpu)
                if discriminator_config.use_dual_discrimination:
                    phase_real_img_flame = (phase_real_img_flame.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                else:
                    phase_real_img_flame = [None] * len(phase_real_mesh)
            else:
                phase_real_img, phase_real_c, phase_real_mesh = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device, non_blocking=True).split(batch_gpu)
                phase_real_mesh = phase_real_mesh.to(device, non_blocking=True).float().split(batch_gpu)
                if discriminator_config.use_dual_discrimination:
                    phase_real_img_flame = [loss.G.render_flame_mesh(x) for x in phase_real_mesh]
                else:
                    phase_real_img_flame = [None] * len(phase_real_mesh)

            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            if dataset_config.precomputed_flame_renderings and gen_sample_iterator is not None:
                # Use prefetching dataloader for generator conditioning when precomputed flames are enabled
                gen_c_batch, gen_mesh_batch, gen_img_flame_batch = next(gen_sample_iterator)
                gen_c_batch = gen_c_batch.pin_memory().to(device, non_blocking=True)
                gen_mesh_batch = gen_mesh_batch.pin_memory().to(device, non_blocking=True)
                if discriminator_config.use_dual_discrimination:
                    gen_img_flame_batch = gen_img_flame_batch.pin_memory().to(device, non_blocking=True).to(torch.float32)
                    gen_img_flame_batch = gen_img_flame_batch / 127.5 - 1
                else:
                    gen_img_flame_batch = None

                all_gen_c = [x.split(batch_gpu) for x in gen_c_batch.split(batch_size)]
                all_gen_mesh = [x.float().split(batch_gpu) for x in gen_mesh_batch.split(batch_size)]
                if gen_img_flame_batch is not None:
                    all_gen_img_flame = [x.split(batch_gpu) for x in gen_img_flame_batch.split(batch_size)]
                else:
                    all_gen_img_flame = None
                all_gen_weights = [[None] * batch_size] * len(phases)
            else:
                # Random index fallback
                indices = np.random.randint(len(training_set), size=len(phases) * batch_size)
                if training_set._config.use_flame_cameras:
                    all_gen_c = [training_set.get_camera_parameters(i) for i in indices]
                else:
                    all_gen_c = [training_set.get_label(i) for i in indices]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device, non_blocking=True)
                all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

                # OPTION 1 (DGGHEAD-2): random sampling of meshes [WORKS]
                # if dataset_config.sample_weights_path is not None:
                #     all_mesh_and_weight = [training_set.get_flame_parameters(np.random.randint(len(training_set)), return_weight=True) for _ in range(len(phases) * batch_size)]
                #     all_gen_mesh = [x[0] for x in all_mesh_and_weight]
                #     all_gen_weights = [x[1] for x in all_mesh_and_weight]
                #     all_gen_weights = torch.from_numpy(np.stack(all_gen_weights)).float().pin_memory().to(device)
                #     all_gen_weights = [phase_gen_weight.split(batch_gpu) for phase_gen_weight in all_gen_weights.split(batch_size)]
                # else:
                #     all_gen_mesh = [training_set.get_flame_parameters(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
                #     all_gen_weights = [[None] * batch_size] * len(phases)

                all_gen_weights = [[None] * batch_size] * len(phases)

                # OPTION 2: joint sampling of (camera, mesh) [WORKS BETTER, cuz in opt 1 D can overfit to real's (camera, mesh) distribution]
                all_gen_mesh = [training_set.get_flame_parameters(i) for i in indices]
                all_gen_mesh = torch.from_numpy(np.stack(all_gen_mesh)).float().pin_memory().to(device, non_blocking=True)
                all_gen_mesh = [phase_gen_mesh.split(batch_gpu) for phase_gen_mesh in all_gen_mesh.split(batch_size)]

                if dataset_config.precomputed_flame_renderings and discriminator_config.use_dual_discrimination:
                    all_gen_img_flame = [training_set.get_flame_rendering(i) for i in indices]
                    all_gen_img_flame = torch.from_numpy(np.stack(all_gen_img_flame)).pin_memory().to(device, non_blocking=True).to(torch.float32)
                    all_gen_img_flame = all_gen_img_flame / 127.5 - 1
                    all_gen_img_flame = [phase_gen_img_flame.split(batch_gpu) for phase_gen_img_flame in all_gen_img_flame.split(batch_size)]
                else:
                    all_gen_img_flame = None

                # OPTION 3 (DGGHEAD-3): sample the same meshes as in the corresponding real samples  
                # all_gen_mesh = [phase_real_mesh for _ in range(len(phases))]



        # Execute training phases.
        for idx_phase, (phase, phase_gen_z, phase_gen_c, phase_gen_mesh, phase_gen_weight) in enumerate(zip(phases, all_gen_z, all_gen_c, all_gen_mesh, all_gen_weights)):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            if experiment_config.optimizer_config.freeze_generator and phase.name in ['Gmain', 'Gboth']:
                for k, p in phase.module.named_parameters():
                    if "super_resolution" not in k:
                        p.requires_grad_(False)

            if all_gen_img_flame is not None:
                phase_gen_img_flame = all_gen_img_flame[idx_phase]
            else:
                phase_gen_img_flame = [None] * len(phase_gen_z)

            for real_img, real_c, gen_z, gen_c, gen_mesh, real_img_flame, real_mesh, gen_weight, gen_img_flame in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_gen_mesh, phase_real_img_flame, phase_real_mesh, phase_gen_weight, phase_gen_img_flame):
                # Special paired sampling for G_id_consistency: duplicate z and shapecode across halves
                if phase.name == 'G_id_consistency':
                    bs = gen_z.shape[0]
                    # same z for both halves
                    gen_z = torch.cat([gen_z[:bs//2], gen_z[:bs//2]], dim=0)
                    # same shape code for both halves: achieved in loss by overriding second half's shapecode
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gen_mesh=gen_mesh, gain=phase.interval, cur_nimg=cur_nimg, real_img_flame=real_img_flame, gen_weight=gen_weight, real_mesh=real_mesh, gen_img_flame=gen_img_flame)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, (k, b) in zip(G_ema.buffers(), G.named_buffers()):
                if b.shape != b_ema.shape and ('flame_vertices' in k or 'uv_grid' in k or 'maintenance_' in k):
                    setattr(G_ema, k, b.clone())
                else:
                    b_ema.copy_(b)
            if isinstance(G_ema, TriPlaneGenerator):
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Profiler
        if batch_idx == profile_batch_idx:
            profiler.__exit__(*sys.exc_info())
            print(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        logger_bundle.log_metrics({
            'Progress/n_samples_seen': cur_nimg,
            'Progress/n_batches_seen': batch_idx
        }, step=cur_nimg)

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(constant(0, device=device)))

        if use_vis_window:
            with torch.no_grad():
                c_valid = torch.tensor(training_set.get_label(0), device=device).unsqueeze(0)
                rendering_dict = G.forward(z_valid, c_valid)
                vis_img_buffer[:] = (rendering_dict['image'][0].permute(1, 2, 0).cpu().numpy()[..., :3] + 1) / 2

        if hasattr(G, '_cnn_adaptor'):
            G._cnn_adaptor.progressive_update(cur_nimg / 1000)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # ----------------------------------------------------------
        # IMPORTANT: EVERYTHING BELOW HERE IS ONLY EXECUTED ONCE "PER TICK"
        # ----------------------------------------------------------

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()

        if rank == 0:
            fields = []
            fields += [f"tick {cur_tick:<5d}"]
            fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
            fields += [f"time {format_time(tick_end_time - start_time):<12s}"]
            fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
            fields += [f"sec/kimg {(tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3:<7.2f}"]
            fields += [f"maintenance {maintenance_time:<6.1f}"]
            fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2 ** 30:<6.2f}"]
            fields += [f"gpumem {torch.cuda.max_memory_allocated(device) / 2 ** 30:<6.2f}"]
            fields += [f"reserved {torch.cuda.max_memory_reserved(device) / 2 ** 30:<6.2f}"]
            fields += [f"augment {float(augment_pipe.p.cpu()) if augment_pipe is not None else 0:.3f}"]
            print(' '.join(fields))

            logger_bundle.log_metrics({
                'Progress/tick': cur_tick,
                'Progress/kimg': cur_nimg / 1e3,
                'Timing/total_sec': tick_end_time - start_time,
                'Timing/sec_per_tick': tick_end_time - tick_start_time,
                'Timing/sec_per_kimg': (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3,
                'Timing/maintenance_sec': maintenance_time,
                'Resources/cpu_mem_gb': psutil.Process(os.getpid()).memory_info().rss / 2 ** 30,
                'Resources/peak_gpu_mem_gb': torch.cuda.max_memory_allocated(device) / 2 ** 30,
                'Resources/peak_gpu_mem_reserved_gb': torch.cuda.max_memory_reserved(device) / 2 ** 30,
                'Progress/augment': float(augment_pipe.p.cpu()) if augment_pipe is not None else 0,
                'Timing/total_hours': (tick_end_time - start_time) / (60 * 60),
                'Timing/total_days': (tick_end_time - start_time) / (24 * 60 * 60)
            },
                step=cur_nimg)

        torch.cuda.reset_peak_memory_stats()

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or ((cur_tick % 4) == 0 and (cur_nimg / 1e3) < 100) or (cur_tick % image_snapshot_ticks == 0)):
            with torch.no_grad():
                # Stream outputs to CPU to minimize peak GPU memory
                images_cpu = []
                images_raw_cpu = []
                images_depth_cpu = []
                collect_flame_once = (
                    experiment_config.model_config.generator_config.use_flame_rasterization
                    and not dataset_config.precomputed_flame_renderings
                    and cur_tick < 10
                )
                if collect_flame_once:
                    images_flame_cpu = []

                for z, c, mesh in zip(grid_z, grid_c, grid_meshes):
                    o = G_ema(z=z, c=c, flame_params=mesh, noise_mode='const', c2=mesh, also_render_mouth=True)
                    images_cpu.append(o['image'].detach().to('cpu'))
                    images_raw_cpu.append(o['image_raw'].detach().to('cpu'))
                    depth_cpu = o['image_depth'].detach().to('cpu')
                    images_depth_cpu.append(-depth_cpu)
                    if collect_flame_once and ('image_flame' in o):
                        images_flame_cpu.append(o['image_flame'].detach().to('cpu'))
                    del o

                images = torch.cat(images_cpu).numpy()
                images_raw = torch.cat(images_raw_cpu).numpy()
                images_depth = torch.cat(images_depth_cpu).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1], grid_size=grid_size)
                # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_raw.png'), drange=[-1, 1], grid_size=grid_size)
                save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()],
                                grid_size=grid_size)

                if collect_flame_once and len(images_flame_cpu) > 0:
                    images_flame = torch.cat(images_flame_cpu).numpy()
                    save_image_grid(images_flame, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_flame.png'), drange=[-1, 1],
                                    grid_size=grid_size)

                if (experiment_config.model_config.generator_config.use_flame_rasterization or dataset_config.precomputed_flame_renderings) and (flame_grid is not None and flame_grid_z is not None and flame_grid_c is not None and flame_grid_size is not None):
                    # expressions grid
                    images_flame_grid_cpu = []
                    mouth_enabled = (
                        experiment_config.model_config.generator_config.use_mouth_branch
                        or experiment_config.model_config.generator_config.use_extended_uv_generation
                    )
                    if mouth_enabled:
                        images_mouth_cpu = []
                        images_wo_mouth_cpu = []
                    for z, c, mesh in zip(flame_grid_z, flame_grid_c, flame_grid):
                        o = G_ema(z=z, c=c, flame_params=mesh, noise_mode='const', c2=mesh, also_render_mouth=True)
                        images_flame_grid_cpu.append(o['image'].detach().to('cpu'))
                        if mouth_enabled:
                            images_mouth_cpu.append(o['image_mouth'].detach().to('cpu'))
                            images_wo_mouth_cpu.append(o['image_wo_mouth'].detach().to('cpu'))
                        del o
                    images_flame_grid = torch.cat(images_flame_grid_cpu).numpy()
                    save_image_grid(images_flame_grid, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_expressions.png'), drange=[-1, 1],
                                    grid_size=flame_grid_size, truncate_gh=flame_grid_size[1] // 2)

                    if mouth_enabled:
                        images_mouth = torch.cat(images_mouth_cpu).numpy()
                        save_image_grid(images_mouth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_expressions_mouth.png'), drange=[-1, 1],
                                        grid_size=flame_grid_size, truncate_gh=flame_grid_size[1] // 2)

                        images_wo_mouth = torch.cat(images_wo_mouth_cpu).numpy()
                        save_image_grid(images_wo_mouth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_expressions_wo_mouth.png'), drange=[-1, 1],
                                        grid_size=flame_grid_size, truncate_gh=flame_grid_size[1] // 2)

                    # expressions point cloud mode
                    images_flame_grid_pc_cpu = []
                    if mouth_enabled:
                        images_mouth_pc_cpu = []
                    for z, c, mesh in zip(flame_grid_z, flame_grid_c, flame_grid):
                        o = G_ema(z=z, c=c, flame_params=mesh, noise_mode='const', c2=mesh, inference_options={'fixed_scale' : -7.5, 'override_color' : None}, also_render_mouth=True)
                        images_flame_grid_pc_cpu.append(o['image'].detach().to('cpu'))
                        if mouth_enabled:
                            images_mouth_pc_cpu.append(o['image_mouth'].detach().to('cpu'))
                        del o
                    images_flame_grid = torch.cat(images_flame_grid_pc_cpu).numpy()
                    save_image_grid(images_flame_grid, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_expressions_point_cloud.png'), drange=[-1, 1],
                                    grid_size=flame_grid_size, truncate_gh=flame_grid_size[1] // 2)

                    if mouth_enabled:
                        images_mouth = torch.cat(images_mouth_pc_cpu).numpy()
                        save_image_grid(images_mouth, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_expressions_point_cloud_mouth.png'), drange=[-1, 1],
                                        grid_size=flame_grid_size, truncate_gh=flame_grid_size[1] // 2)

                # Free cached GPU memory after snapshot generation on rank 0
                torch.cuda.empty_cache()

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=asdict(dataset_config))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        try: 
                            check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        except Exception as e:
                            print(f"Error checking DDP consistency: {e}")
                            print(f"Module: {module}. Rank: {rank}. Device: {device}. Num GPUs: {num_gpus}")
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    if hasattr(module, '_logger_bundle'):
                        module._logger_bundle = None  # Don't persist wandb loggers. Otherwise, can get error during unpickling
                snapshot_data[name] = module
                del module  # conserve memory

            if rank == 0:
                if use_gaussians:
                    checkpoint_name = model_manager._checkpoints_folder.substitute(model_manager._checkpoint_name_format, cur_nimg // 1000)
                    snapshot_pkl = f"{model_manager._checkpoints_folder.get_location()}/{checkpoint_name}"
                    ensure_directory_exists_for_file(snapshot_pkl)
                else:
                    snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')

                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        metrics = experiment_config.train_setup.metrics
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
                evaluation_results = dict()
                for metric in metrics:
                    result_dict = calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                              dataset_kwargs=dataset_config.get_eval_dict(), num_gpus=1, rank=rank,
                                              device=device)
                    report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)

                    evaluation_results[metric] = result_dict.results[metric]

                evaluation_config = GGHeadEvaluationConfig(checkpoint=cur_nimg // 1000, load_ema=True)
                evaluation_result = GGHeadEvaluationResult(**evaluation_results)
                model_manager.store_evaluation_result(evaluation_config, evaluation_result)
        del snapshot_data  # conserve memory
        if rank == 0:
            torch.cuda.empty_cache()

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
                logger_bundle.log_metrics({
                    'Timing/' + phase.name: value
                }, step=cur_nimg)
            # training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()

        for name, value in stats_dict.items():
            logger_bundle.log_metrics({
                name: value.mean,
                'Progress/n_samples_seen': cur_nimg,
                'Progress/n_batches_seen': batch_idx
            }, step=cur_nimg)
        for name, value in stats_metrics.items():
            logger_bundle.log_metrics({
                f'Metrics/{name}': value,
                'Progress/n_samples_seen': cur_nimg,
                'Progress/n_batches_seen': batch_idx
            }, step=cur_nimg)

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------


# EG3D OVERRIDES #


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data (support datasets that may return extra items like precomputed flames)
    vals = [training_set[i] for i in grid_indices]
    triples = [(v[0], v[1], v[2]) for v in vals]
    images, labels, meshes = zip(*triples)
    # images_meshes = [training_set.get_rendered_mesh(i) for i in grid_indices]
    # images_meshes = np.transpose(np.stack(images_meshes), (0, 3, 1, 2))
    images_meshes = None

    return (gw, gh), np.stack(images), np.stack(labels), np.stack(meshes), images_meshes

import PIL
def save_image_grid(img, fname, drange, grid_size, truncate_gw=0, truncate_gh=0):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    if truncate_gw > 0:
        img = img[:, :truncate_gw, :, :, :]
        gw = truncate_gw
    if truncate_gh > 0:
        img = img[:truncate_gh, :, :, :, :]
        gh = truncate_gh
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
