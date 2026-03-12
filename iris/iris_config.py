"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

import torch

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PixelSamplerConfig

from iris.data.dataparsers import GenieBlenderDataParserConfig, GenieNerfstudioDataParserConfig
from iris.data.datamanagers import GenieDataManagerConfig
from iris.iris_trainer import IrisTrainerConfig
from iris.iris_model import IrisModelConfig
from iris.sampler.sampler_algorithms import GaussianIntersectionSamplerConfig
from iris.utils.schedulers import ChainedSchedulerConfig, CosineAnnealingSchedulerConfig, CustomExponentialLRSchedulerConfig, ExponentialSchedulerConfig
from iris.iris_pipeline import IrisPipelineConfig

MAX_NUM_ITERATIONS = 30_000

nerf_synth_transform_matrix = torch.tensor([
    [1.0, 0.0, 0.0, 1.5],
    [0.0, 1.0, 0.0, 1.5],
    [0.0, 0.0, 1.0, 1.5],
])
nerf_synth_scale_factor = 1.0 / 3.0

big_scenes_transform_matrix_1 = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
big_scenes_transform_matrix_2 = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])
big_scenes_scale_factor = 1.0

iris = MethodSpecification(
    config=IrisTrainerConfig(
        method_name="iris",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=MAX_NUM_ITERATIONS,
        freeze_means_step=31000,
        pipeline=IrisPipelineConfig(
            datamanager=GenieDataManagerConfig(
                dataparser=GenieBlenderDataParserConfig(
                    gauss_transform_matrix=nerf_synth_transform_matrix,
                    gauss_scale_factor=nerf_synth_scale_factor,
                    alpha_color=None
                ),
                pixel_sampler=PixelSamplerConfig(
                    rejection_sample_mask=False,
                ),
                train_num_rays_per_batch=2**16,
                eval_num_rays_per_batch=2**16,
            ),
            model=IrisModelConfig(
                sampler=GaussianIntersectionSamplerConfig(
                    renderer_param=6.25,
                    # renderer_param=11.3449,
                    use_tri_sampling=False,
                    hits_per_ray=128,
                    transform_matrix=nerf_synth_transform_matrix,
                    scale_factor=nerf_synth_scale_factor,
                ),
                densify=False,
                prune=False,
                eval_num_rays_per_chunk=2**16,
                near_plane=2.0,
                far_plane=6.0,
                background_color="random",
                disable_scene_contraction=True,
                cone_angle=0.0,
                use_per_gauss_weight=True,
                gauss_transform_matrix=nerf_synth_transform_matrix,
                gauss_scale_factor=nerf_synth_scale_factor,
                # near_plane=0.0,
                # far_plane=1e3,
                # background_color="random",
                # disable_scene_contraction=False,
                # cone_angle=1.0 / 256.0,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-06),
                "scheduler": ExponentialSchedulerConfig(max_steps=20000, min_lr=1e-4, init_lr=1e-2),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=10000, min_lr=1e-6, init_lr=1e-5),
            },
            "log_covs": {
                "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=MAX_NUM_ITERATIONS, min_lr=5e-4, init_lr=5e-5),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=10000, min_lr=1e-4, init_lr=1e-3),
            },
            "weights": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=20000, min_lr=1e-5, init_lr=1e-3),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Gaussian Splatting Encoded Neural Radiance Fields",
)

iris_real = MethodSpecification(
    config=IrisTrainerConfig(
        method_name="iris-real",
        steps_per_eval_image=31000,
        steps_per_eval_batch=500,
        max_num_iterations=MAX_NUM_ITERATIONS,
        freeze_means_step=31000,
        pipeline=IrisPipelineConfig(
            datamanager=GenieDataManagerConfig(
                dataparser=GenieNerfstudioDataParserConfig(
                    gauss_transform_matrix=big_scenes_transform_matrix_1,
                    gauss_scale_factor=big_scenes_scale_factor,
                    eval_mode="interval",
                    eval_interval=8,
                ),
                pixel_sampler=PixelSamplerConfig(
                    rejection_sample_mask=False,
                ),
                train_num_rays_per_batch=2**13,
                eval_num_rays_per_batch=2**13,
            ),
            model=IrisModelConfig(
                sampler=GaussianIntersectionSamplerConfig(
                    # renderer_param=6.25,
                    renderer_param=11.3449,
                    use_tri_sampling=False,
                    hits_per_ray=256,
                    transform_matrix=big_scenes_transform_matrix_2,
                    scale_factor=big_scenes_scale_factor,
                ),
                densify=False,
                prune=False,
                eval_num_rays_per_chunk=2**13,
                near_plane=0.05,
                far_plane=1e3,
                background_color="black",
                disable_scene_contraction=False,
                cone_angle=0.0,
                use_per_gauss_weight=True,
                gauss_transform_matrix=big_scenes_transform_matrix_2,
                gauss_scale_factor=big_scenes_scale_factor,
                encodings_dist_threshold=2.0,
                # near_plane=0.0,
                # far_plane=1e3,
                # background_color="random",
                # disable_scene_contraction=False,
                # cone_angle=1.0 / 256.0,
                ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-06),
                "scheduler": ExponentialSchedulerConfig(max_steps=20000, min_lr=1e-4, init_lr=1e-2),
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=10000, min_lr=1e-6, init_lr=1e-5),
            },
            "log_covs": {
                "optimizer": AdamOptimizerConfig(lr=5e-5, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=MAX_NUM_ITERATIONS, min_lr=5e-4, init_lr=5e-5),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=10000, min_lr=1e-4, init_lr=1e-3),
            },
            "weights": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialSchedulerConfig(max_steps=20000, min_lr=1e-5, init_lr=1e-3),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="viewer",
    ),
    description="Gaussian Splatting Encoded Neural Radiance Fields",
)
