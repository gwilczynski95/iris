"""
Implementation of Iris.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Callable, Any

import nerfacc
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.cuda.amp import GradScaler

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, profiler

from iris.field.field import IrisFastField
from iris.sampler.sampler_algorithms import GaussianIntersectionSampler, GaussianIntersectionSamplerConfig
from iris.utils.viewer_utils import ViewerGaussianSplats, ViewerPointCloud, ViewerAABB

@dataclass
class IrisModelConfig(ModelConfig):
    """Iris Model Config"""

    _target: Type = field(
        default_factory=lambda: IrisModel
    )  # We can't write `NGPModel` directly, because `NGPModel` doesn't exist yet
    """target class to instantiate"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: Union[int, List[int]] = 128
    """Resolution of the grid used for the field."""
    alpha_thre: float = 0.0
    """Threshold for opacity skipping."""
    cone_angle: float = 0.0
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = 0.005
    """Minimum step size for rendering."""
    near_plane: float = 0.0
    """How far along ray to start sampling."""
    far_plane: float = 1e10
    """How far along ray to stop sampling."""
    use_gradient_scaling: bool = True
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    appearance_embedding_dim: int = 32
    """Dimension of the appearance embedding."""
    background_color: Literal["random", "black", "white"] = "white"
    """
    The color that is given to masked areas.
    These areas are used to force the density in those regions to be zero.
    """
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""
    sampler: GaussianIntersectionSamplerConfig = field(default_factory=lambda: GaussianIntersectionSampler())
    """KNN algorithm to use for nearest neighbor search."""
    max_gb: int = 20
    """Maximum amount of GPU memory to use for densification."""
    densify: bool = True
    """Whether to densify points or not. If False, the model will not densify."""
    prune: bool = True
    """Whether to prune the model or not. If False, the model will not prune."""
    unfreeze_means: bool = True
    """Whether to unfreeze the means of the encoder or not."""
    unfreeze_covs: bool = True
    """Whether to unfreeze the covariance matrices of the encoder or not."""
    use_per_gauss_weight: bool = False
    """Whether to use per gauss weight or not."""
    gauss_transform_matrix: torch.Tensor = None
    """Transform matrix to apply to the Gaussian means."""
    gauss_scale_factor: float = 1.0
    """Scale factor to apply to the Gaussian means."""
    encodings_dist_threshold: float = 2.0
    """Threshold for the encoding distance."""
    rca_n_neighbors: int = 2
    "No of neighbors on ray"
    
class IrisModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: IrisModelConfig
    field: IrisFastField

    def __init__(self, config: IrisModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Get seed points
        seed_points = self.kwargs.get("seed_points", None)
            
        # Initilize field
        self.field = IrisFastField(
            aabb=self.scene_box.aabb,
            appearance_embedding_dim=self.config.appearance_embedding_dim if self.config.use_appearance_embedding else 0,
            num_images=self.num_train_data,
            spatial_distortion=scene_contraction,
            seed_points=seed_points,
            densify=self.config.densify,
            prune=self.config.prune,
            unfreeze_means=self.config.unfreeze_means,
            unfreeze_covs=self.config.unfreeze_covs,
            use_per_gauss_weight=self.config.use_per_gauss_weight,
            gauss_transform_matrix=self.config.gauss_transform_matrix,
            gauss_scale_factor=self.config.gauss_scale_factor,
            dist_threshold=self.config.encodings_dist_threshold,
            n_neighbors=self.config.rca_n_neighbors
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)
        
        # Auto step size: ~1000 samples in the base level grid
        if self.config.render_step_size is None:
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000

        # Sampler
        self.sampler = self.config.sampler.setup()
        self.sampler.set_aabb(self.scene_box.aabb)
        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # Losses
        self.rgb_loss = F.smooth_l1_loss

        # Metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Point Cloud Viewer
        # self.viewer_point_cloud_handle = ViewerPointCloud(
        #     name="means", 
        #     aabb=self.scene_box, 
        #     points=self.field.mlp_base.encoder.gauss_params["means"].detach().cpu().numpy(),
        #     confidence=self.field.mlp_base.encoder.confidence.detach().cpu().numpy(),
        # )
        # self.viewer_gaussian_splats_handle = ViewerGaussianSplats(
        #     name="gausses", 
        #     aabb=self.scene_box, 
        #     means=self.field.mlp_base.encoder.gauss_params["means"].detach().cpu().numpy(),
        #     covariances=torch.exp(self.field.mlp_base.encoder.gauss_params["log_covs"]).detach().cpu().numpy(),
        #     quats=self.field.mlp_base.encoder.gauss_params["quats"].detach().cpu().numpy(),
        #     confidence=self.field.mlp_base.encoder.confidence.detach().cpu().numpy()
        # )
        self.viewer_aabb_handle = ViewerAABB(
            name="aabb",
            aabb=self.scene_box,
        )

        self.grad_scaler = GradScaler(2**10)
    
    def occ_eval_fn(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = 128**3
        density_list = []
        for i in range(0, x.shape[0], chunk_size):
            chunk_x = x[i : i + chunk_size]
            density_list.append(self.field.density_fn(chunk_x))
        
        return torch.cat(density_list, dim=0) * self.config.render_step_size

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_geometry(step: int):
            self.sampler.update_geometry(
                means=self.field.mlp_base.encoder.gauss_params["means"].detach(),
                log_covs=self.field.mlp_base.encoder.gauss_params["log_covs"].detach(),
                rots=self.field.mlp_base.encoder.quats.detach(),
            )

        return [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=20,
                func=update_geometry,
            ),
        ]

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")

        fields = []
        for name, param in self.field.named_parameters():
            if name == "mlp_base.encoder.gauss_params.means":
                param_groups["means"] = [param]
            elif name == "mlp_base.encoder.gauss_params.log_covs":
                param_groups["log_covs"] = [param]
            elif name == "mlp_base.encoder.gauss_params.quats":
                param_groups["quats"] = [param]
            elif name == "mlp_base.encoder.gauss_params.weights":
                param_groups["weights"] = [param]
            else:
                fields.append(param)

        param_groups["fields"] = fields

        return param_groups
    
    def densify_points(self, optimizers: Dict[str, torch.optim.Optimizer]) -> bool:
        # Check memory usage before densifying
        used_gb = torch.cuda.memory_reserved() / 1e9
        if used_gb > self.config.max_gb:
            print(f"[Densification] Skipped: CUDA memory usage {used_gb:.2f}GB > {self.config.max_gb}GB")
            return False
        
        if self.config.densify:
            # Calculate scene extent for splitting threshold
            extent = self.scene_aabb[3:] - self.scene_aabb[:3]
            scene_extent = extent.max().item()
            
            self.field.mlp_base.encoder.densify_and_split(optimizers, scene_extent=scene_extent)
            return True
            
        return False

    @profiler.time_function
    def get_outputs(self, ray_bundle: RayBundle, direction_transform: torch.Tensor = None):
        assert self.field is not None
        num_rays = len(ray_bundle)
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.config.near_plane,
                far_plane=self.config.far_plane,
            )
        
        if direction_transform is not None:
            gaussian_indices = ray_samples.metadata["gaussian_indices"]
            closest_direction_transforms = direction_transform[gaussian_indices]

        else:
            closest_direction_transforms = None
        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        
        if ray_samples.frustums.starts.numel() == 0:
            ray_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
            packed_info = nerfacc.pack_info(ray_indices, num_rays)
            rgb_output = torch.tensor([[0., 0., 0.]], device="cuda", dtype=torch.float32)
            weights = torch.tensor([[0]], device="cuda", dtype=torch.float32)
            t_hits = torch.tensor([[self.config.near_plane + 1e-3]], device="cuda", dtype=torch.float32)
        else:
            field_outputs = self.field(ray_samples, direction_transform=closest_direction_transforms, ray_indices=ray_indices)
            rgb_output = field_outputs[FieldHeadNames.RGB]
            transmittance = nerfacc.render_transmittance_from_alpha(
                alphas=field_outputs["alpha"][..., 0],
                packed_info=packed_info,
                n_rays=num_rays
            )[..., None]
            weights = transmittance * field_outputs["alpha"]
            t_hits = ray_samples.frustums.gaussian_t
            
        rgb = self.renderer_rgb(
            rgb=rgb_output,
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        accumulation = self.renderer_accumulation(
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays
        )
        
        # calculate depth
        depth = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=t_hits,
            ray_indices=ray_indices,
            n_rays=num_rays
        )

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "mip_loss": torch.zeros_like(depth),
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        rgb_loss = self.rgb_loss(image, pred_rgb)
        mip_loss = outputs["mip_loss"].mean() * 1e-3

        loss = rgb_loss + mip_loss
        if self.config.use_gradient_scaling:
            loss = self.grad_scaler.scale(loss)

        loss_dict = {"loss": loss}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        image = image[:rgb.shape[0], :rgb.shape[1], ...]  # Ensure image and rgb have the same batch size

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None, direction_transform: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Assumes a ray-based model.

        Args:
            camera: generates raybundle
        """
        if not self.sampler._geometry_set:
            self.sampler.update_geometry(
                means=self.field.mlp_base.encoder.gauss_params["means"].detach(),
                log_covs=self.field.mlp_base.encoder.gauss_params["log_covs"].detach(),
                rots=self.field.mlp_base.encoder.quats.detach(),
            )
        return self.get_outputs_for_camera_ray_bundle(
            camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box), direction_transform=direction_transform
        )

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, direction_transform: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        from collections import defaultdict
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle, direction_transform=direction_transform)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def forward(self, ray_bundle: Union[RayBundle, Cameras], direction_transform: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, direction_transform=direction_transform)
