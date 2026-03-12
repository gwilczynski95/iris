from dataclasses import dataclass, field
import torch
from typing import Optional, Tuple, Type

# Nerfstudio imports
from nerfstudio.cameras.rays import RayBundle, RaySamples, Frustums
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.model_components.ray_samplers import Sampler
from nerfstudio.utils import profiler

# Your custom renderer import
from iris.sampler import optix_sampler
from iris.sampler.utils import GenieFrustums

def intersect_unit_cube(origins: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes ray intersection with the unit cube [0, 1]^3 using the Slab Method.
    
    Args:
        origins: [N, 3] Normalized ray origins
        directions: [N, 3] Normalized ray directions
        
    Returns:
        t_entry: [N, 1] Distance where ray enters the cube
        t_exit: [N, 1] Distance where ray exits the cube
    """
    # Define Unit Cube bounds (0.0 to 1.0)
    zeros = torch.zeros_like(origins)
    ones = torch.ones_like(origins)
    
    # Inverse direction (handle division by zero gracefully via IEEE 754 infinity)
    inv_d = 1.0 / directions

    t0 = (zeros - origins) * inv_d
    t1 = (ones - origins) * inv_d

    # Swap so t0 is min and t1 is max
    t_min_vals = torch.minimum(t0, t1)
    t_max_vals = torch.maximum(t0, t1)
    
    # Largest of the mins is entry, Smallest of the maxs is exit
    t_entry = t_min_vals.max(dim=-1, keepdim=True)[0]
    t_exit = t_max_vals.min(dim=-1, keepdim=True)[0]
    
    return t_entry, t_exit

@dataclass
class GaussianIntersectionSamplerConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: GaussianIntersectionSampler)
    """Configuration for OptiX KNN algorithm."""
    renderer_param: float = 11.3449
    device: str = 'cuda'
    hits_per_ray: int = 128
    """Number of hits per ray to sample."""
    use_tri_sampling: bool = False
    """If true, samples entry, mid, and exit points for each Gaussian intersection."""
    transform_matrix: torch.Tensor = None
    """Transform matrix to apply to the Gaussian means."""
    scale_factor: float = 1.0
    """Scale factor to apply to the Gaussian means."""
    
class GaussianIntersectionSampler(Sampler):
    def __init__(
        self, 
        config: GaussianIntersectionSamplerConfig
    ):
        """
        Args:
            config: The configuration for the sampler.
        """
        super().__init__()
        self.config = config
        
        # Initialize the custom renderer lazily
        self.renderer = optix_sampler.CPyOptiXIrisRenderer(self.config.renderer_param, self.config.hits_per_ray)
        
        # Keep track if geometry has been set at least once to prevent crashes
        self._geometry_set = False
        self.aabb = None
        self.transform_matrix = config.transform_matrix.cuda()
        self.scale_factor = config.scale_factor
    
    def set_aabb(self, aabb: torch.Tensor):
        self.aabb = aabb

    @profiler.time_function
    def update_geometry(self, means: torch.Tensor, log_covs: torch.Tensor, rots: torch.Tensor):
        """
        Updates the geometry in the OptiX renderer.
        Can be called manually or via a TrainingCallback.
        """
        # Ensure inputs are contiguous and on the correct device if necessary.
        # Assuming the C++ extension expects CUDA tensors given it is OptiX.
        assert means.is_cuda, "means must be on GPU"
        assert log_covs.is_cuda, "log_covs must be on GPU"
        
        scales = torch.sqrt(torch.exp(log_covs)).contiguous()
        rots = rots / torch.linalg.norm(rots, dim=-1, keepdim=True)
        rots = rots.contiguous()
        self.scales = scales

        self.renderer.SetGeometry(
            means.contiguous(), 
            scales, 
            rots
        )
        self._geometry_set = True

    @profiler.time_function
    def _sample_layer(self, origins: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the OptiX sampler multiple times, peeling back layers of geometry.
        """
        with profiler.time_function("optix_sample"):
            t_data, _delta, indices_data = self.renderer.Sample(
                origins,
                directions,
            )
            
        # Permute to [N_rays, Buffer_Size]
        t_data = t_data.permute(1, 0)
        delta_data = _delta.permute(1, 0)
        indices_data = indices_data.permute(1, 0).long()
        
        return t_data, delta_data, indices_data

    @profiler.time_function
    def generate_ray_samples(
            self, 
            ray_bundle: RayBundle, 
            near_plane: float = 0.0,
            far_plane: Optional[float] = None,
            num_samples: Optional[int] = None
        ) -> Tuple[RaySamples, torch.Tensor]:
        if not self._geometry_set or self.renderer is None:
            raise RuntimeError("update_geometry must be called before sampling.")

        # --- 1. Prepare Inputs & Get Raw Hits ---
        origins, directions, num_rays, device = (
            ray_bundle.origins.contiguous(),
            ray_bundle.directions.contiguous(),
            ray_bundle.origins.shape[0],
            ray_bundle.origins.device,
        )
        origins_optix, directions_optix = self._prepare_inputs_for_optix(
            origins, directions, num_rays, device, far_plane
        )
        t_data, delta_data, indices_data = self._sample_layer(
            origins_optix.contiguous(), directions_optix.contiguous()
        )

        valid_mask = t_data != float("inf")
        if not valid_mask.any():
            return self._get_empty_samples(ray_bundle)

        # --- 2. Augment and Sort Samples ---
        raw_hits_t = t_data[valid_mask]
        raw_hits_delta = delta_data[valid_mask]
        raw_hits_indices = indices_data[valid_mask]

        ray_indices_grid = torch.arange(num_rays, device=device).unsqueeze(1).expand(-1, t_data.shape[1])
        raw_hits_ray_indices = ray_indices_grid[valid_mask]

        # --- FIX: Calculate a physically-based sample width ---
        samples_width_norm = self.scales[raw_hits_indices].max(dim=-1)[0]
        if self.aabb is not None:
            valid_directions_optix = directions_optix[raw_hits_ray_indices]
            dilation = 1.0 / (torch.linalg.vector_norm(valid_directions_optix, dim=-1) + 1e-8)
            samples_width_world = samples_width_norm * dilation
        else:
            samples_width_world = samples_width_norm

        if self.config.use_tri_sampling:
            original_dtype = raw_hits_t.dtype
            t_mid_double, delta_double = raw_hits_t.double(), raw_hits_delta.double()
            aug_t = torch.cat([t_mid_double - delta_double, t_mid_double, t_mid_double + delta_double]).to(
                original_dtype
            )
            aug_indices = raw_hits_indices.repeat(3)
            aug_ray_indices = raw_hits_ray_indices.repeat(3)
            aug_width = samples_width_world.repeat(3)

            sorting_key = aug_ray_indices.double() * (aug_t.max() + 1.0).double() + aug_t.double()
            sort_indices = torch.argsort(sorting_key)
            sorted_t, sorted_indices, sorted_ray_indices, sorted_width = (
                aug_t[sort_indices],
                aug_indices[sort_indices],
                aug_ray_indices[sort_indices],
                aug_width[sort_indices],
            )
        else:
            sorted_t, sorted_indices, sorted_ray_indices, sorted_width, sorted_delta = (
                raw_hits_t,
                raw_hits_indices,
                raw_hits_ray_indices,
                samples_width_world,
                raw_hits_delta,
            )

        t_starts = sorted_t - sorted_delta
        t_ends = sorted_t + sorted_delta

        valid_interval = t_ends > t_starts
        if not valid_interval.all():
            if not valid_interval.any(): return self._get_empty_samples(ray_bundle)
            final_starts, final_ends, final_indices, final_ray_indices, final_t = (
                t_starts[valid_interval],
                t_ends[valid_interval],
                sorted_indices[valid_interval],
                sorted_ray_indices[valid_interval],
                sorted_t[valid_interval].unsqueeze(-1)
            )
        else:
            final_starts, final_ends, final_indices, final_ray_indices, final_t = (
                t_starts,
                t_ends,
                sorted_indices,
                sorted_ray_indices,
                sorted_t.unsqueeze(-1)
            )

        # --- 4. Construct Final RaySamples Object ---
        starts, ends = final_starts.unsqueeze(-1), final_ends.unsqueeze(-1)
        # to the world space from the gaussian space
        starts_world = starts / self.scale_factor
        ends_world = ends / self.scale_factor
        t_world = final_t / self.scale_factor
        packed_origins, packed_directions = origins[final_ray_indices], directions[final_ray_indices]
        packed_pixel_area = ray_bundle.pixel_area[final_ray_indices] if ray_bundle.pixel_area is not None else None
        packed_camera_indices = (
            ray_bundle.camera_indices[final_ray_indices] if ray_bundle.camera_indices is not None else None
        )
        frustums = GenieFrustums(
            origins=packed_origins, directions=packed_directions, starts=starts_world, ends=ends_world, pixel_area=packed_pixel_area, gaussian_t=t_world
        )
        metadata = {"gaussian_indices": final_indices.unsqueeze(-1)}
        ray_samples = RaySamples(frustums=frustums, camera_indices=packed_camera_indices, metadata=metadata)

        return ray_samples, final_ray_indices

    def _prepare_inputs_for_optix(self, origins, directions, num_rays, device, far_plane):
        """Helper to handle AABB intersection and normalization."""
        
        origins_hom = torch.cat(
            [origins, torch.ones([origins.shape[0], 1], device=device, dtype=origins.dtype)],
            dim=-1
        )
        origins_optix = origins_hom @ self.transform_matrix.T
        origins_optix = origins_optix * self.scale_factor
        
        directions_hom = torch.cat(
            [directions, torch.zeros([directions.shape[0], 1], device=device, dtype=directions.dtype)],
            dim=-1
        )
        directions_optix = directions_hom @ self.transform_matrix.T
        
        return origins_optix, directions_optix
    
    def _get_empty_samples(self, ray_bundle: RayBundle) -> Tuple[RaySamples, torch.Tensor]:
        """
        Returns empty samples to prevent MLP crashes when no samples are found.
        This keeps shapes valid for subsequent operations.
        """
        device = ray_bundle.origins.device
        ray_indices = torch.empty((0,), dtype=torch.long, device=device)
        
        starts = torch.zeros((1, 1), device=device)
        ends = torch.ones((1, 1), device=device) * 1e-9

        frustums = GenieFrustums(
            origins=ray_bundle.origins[0:0],
            directions=ray_bundle.directions[0:0],
            starts=starts,
            ends=ends,
            pixel_area=ray_bundle.pixel_area[0:0] if ray_bundle.pixel_area is not None else None,
            gaussian_t=torch.empty((0, 1), dtype=torch.float, device=device)
        )
        
        metadata = {
            "gaussian_indices": torch.empty((0, 1), dtype=torch.long, device=device)
        }

        ray_samples = RaySamples(
            frustums=frustums,
            camera_indices=ray_bundle.camera_indices[0:0] if ray_bundle.camera_indices is not None else None,
            metadata=metadata
        )
        
        return ray_samples, ray_indices