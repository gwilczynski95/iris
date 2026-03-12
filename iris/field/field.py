"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.encodings import SHEncoding
from nerfstudio.utils import profiler

from iris.field.mlp import MLP, FastMLPWithHashEncoding


class IrisFastField(Field):
    
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        n_features_per_gauss: int = 32,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        seed_points: Optional[Tensor] = None,
        densify: bool = True,
        prune: bool = True,
        unfreeze_means: bool = False,
        unfreeze_covs: bool = False,
        use_per_gauss_weight: bool = False,
        gauss_transform_matrix: torch.Tensor = None,
        gauss_scale_factor: float = 1.0,
        dist_threshold: float = 2.0,
        n_neighbors: int = 2
    ) -> None:
        super().__init__()
        
        self.register_buffer("aabb", aabb)
        self.register_buffer("n_features_per_gauss", torch.tensor(n_features_per_gauss))
        self.register_buffer("gauss_transform_matrix", gauss_transform_matrix)
        self.register_buffer("gauss_scale_factor", torch.tensor(gauss_scale_factor))
        self.register_buffer("dist_threshold", torch.tensor(dist_threshold))
        self.geo_feat_dim = geo_feat_dim
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None

        self.use_average_appearance_embedding = use_average_appearance_embedding
        # self.use_pred_normals = use_pred_normals
        self.step = 0

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )
        
        self.mlp_base = FastMLPWithHashEncoding(
            n_features_per_gauss=n_features_per_gauss,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            seed_points=seed_points,
            densify=densify,
            prune=prune,
            unfreeze_means=unfreeze_means,
            unfreeze_covs=unfreeze_covs,
            spatial_distortion=self.spatial_distortion,
            use_per_gauss_weight=use_per_gauss_weight,
            dist_threshold=dist_threshold,
            n_neighbors=n_neighbors
        )

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )
    
    @profiler.time_function
    def get_sampling_positions(self, ray_samples: RaySamples) -> Tensor:
        """Computes and returns the sampling positions."""
        contracted_positions = None
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_gaussian_positions()  # gaussian positions in world coordinates
            positions = torch.cat([positions, torch.ones([positions.shape[0], 1], device=positions.device)], dim=-1)
            positions = positions @ self.gauss_transform_matrix.T
            positions = positions * self.gauss_scale_factor
            contracted_positions = self.spatial_distortion(positions)
            contracted_positions = (contracted_positions + 2.0) / 4.0
        else:
            positions = ray_samples.frustums.get_gaussian_positions()
            positions = torch.cat([positions, torch.ones([positions.shape[0], 1], device=positions.device)], dim=-1)
            positions = positions @ self.gauss_transform_matrix.T
            positions = positions * self.gauss_scale_factor
            contracted_positions = positions
        selector = ((contracted_positions > 0.0) & (contracted_positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        return positions, contracted_positions
    
    @profiler.time_function
    def get_density(self, ray_samples: RaySamples, ray_indices: Optional[torch.Tensor] = None) -> Tuple[Tensor, Tensor]:
        positions, contracted_positions = self.get_sampling_positions(ray_samples)
        if positions.numel() == 0:
            return torch.zeros([1, 1], device=ray_samples.frustums.directions.device), torch.zeros([1, self.geo_feat_dim], device=ray_samples.frustums.directions.device), torch.zeros([1, 1], device=ray_samples.frustums.directions.device)

        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)

        assert positions_flat.numel() > 0, "positions_flat is empty."
        h, alphas = self.mlp_base(positions_flat, ray_samples.metadata["gaussian_indices"], ray_indices)
        h = h.view(*ray_samples.frustums.shape, -1)
        alphas = alphas.view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation
        
        density = trunc_exp(density_before_activation.to(positions) - 1)
        selector = ((contracted_positions > 0.0) & (contracted_positions < 1.0)).all(dim=-1)
        density = density * selector[..., None]
        return density, base_mlp_out, alphas

    @profiler.time_function
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None, direction_transform: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        
        if direction_transform is not None:
            directions = ray_samples.frustums.directions
            rotated_dirs = torch.einsum("...ij,...j->...i", direction_transform.squeeze(), directions)
            directions = rotated_dirs
        else:
            directions = ray_samples.frustums.directions
        
        directions = get_normalized_directions(directions)
        directions_flat = directions.reshape(-1, 3)
        if directions_flat.numel() == 0:
            outputs.update({FieldHeadNames.RGB: torch.empty([0, 3], device=ray_samples.frustums.directions.device)})
            return outputs
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )
        else:
            embedded_appearance = None

        # # predicted normals
        # if self.use_pred_normals:
        #     positions = ray_samples.frustums.get_positions()

        #     positions_flat = self.position_encoding(positions.view(-1, 3))
        #     pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

        #     x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
        #     outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs

    def forward(self, ray_samples: RaySamples, direction_transform: Optional[torch.Tensor] = None, compute_normals: bool = False, ray_indices: Optional[torch.Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding, alphas = self.get_density(ray_samples, ray_indices=ray_indices)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding, direction_transform=direction_transform)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        effective_density = density * alphas
        final_alpha = 1. - torch.exp(-effective_density)
        field_outputs["alpha"] = final_alpha
        if torch.isnan(final_alpha).any() or torch.isnan(density).any() or torch.isnan(density_embedding).any() or torch.isnan(alphas).any():
            stop = 1

        return field_outputs
