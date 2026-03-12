"""
Encoding functions
"""

from abc import abstractmethod
from typing import Optional, Callable, Union, Dict, List

import numpy as np
import torch
from torch import Tensor, nn

from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.utils import profiler

from iris.utils.utils import quat_to_rotmat


class FastSplashEncoding(nn.Module):
    def __init__(
        self,
        n_gausses: int = 10000,
        n_features_per_gauss: int = 32,
        gaussians: Optional[Tensor] = None,
        densify: bool = True,
        prune: bool = True,
        unfreeze_means: bool = True,
        unfreeze_covs: bool = True,
        spatial_distortion: Optional[SpatialDistortion] = None,
        device: str = 'cuda',
        empty_as_hash: bool = False,
        use_per_gauss_weight: bool = False,
        dist_threshold: float = 2.0,
        n_neighbors: int = 2
    ):
        super().__init__()
        
        self.n_features_per_gauss = n_features_per_gauss
        self.densify_gausses = densify
        self.prune_gausses = prune
        self.unfreeze_gausses = unfreeze_means
        self.unfreeze_covs = unfreeze_covs
        self.device = device
        self.empty_as_hash = empty_as_hash
        self.use_per_gauss_weight = use_per_gauss_weight
        self.spatial_distortion = spatial_distortion
        self.dist_threshold = dist_threshold
        
        self.N_NEIGHBORS = n_neighbors
        self.WINDOW_SIZE = (2 * n_neighbors) + 1
        self.SHARPNESS = 1.0
        
        means = gaussians["points3D_xyz"]
        if means is not None and isinstance(means, np.ndarray):
            means = torch.tensor(means, dtype=torch.float32, device=self.device)
        elif means is not None and isinstance(means, Tensor):
            means = means.to(device=self.device)
        else:
            means = self.init_mean(n_gausses)
        
        opacity = gaussians["points3D_opacity"]
        if opacity is not None and isinstance(opacity, np.ndarray):
            opacity = torch.tensor(opacity, dtype=torch.float32, device=self.device, requires_grad=True)
        elif opacity is not None and isinstance(opacity, Tensor):
            opacity = opacity.to(device=self.device, requires_grad=True)
        else:
            opacity = torch.ones_like(means[:, 0], device=self.device, requires_grad=True)
        if opacity.dim() == 1:
            opacity = opacity.unsqueeze(-1)

        scales_tensor = gaussians.get("points3D_scale", None)
        if scales_tensor is not None:
            log_covs_tensor = torch.log(torch.square(scales_tensor))
        else:
            log_covs_tensor = torch.log(torch.ones(means.shape[0], 3, device=self.device) * 0.0000004)

        quats_tensor = gaussians.get("points3D_quat", None)
        if quats_tensor is not None:
            quats_tensor = gaussians["points3D_quat"]
        else:
            quats_tensor = torch.zeros(means.shape[0], 4, device=self.device)
            quats_tensor[:, 0] = 1.0
        
        if spatial_distortion is not None:
            contracted_means = self.spatial_distortion(means)
            contracted_means = (contracted_means + 2.0) / 4.0
            self.contracted_means = contracted_means
        else:
            self.contracted_means = means
        
        self.means_hash = HashEncoding(max_res=8192, log2_hashmap_size=21)
        
        self.total_gaus = means.shape[0]
        means = nn.Parameter(means)
        self.register_buffer("feats", self.means_hash(self.contracted_means))
        log_covs = nn.Parameter(log_covs_tensor)
        quats = nn.Parameter(quats_tensor)
        self.confidence = torch.ones_like(means[:, 0], device=self.device, requires_grad=False)
        weights = nn.Parameter(opacity)
        self.gauss_params = torch.nn.ParameterDict({
            "means": means,
            "log_covs": log_covs,
            "quats": quats,
            "weights": weights,
        })

        # Gradient accumulation buffers
        self.xyz_gradient_accum = torch.zeros(self.total_gaus, device=self.device)
        self.denom = torch.zeros(self.total_gaus, device=self.device)

        if self.unfreeze_gausses:
            self.gauss_params["means"].register_hook(self._grad_hook)
    
    def _grad_hook(self, grad):
        if grad.shape[0] == self.xyz_gradient_accum.shape[0]:
            self.xyz_gradient_accum += grad.norm(dim=-1)
            self.denom += 1
    
    def init_mean(self, N):
        print(f'Total number of gauss: {N}')
        pts = np.random.randn(N, 3)
        r = np.sqrt(np.random.rand(N, 1))
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        pts = pts * 0.5 + 0.5 # [0.25 ... 0.75]
        
        return torch.tensor(pts, dtype=torch.float32, device=self.device)

    def init_mean_unifrom(self, N, thickness: float = 0.05, box_min: float = 0.1, box_max: float = 0.9):
        """Initialize means uniformly on the outer shell (thickness) of a box [box_min,box_min,box_min]-[box_max,box_max,box_max], excluding top and bottom faces."""
        print(f'Total number of gauss: {N}')
        pts = []
        batch = int(N * 1.5)  # oversample to ensure enough points
        box_size = box_max - box_min
        while sum(len(p) for p in pts) < N:
            samples = np.random.rand(batch, 3)
            samples = samples * box_size + box_min  # scale to [box_min, box_max]
            # Only select points near the sides (x or y near boundary), not z
            mask = ((samples[:, 0] <= box_min + thickness) | (samples[:, 0] >= box_max - thickness) |
                    (samples[:, 1] <= box_min + thickness) | (samples[:, 1] >= box_max - thickness))
            shell_pts = samples[mask]
            if len(shell_pts) > 0:
                pts.append(shell_pts)
        pts = np.concatenate(pts, axis=0)
        if pts.shape[0] > N:
            pts = pts[:N]
        return torch.tensor(pts, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        param_fn: Callable[[str, Tensor], Tensor],
        optimizer_fn: Callable[[str, Tensor], Tensor],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        names: Union[List[str], None] = None,
    ):
        """Update the parameters and the state in the optimizers with defined functions.

        Args:
            param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            params: A dictionary of parameters.
            optimizers: A dictionary of optimizers, each corresponding to a parameter.
            names: A list of key names to update. If None, update all. Default: None.
        """
        if names is None:
            # If names is not provided, update all parameters
            names = list(params.keys())

        for name in names:
            param = params[name]
            new_param = param_fn(name, param)
            params[name] = new_param
            if name not in optimizers:
                assert not param.requires_grad, (
                    f"Optimizer for {name} is not found, but the parameter is trainable."
                    f"Got requires_grad={param.requires_grad}"
                )
                continue
            optimizer = optimizers[name]
            for i in range(len(optimizer.param_groups)):
                param_state = optimizer.state[param]
                del optimizer.state[param]
                for key in param_state.keys():
                    if key != "step":
                        v = param_state[key]
                        param_state[key] = optimizer_fn(key, v)
                optimizer.param_groups[i]["params"] = [new_param]
                optimizer.state[new_param] = param_state
    
    def densify_and_split(self, optimizers: Dict[str, torch.optim.Optimizer], scene_extent: float, grad_threshold: float = 0.05):
        """
        Densify gaussians based on accumulated gradients:
        - Clone: High gradient, small scale.
        - Split: High gradient, large scale.
        """
        if not self.densify_gausses:
            return

        # Safety check for max gaussians to prevent explosion
        if self.total_gaus > 2000000:
             print(f"Skipping densification: reached {self.total_gaus} gaussians (limit 2M).")
             # Reset accumulators even if we skip to avoid stale gradients piling up
             self.xyz_gradient_accum.zero_()
             self.denom.zero_()
             return

        grads = self.xyz_gradient_accum / self.denom.clamp(min=1)
        grads[self.denom == 0] = 0.0
        
        # Reset accumulators
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()

        # Identify candidates
        selected_pts_mask = torch.where(grads >= grad_threshold, True, False)
        # Exclude points with invalid scales
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(torch.exp(self.log_covs), dim=1).values > 0.0)

        if not selected_pts_mask.any():
            return

        print(f"Densifying: found {selected_pts_mask.sum()} candidates.")

        # Determine scale (std)
        scales = torch.sqrt(torch.exp(self.log_covs))
        max_scale = torch.max(scales, dim=1).values
        
        percent_max_extent = 0.01 * scene_extent
        
        split_mask = torch.logical_and(selected_pts_mask, max_scale > percent_max_extent)
        clone_mask = torch.logical_and(selected_pts_mask, ~split_mask)
        
        # --- Prepare new parameters to append ---
        
        new_means_list = []
        new_covs_list = []
        new_quats_list = []
        new_conf_list = []
        new_weights_list = []
        # 1. Clone
        if clone_mask.any():
            new_means_list.append(self.means[clone_mask])
            new_covs_list.append(self.log_covs[clone_mask])
            new_quats_list.append(self.quats[clone_mask])
            new_conf_list.append(self.confidence[clone_mask])
            new_weights_list.append(self.weights[clone_mask])
        # 2. Split (Append new copies)
        if split_mask.any():
            # Sample new positions for the copy
            stds = scales[split_mask]
            means = self.means[split_mask]
            samples = torch.randn_like(means) * stds
            
            # Rotate samples
            quats = self.quats[split_mask]
            quats = quats / quats.norm(dim=-1, keepdim=True)
            R = quat_to_rotmat(quats)
            # R is (N, 3, 3), samples is (N, 3)
            rotated_samples = torch.bmm(R, samples.unsqueeze(-1)).squeeze(-1)
            
            new_means_split = means + rotated_samples
            # Reduce scale by 1.6 (log variance -= 2*log(1.6))
            new_covs_split = self.log_covs[split_mask] - 2 * np.log(1.6)
            
            new_means_list.append(new_means_split)
            new_covs_list.append(new_covs_split)
            new_quats_list.append(quats)
            new_conf_list.append(self.confidence[split_mask])
            new_weights_list.append(self.weights[split_mask])
        if not new_means_list:
            return

        new_means_append = torch.cat(new_means_list, dim=0)
        new_covs_append = torch.cat(new_covs_list, dim=0)
        new_quats_append = torch.cat(new_quats_list, dim=0)
        new_conf_append = torch.cat(new_conf_list, dim=0)
        new_weights_append = torch.cat(new_weights_list, dim=0)
        # --- Update Parameters ---

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == 'means':
                new_param = nn.Parameter(torch.cat([p, new_means_append], dim=0), requires_grad=self.means.requires_grad)
                if self.unfreeze_gausses:
                    new_param.register_hook(self._grad_hook)
                return new_param
            elif name == 'log_covs':
                # For split mask, we need to modify existing values in p
                # Since we can't modify p in-place easily without affecting optimizer state logic if we replace it,
                # we construct the new tensor with modified values.
                
                # Clone p to avoid modifying the original tensor in place before concatenation if needed
                p_mod = p.clone()
                if split_mask.any():
                    p_mod[split_mask] -= 2 * np.log(1.6)
                
                new_param = nn.Parameter(torch.cat([p_mod, new_covs_append], dim=0), requires_grad=self.log_covs.requires_grad)
                return new_param
            elif name == 'quats':
                new_param = nn.Parameter(torch.cat([p, new_quats_append], dim=0), requires_grad=self.quats.requires_grad)
                return new_param
            elif name == 'weights':
                new_param = nn.Parameter(torch.cat([p, new_weights_append], dim=0), requires_grad=self.weights.requires_grad)
                return new_param
            return p

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            # Append zeros for new parameters
            # For existing parameters, we keep the state. 
            # Note: For split, we modified the parameter value, but optimizer state (momentum) 
            # usually should be kept or reset. Keeping it is standard for simple implementations.
            zeros = torch.zeros((new_means_append.shape[0], *v.shape[1:]), device=self.device)
            return torch.cat([v, zeros], dim=0)

        self._update_param_with_optimizer(param_fn, optimizer_fn, self.gauss_params, optimizers)

        self.confidence = torch.cat([self.confidence, new_conf_append], dim=0)
        self.total_gaus = self.means.shape[0]
        
        # Resize accumulators
        self.xyz_gradient_accum = torch.zeros(self.total_gaus, device=self.device)
        self.denom = torch.zeros(self.total_gaus, device=self.device)
        
        self.feats = self.means_hash(self.means)
        print(f"Densified to {self.total_gaus} gaussians (Cloned: {clone_mask.sum()}, Split: {split_mask.sum()})")

    def prune(self, optimizers: Dict[str, torch.optim.Optimizer], threshold: float=0.1, weight_threshold: float=0.05):
        """
        Remove all means, feats, log_covs, quats, and confidence entries with confidence lower than threshold.
        """

        if self.prune_gausses:

            mask = self.confidence >= threshold
            # if self.use_per_gauss_weight:
            #     weight_mask = torch.sigmoid(self.weights.squeeze(-1)) >= weight_threshold
            # else:
            weight_mask = torch.ones_like(self.confidence).bool()
            mask = torch.logical_and(mask, weight_mask)
            def param_fn(name: str, p: Tensor) -> Tensor:
                new_param = torch.nn.Parameter(p[mask], requires_grad=p.requires_grad)
                if name == 'means' and self.unfreeze_gausses:
                    new_param.register_hook(self._grad_hook)
                return new_param

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                return v[mask]

            self._update_param_with_optimizer(param_fn, optimizer_fn, self.gauss_params, optimizers)

            # Only keep entries where mask is True
            self.confidence = self.confidence[mask]
            self.contracted_means = self.contracted_means[mask]
            self.total_gaus = self.means.shape[0]
            
            # Resize accumulators
            self.xyz_gradient_accum = torch.zeros(self.total_gaus, device=self.device)
            self.denom = torch.zeros(self.total_gaus, device=self.device)
            
            self.feats = self.means_hash(self.contracted_means)
            
            # Refit KNN with new means
            print(f"Pruned to {self.means.shape[0]} gaussians.")

    def reinitialize_params(self, n_gausses: int) -> None:
        """
        Reinitialize the means, feats, log_covs, and confidence with new random values, and refit KNN.
        """
        self.gauss_params["means"] = nn.Parameter(self.init_mean(n_gausses))
        self.feats = self.means_hash(self.means)
        self.gauss_params["log_covs"] = nn.Parameter(torch.log(torch.ones(n_gausses, 3, device=self.device) * 0.0001), requires_grad=self.log_covs.requires_grad)
        quats = torch.zeros(n_gausses, 4, device=self.device)
        quats[:, 0] = 1.0
        self.gauss_params["quats"] = nn.Parameter(quats, requires_grad=self.quats.requires_grad)
        self.confidence = torch.ones(n_gausses, device=self.device)
        if self.use_per_gauss_weight:
            weights = torch.normal(0., 1., size=(n_gausses, 1), device=self.device)
            self.gauss_params["weights"] = nn.Parameter(weights, requires_grad=self.weights.requires_grad)
        else:
            self.gauss_params["weights"] = None
        self.gauss_params["feats"] = nn.Parameter(torch.normal(0., 1., size=(n_gausses, self.n_features_per_gauss), device=self.device, requires_grad=True))
        self.total_gaus = n_gausses
        print(f"Reinitialized to {n_gausses} gaussians.")
        
    def unfreeze_means(self):
        if self.unfreeze_gausses:
            self.gauss_params["means"].requires_grad_(True)

    def freeze_means(self):
        if self.unfreeze_gausses:
            self.gauss_params["means"].requires_grad_(False)

    def get_out_dim(self) -> int:
        return self.n_features_per_gauss
    
    @property
    def means(self) -> Tensor:
        return self.gauss_params["means"]
    
    @property
    def log_covs(self) -> Tensor:
        return self.gauss_params["log_covs"]

    @property
    def quats(self) -> Tensor:
        return self.gauss_params["quats"]
    
    @property
    def weights(self) -> Tensor:
        return self.gauss_params["weights"]

    def get_mahalanobis_features(self, samples, gauss_indices, ray_indices):
        """
        Loop-free, fully vectorized Sliding Window Attention.
        Includes Spatial Distance Masking and NaN safeguards.
        """
        B = samples.shape[0]
        device = samples.device
        
        gauss_indices = gauss_indices.view(-1)
        ray_indices = ray_indices.view(-1)

        padded_gauss_idx = torch.nn.functional.pad(gauss_indices, (self.N_NEIGHBORS, self.N_NEIGHBORS), mode='constant', value=0)
        padded_ray_idx = torch.nn.functional.pad(ray_indices, (self.N_NEIGHBORS, self.N_NEIGHBORS), mode='constant', value=-1)
        
        win_gauss_idx = padded_gauss_idx.unfold(0, self.WINDOW_SIZE, 1) # [B, W]
        win_ray_idx = padded_ray_idx.unfold(0, self.WINDOW_SIZE, 1)     # [B, W]
        
        win_means = self.means[win_gauss_idx]
        win_log_covs = self.log_covs[win_gauss_idx]
        win_scales_quad = torch.exp(win_log_covs).clamp(min=1e-6)  # clamp to prevent explosion in inversion
        
        win_quats = self.quats[win_gauss_idx]
        
        flat_idx = win_gauss_idx.reshape(-1)
        flat_feats = self.feats[flat_idx]
        win_feats = flat_feats.view(B, self.WINDOW_SIZE, -1)

        if self.use_per_gauss_weight:
            win_weights_param = self.weights[win_gauss_idx]
            win_opacities = torch.sigmoid(win_weights_param).squeeze(-1)
        else:
            win_opacities = torch.ones((B, self.WINDOW_SIZE), device=device)

        diff_world = samples.unsqueeze(1) - win_means # [B, W, 3]
        euclidean_dists = torch.norm(diff_world, dim=-1) # [B, W]
        
        # Rotate to Local Space
        win_quats = torch.nn.functional.normalize(win_quats, dim=-1)
        R = quat_to_rotmat(win_quats) 
        R_T = R.transpose(-1, -2)
        diff_local = torch.matmul(R_T, diff_world.unsqueeze(-1)).squeeze(-1)
        
        inv_scale_sq = 1.0 / win_scales_quad
        win_maha_sq = torch.sum(diff_local.pow(2) * inv_scale_sq, dim=-1) # [B, W]
        
        ray_mask = (ray_indices.unsqueeze(1) == win_ray_idx)
        dist_mask = (euclidean_dists < self.dist_threshold)
        valid_mask = ray_mask & dist_mask
        
        logits = -0.5 * win_maha_sq * self.SHARPNESS
        logits = logits.masked_fill(~valid_mask, -1e9)
        
        weights = torch.softmax(logits, dim=1) # [B, W]
        aggregated_features = torch.sum(weights.unsqueeze(-1) * win_feats, dim=1)  # weighted sum of features
        
        win_raw_alphas = torch.exp(logits) * win_opacities  # calculate raw alpha for window
        point_alpha = torch.sum(weights * win_raw_alphas, dim=1).unsqueeze(1)  # weighted sum for alpha
        
        center_idx = self.N_NEIGHBORS
        diff_world_curr = diff_world[:, center_idx, :]
        distances = torch.norm(diff_world_curr, dim=-1)
        
        return aggregated_features, distances, point_alpha

    def forward(self, coords, closest_gauss_indices=None, ray_indices=None):
        """
        This is an implementation for the old representation of data, which means rot = [1, 0, 0, 0] and we have only means and log_covs
        """
        assert closest_gauss_indices is not None, "closest_gauss_indices must be provided"
        
        if self.training:
            self.feats = self.means_hash(self.contracted_means)
        
        splash_feats, self.distances, alphas = self.get_mahalanobis_features(coords, closest_gauss_indices, ray_indices)

        if self.training:
            self.confidence -= 0.001
            self.confidence[closest_gauss_indices] += 0.01
            self.confidence.clamp_(min=0.0, max=1.0)

        return splash_feats, alphas
