import numpy as np
import torch
import trimesh

from nerfstudio.viewer.viewer_elements import ViewerElement
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from viser import ViserServer
from viser._scene_handles import PointCloudHandle, GaussianSplatHandle


class ViewerPointCloud(ViewerElement[bool]):
    """A point cloud in the viewer

    Args:
        name: The name of the point cloud
        visible: If the point cloud is visible
    """

    scene_handle: PointCloudHandle

    def __init__(self, name: str, aabb: SceneBox, points: np.ndarray, confidence: np.ndarray, visible: bool = True):
        self.aabb = aabb
        self.points = points
        self.confidence = confidence
        super().__init__(name, visible=visible)

    def update(self, points: np.ndarray, confidence: np.ndarray) -> None:
        """Update the point cloud with new points."""
        self.points = points
        self.confidence = confidence
        try:
            if self.viser_server is not None:
                self._create_scene_handle(self.viser_server)
        except AttributeError as e:
            pass

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        points = self.points.reshape(-1, self.points.shape[-1])
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy()
        aabb_min, aabb_max = aabb[0] * VISER_NERFSTUDIO_SCALE_RATIO, aabb[1] * VISER_NERFSTUDIO_SCALE_RATIO
        points = points * (aabb_max - aabb_min) + aabb_min

        pcd = trimesh.PointCloud(points)

        color_coeffs = np.random.uniform(0.4, 1.0, size=(pcd.vertices.shape[0]))
        colors = np.tile((0, 255, 255), pcd.vertices.shape[0]).reshape(-1, 3) * color_coeffs[:, None]
        colors[:, 1] *= (1 - self.confidence)
        colors[:, 2] *= self.confidence
        
        self.scene_handle = viser_server.scene.add_point_cloud(
            f"/{self.name}",
            points=pcd.vertices,
            colors=colors,
            point_size=0.02,
            point_shape="circle"
        )
        
    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)


class ViewerOccupancyGrid(ViewerElement[bool]):
    """A occupancy grid in the viewer

    Args:
        name: The name of the occupancy grid
        visible: If the occupancy grid is visible
    """

    def __init__(self, name: str, occ_grid: torch.Tensor, aabb: SceneBox, visible: bool = True):
        self.aabb = aabb
        self.occ_grid = occ_grid
        super().__init__(name, visible=visible)

    def update(self, occ_grid: torch.Tensor) -> None:
        """Update the occupancy grid with new occupancy values."""
        self.occ_grid = occ_grid
        try:
            if self.viser_server is not None:
                self._create_scene_handle(self.viser_server)
        except AttributeError as e:
            pass

    def _create_scene_handle(self, viser_server: ViserServer) -> None:

        if torch.any(self.occ_grid):
            # Generate voxel grid indices
            res = self.occ_grid.shape[0]
            device = self.occ_grid.device

            grid_coords = torch.stack(torch.meshgrid(
                torch.arange(res, device=device),
                torch.arange(res, device=device),
                torch.arange(res, device=device),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)  # (res^3, 3)

            # Select occupied voxels
            occupied_indices = grid_coords[self.occ_grid.view(-1)]  # (N, 3)

            # Convert to world coordinates
            aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
            voxel_size = (aabb[1] - aabb[0]) / res
            occupied_centers = aabb[0] + (occupied_indices.cpu().numpy() + 0.5) * voxel_size  # (N, 3)

            # Convert to NumPy and visualize using trimesh + your viewer
            occupied_cloud = trimesh.PointCloud(occupied_centers)
            viser_server.scene.add_point_cloud(
                "/occupied_voxels",
                points=occupied_cloud.vertices,
                colors=np.tile((255, 0, 0), occupied_cloud.vertices.shape[0]).reshape(-1, 3),  # red color
                point_size=0.03,
                point_shape="circle",
                visible=True
            )

    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)


class ViewerGaussianSplats(ViewerElement[bool]):
    """A point cloud in the viewer

    Args:
        name: The name of the point cloud
        visible: If the point cloud is visible
    """

    scene_handle: GaussianSplatHandle

    def __init__(self, name: str, aabb: SceneBox, means: np.ndarray, covariances: np.ndarray, quats: np.ndarray, confidence: np.ndarray, visible: bool = True):
        self.aabb = aabb
        self.means = means
        self.covariances = covariances
        self.quats = quats
        self.confidence = confidence
        super().__init__(name, visible=visible)

    def update(self, means: np.ndarray, covariances: np.ndarray, quats: np.ndarray, confidence: np.ndarray) -> None:
        """Update the point cloud with new points."""
        self.means = means
        self.covariances = covariances
        self.quats = quats
        self.confidence = confidence
        try:
            if self.viser_server is not None:
                self._create_scene_handle(self.viser_server)
        except AttributeError as e:
            pass

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        means = self.means.reshape(-1, self.means.shape[-1])
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy()
        aabb_min, aabb_max = aabb[0] * VISER_NERFSTUDIO_SCALE_RATIO, aabb[1] * VISER_NERFSTUDIO_SCALE_RATIO
        means = means * (aabb_max - aabb_min) + aabb_min

        # Convert scales and quats to covariance matrices
        scales = np.sqrt(self.covariances) * VISER_NERFSTUDIO_SCALE_RATIO
        quats = self.quats

        # Normalize quaternions
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # Create rotation matrices (assuming w, x, y, z convention)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        R = np.zeros((quats.shape[0], 3, 3))
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        # Compute covariance matrices: Sigma = R * S^2 * R^T
        S_squared = scales ** 2
        # Multiply columns of R by S_squared (broadcasting)
        R_scaled = R * S_squared[:, None, :]
        covariances = R_scaled @ np.transpose(R, (0, 2, 1))

        color_coeffs = np.random.uniform(0.4, 1.0, size=(means.shape[0]))
        colors = np.tile((0, 1, 1), means.shape[0]).reshape(-1, 3) * color_coeffs[:, None]
        colors[:, 1] *= (1 - self.confidence)
        colors[:, 2] *= self.confidence

        opacities = np.ones((means.shape[0], 1))

        self.scene_handle = viser_server.scene.add_gaussian_splats(
            f"/{self.name}",
            centers=means,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities
        )

    def install(self, viser_server: ViserServer) -> None:
        self.viser_server = viser_server
        self._create_scene_handle(viser_server)


class ViewerAABB(ViewerElement[bool]):
    """A bounding box in the viewer

    Args:
        name: The name of the aabb
        visible: If the aabb is visible
    """

    def __init__(self, name: str, aabb: SceneBox, visible: bool = True):
        self.aabb = aabb
        super().__init__(name, visible=visible)

    def _create_scene_handle(self, viser_server: ViserServer) -> None:
        aabb =  self.aabb.aabb.reshape(2, 3).detach().cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
        mesh = trimesh.creation.box(tuple((aabb[1] - aabb[0])))
        viser_server.scene.add_mesh_simple(
            name=f"/{self.name}",
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(0, 0, 0),
            wireframe=True,
            visible=False
        )

    def install(self, viser_server: ViserServer) -> None:
        self._create_scene_handle(viser_server)