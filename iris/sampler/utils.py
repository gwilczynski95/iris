from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float
import torch
from torch import Tensor

from nerfstudio.cameras.rays import Frustums, TORCH_DEVICE

@dataclass
class IrisFrustums(Frustums):
    """Describes region of space as a frustum."""
    gaussian_t: Float[Tensor, "*bs 1"] = None
    """Where the gaussian was hit"""

    def get_gaussian_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        if self.gaussian_t is None:
            return self.get_positions()
        pos = self.origins + self.directions * self.gaussian_t
        return pos
    
    @classmethod
    def get_mock_frustum(cls, device: Optional[TORCH_DEVICE] = "cpu") -> "IrisFrustums":
        """Helper function to generate a placeholder frustum.

        Returns:
            A size 1 frustum with meaningless values.
        """
        return IrisFrustums(
            origins=torch.ones((1, 3)).to(device),
            directions=torch.ones((1, 3)).to(device),
            starts=torch.ones((1, 1)).to(device),
            ends=torch.ones((1, 1)).to(device),
            pixel_area=torch.ones((1, 1)).to(device),
            gaussian_t=torch.ones((1, 1)).to(device),
        )
