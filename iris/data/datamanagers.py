from dataclasses import dataclass, field
from typing import Literal, Type, Union

import torch

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader

@dataclass
class GenieDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the GenieDataManager."""
    _target: Type = field(default_factory=lambda: GenieDataManager)
    """Target class to instantiate."""

class GenieDataManager(VanillaDataManager):
    config: GenieDataManagerConfig
    def __init__(
        self,
        config: GenieDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def setup_train(self):
        super().setup_train()
        self.train_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
    
    def next_train_image(self, step: int):
        for camera, batch in self.train_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more training images")
