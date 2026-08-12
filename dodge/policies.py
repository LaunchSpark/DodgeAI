"""Custom Stable-Baselines3 policy components for Dodge."""

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .observe import (
    CELL_TYPES,
    FRAME_OBSERVATION_SIZE,
    FRAME_STACK_SIZE,
    GRID_SIZE,
    OBSERVATION_SIZE,
)


ARCHITECTURES = ("hybrid-cnn", "mlp")
HYBRID_FEATURES_DIM = 160


class DodgeHybridExtractor(BaseFeaturesExtractor):
    """Encode one-hot spatial grids with a CNN and coordinates with an MLP."""

    def __init__(self, observation_space):
        if observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                f"DodgeHybridExtractor expects {OBSERVATION_SIZE} values, "
                f"got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=HYBRID_FEATURES_DIM)
        channels = FRAME_STACK_SIZE * CELL_TYPES
        self.grid_net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.coordinate_net = nn.Sequential(
            nn.Linear(FRAME_STACK_SIZE * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    @staticmethod
    def policy_inputs(observations):
        """Split flattened frames into coordinates and one-hot grid channels."""
        batch = observations.shape[0]
        frames = observations.reshape(batch, FRAME_STACK_SIZE, FRAME_OBSERVATION_SIZE)
        coordinates = frames[:, :, :2].reshape(batch, FRAME_STACK_SIZE * 2)
        cells = torch.round(frames[:, :, 2:] * CELL_TYPES).long()
        cells = cells.clamp(0, CELL_TYPES).reshape(
            batch, FRAME_STACK_SIZE, GRID_SIZE, GRID_SIZE
        )
        # Empty cells become all-zero channels. The remaining eight channels
        # preserve entity type and spatial neighborhood for each decision frame.
        grids = F.one_hot(cells, num_classes=CELL_TYPES + 1)[..., 1:]
        grids = grids.permute(0, 1, 4, 2, 3).reshape(
            batch, FRAME_STACK_SIZE * CELL_TYPES, GRID_SIZE, GRID_SIZE
        )
        return coordinates, grids.to(dtype=observations.dtype)

    def forward(self, observations):
        coordinates, grids = self.policy_inputs(observations)
        return torch.cat(
            (self.grid_net(grids), self.coordinate_net(coordinates)),
            dim=1,
        )


def policy_kwargs(architecture):
    if architecture == "mlp":
        return {}
    if architecture == "hybrid-cnn":
        return {
            "features_extractor_class": DodgeHybridExtractor,
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            "activation_fn": nn.ReLU,
        }
    raise ValueError(f"Unknown policy architecture: {architecture}")


def architecture_name(model):
    if isinstance(model.policy.features_extractor, DodgeHybridExtractor):
        return "hybrid-cnn"
    return "mlp"
