"""Custom Stable-Baselines3 policy components for Dodge."""

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .observe import (
    ARENA_SIZE,
    CELL_SIZE,
    CELL_TYPES,
    FRAME_OBSERVATION_SIZE,
    FRAME_STACK_SIZE,
    GRID_SIZE,
    OBSERVATION_SIZE,
)


ARCHITECTURES = ("hybrid-cnn", "mlp")
HYBRID_FEATURES_DIM = 160

# Hazard channels come first, in the same order and count as before, so a
# checkpoint trained without the player channels migrates by copying into the
# leading channels. A convolution learns an arbitrary combination across its
# inputs, so this ordering costs nothing.
HAZARD_CHANNELS = FRAME_STACK_SIZE * CELL_TYPES
PLAYER_CHANNELS = FRAME_STACK_SIZE
HYBRID_INPUT_CHANNELS = HAZARD_CHANNELS + PLAYER_CHANNELS


def _player_channels(coordinates, dtype):
    """Mark the player's own cell in a grid, one channel per decision frame.

    The hazard map says nothing about where the agent is, so before this every
    convolution reasoned about threats without knowing what they were near. The
    player's position was only reachable through a separate branch, after the
    spatial layout had already been flattened away. Painting it into the grid
    puts player-relative geometry inside the convolution's receptive field,
    which is the whole game.

    Costs nothing in the environment: these coordinates are already part of the
    observation.
    """
    scale = (ARENA_SIZE - 1) / CELL_SIZE
    cells = (coordinates * scale).long().clamp(0, GRID_SIZE - 1)
    flat = cells[..., 1] * GRID_SIZE + cells[..., 0]
    batch, frames = flat.shape
    channels = torch.zeros(batch, frames, GRID_SIZE * GRID_SIZE,
                           dtype=dtype, device=coordinates.device)
    channels.scatter_(2, flat.unsqueeze(-1), 1.0)
    return channels.reshape(batch, frames, GRID_SIZE, GRID_SIZE)


class DodgeHybridExtractor(BaseFeaturesExtractor):
    """Encode one-hot spatial grids with a CNN and coordinates with an MLP."""

    def __init__(self, observation_space):
        if observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError(
                f"DodgeHybridExtractor expects {OBSERVATION_SIZE} values, "
                f"got {observation_space.shape}"
            )
        super().__init__(observation_space, features_dim=HYBRID_FEATURES_DIM)
        self.grid_net = nn.Sequential(
            nn.Conv2d(HYBRID_INPUT_CHANNELS, 32, kernel_size=3, padding=1),
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
        grids = grids.to(dtype=observations.dtype)
        player = _player_channels(frames[:, :, :2], observations.dtype)
        return coordinates, torch.cat((grids, player), dim=1)

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
