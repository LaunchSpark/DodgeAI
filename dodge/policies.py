"""Custom Stable-Baselines3 policy components for Dodge."""

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .observe import (
    ARENA_SIZE,
    CELL_SIZE,
    CELL_TYPES,
    ENEMY_BLAST,
    ENEMY_KAMIKAZE,
    ENEMY_NORMAL,
    HAZARD_LETHAL,
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
# Raw coordinates for both frames, plus the derived senses.
COORDINATE_INPUTS = FRAME_STACK_SIZE * 2 + 6


# Channel offsets within one frame's eight that can kill the agent. Pickups and
# a pattern's warning telegraph are excluded: they occupy cells but do not end
# the episode, so treating them as threats would point the agent away from free
# help and away from ground that is still safe to stand on.
LETHAL_OFFSETS = (ENEMY_NORMAL - 1, ENEMY_KAMIKAZE - 1,
                  ENEMY_BLAST - 1, HAZARD_LETHAL - 1)
SENSE_FEATURES = 6
FAR_AWAY = 4.0


def _sense_features(coordinates, grids):
    """Nearest-threat geometry and how fast it is closing.

    Everything here is a function of values already in the observation, so it
    adds no work to the environment. It is supplied anyway because the shape of
    the function matters: reading "a threat is two cells up-left and gaining"
    off a pair of one-hot grids is a lot to ask of three convolutions, and
    dodging is exactly that question.

    Returns velocity, the nearest lethal cell's offset and distance, and the
    change in that distance between the two decision frames.
    """
    batch = coordinates.shape[0]
    device, dtype = coordinates.device, coordinates.dtype
    player = coordinates.reshape(batch, FRAME_STACK_SIZE, 2) * ((ARENA_SIZE - 1) / CELL_SIZE)

    axis = torch.arange(GRID_SIZE, device=device, dtype=dtype)
    cell_x = axis.reshape(1, 1, GRID_SIZE).expand(1, GRID_SIZE, GRID_SIZE)
    cell_y = axis.reshape(1, GRID_SIZE, 1).expand(1, GRID_SIZE, GRID_SIZE)

    offsets, distances = [], []
    for frame in range(FRAME_STACK_SIZE):
        base = frame * CELL_TYPES
        lethal = sum(grids[:, base + offset] for offset in LETHAL_OFFSETS)
        occupied = lethal > 0

        dx = cell_x - player[:, frame, 0].reshape(batch, 1, 1)
        dy = cell_y - player[:, frame, 1].reshape(batch, 1, 1)
        distance = torch.sqrt(dx * dx + dy * dy)
        # Empty cells are pushed past any real distance so they never win the
        # minimum; a frame with no threat at all reports FAR_AWAY.
        masked = torch.where(occupied, distance,
                             torch.full_like(distance, GRID_SIZE * FAR_AWAY))
        flat = masked.reshape(batch, -1)
        best, index = flat.min(dim=1)
        nearest_x = (index % GRID_SIZE).to(dtype) - player[:, frame, 0]
        nearest_y = torch.div(index, GRID_SIZE, rounding_mode="floor").to(dtype) \
            - player[:, frame, 1]

        empty = best > GRID_SIZE * 2
        scale = float(GRID_SIZE)
        offsets.append(torch.stack((
            torch.where(empty, torch.zeros_like(nearest_x), nearest_x / scale),
            torch.where(empty, torch.zeros_like(nearest_y), nearest_y / scale),
        ), dim=1))
        distances.append(torch.where(empty, torch.full_like(best, FAR_AWAY),
                                     best / scale))

    velocity = (player[:, -1] - player[:, 0]) / GRID_SIZE
    closing = (distances[0] - distances[-1]).unsqueeze(1)
    return torch.cat((velocity, offsets[-1], distances[-1].unsqueeze(1), closing),
                     dim=1)


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
            nn.Linear(COORDINATE_INPUTS, 32),
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
        senses = _sense_features(coordinates, grids)
        return torch.cat(
            (self.grid_net(grids),
             self.coordinate_net(torch.cat((coordinates, senses), dim=1))),
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
