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

# Every activation leaks rather than clamping to zero.
#
# A plain ReLU unit whose pre-activation is negative for every observation
# receives exactly zero gradient, so it can never come back. Measured on a
# 2.3M-step run: 56% of the 160 shared features were permanently dead and
# *every* unit in the policy trunk was, which left the network emitting one
# fixed action distribution regardless of what it saw. Nothing in the loss can
# recover from that, and nothing in the existing telemetry showed it -- entropy
# stayed near uniform the whole time, because a constant distribution is still
# a high-entropy one.
#
# A small negative slope keeps the gradient alive without changing the shape of
# anything, so old checkpoints still load.
NEGATIVE_SLOPE = 0.01


def activation():
    return nn.LeakyReLU(NEGATIVE_SLOPE)

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
    scale = float(GRID_SIZE)
    player = coordinates.reshape(batch, FRAME_STACK_SIZE, 2) \
        * ((ARENA_SIZE - 1) / CELL_SIZE)

    # Both frames are handled in one batched pass rather than a Python loop.
    # At the batch sizes rollout collection uses, this is dominated by kernel
    # launches, not arithmetic, so the number of operations is what costs.
    cell_x, cell_y, lethal_index = _grid_axes(device, dtype)
    hazard = grids[:, :HAZARD_CHANNELS].reshape(
        batch, FRAME_STACK_SIZE, CELL_TYPES, GRID_SIZE, GRID_SIZE)
    occupied = hazard.index_select(2, lethal_index).amax(dim=2) > 0

    dx = cell_x - player[:, :, 0].reshape(batch, FRAME_STACK_SIZE, 1, 1)
    dy = cell_y - player[:, :, 1].reshape(batch, FRAME_STACK_SIZE, 1, 1)
    # Squared distance is enough to find the nearest cell; the square root is
    # then taken once per frame instead of once per cell.
    squared = dx * dx + dy * dy
    beyond = float(GRID_SIZE * GRID_SIZE * 16)
    masked = torch.where(occupied, squared, squared.new_full((), beyond))
    best, index = masked.reshape(batch, FRAME_STACK_SIZE, -1).min(dim=2)

    empty = best >= beyond
    distance = torch.where(empty, best.new_full((), FAR_AWAY),
                           torch.sqrt(best.clamp(max=beyond - 1.0)) / scale)
    nearest_x = (index % GRID_SIZE).to(dtype) - player[:, :, 0]
    nearest_y = torch.div(index, GRID_SIZE, rounding_mode="floor").to(dtype) \
        - player[:, :, 1]
    zero = nearest_x.new_zeros(())
    nearest_x = torch.where(empty, zero, nearest_x / scale)
    nearest_y = torch.where(empty, zero, nearest_y / scale)

    velocity = (player[:, -1] - player[:, 0]) / scale
    closing = (distance[:, 0] - distance[:, -1]).unsqueeze(1)
    return torch.cat((
        velocity,
        nearest_x[:, -1].unsqueeze(1), nearest_y[:, -1].unsqueeze(1),
        distance[:, -1].unsqueeze(1), closing,
    ), dim=1)


_GRID_AXIS_CACHE = {}


def _grid_axes(device, dtype):
    """Cell coordinate grids and the lethal channel index, built once per device."""
    key = (device, dtype)
    cached = _GRID_AXIS_CACHE.get(key)
    if cached is None:
        axis = torch.arange(GRID_SIZE, device=device, dtype=dtype)
        cached = (
            axis.reshape(1, 1, 1, GRID_SIZE),
            axis.reshape(1, 1, GRID_SIZE, 1),
            torch.tensor(LETHAL_OFFSETS, device=device, dtype=torch.long),
        )
        _GRID_AXIS_CACHE[key] = cached
    return cached


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
            activation(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            activation(),
        )
        self.coordinate_net = nn.Sequential(
            nn.Linear(COORDINATE_INPUTS, 32),
            activation(),
            nn.Linear(32, 32),
            activation(),
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
            "activation_fn": lambda: nn.LeakyReLU(NEGATIVE_SLOPE),
        }
    raise ValueError(f"Unknown policy architecture: {architecture}")


def dead_unit_fraction(policy, observations):
    """Share of policy-trunk units that output zero for every observation given.

    A unit that is zero across a whole rollout is one a plain ReLU could never
    revive, and once they all are the policy is a constant. Cheap enough to run
    once per update, and the only signal that catches this: entropy does not,
    since a constant distribution can be a high-entropy one.
    """
    with torch.no_grad():
        features = policy.extract_features(observations)
        if isinstance(features, tuple):     # separate pi/vf extractors
            features = features[0]
        latent = policy.mlp_extractor.policy_net(features)
        return float((latent == 0).all(dim=0).to(torch.float32).mean())


def architecture_name(model):
    if isinstance(model.policy.features_extractor, DodgeHybridExtractor):
        return "hybrid-cnn"
    return "mlp"
