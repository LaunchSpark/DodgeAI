"""Custom Stable-Baselines3 policy components for Dodge."""

from functools import partial

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .observe import (
    ARENA_SIZE,
    CELL_SIZE,
    CELL_TYPES,

    ENEMY_BLAST,
    ENEMY_KAMIKAZE,
    ENEMY_NORMAL,
    HAZARD_LETHAL,
    CELL_COUNT,
    PLAYER_VALUES,
    GRID_SIZE,
    OBSERVATION_SIZE,
)


# Ordered best-first; the head of this list is what everything defaults to.
#
# velocity-flow leads because it is the shape the failures drove us to. It
# emits a danger field per future instant instead of nine logits, which makes
# "lethal now, clear in two seconds" expressible -- the sentence a sweeping
# pattern requires. It reads a velocity per occupied cell rather than inferring
# motion from a history frame. It sees the whole arena rather than a 44px
# window, carries a sixth of hybrid-cnn parameters, has no dense layer to die
# in, and its internal state is a picture you can watch.
ARCHITECTURES = ("velocity-flow", "velocity-flow-v2", "hybrid-cnn", "mlp")

# Exact class names, not isinstance: v2 subclasses v1, so isinstance would
# report every v2 checkpoint as a v1 one.
EXTRACTOR_ARCHITECTURE = {
    "VelocityFlowV2Extractor": "velocity-flow-v2",
    "VelocityFlowExtractor": "velocity-flow",
    "DodgeHybridExtractor": "hybrid-cnn",
}
DEFAULT_ARCHITECTURE = ARCHITECTURES[0]
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
HAZARD_CHANNELS = CELL_TYPES
PLAYER_CHANNELS = 1
VELOCITY_CHANNELS = 2
HYBRID_INPUT_CHANNELS = HAZARD_CHANNELS + PLAYER_CHANNELS + VELOCITY_CHANNELS
# Player state (px, py, dpx, dpy) plus the derived senses.
COORDINATE_INPUTS = PLAYER_VALUES + 6


# Channel offsets within one frame's eight that can kill the agent. Pickups and
# a pattern's warning telegraph are excluded: they occupy cells but do not end
# the episode, so treating them as threats would point the agent away from free
# help and away from ground that is still safe to stand on.
LETHAL_OFFSETS = (ENEMY_NORMAL - 1, ENEMY_KAMIKAZE - 1,
                  ENEMY_BLAST - 1, HAZARD_LETHAL - 1)
SENSE_FEATURES = 6
FAR_AWAY = 4.0


def _sense_features(coordinates, grids):
    """Nearest-threat geometry, and how fast it is actually closing.

    Every term here is now exact rather than inferred. Velocity comes straight
    from the observation, and the closing rate is the real relative speed along
    the line to the threat -- the threat's own velocity, read from its cell,
    minus the player's, projected onto that line. Before, closing had to be the
    change in distance between two quantised frames, which is a noisy estimate
    of a quantity the game knows precisely.
    """
    batch = coordinates.shape[0]
    device, dtype = coordinates.device, coordinates.dtype
    scale = float(GRID_SIZE)
    player = coordinates[:, :2] * ((ARENA_SIZE - 1) / CELL_SIZE)
    player_velocity = coordinates[:, 2:4]

    cell_x, cell_y, lethal_index = _grid_axes(device, dtype)
    hazard = grids[:, :HAZARD_CHANNELS]
    occupied = hazard.index_select(1, lethal_index).amax(dim=1) > 0

    dx = cell_x - player[:, 0].reshape(batch, 1, 1)
    dy = cell_y - player[:, 1].reshape(batch, 1, 1)
    squared = dx * dx + dy * dy
    beyond = float(GRID_SIZE * GRID_SIZE * 16)
    masked = torch.where(occupied, squared, squared.new_full((), beyond))
    best, index = masked.reshape(batch, -1).min(dim=1)

    empty = best >= beyond
    distance = torch.where(empty, best.new_full((), FAR_AWAY),
                           torch.sqrt(best.clamp(max=beyond - 1.0)) / scale)
    column = (index % GRID_SIZE).to(dtype)
    row = torch.div(index, GRID_SIZE, rounding_mode="floor").to(dtype)
    nearest_x = column - player[:, 0]
    nearest_y = row - player[:, 1]

    # Closing speed: relative velocity projected onto the line to the threat.
    # Positive means the gap is shrinking.
    rows = torch.arange(batch, device=device)
    flat = grids.reshape(batch, grids.shape[1], -1)
    threat_vx = flat[rows, HAZARD_CHANNELS + PLAYER_CHANNELS, index]
    threat_vy = flat[rows, HAZARD_CHANNELS + PLAYER_CHANNELS + 1, index]
    relative_x = threat_vx - player_velocity[:, 0]
    relative_y = threat_vy - player_velocity[:, 1]
    length = torch.sqrt(nearest_x * nearest_x + nearest_y * nearest_y).clamp(min=1e-6)
    closing = -(relative_x * nearest_x + relative_y * nearest_y) / length

    zero = nearest_x.new_zeros(())
    nearest_x = torch.where(empty, zero, nearest_x / scale)
    nearest_y = torch.where(empty, zero, nearest_y / scale)
    closing = torch.where(empty, zero, closing)
    return torch.stack((
        player_velocity[:, 0], player_velocity[:, 1],
        nearest_x, nearest_y, distance, closing,
    ), dim=1)


_GRID_AXIS_CACHE = {}
_LEVEL_CACHE = {}


def _entity_levels(device, dtype):
    """Entity type of each one-hot channel, (1, CELL_TYPES, 1, 1), per device."""
    key = (device, dtype)
    cached = _LEVEL_CACHE.get(key)
    if cached is None:
        cached = torch.arange(1, CELL_TYPES + 1, device=device, dtype=dtype)
        cached = _LEVEL_CACHE[key] = cached.reshape(1, CELL_TYPES, 1, 1)
    return cached


def _grid_axes(device, dtype):
    """Cell coordinate grids and the lethal channel index, built once per device."""
    key = (device, dtype)
    cached = _GRID_AXIS_CACHE.get(key)
    if cached is None:
        axis = torch.arange(GRID_SIZE, device=device, dtype=dtype)
        cached = (
            axis.reshape(1, 1, GRID_SIZE),
            axis.reshape(1, GRID_SIZE, 1),
            torch.tensor(LETHAL_OFFSETS, device=device, dtype=torch.long),
        )
        _GRID_AXIS_CACHE[key] = cached
    return cached


def _player_channels(coordinates, dtype):
    """Mark the player's own cell in the grid.

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
    cells = (coordinates[:, :2] * scale).long().clamp(0, GRID_SIZE - 1)
    flat = cells[:, 1] * GRID_SIZE + cells[:, 0]
    batch = flat.shape[0]
    channels = torch.zeros(batch, 1, GRID_SIZE * GRID_SIZE,
                           dtype=dtype, device=coordinates.device)
    channels.scatter_(2, flat.reshape(batch, 1, 1), 1.0)
    return channels.reshape(batch, 1, GRID_SIZE, GRID_SIZE)


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
        """Decode the flat observation into player state and grid channels.

        Channels, in order: eight one-hot entity types, the player's own cell,
        then the velocity of whatever occupies each cell, one plane per axis.
        Eleven in total, from a single frame -- the history stack is gone
        because the velocity it existed to imply is now given outright.
        """
        batch = observations.shape[0]
        coordinates = observations[:, :PLAYER_VALUES]
        start = PLAYER_VALUES
        cells = torch.round(
            observations[:, start:start + CELL_COUNT] * CELL_TYPES
        ).clamp(0, CELL_TYPES).reshape(batch, 1, GRID_SIZE, GRID_SIZE)
        # Empty cells become all-zero channels; the rest carry entity type.
        # One comparison per type, written straight into NCHW: the same 0/1
        # planes F.one_hot gave, without its int64 detour, the min/max checks
        # it runs over the input, or the permuted copy back to channels-first.
        levels = _entity_levels(observations.device, observations.dtype)
        grids = (cells == levels).to(dtype=observations.dtype)

        velocity = observations[:, start + CELL_COUNT:].reshape(
            batch, VELOCITY_CHANNELS, GRID_SIZE, GRID_SIZE)
        player = _player_channels(coordinates, observations.dtype)
        return coordinates, torch.cat((grids, player, velocity), dim=1)

    def forward(self, observations):
        coordinates, grids = self.policy_inputs(observations)
        senses = _sense_features(coordinates, grids)
        return torch.cat(
            (self.grid_net(grids),
             self.coordinate_net(torch.cat((coordinates, senses), dim=1))),
            dim=1,
        )


def policy_class(architecture):
    """The SB3 policy class, which is not always the default one.

    velocity-flow replaces the action head entirely -- its logits are readings
    of a danger field rather than a learned projection -- so it needs its own
    ActorCriticPolicy subclass, not just a features extractor.
    """
    if architecture in ("velocity-flow", "velocity-flow-v2"):
        from .velocity import VelocityFlowPolicy
        return VelocityFlowPolicy
    return "MlpPolicy"


def policy_kwargs(architecture):
    if architecture == "mlp":
        return {}
    if architecture in ("velocity-flow", "velocity-flow-v2"):
        from .velocity import VelocityFlowExtractor, VelocityFlowV2Extractor
        return {
            "features_extractor_class": (VelocityFlowExtractor
                                         if architecture == "velocity-flow"
                                         else VelocityFlowV2Extractor),
            # No policy trunk: the field readings are the logits, so anything
            # between them and the distribution would be a head learning to
            # second-guess the controller.
            "net_arch": {"pi": [], "vf": [64]},
            # partial, not a lambda: checkpoints pickle this, and a lambda
            # is stored by value -- tying every saved policy to this file.
            "activation_fn": partial(nn.LeakyReLU, NEGATIVE_SLOPE),
        }
    if architecture == "hybrid-cnn":
        return {
            "features_extractor_class": DodgeHybridExtractor,
            "net_arch": {"pi": [128, 64], "vf": [128, 64]},
            # partial, not a lambda: checkpoints pickle this, and a lambda
            # is stored by value -- tying every saved policy to this file.
            "activation_fn": partial(nn.LeakyReLU, NEGATIVE_SLOPE),
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
    return EXTRACTOR_ARCHITECTURE.get(
        type(model.policy.features_extractor).__name__, "mlp")


def architecture_from_kwargs(policy_kwargs):
    """The architecture a saved checkpoint describes, before it is built.

    Loading needs this to decide which hyperparameters to impose, and it cannot
    ask the model -- the model does not exist yet.
    """
    extractor = (policy_kwargs or {}).get("features_extractor_class")
    return EXTRACTOR_ARCHITECTURE.get(getattr(extractor, "__name__", ""), "mlp")
