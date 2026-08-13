"""A convolutional danger field, read by a fixed controller.

The network's only output is a scalar danger per arena cell. It never chooses
an action. Choosing is arithmetic: simulate where each of the nine actions
takes the player, read the field along those paths, and let the totals be the
logits. Nothing about the controller is learned, so all the learning pressure
lands on the field.

Why this shape rather than the hybrid extractor's:

* **No flatten.** The hybrid CNN ends `Flatten -> Linear(4096, 128)`, which is
  524k of its ~600k parameters -- 87% of the model in the one layer that
  discards the spatial structure the convolutions just built. It is also the
  layer that died. A field output is fully convolutional, and the whole trunk
  here is ~80k parameters.

* **Translation equivariance survives to the output.** The hybrid CNN has to
  learn "dodge a threat closing from the left" separately at each absolute
  position, because the flatten binds features to locations. A field is
  computed the same way everywhere, so one learned pattern covers the arena.

* **It can be looked at.** The output is a picture. After a policy collapse
  that hid behind healthy-looking entropy for two million steps, an
  architecture whose internal state renders is worth something by itself.

The known limitation, established by the analytic version in gradient.py: a
scalar over positions cannot express pursuit. "Step sideways because it is
chasing you" is a fact about trajectories, not about any cell. Sampling the
field along each action's path recovers some of that -- the path is a
trajectory even if the field is not -- but a field that has to serve every
action at once cannot encode where a threat will be *because of* the action
taken. If that ceiling binds, the fix is a field conditioned on time, not a
deeper trunk.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from .actions import ACTION_MASKS
from .gradient import simulate_player
from .observe import ARENA_SIZE, CELL_SIZE, GRID_SIZE
from .policies import (HYBRID_INPUT_CHANNELS, NEGATIVE_SLOPE, COORDINATE_INPUTS,
                       DodgeHybridExtractor, _sense_features, activation)

N_ACTIONS = len(ACTION_MASKS)

# Frames ahead at which each action's path is sampled. The first is the
# decision interval itself; the rest are the coast after the button is
# released, which is what the environment actually does.
SAMPLE_FRAMES = (2, 4, 8, 12, 18, 24)
# Later samples are discounted: they are both less certain and further away.
SAMPLE_DECAY = 0.93

# Initial logit sharpness. Low enough that the first rollouts are near
# uniform -- committing hard to a randomly initialised field would be
# committing to noise -- and learnable, so the policy sharpens as the
# field becomes worth trusting.
INITIAL_TEMPERATURE = 0.5

TRUNK_CHANNELS = 32
VALUE_FEATURES = 64
FIELD_FEATURES_DIM = N_ACTIONS + VALUE_FEATURES


def action_offsets(action_repeat=4, frames=SAMPLE_FRAMES):
    """Where each action moves the player, in pixels, starting from rest.

    Taken from the cart's own movement model, which gradient.py verified
    against the live game to 0.000px over 24 frames. Starting from rest is an
    approximation -- the real player carries velocity -- but it keeps the
    offsets a constant, so the controller stays a lookup rather than a
    simulation inside the forward pass.
    """
    horizon = max(frames)
    # Run from the middle of the arena and read the result as a delta.
    # simulate_player clamps to 2..125, so starting at the origin pins every
    # leftward and upward action against the wall -- which made "left" and
    # "idle" the same offset, and gave all nine a spurious +2 bias.
    origin = ARENA_SIZE / 2.0
    offsets = np.empty((N_ACTIONS, len(frames), 2), dtype=np.float32)
    for action, mask in enumerate(ACTION_MASKS):
        path = simulate_player(origin, origin, 0.0, 0.0, mask, horizon,
                               held=action_repeat)
        offsets[action] = [path[frame - 1] - origin for frame in frames]
    return offsets


def _sample_weights(frames=SAMPLE_FRAMES, decay=SAMPLE_DECAY):
    weights = np.array([decay ** frame for frame in frames], dtype=np.float32)
    return weights / weights.sum()


class DodgeFieldExtractor(BaseFeaturesExtractor):
    """Produce a danger field, then read it along each action's path."""

    def __init__(self, observation_space, action_repeat=4):
        super().__init__(observation_space, features_dim=FIELD_FEATURES_DIM)
        self.field_net = nn.Sequential(
            nn.Conv2d(HYBRID_INPUT_CHANNELS, 32, 3, padding=1),
            activation(),
            nn.Conv2d(32, 64, 3, padding=1),
            activation(),
            # Dilated, to reach across the arena without another downsample.
            # Receptive field ends at 11 cells, 44px, a third of the width.
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            activation(),
            nn.Conv2d(64, TRUNK_CHANNELS, 3, padding=1),
            activation(),
        )
        self.field_head = nn.Conv2d(TRUNK_CHANNELS, 1, 1)
        self.value_net = nn.Sequential(
            nn.Linear(TRUNK_CHANNELS + COORDINATE_INPUTS, VALUE_FEATURES),
            activation(),
            nn.Linear(VALUE_FEATURES, VALUE_FEATURES),
            activation(),
        )
        self.register_buffer("offsets",
                             torch.as_tensor(action_offsets(action_repeat)))
        self.register_buffer("sample_weights",
                             torch.as_tensor(_sample_weights()))

    def field(self, observations):
        """The danger field alone, for rendering and for inspection."""
        _, grids = DodgeHybridExtractor.policy_inputs(observations)
        return self.field_head(self.field_net(grids))

    def forward(self, observations):
        coordinates, grids = DodgeHybridExtractor.policy_inputs(observations)
        trunk = self.field_net(grids)
        field = self.field_head(trunk)

        # Current-frame player position, in pixels.
        position = coordinates[:, -2:] * (ARENA_SIZE - 1)
        # (batch, actions, samples, 2)
        points = position[:, None, None, :] + self.offsets[None]
        # grid_sample wants normalised coordinates, and (x, y) order.
        normalised = points / (GRID_SIZE * CELL_SIZE / 2.0) - 1.0
        batch = observations.shape[0]
        sampled = F.grid_sample(
            field, normalised.reshape(batch, N_ACTIONS, len(SAMPLE_FRAMES), 2),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        # (batch, 1, actions, samples) -> danger per action along its path
        danger = (sampled.squeeze(1) * self.sample_weights).sum(dim=-1)

        senses = _sense_features(coordinates, grids)
        pooled = trunk.mean(dim=(2, 3))
        value = self.value_net(torch.cat((pooled, coordinates, senses), dim=1))
        # Negated: the controller prefers low danger, and these become logits.
        return torch.cat((-danger, value), dim=1)


class _FieldLogits(nn.Module):
    """Turn path danger into action logits, with a learnable sharpness.

    The field's units are arbitrary -- nothing pins its scale -- so without a
    temperature the initial policy could be anywhere between uniform and
    one-hot, and entropy would be set by initialisation rather than by
    learning. Starting small keeps the first rollouts exploratory.
    """

    EPSILON = 1e-3

    def __init__(self, temperature=INITIAL_TEMPERATURE):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))

    def forward(self, latent):
        danger = latent[:, :N_ACTIONS]
        # Standardise across the nine actions, so only the *shape* of the field
        # reaches the distribution and its magnitude does not. Untouched, a
        # freshly initialised field spreads its readings by 0.03 -- roughly
        # thirty times too little to move a softmax -- so the policy would sit
        # at uniform until the field grew, learning nothing in the meantime.
        # Standardising also stops a field that drifts large from silently
        # sharpening the policy toward greedy.
        danger = danger - danger.mean(dim=1, keepdim=True)
        spread = danger.std(dim=1, keepdim=True).clamp_min(self.EPSILON)
        return danger / spread * self.log_temperature.exp()


class DodgeFieldPolicy(ActorCriticPolicy):
    """ActorCritic whose logits come from the field, not from a head."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = _FieldLogits()
        # The optimizer was built over the head that was just replaced.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
