"""VelocityFlow: a convolutional danger field, read by a fixed controller.

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

* **Motion is given, not inferred.** Its input is one frame carrying a velocity
  per occupied cell, so nothing has to be recovered from the difference between
  two quantised grids -- a difference that was usually zero, since everything
  in this game moves under one cell per decision.

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

from . import speed
from .actions import ACTION_MASKS
from .gradient import simulate_player
from .observe import ARENA_SIZE, CELL_SIZE, GRID_SIZE
from .policies import (HYBRID_INPUT_CHANNELS, NEGATIVE_SLOPE, COORDINATE_INPUTS,
                       DodgeHybridExtractor, _sense_features, activation)

N_ACTIONS = len(ACTION_MASKS)

# Frames ahead at which each action is judged, and equally the times the field
# is predicted for: slice k answers "how dangerous is this cell at frame
# SAMPLE_FRAMES[k]".
#
# Stretched to nearly two seconds because the manoeuvres that were failing take
# that long. Pattern 11 sweeps two lethal bars across the arena at 0.7px a
# frame; playing it means letting the near bar pass and moving into the space
# behind it, which needs the bar's position seconds out, not the 17px of travel
# a 24-frame horizon could see.
SAMPLE_FRAMES = (4, 12, 24, 48, 72, 108)
# Gentle enough that the far samples still carry weight: at the old 0.93 the
# 108-frame sample would be discounted to 0.0004 and the horizon would be
# decorative.
SAMPLE_DECAY = 0.985

# Initial logit sharpness. Low enough that the first rollouts are near
# uniform -- committing hard to a randomly initialised field would be
# committing to noise -- and learnable, so the policy sharpens as the
# field becomes worth trusting.
INITIAL_TEMPERATURE = 0.5

TRUNK_CHANNELS = 32
CONTEXT_CHANNELS = 16
CONTEXT_GRID = 4
VALUE_FEATURES = 64
FIELD_FEATURES_DIM = N_ACTIONS + VALUE_FEATURES

# --- velocity-flow-v2 ---------------------------------------------------
# A coarse arena map for the critic. v1 hands the value function
# trunk.mean(dim=(2, 3)) -- 32 channel means over the whole 32x32 field --
# so its entire view of the layout is an average, and "am I about to die"
# is almost entirely a question of layout. Eight channels on a 4x4 grid is
# 128 numbers and a 1x1 conv: negligible against the trunk, and enough to
# tell threats converging on the player from threats in the far corner.
VALUE_SUMMARY_CHANNELS = 8
VALUE_SUMMARY_GRID = 4
VALUE_SUMMARY_DIM = VALUE_SUMMARY_CHANNELS * VALUE_SUMMARY_GRID ** 2

# Run the trunk's three heavy convolutions at 16x16 instead of 32x32 and
# interpolate back. By multiply-accumulate count that is roughly 79M -> 22M
# per sample, and the optimizer is where the wall-clock goes: a rollout
# spends about five seconds learning against one collecting.
#
# It is the one change here that trades against the design thesis, so it is
# a flag rather than a fact. The field it draws is smoother, and if what
# makes the controller work is precision at a threat's edge, this is what
# costs it. Build the extractor with half_resolution=False to A/B it.
HALF_RESOLUTION = True


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


class VelocityFlowExtractor(BaseFeaturesExtractor):
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
        # Arena-wide context, folded back in at full resolution. The trunk's
        # receptive field is 11 cells, 44px; pattern 11's two bars start 151px
        # apart, so no output cell could see both and "which half is safe" was
        # not a question the field could ask. Pooling to 4x4 and broadcasting
        # back gives every cell the whole arena's summary while the field stays
        # fully convolutional.
        self.context_net = nn.Sequential(
            nn.Conv2d(TRUNK_CHANNELS, 32, 3, padding=1),
            activation(),
            nn.Conv2d(32, CONTEXT_CHANNELS, 3, padding=1),
            activation(),
        )
        # One output per time slice rather than one field.
        self.field_head = nn.Conv2d(TRUNK_CHANNELS + CONTEXT_CHANNELS,
                                    len(SAMPLE_FRAMES), 1)
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

    def _value_features(self, trunk, coordinates, senses):
        """What the critic sees. Overridden by v2, which adds a coarse map."""
        pooled = trunk.mean(dim=(2, 3))
        return self.value_net(torch.cat((pooled, coordinates, senses), dim=1))

    def _field_from_trunk(self, trunk):
        context = self.context_net(F.adaptive_avg_pool2d(trunk, CONTEXT_GRID))
        # Bilinear, not nearest. Nearest holds each of the 4x4 context values
        # flat across an 8x8 cell block -- 32x32px of a 128px arena -- so the
        # context arrived as sixteen hard-edged tiles that swamped the trunk's
        # per-cell detail. A global summary should be a smooth low-frequency
        # surface; the step at a block edge is not a fact about danger, and the
        # controller samples across those edges.
        context = speed.upsample(context, trunk.shape[-2:])
        return self.field_head(torch.cat((trunk, context), dim=1))

    def field(self, observations):
        """Danger per cell per time slice: (batch, len(SAMPLE_FRAMES), 32, 32).

        Slice 0 is the nearest future and is what the watch overlay draws.
        """
        _, grids = DodgeHybridExtractor.policy_inputs(observations)
        return self._field_from_trunk(self.field_net(speed.trunk_layout(grids)))

    def forward(self, observations):
        coordinates, grids = DodgeHybridExtractor.policy_inputs(observations)
        trunk = self.field_net(speed.trunk_layout(grids))
        field = self._field_from_trunk(trunk)

        # Player position, in pixels. The leading pair, not the trailing one:
        # the coordinate block is (px, py, dpx, dpy), and it used to be (px, py)
        # for each of two stacked frames, where the tail was the current
        # position. Taking the tail now samples the field at the player's
        # velocity, which is nowhere near the player.
        position = coordinates[:, :2] * (ARENA_SIZE - 1)
        # (batch, actions, samples, 2)
        points = position[:, None, None, :] + self.offsets[None]
        # grid_sample wants normalised coordinates, and (x, y) order.
        normalised = points / (GRID_SIZE * CELL_SIZE / 2.0) - 1.0
        batch = observations.shape[0]
        # Samples every slice at every point: (batch, slices, actions, samples).
        sampled = F.grid_sample(
            field, normalised.reshape(batch, N_ACTIONS, len(SAMPLE_FRAMES), 2),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        # Each sample must be read from the slice for its own arrival time --
        # where the player will be at frame t, judged by the field predicted
        # for frame t. That pairing is the diagonal; taking any other entry
        # would ask whether a cell is safe at a time the player is not there.
        # It is what lets "lethal now, clear in two seconds" be expressed at
        # all, which a single field per cell cannot say.
        paired = sampled.permute(0, 2, 1, 3).diagonal(dim1=2, dim2=3)
        danger = (paired * self.sample_weights).sum(dim=-1)

        senses = _sense_features(coordinates, grids)
        value = self._value_features(trunk, coordinates, senses)
        # Negated: the controller prefers low danger, and these become logits.
        return torch.cat((-danger, value), dim=1)


class _HalfResolutionTrunk(nn.Module):
    """The v1 trunk with its three heavy convolutions at half resolution.

    The stem stays at full resolution so the first layer still sees every 4px
    cell, and the result is interpolated back before the field head, so the
    field itself is unchanged in shape. Bilinear rather than nearest for the
    same reason the context branch is: a step at a block edge is not a fact
    about danger, and the controller samples across those edges.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(HYBRID_INPUT_CHANNELS, 32, 3, padding=1),
            activation(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            activation(),
            # Still dilated, and at half resolution its reach in arena terms
            # doubles: 11 cells of 8px rather than of 4px.
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            activation(),
            nn.Conv2d(64, TRUNK_CHANNELS, 3, padding=1),
            activation(),
        )

    def forward(self, grids):
        stem = self.stem(grids)
        body = self.body(F.avg_pool2d(stem, 2))
        return speed.upsample(body, stem.shape[-2:])


class VelocityFlowV2Extractor(VelocityFlowExtractor):
    """v1, with a critic that can see space and a cheaper trunk.

    Two independent changes, kept in one architecture so they can be measured
    against v1 as a package:

    * The value function receives a 4x4 arena summary as well as the global
      mean, because a mean over 32x32 cells cannot say *where* anything is.
    * The trunk's middle runs at 16x16, which is the only structural lever on
      a training cycle whose time is dominated by the optimizer.
    """

    def __init__(self, observation_space, action_repeat=4,
                 half_resolution=HALF_RESOLUTION):
        super().__init__(observation_space, action_repeat=action_repeat)
        self.half_resolution = half_resolution
        if half_resolution:
            self.field_net = _HalfResolutionTrunk()
        # 1x1, so the summary is a channel projection of each cell rather than
        # another receptive field -- the trunk has already done that work.
        self.value_pool = nn.Conv2d(TRUNK_CHANNELS, VALUE_SUMMARY_CHANNELS, 1)
        self.value_net = nn.Sequential(
            nn.Linear(TRUNK_CHANNELS + VALUE_SUMMARY_DIM + COORDINATE_INPUTS,
                      VALUE_FEATURES),
            activation(),
            nn.Linear(VALUE_FEATURES, VALUE_FEATURES),
            activation(),
        )

    def value_summary(self, trunk):
        """A coarse map of the arena for the critic: (batch, 8 * 4 * 4)."""
        return F.adaptive_avg_pool2d(self.value_pool(trunk),
                                     VALUE_SUMMARY_GRID).flatten(1)

    def _value_features(self, trunk, coordinates, senses):
        pooled = trunk.mean(dim=(2, 3))
        return self.value_net(
            torch.cat((pooled, self.value_summary(trunk), coordinates, senses),
                      dim=1))


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


class VelocityFlowPolicy(ActorCriticPolicy):
    """ActorCritic whose logits come from the field, not from a head.

    forward and evaluate_actions are Stable-Baselines3's, op for op, minus the
    torch.distributions object they build on the way. That object is the same
    arithmetic -- normalise the logits, softmax, gather -- wrapped in argument
    validation that costs a device sync per check on CUDA, and SB3 builds one
    per call. Rollout forwards on CUDA additionally replay as a CUDA graph.
    """

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = _FieldLogits()
        # The optimizer was built over the head that was just replaced.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        speed.configure_cudnn()
        self._graphs = speed.GraphCache()

    def __getstate__(self):
        # Captured graphs belong to one process and one device.
        state = self.__dict__.copy()
        state.pop("_graphs", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._graphs = speed.GraphCache()

    def _distribution_inputs(self, obs):
        """Normalised log-probabilities and values.

        log_pmf is exactly what torch's Categorical stores as its logits.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)
        log_pmf = logits - logits.logsumexp(dim=-1, keepdim=True)
        return log_pmf, values

    def _sampling_inputs(self, obs):
        log_pmf, values = self._distribution_inputs(obs)
        return log_pmf, F.softmax(log_pmf, dim=-1), values

    def _parameter_signature(self):
        return tuple(t.data_ptr() for t in self.parameters()) + tuple(
            t.data_ptr() for t in self.buffers())

    def forward(self, obs, deterministic=False):
        graph = None
        if (speed.CUDA_GRAPHS and obs.is_cuda and not self.training
                and not torch.is_grad_enabled()):
            graph = self._graphs.lookup(self._sampling_inputs, obs,
                                        self._parameter_signature())
        if graph is not None:
            log_pmf, probs, values = graph(obs)
            # The graph's outputs are overwritten by its next replay.
            values = values.clone()
        else:
            log_pmf, probs, values = self._sampling_inputs(obs)
        # Categorical.mode and Categorical.sample, as SB3 calls them.
        if deterministic:
            actions = torch.argmax(probs, dim=1)
        else:
            actions = torch.multinomial(probs.reshape(-1, N_ACTIONS), 1, True).T
            actions = actions.reshape(probs.shape[:-1])
        log_prob = self._log_prob(log_pmf, actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    @staticmethod
    def _log_prob(log_pmf, actions):
        """Categorical.log_prob, without the support check."""
        value = actions.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, log_pmf)
        return log_pmf.gather(-1, value[..., :1]).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        log_pmf, values = self._distribution_inputs(obs)
        log_prob = self._log_prob(log_pmf, actions)
        # Categorical.entropy, op for op.
        clamped = torch.clamp(log_pmf, min=torch.finfo(log_pmf.dtype).min)
        entropy = -(clamped * F.softmax(log_pmf, dim=-1)).sum(-1)
        return values, log_prob, entropy
