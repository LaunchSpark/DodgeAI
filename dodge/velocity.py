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
from .critic import CRITIC_FEATURE_COUNT, CRITIC_GROUPS, WIDE_OBSERVATION_SIZE
from .critic import critic_layout as current_layout
from .observe import ARENA_SIZE, CELL_SIZE, GRID_SIZE, OBSERVATION_SIZE, VELOCITY_SCALE
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

# --- velocity-flow-v3 ---------------------------------------------------
# Frames each slice's probe holds its button, one per SAMPLE_FRAMES entry.
# A held button and its coast end exactly 2.5px a frame of hold away, so a
# single hold stops every path about twenty frames after release: at v2's
# four, the 24-, 48-, 72- and 108-frame slices all read one point 10px out.
# Graduated, the six slices reach 4, 9, 15, 20, 25 and 30px -- 42px on a
# diagonal, which the cart does not normalise -- so the far slices ask about
# the corridor an action opens rather than the same cell four times.
SLICE_HOLDS = (4, 4, 6, 8, 10, 12)
# The bound on a logit once the field is no longer standardised per state.
# Nine actions at +-5.5 is already all but deterministic; the cap is there so
# a field that grows cannot buy certainty the entropy bonus never priced.
LOGIT_CAP = 5.5
# The critic-only feature groups v3's critic reads, in layout order. Left out:
# next_pattern and next_enemy, which the probe found worth +0.001 and +0.0000
# explained variance (docs/superpowers/plans/2026-11-01-critic-probe-result.txt).
# The engine still produces them; the critic just does not look.
V3_CRITIC_GROUPS = ("clock", "timers", "active_pattern", "nearest_enemies")


def action_offsets(action_repeat=4, frames=SAMPLE_FRAMES, holds=None):
    """Where each action moves the player, in pixels, starting from rest.

    Taken from the cart's own movement model, which gradient.py verified
    against the live game to 0.000px over 24 frames. Starting from rest keeps
    the offsets a constant, so the controller stays a lookup rather than a
    simulation inside the forward pass; v3 adds the carried velocity back as a
    second constant (see velocity_carry).

    `holds` gives each sample its own hold; by default every sample holds for
    action_repeat frames.
    """
    holds = (action_repeat,) * len(frames) if holds is None else holds
    # Run from the middle of the arena and read the result as a delta.
    # simulate_player clamps to 2..125, so starting at the origin pins every
    # leftward and upward action against the wall -- which made "left" and
    # "idle" the same offset, and gave all nine a spurious +2 bias.
    origin = ARENA_SIZE / 2.0
    offsets = np.empty((N_ACTIONS, len(frames), 2), dtype=np.float32)
    for action, mask in enumerate(ACTION_MASKS):
        for sample, (frame, held) in enumerate(zip(frames, holds)):
            path = simulate_player(origin, origin, 0.0, 0.0, mask, frame, held=held)
            offsets[action, sample] = path[frame - 1] - origin
    return offsets


def velocity_carry(frames=SAMPLE_FRAMES):
    """How far one pixel a frame of carried velocity has moved the player by each frame.

    Movement is linear in the starting velocity until a wall, so a player
    moving at v lands at its at-rest offset plus v times this: 4(1 - 0.8^t)
    for the cart's friction. Read off simulate_player rather than written as
    that formula, so it cannot disagree with the movement model.
    """
    origin = ARENA_SIZE / 2.0
    path = simulate_player(origin, origin, 1.0, 0.0, 0, max(frames))
    return np.array([path[frame - 1, 0] - origin for frame in frames], dtype=np.float32)


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

    def _read_points(self, coordinates):
        """Where each action's path is read, in pixels.

        (batch, actions, samples, 2), in (x, y) order.
        """
        # Player position, in pixels. The leading pair, not the trailing one:
        # the coordinate block is (px, py, dpx, dpy), and it used to be (px, py)
        # for each of two stacked frames, where the tail was the current
        # position. Taking the tail now samples the field at the player's
        # velocity, which is nowhere near the player.
        position = coordinates[:, :2] * (ARENA_SIZE - 1)
        # (batch, actions, samples, 2)
        return position[:, None, None, :] + self.offsets[None]

    def forward(self, observations):
        coordinates, grids = DodgeHybridExtractor.policy_inputs(observations)
        trunk = self.field_net(speed.trunk_layout(grids))
        field = self._field_from_trunk(trunk)

        # grid_sample wants normalised coordinates, and (x, y) order.
        normalised = (self._read_points(coordinates)
                      / (GRID_SIZE * CELL_SIZE / 2.0) - 1.0)
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


class _DilatedConv2d(nn.Conv2d):
    """nn.Conv2d(..., 3, padding=2, dilation=2), run by speed.dilated_conv.

    A subclass rather than a wrapper so the parameters keep their place and
    their names -- field_net.body.2.weight, as every checkpoint stores them --
    and so the constructor draws its initialisation from the RNG exactly as
    the plain nn.Conv2d did, which a seeded fresh policy depends on.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 3, padding=2, dilation=2)

    def forward(self, features):
        return speed.dilated_conv(self, features)


class _HalfResolutionTrunk(nn.Module):
    """The v1 trunk with its three heavy convolutions at half resolution.

    The stem stays at full resolution so the first layer still sees every 4px
    cell, and the result is interpolated back before the field head, so the
    field itself is unchanged in shape. Bilinear rather than nearest for the
    same reason the context branch is: a step at a block edge is not a fact
    about danger, and the controller samples across those edges.

    That interpolation is what forward() returns, for field() and the watch
    overlay. The extractor's own forward pass never builds it: everything it
    reads from the 32x32 trunk is a fixed linear map of half(), and it applies
    those maps to the 16x16 body directly (see VelocityFlowV2Extractor).
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
            _DilatedConv2d(64, 64),
            activation(),
            nn.Conv2d(64, TRUNK_CHANNELS, 3, padding=1),
            activation(),
        )

    def half(self, grids):
        """The trunk before it is interpolated back: (batch, 32, 16, 16)."""
        return self.body(speed.halve(self.stem(grids)))

    def forward(self, grids):
        return speed.upsample(self.half(grids), grids.shape[-2:])


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
            self._register_trunk_maps()
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

    # --- the half-resolution forward pass, without the 32x32 trunk --------
    #
    # With half_resolution the trunk is bilinear_upsample(body), body 16x16,
    # and building it was most of what an update spent outside the
    # convolutions: the upsample itself, both 32x32 1x1 heads, the adaptive
    # pool and the grid_sample, forward and backward -- about half of a 56ms
    # minibatch on a GTX 1070. Yet the forward pass reads that trunk in only
    # four ways, and every one of them is linear in it:
    #
    # * its 4x4 adaptive average, for the context branch;
    # * its mean, and value_pool's 4x4 average, for the critic;
    # * the field head's 1x1 projection of it, at 54 points per sample.
    #
    # Upsampling, pooling and bilinear reading all act on each axis of each
    # channel separately, so each of these is a small matrix per axis applied
    # to the 16x16 body, and the 1x1 projections commute with all of them --
    # every one of these maps has rows that sum to 1, so a bias passes through
    # unchanged. The results are the same numbers, summed in a different
    # order. tests/test_velocity_trunk.py holds them to a frozen copy of the
    # route that built the trunk.
    #
    # field() still builds the 32x32 field. It is what the overlay draws, and
    # it is not on the training path.

    def _register_trunk_maps(self):
        assert CONTEXT_GRID == VALUE_SUMMARY_GRID, \
            "the context and the critic's summary share one pooled trunk"
        half, full = GRID_SIZE // 2, GRID_SIZE
        # Per axis: bilinear 16 -> 32 (as speed.upsample does it, so built
        # from F.interpolate itself rather than from a formula that could
        # disagree with it at the edges), 4 -> 32 likewise for the context,
        # and adaptive average pooling 32 -> 4.
        trunk_up = _axis_map(lambda x: F.interpolate(
            x[..., None], size=(full, 1), mode="bilinear",
            align_corners=False)[..., 0], half)
        context_up = _axis_map(lambda x: F.interpolate(
            x[..., None], size=(full, 1), mode="bilinear",
            align_corners=False)[..., 0], CONTEXT_GRID)
        pool = _axis_map(lambda x: F.adaptive_avg_pool1d(x, CONTEXT_GRID), full)
        # The buffers below are not persistent: derived constants, so
        # checkpoints neither carry nor expect them, and one saved before they
        # existed loads as is.
        #
        # (16 -> 4): adaptive_avg_pool2d(upsample(body), 4) per axis. Not a
        # plain 4x4 pool of the body: each 8-cell bin of the upsampled trunk
        # reaches a quarter of a cell into its neighbour's half of the body.
        # Stored as the 2-D map on the flattened 16x16, (16, 256), the
        # Kronecker product of the per-axis one: four times the arithmetic of
        # applying it an axis at a time, but one plain GEMM with no transposes
        # either way through autograd -- 0.5ms forward and backward at
        # minibatch 512 on a GTX 1070, against 1.6ms for the separable einsum.
        axis_pool = pool @ trunk_up
        self.register_buffer("trunk_pool", torch.kron(axis_pool, axis_pool).float(),
                             persistent=False)
        # (32 -> 16 + 4): a point's bilinear weights over the 32 cells of one
        # axis, carried back to the body's 16 and the context's 4.
        self.register_buffer("read_basis",
                             torch.cat((trunk_up, context_up), dim=1).float(),
                             persistent=False)
        self.register_buffer("cell_index", torch.arange(full, dtype=torch.float32),
                             persistent=False)

    def _cell_weights(self, cells):
        """grid_sample's bilinear weights, per axis, over the 32 cells.

        (..., 2) points in cell units -> (..., 2, 32). A point in cell units is
        where grid_sample puts a normalised coordinate g, for align_corners=
        False: ((g + 1) * 32 - 1) / 2, which for the controller's pixel points
        is simply pixels / CELL_SIZE - 0.5 -- cell i's centre is at i.

        grid_sample, with border padding, clamps that into [0, 31] and weights
        the cell below it by 1 - f and the one above by f. That is the hat
        function max(0, 1 - |u - i|) over the cell indices i: 1 - f at the cell
        below, f at the one above, zero at every other. At the far edge the
        clamp puts the point on cell 31 with f = 0, so the cell above, which
        does not exist, would carry no weight -- exactly grid_sample's
        zero-weighted out-of-range corner. Written as arithmetic rather than
        as floor and two index comparisons because the comparisons were the
        slower: 240 -> 60us at batch 8 on the CPU.
        """
        cells = cells.clamp(0, GRID_SIZE - 1)
        return (1 - (cells[..., None] - self.cell_index).abs()).clamp_min(0)

    def _path_danger(self, body, context, coordinates):
        """forward()'s danger, read without the field: (batch, actions).

        Slice k is only ever read at the points sample k lands on, so the
        field head is applied per slice at those points and nowhere else.
        """
        # Slice-major, (batch, samples, actions, 2): sample k's nine points
        # are all read from slice k, so they batch against it. In cell units
        # directly -- the same number as normalising them for grid_sample and
        # having it unnormalise them, without the four kernels that round trip
        # costs an eager forward.
        cells = self._read_points(coordinates).transpose(1, 2) / CELL_SIZE - 0.5
        # (batch, samples, actions, axis, 16 + 4); axis 0 is x, the column.
        weights = self._cell_weights(cells) @ self.read_basis
        half = GRID_SIZE // 2
        body_x, body_y = weights[..., 0, :half], weights[..., 1, :half]
        context_x, context_y = weights[..., 0, half:], weights[..., 1, half:]

        # The field head is one 1x1 conv over cat(trunk, context): its trunk
        # half projects the body before upsampling, and its context half the
        # 4x4 context before upsampling, since both commute with a per-channel
        # interpolation. Slice k is then read at sample k's points only -- the
        # diagonal forward() takes from grid_sample's every-slice-every-point.
        head = self.field_head.weight
        body_field = F.conv2d(body, head[:, :TRUNK_CHANNELS])
        context_field = F.conv2d(context, head[:, TRUNK_CHANNELS:])

        def read(field, rows, columns):
            # (b, k, a, h) @ (b, k, h, w): each point's weighted row, then its
            # weighted column. A batched matmul, measured faster here than the
            # equivalent three-operand einsum.
            return ((rows @ field) * columns).sum(dim=-1)

        # (batch, samples, actions)
        paired = (read(body_field, body_y, body_x)
                  + read(context_field, context_y, context_x)
                  + self.field_head.bias[:, None])
        return (paired * self.sample_weights[:, None]).sum(dim=1)

    def _danger_and_value_inputs(self, observations):
        """The half-resolution forward pass, up to the critic's first layer.

        Returns the path danger (batch, actions) and what value_net reads.
        Split out so a subclass can widen the critic's input without touching
        the actor.
        """
        coordinates, grids = DodgeHybridExtractor.policy_inputs(observations)
        body = self.field_net.half(speed.trunk_layout(grids))
        # adaptive_avg_pool2d(trunk, 4), for the context and the critic alike.
        pooled = F.linear(body.flatten(2), self.trunk_pool).unflatten(
            2, (CONTEXT_GRID, CONTEXT_GRID))
        context = self.context_net(pooled)
        danger = self._path_danger(body, context, coordinates)

        senses = _sense_features(coordinates, grids)
        # The trunk's mean is the mean of its sixteen equal 8x8 bins, and
        # value_pool, being 1x1, commutes with pooling them.
        inputs = torch.cat(
            (pooled.mean(dim=(2, 3)), self.value_pool(pooled).flatten(1),
             coordinates, senses), dim=1)
        return danger, inputs

    def forward(self, observations):
        if not self.half_resolution:
            return super().forward(observations)
        danger, inputs = self._danger_and_value_inputs(observations)
        return torch.cat((-danger, self.value_net(inputs)), dim=1)


class VelocityFlowV3Extractor(VelocityFlowV2Extractor):
    """v2's trunk, read where each action actually takes the player, with a
    critic that also reads hidden game state.

    The observation is the actor's OBSERVATION_SIZE values followed by
    CRITIC_FEATURE_COUNT the actor must never see. The actor path slices the
    first block off before anything else: policy_inputs reads the velocity
    planes to the end of the vector, so the extra columns would otherwise be
    folded into them. The critic reads only `critic_groups` of them,
    V3_CRITIC_GROUPS by default, recorded in the checkpoint so the columns a
    network was trained on are the ones it is given back.

    Two of v3's three actor changes live here, each a flag so an ablation can
    isolate it. (The third, centred logits, is the policy's: see
    VelocityFlowV3Policy.)

    * carry_velocity: the read points include the player's own momentum. At
      the terminal 2.5px a frame the coast alone is 10px, as far as v2's whole
      action moves it, so from rest the controller read each action's danger
      at the point the action to one side of it actually lands on.
    * graduated_holds: each slice holds its button for SLICE_HOLDS frames, so
      the six slices read six distances instead of four at one.

    With both off, the actor is v2's.
    """

    def __init__(self, observation_space, action_repeat=4, critic_layout=None,
                 carry_velocity=True, graduated_holds=True,
                 critic_groups=V3_CRITIC_GROUPS):
        # The layout the checkpoint was trained with; None for a fresh network.
        # Same width with a different meaning would otherwise load silently.
        if critic_layout is not None and critic_layout != current_layout():
            raise ValueError(
                "this checkpoint was trained with a different critic feature "
                f"layout ({critic_layout}) than the engine now produces "
                f"({current_layout()})")
        if observation_space.shape != (WIDE_OBSERVATION_SIZE,):
            raise ValueError(
                f"VelocityFlowV3Extractor expects {WIDE_OBSERVATION_SIZE} values "
                f"(the actor's and the critic's), got {observation_space.shape}; "
                "build the environment with critic_features=True")
        unknown = [name for name in critic_groups if name not in CRITIC_GROUPS]
        if unknown:
            raise ValueError(f"no critic group {', '.join(unknown)}; the engine "
                             f"produces {', '.join(CRITIC_GROUPS)}")
        super().__init__(observation_space, action_repeat=action_repeat,
                         half_resolution=True)
        self.critic_groups = list(critic_groups)
        columns = [column for name in critic_groups
                   for column in range(*CRITIC_GROUPS[name])]
        self.critic_width = len(columns)
        self.register_buffer(
            "critic_index",
            torch.as_tensor(columns, dtype=torch.long) + OBSERVATION_SIZE,
            persistent=False)
        first = self.value_net[0]
        wide = nn.Linear(first.in_features + self.critic_width, first.out_features)
        with torch.no_grad():
            # Zero, so the critic starts out computing exactly what v2's does
            # and learns to use the new columns rather than being thrown by them.
            wide.weight.zero_()
            wide.weight[:, :first.in_features] = first.weight
            wide.bias.copy_(first.bias)
        self.value_net[0] = wide

        self.carry_velocity = carry_velocity
        self.graduated_holds = graduated_holds
        if graduated_holds:
            # offsets is a persistent buffer, so a checkpoint carries whichever
            # set it was trained with.
            with torch.no_grad():
                self.offsets.copy_(torch.as_tensor(action_offsets(holds=SLICE_HOLDS)))
        if carry_velocity:
            # Not persistent: a derived constant, like the trunk maps.
            self.register_buffer("carry", torch.as_tensor(velocity_carry()),
                                 persistent=False)

    def _read_points(self, coordinates):
        points = super()._read_points(coordinates)
        if not self.carry_velocity:
            return points
        velocity = coordinates[:, 2:4] * VELOCITY_SCALE
        return points + velocity[:, None, None, :] * self.carry[None, None, :, None]

    def _danger_and_value_inputs(self, observations):
        danger, inputs = super()._danger_and_value_inputs(
            observations[:, :OBSERVATION_SIZE])
        critic = observations.index_select(1, self.critic_index)
        return danger, torch.cat((inputs, critic), dim=1)

    def field(self, observations):
        return super().field(observations[:, :OBSERVATION_SIZE])


def _axis_map(operation, size):
    """The matrix of a linear map along one axis, (out, size), in float64.

    operation takes (1, channels, size) and returns (1, channels, out); fed
    the identity, channel c comes back as the image of the c-th unit vector.
    """
    return operation(torch.eye(size, dtype=torch.float64)[None])[0].T


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


class _CentredFieldLogits(nn.Module):
    """v3's logits: centred danger over one scale shared by every state.

    _FieldLogits divides each state by its own spread, which pins every
    state's logits to unit variance -- an empty arena included. The most
    uniform policy that constraint allows puts one action 2.67 sigma below the
    other eight, so a calm state could only look calm by sacrificing an
    action, and v2 trained one diagonal into exactly that role. Here a flat
    field gives flat logits, and the field's own contrast is the per-state
    temperature.

    The scale is still taken out, so a field that drifts large does not
    silently sharpen the policy, but as one number: the RMS of the centred
    danger across the states of past updates. It holds still for a whole
    update -- the rollout's log-probabilities and every epoch's are taken
    under one scale, or the ratio starts off 1 and approx_kl measures the
    scale rather than the policy. observe() gathers what the update sees and
    commit() folds it in once the update is over.
    """

    EPSILON = 1e-3
    # Weight of one update's statistics in the running mean square.
    MOMENTUM = 0.2

    def __init__(self, temperature=INITIAL_TEMPERATURE, cap=LOGIT_CAP):
        super().__init__()
        self.cap = cap
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))
        # Persistent: a reloaded policy must read its field at the scale it
        # was trained at.
        self.register_buffer("rms", torch.ones(()))
        self.register_buffer("rms_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("pending_sum", torch.zeros(()), persistent=False)
        # Host-side, so neither check below costs a device sync.
        self._pending_count = 0
        self._ready = None

    @staticmethod
    def _centred(latent):
        danger = latent[:, :N_ACTIONS]
        return danger - danger.mean(dim=1, keepdim=True)

    def forward(self, latent):
        scaled = (self._centred(latent) / self.rms.clamp_min(self.EPSILON)
                  * self.log_temperature.exp())
        return self.cap * torch.tanh(scaled / self.cap)

    def ready(self):
        if self._ready is None:
            self._ready = bool(self.rms_ready)
        return self._ready

    def initialise(self, latent):
        """Take the scale from one batch: the first rollout's first step."""
        with torch.no_grad():
            self.rms.copy_(self._centred(latent).square().mean().sqrt())
            self.rms_ready.fill_(True)
        self._ready = True

    def observe(self, latent):
        with torch.no_grad():
            self.pending_sum += self._centred(latent).square().mean(dim=1).sum()
        self._pending_count += latent.shape[0]

    def pending(self):
        return self._pending_count > 0

    def commit(self):
        if not self._pending_count:
            return
        with torch.no_grad():
            seen = self.pending_sum / self._pending_count
            if self.ready():
                seen = torch.lerp(self.rms.square(), seen, self.MOMENTUM)
            self.rms.copy_(seen.sqrt())
            self.rms_ready.fill_(True)
            self.pending_sum.zero_()
        self._pending_count = 0
        self._ready = True

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._ready = None
        self._pending_count = 0
        self.pending_sum.zero_()


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
        self.action_net = self._field_logits()
        # The optimizer was built over the head that was just replaced.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        speed.configure_cudnn()
        self._graphs = speed.GraphCache()

    def _field_logits(self):
        return _FieldLogits()

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


class VelocityFlowV3Policy(VelocityFlowPolicy):
    """The v2 policy with v3's centred logits, taking wide or narrow observations.

    Evaluation and the watch tools hold observations without the critic's
    columns, so they are padded with zeros here, where SB3 checks the shape.

    With centred_logits=False the head is v2's _FieldLogits, so an ablation
    that keeps only the extractor's changes trains everything else as v2's
    head does. With it on, the entropy bonus is computed from the logits of a
    detached field: the same number, but its gradient reaches the sharpness
    alone. The bonus then sets how sharp the policy is without rewarding the
    field for flattening the danger it reports -- which, with nothing
    standardising the field per state any more, would be the cheap way to
    raise entropy.
    """

    def __init__(self, *args, centred_logits=True, **kwargs):
        # Before the base constructor, which calls _build.
        self.centred_logits = centred_logits
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # SB3 re-initialises every Linear in the extractor once it is built,
        # which undoes the zeros the extractor started the new columns with.
        width = self.features_extractor.critic_width
        if width:
            with torch.no_grad():
                self.features_extractor.value_net[0].weight[:, -width:] = 0

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(),
                "centred_logits": self.centred_logits}

    def _field_logits(self):
        return _CentredFieldLogits() if self.centred_logits else _FieldLogits()

    def obs_to_tensor(self, observation):
        if (isinstance(observation, np.ndarray)
                and observation.shape[-1:] == (OBSERVATION_SIZE,)):
            pad = [(0, 0)] * (observation.ndim - 1) + [(0, CRITIC_FEATURE_COUNT)]
            observation = np.pad(observation, pad)
        return super().obs_to_tensor(observation)

    def end_update(self):
        """Fold the update's statistics into the logit scale.

        DashboardPPO calls this once train() returns, so the checkpoint saved
        after an update and the rollout that follows it read the same scale.
        """
        if self.centred_logits:
            self.action_net.commit()

    def forward(self, obs, deterministic=False):
        if self.centred_logits:
            head = self.action_net
            # A plain PPO never calls end_update; catch its update up here,
            # before this rollout's first log-probability is taken.
            if head.pending():
                head.commit()
            if not head.ready():
                with torch.no_grad():
                    head.initialise(self.mlp_extractor.forward_actor(
                        self.extract_features(obs)))
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        if not self.centred_logits:
            return super().evaluate_actions(obs, actions)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        head = self.action_net
        head.observe(latent_pi)
        log_pmf = head(latent_pi)
        log_pmf = log_pmf - log_pmf.logsumexp(dim=-1, keepdim=True)
        log_prob = self._log_prob(log_pmf, actions)
        bonus = head(latent_pi.detach())
        bonus = bonus - bonus.logsumexp(dim=-1, keepdim=True)
        clamped = torch.clamp(bonus, min=torch.finfo(bonus.dtype).min)
        entropy = -(clamped * F.softmax(bonus, dim=-1)).sum(-1)
        return values, log_prob, entropy
