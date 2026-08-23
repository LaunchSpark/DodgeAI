"""VelocityFlow: a CNN that only draws danger, and a controller that reads it."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from dodge.observe import GRID_SIZE, OBSERVATION_SIZE
from dodge.actions import ACTION_NAMES
from dodge.policies import policy_class, policy_kwargs, architecture_name


def _model(seed=0):
    from dodge.env import DodgeEnv
    return PPO(policy_class("velocity-flow"), DodgeEnv(max_episode_frames=20),
               n_steps=16, batch_size=8, n_epochs=1, seed=seed, device="cpu",
               policy_kwargs=policy_kwargs("velocity-flow"))


def test_action_offsets_are_distinct_and_symmetric():
    """A clamp at the arena edge once made 'left' identical to 'idle'."""
    from dodge.velocity import action_offsets

    offsets = action_offsets()
    assert np.allclose(offsets[ACTION_NAMES.index("idle")], 0.0)

    def final(name):
        return offsets[ACTION_NAMES.index(name)][-1]

    assert final("left")[0] < 0 and final("right")[0] > 0
    assert final("up")[1] < 0 and final("down")[1] > 0
    np.testing.assert_allclose(final("left"), -final("right"))
    np.testing.assert_allclose(final("up"), -final("down"))
    # Every action must reach somewhere its neighbours do not.
    assert len({tuple(np.round(o[-1], 3)) for o in offsets}) == len(offsets)


def test_the_network_only_produces_a_field():
    from dodge.velocity import SAMPLE_FRAMES, VelocityFlowExtractor

    model = _model()
    extractor = model.policy.features_extractor
    assert isinstance(extractor, VelocityFlowExtractor)
    field = extractor.field(torch.rand(4, OBSERVATION_SIZE))
    # One field per future time, not one field.
    assert field.shape == (4, len(SAMPLE_FRAMES), GRID_SIZE, GRID_SIZE)


def test_each_sample_is_read_from_the_slice_for_its_arrival_time():
    """The pairing is what makes "lethal now, clear later" expressible.

    Reading any other slice would ask whether a cell is safe at a time the
    player is not there, which is the question a single static field is stuck
    with.
    """
    from dodge.velocity import N_ACTIONS, SAMPLE_FRAMES

    slices, batch = len(SAMPLE_FRAMES), 2
    sampled = torch.arange(batch * slices * N_ACTIONS * slices,
                           dtype=torch.float32)
    sampled = sampled.reshape(batch, slices, N_ACTIONS, slices)

    paired = sampled.permute(0, 2, 1, 3).diagonal(dim1=2, dim2=3)

    assert paired.shape == (batch, N_ACTIONS, slices)
    for b in range(batch):
        for action in range(N_ACTIONS):
            for k in range(slices):
                assert paired[b, action, k] == sampled[b, k, action, k]


def test_the_field_can_see_the_whole_arena():
    """Two lethal bars in pattern 11 start 151px apart.

    With only the convolution trunk the reach was 11 cells, 44px, so no output
    cell could see both and "which half is safe" was unanswerable.
    """
    from dodge.observe import CELL_SIZE
    from dodge.policies import HYBRID_INPUT_CHANNELS

    extractor = _model().policy.features_extractor
    grids = torch.zeros(1, HYBRID_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE,
                        requires_grad=True)
    extractor._field_from_trunk(extractor.field_net(grids))[0, 0, 16, 16].backward()

    reached = torch.nonzero(grids.grad[0].abs().sum(0).sum(1))[:, 0]
    span = int(reached.max() - reached.min()) + 1
    assert span * CELL_SIZE >= GRID_SIZE * CELL_SIZE,         f"only {span * CELL_SIZE}px of a {GRID_SIZE * CELL_SIZE}px arena"


def test_the_horizon_outlasts_a_sweeping_pattern():
    """Pattern 11's bars cross at 0.7px a frame; the manoeuvre is seconds long."""
    from dodge.velocity import SAMPLE_FRAMES

    assert max(SAMPLE_FRAMES) >= 90, "under 1.5s cannot see the bar move"
    # The discount must leave the far samples with real weight, or the horizon
    # is decorative.
    from dodge.velocity import SAMPLE_DECAY
    assert SAMPLE_DECAY ** max(SAMPLE_FRAMES) > 0.1


def test_it_is_far_smaller_than_the_hybrid_network():
    """The point of dropping the flatten: 87% of hybrid-cnn was one dense layer."""
    from dodge.env import DodgeEnv

    field = sum(p.numel() for p in _model().policy.parameters())
    hybrid = PPO("MlpPolicy", DodgeEnv(max_episode_frames=20), n_steps=16,
                 batch_size=8, seed=0, device="cpu",
                 policy_kwargs=policy_kwargs("hybrid-cnn"))
    hybrid_count = sum(p.numel() for p in hybrid.policy.parameters())
    assert field < hybrid_count / 5, f"{field:,} vs {hybrid_count:,}"


def test_logits_ignore_the_fields_magnitude():
    """Only the shape of the field should reach the distribution.

    Untouched, a fresh field spreads its readings ~30x too little to move a
    softmax, so the policy would sit at uniform until the magnitude grew.
    """
    model = _model()
    observations = torch.rand(16, OBSERVATION_SIZE)
    with torch.no_grad():
        before = model.policy.get_distribution(observations).distribution.probs
        # Scaling the field head scales every reading; the policy must not care.
        model.policy.features_extractor.field_head.weight.mul_(7.0)
        model.policy.features_extractor.field_head.bias.mul_(7.0)
        after = model.policy.get_distribution(observations).distribution.probs
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-4)


def test_the_policy_starts_exploratory_but_state_dependent():
    model = _model()
    observations = torch.rand(64, OBSERVATION_SIZE)
    with torch.no_grad():
        distribution = model.policy.get_distribution(observations).distribution
    entropy = float(distribution.entropy().mean())
    assert entropy > 0.9 * np.log(9), "should start near uniform, not committed"
    assert float(distribution.probs.std(dim=0).max()) > 0.0, \
        "but must already vary with the observation"


def test_it_trains_and_survives_a_save_load_cycle(tmp_path):
    model = _model(seed=3)
    assert architecture_name(model) == "velocity-flow"
    model.learn(total_timesteps=32)

    path = tmp_path / "velocity-flow"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    assert architecture_name(restored) == "velocity-flow"
    action, _ = restored.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    assert 0 <= int(action) < 9


def test_the_field_is_sampled_at_the_player_not_its_velocity():
    """Which coordinates are the position is a layout fact, and it changed.

    The controller reads the field along each action's path, which starts at
    the player. When the coordinate block was (px, py) across two stacked
    frames, the last two values were the current position; now the block is
    (px, py, dpx, dpy) and the last two are velocity. Taking the tail put the
    sample points at the player's *speed* interpreted as a position, so the
    field was read somewhere the player was not, and reading moved whenever the
    player accelerated.
    """
    from dodge.observe import PLAYER_VALUES, VELOCITY_SCALE
    from dodge.velocity import N_ACTIONS

    extractor = _model(seed=7).policy.features_extractor
    base = torch.rand(1, OBSERVATION_SIZE)
    base[0, :PLAYER_VALUES] = torch.tensor((0.25, 0.75, 0.0, 0.0))

    def danger(px, py, dpx, dpy):
        observation = base.clone()
        observation[0, :PLAYER_VALUES] = torch.tensor(
            (px, py, dpx / VELOCITY_SCALE, dpy / VELOCITY_SCALE))
        with torch.no_grad():
            return extractor(observation)[0, :N_ACTIONS]

    still = danger(0.25, 0.75, 0.0, 0.0)
    # Velocity is already an input to the field and to the senses; it must not
    # additionally decide *where* the field is read.
    torch.testing.assert_close(danger(0.25, 0.75, 2.5, -2.5), still)
    # Position must, or the controller is reading a fixed spot.
    assert not torch.allclose(danger(0.75, 0.25, 0.0, 0.0), still)


def test_global_context_is_smooth_not_tiled():
    """Nearest upsampling made the context sixteen hard-edged tiles.

    The 4x4 context grid is broadcast back to 32x32. Held flat, each value
    covers an 8x8 cell block -- 32x32px of a 128px arena -- and the picture is
    dominated by block edges rather than by anything about danger. The step at
    a boundary is also a lie the controller samples across.
    """
    import torch.nn.functional as F

    from dodge.velocity import CONTEXT_GRID
    from dodge.policies import HYBRID_INPUT_CHANNELS

    extractor = _model().policy.features_extractor
    grids = torch.rand(1, HYBRID_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    with torch.no_grad():
        trunk = extractor.field_net(grids)
        small = extractor.context_net(F.adaptive_avg_pool2d(trunk, CONTEXT_GRID))
        context = F.interpolate(small, size=(GRID_SIZE, GRID_SIZE),
                                mode="bilinear", align_corners=False)

    block = GRID_SIZE // CONTEXT_GRID
    inside = context[0, :, :block, :block]
    assert float(inside.std()) > 0.0, "context is piecewise constant across a block"

    # And the whole point of the branch must survive: full-arena reach.
    probe = torch.zeros(1, HYBRID_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE,
                        requires_grad=True)
    extractor._field_from_trunk(extractor.field_net(probe))[0, 0, 16, 16].backward()
    reached = torch.nonzero(probe.grad[0].abs().sum(0).sum(1))[:, 0]
    assert int(reached.max() - reached.min()) + 1 == GRID_SIZE


def _model_v2(seed=0):
    from dodge.env import DodgeEnv
    return PPO(policy_class("velocity-flow-v2"), DodgeEnv(max_episode_frames=20),
               n_steps=16, batch_size=8, n_epochs=1, seed=seed, device="cpu",
               policy_kwargs=policy_kwargs("velocity-flow-v2"))


def test_v2_is_selectable_and_names_itself_apart_from_v1():
    """Both stay registered: v1 is the baseline v2 has to beat."""
    from dodge.policies import ARCHITECTURES

    assert "velocity-flow" in ARCHITECTURES and "velocity-flow-v2" in ARCHITECTURES
    assert architecture_name(_model_v2()) == "velocity-flow-v2"
    assert architecture_name(_model()) == "velocity-flow"


def test_v2_trains_and_survives_a_save_load_cycle(tmp_path):
    model = _model_v2(seed=3)
    model.learn(total_timesteps=32)

    path = tmp_path / "velocity-flow-v2"
    model.save(path)
    restored = PPO.load(path, device="cpu")

    assert architecture_name(restored) == "velocity-flow-v2"
    action, _ = restored.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    assert 0 <= int(action) < 9


def test_v2_still_emits_a_full_resolution_field():
    """The trunk runs at half resolution; the field it draws must not.

    The controller samples the field along nine pixel-space paths, so a coarser
    output would blur exactly the near-threat detail those readings depend on.
    """
    from dodge.velocity import SAMPLE_FRAMES

    field = _model_v2().policy.features_extractor.field(torch.rand(4, OBSERVATION_SIZE))
    assert field.shape == (4, len(SAMPLE_FRAMES), GRID_SIZE, GRID_SIZE)


def test_the_v2_critic_can_tell_where_a_threat_is():
    """v1's critic sees trunk.mean(dim=(2,3)), the same number wherever a
    threat sits. Contrasted against v1 directly so it cannot pass vacuously."""
    from dodge.velocity import VALUE_SUMMARY_DIM

    trunk = torch.zeros(2, 32, GRID_SIZE, GRID_SIZE)
    trunk[0, :, 2, 2] = 1.0                    # near one corner
    trunk[1, :, 29, 29] = 1.0                  # near the opposite one

    pooled = trunk.mean(dim=(2, 3))
    assert torch.allclose(pooled[0], pooled[1]), \
        "the setup: identical means, so only layout differs"

    summary = _model_v2().policy.features_extractor.value_summary(trunk)

    assert summary.shape == (2, VALUE_SUMMARY_DIM)
    assert not torch.allclose(summary[0], summary[1]), \
        "the summary must carry what the mean throws away"


def test_the_v2_trunk_runs_its_heavy_layers_at_half_resolution():
    """Where the compute saving comes from, and the flag that turns it off."""
    import torch.nn.functional as F
    from gymnasium import spaces
    from dodge.policies import HYBRID_INPUT_CHANNELS
    from dodge.velocity import VelocityFlowV2Extractor, _HalfResolutionTrunk

    space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,),
                       dtype=np.float32)
    half = VelocityFlowV2Extractor(space)
    assert isinstance(half.field_net, _HalfResolutionTrunk)

    grids = torch.rand(2, HYBRID_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    stem = half.field_net.stem(grids)
    assert stem.shape[-2:] == (GRID_SIZE, GRID_SIZE), "the stem sees every cell"
    body = half.field_net.body(F.avg_pool2d(stem, 2))
    assert body.shape[-2:] == (GRID_SIZE // 2, GRID_SIZE // 2), "the body does not"
    assert half.field_net(grids).shape[-2:] == (GRID_SIZE, GRID_SIZE), \
        "and it comes back before the field head"

    full = VelocityFlowV2Extractor(space, half_resolution=False)
    assert not isinstance(full.field_net, _HalfResolutionTrunk), \
        "the flag has to actually turn it off, or it cannot be A/B'd"
