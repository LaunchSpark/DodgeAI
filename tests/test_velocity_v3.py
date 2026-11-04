"""velocity-flow-v3: v2's trunk read where the player goes, and a critic that
also reads hidden state. Its three actor changes can each be turned off."""

import math

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces

from dodge.actions import ACTION_MASKS
from dodge.critic import (CRITIC_ARCHITECTURES, CRITIC_FEATURE_COUNT, CRITIC_FEATURE_VERSION,
                          CRITIC_GROUPS, WIDE_OBSERVATION_SIZE)
from dodge.gradient import simulate_player
from dodge.observe import ARENA_SIZE, OBSERVATION_SIZE, VELOCITY_SCALE
from dodge.policies import (ARCHITECTURES, EXTRACTOR_ARCHITECTURE,
                            architecture_from_kwargs, policy_class, policy_kwargs)
from dodge.training import GPU_ARCHITECTURES, hyperparameters
from dodge.velocity import (LOGIT_CAP, SAMPLE_FRAMES, SLICE_HOLDS, V3_CRITIC_GROUPS,
                            VelocityFlowPolicy, VelocityFlowV2Extractor,
                            VelocityFlowV3Extractor, VelocityFlowV3Policy,
                            action_offsets, velocity_carry)

pytestmark = pytest.mark.skipif(CRITIC_FEATURE_COUNT == 0,
                                reason="the native engine has no critic features")

SPACE = spaces.Box(-1, 1, (WIDE_OBSERVATION_SIZE,), np.float32)
RIGHT, DOWN_RIGHT = ACTION_MASKS.index(2), ACTION_MASKS.index(2 | 8)


def _policy(critic_groups=None, **flags):
    kwargs = policy_kwargs("velocity-flow-v3", **flags)
    if critic_groups is not None:
        kwargs["features_extractor_kwargs"]["critic_groups"] = list(critic_groups)
    return VelocityFlowV3Policy(SPACE, spaces.Discrete(9), lambda _: 3e-4, **kwargs)


def _observations(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, WIDE_OBSERVATION_SIZE, generator=generator) * 2 - 1


def _coordinates(px, py, dpx, dpy):
    return torch.tensor([[px / (ARENA_SIZE - 1), py / (ARENA_SIZE - 1),
                          dpx / VELOCITY_SCALE, dpy / VELOCITY_SCALE]],
                        dtype=torch.float32)


# --- read points ------------------------------------------------------------

def test_read_points_land_where_the_player_does():
    """Each slice's point is where simulate_player puts the player at that
    frame -- carried velocity included -- holding the slice's own hold."""
    extractor = VelocityFlowV3Extractor(SPACE)
    rng = np.random.default_rng(0)
    for _ in range(20):
        # Far enough in that no path reaches a wall: 30px of travel and 10px
        # of coast at most, per axis.
        px, py = rng.uniform(45, 85, 2)
        dpx, dpy = rng.uniform(-2.5, 2.5, 2)
        points = extractor._read_points(_coordinates(px, py, dpx, dpy))[0]
        for action, mask in enumerate(ACTION_MASKS):
            for sample, (frame, held) in enumerate(zip(SAMPLE_FRAMES, SLICE_HOLDS)):
                path = simulate_player(px, py, dpx, dpy, mask, frame, held=held)
                np.testing.assert_allclose(points[action, sample].numpy(),
                                           path[frame - 1], atol=1e-3)


def test_without_carry_the_points_ignore_velocity():
    extractor = VelocityFlowV3Extractor(SPACE, carry_velocity=False)
    moving = extractor._read_points(_coordinates(64, 64, 2.5, -2.5))
    still = extractor._read_points(_coordinates(64, 64, 0.0, 0.0))
    assert torch.equal(moving, still)


def test_velocity_carry_is_the_coast_of_a_unit_velocity():
    expected = [4 * (1 - 0.8 ** frame) for frame in SAMPLE_FRAMES]
    np.testing.assert_allclose(velocity_carry(), expected, rtol=1e-5)


def test_graduated_holds_reach_further_with_each_slice():
    offsets = action_offsets(holds=SLICE_HOLDS)
    right = offsets[RIGHT, :, 0]
    diagonal = np.hypot(*offsets[DOWN_RIGHT].T)
    assert np.all(np.diff(right) > 3.0), right
    assert right[-1] == pytest.approx(30.0, abs=0.1)
    assert diagonal[-1] == pytest.approx(30.0 * math.sqrt(2), abs=0.2)


def test_without_graduated_holds_the_offsets_are_v2s():
    extractor = VelocityFlowV3Extractor(SPACE, graduated_holds=False)
    np.testing.assert_array_equal(extractor.offsets.numpy(), action_offsets(4))


# --- centred logits -----------------------------------------------------------

def _latent(danger):
    danger = torch.as_tensor(danger, dtype=torch.float32).reshape(1, -1)
    return torch.cat((danger, torch.zeros(1, 64)), dim=1)


def _entropy(logits):
    logits = logits.detach()
    return float(-(torch.log_softmax(logits, -1) * torch.softmax(logits, -1)).sum())


def test_a_calm_state_stays_near_uniform():
    """Flat danger means flat logits. v2's per-state std blows the same
    0.01 ripple up to its full spread."""
    ripple = _latent(0.01 * torch.randn(9, generator=torch.Generator().manual_seed(1)))
    centred = _policy().action_net
    centred.rms.fill_(1.0)
    assert _entropy(centred(ripple)) > math.log(9) - 0.001
    standardised = _policy(centred_logits=False).action_net
    assert _entropy(standardised(ripple)) < math.log(9) - 0.05


def test_logits_are_capped():
    logits = _policy().action_net(_latent(torch.linspace(-1e3, 1e3, 9)))
    assert logits.abs().max() <= LOGIT_CAP
    assert logits.abs().max() > LOGIT_CAP - 0.01


def _centred_rms(policy, observations):
    with torch.no_grad():
        danger = policy.extract_features(observations)[:, :9]
    centred = danger - danger.mean(dim=1, keepdim=True)
    return centred.square().mean()


def test_the_first_rollout_sets_the_scale_from_its_own_batch():
    policy = _policy()
    policy.set_training_mode(False)
    observations = _observations(64)
    with torch.no_grad():
        policy(observations)
    assert float(policy.action_net.rms) == pytest.approx(
        float(_centred_rms(policy, observations).sqrt()), rel=1e-5)


def test_the_scale_holds_still_through_every_epoch():
    """Rollout log-probs are taken under one scale; if the epochs that follow
    read another, the ratio starts away from 1 and approx_kl is not about the
    policy at all."""
    policy = _policy()
    policy.set_training_mode(False)
    rollout = _observations(64)
    with torch.no_grad():
        actions, _, old_log_prob = policy(rollout)
    scale = policy.action_net.rms.clone()

    policy.set_training_mode(True)
    other = _observations(64, seed=1) * 0.5
    for _ in range(4):
        policy.evaluate_actions(other, actions)
        _, log_prob, _ = policy.evaluate_actions(rollout, actions)
        torch.testing.assert_close(log_prob, old_log_prob)
        assert torch.equal(policy.action_net.rms, scale)

    # Committed once, after the update, from everything the epochs saw.
    seen = (_centred_rms(policy, other) + _centred_rms(policy, rollout)) / 2
    momentum = policy.action_net.MOMENTUM
    expected = ((1 - momentum) * scale.square() + momentum * seen).sqrt()
    policy.end_update()
    torch.testing.assert_close(policy.action_net.rms, expected)
    policy.end_update()  # nothing new seen: nothing changes
    torch.testing.assert_close(policy.action_net.rms, expected)


def test_a_rollout_commits_what_an_unclosed_update_saw():
    """Plain PPO never calls end_update; the next rollout catches it up."""
    policy = _policy()
    policy.set_training_mode(False)
    rollout = _observations(32)
    with torch.no_grad():
        actions, _, _ = policy(rollout)
    scale = policy.action_net.rms.clone()
    policy.set_training_mode(True)
    policy.evaluate_actions(_observations(32, seed=2) * 0.25, actions)
    policy.set_training_mode(False)
    with torch.no_grad():
        policy(rollout)
    assert not torch.equal(policy.action_net.rms, scale)


def test_the_entropy_bonus_reaches_only_the_sharpness():
    policy = _policy()
    observations = _observations(16)
    with torch.no_grad():
        actions, _, _ = policy(observations)
    _, log_prob, entropy = policy.evaluate_actions(observations, actions)
    entropy.sum().backward()
    head = policy.features_extractor.field_head.weight.grad
    assert head is None or torch.count_nonzero(head) == 0
    assert policy.action_net.log_temperature.grad.abs() > 0

    policy.zero_grad()
    _, log_prob, _ = policy.evaluate_actions(observations, actions)
    log_prob.sum().backward()
    assert torch.count_nonzero(policy.features_extractor.field_head.weight.grad) > 0


# --- flags and ablations --------------------------------------------------------

def test_all_three_changes_are_on_by_default():
    kwargs = policy_kwargs("velocity-flow-v3")
    assert kwargs["centred_logits"] is True
    extractor = kwargs["features_extractor_kwargs"]
    assert extractor["carry_velocity"] is True and extractor["graduated_holds"] is True


NARROW = spaces.Box(-1, 1, (OBSERVATION_SIZE,), np.float32)


def _v2_and_v3(**flags):
    """A v2 policy and a v3 one holding the same weights; v3's critic columns zero."""
    v2 = VelocityFlowPolicy(NARROW, spaces.Discrete(9), lambda _: 3e-4,
                            **policy_kwargs("velocity-flow-v2"))
    v3 = _policy(**flags)
    state = v3.state_dict()
    for name, tensor in v2.state_dict().items():
        if state[name].shape == tensor.shape:
            state[name] = tensor
        else:  # value_net[0]: v2's columns lead, the critic's follow
            state[name][:, :tensor.shape[1]] = tensor
    v3.load_state_dict(state)
    return v2, v3


def test_with_every_change_off_v3_is_v2():
    v2, v3 = _v2_and_v3(carry_velocity=False, graduated_holds=False, centred_logits=False)
    observations = _observations(16)
    with torch.no_grad():
        expected = v2._distribution_inputs(observations[:, :OBSERVATION_SIZE])
        actual = v3._distribution_inputs(observations)
    for want, got in zip(expected, actual):
        torch.testing.assert_close(got, want)


def test_velocity_only_differs_from_v2_only_when_moving():
    v2, v3 = _v2_and_v3(graduated_holds=False, centred_logits=False)
    observations = _observations(16)
    still = observations.clone()
    still[:, 2:4] = 0
    with torch.no_grad():
        torch.testing.assert_close(v3._distribution_inputs(still)[0],
                                   v2._distribution_inputs(still[:, :OBSERVATION_SIZE])[0])
        assert not torch.allclose(v3._distribution_inputs(observations)[0],
                                  v2._distribution_inputs(observations[:, :OBSERVATION_SIZE])[0])


# --- the critic's columns ------------------------------------------------------

DROPPED = ("next_pattern", "next_enemy")


def _columns(groups):
    return [OBSERVATION_SIZE + column for name in groups
            for column in range(*CRITIC_GROUPS[name])]


def test_the_critic_reads_every_group_but_the_two_the_probe_found_useless():
    """next_pattern (+0.001) and next_enemy (+0.0000) explained variance."""
    assert set(V3_CRITIC_GROUPS) == set(CRITIC_GROUPS) - set(DROPPED)
    kwargs = policy_kwargs("velocity-flow-v3")["features_extractor_kwargs"]
    assert kwargs["critic_groups"] == list(V3_CRITIC_GROUPS)
    policy = _policy()
    width = sum(end - start for name, (start, end) in CRITIC_GROUPS.items()
                if name not in DROPPED)
    assert policy.features_extractor.critic_width == width
    v2_width = VelocityFlowV2Extractor(NARROW).value_net[0].in_features
    assert policy.features_extractor.value_net[0].in_features == v2_width + width
    # Zero, so the critic starts out as v2's and learns to use the new columns.
    assert torch.count_nonzero(policy.features_extractor.value_net[0].weight[:, -width:]) == 0


def test_the_dropped_columns_reach_nothing():
    policy = _policy()
    with torch.no_grad():  # make the zero-initialised critic columns live
        policy.features_extractor.value_net[0].weight[:, -1:].fill_(1.0)
        policy.features_extractor.value_net[0].weight[:, -60:].normal_()
    observations = _observations(8)
    dropped, kept = observations.clone(), observations.clone()
    dropped[:, _columns(DROPPED)] = 0.123
    kept[:, _columns(["nearest_enemies"])] = 0.123
    with torch.no_grad():
        base = policy._distribution_inputs(observations)
        torch.testing.assert_close(policy._distribution_inputs(dropped)[1], base[1])
        assert not torch.allclose(policy._distribution_inputs(kept)[1], base[1])


def test_an_unknown_critic_group_is_refused():
    with pytest.raises(ValueError, match="no critic group"):
        VelocityFlowV3Extractor(SPACE, critic_groups=["clock", "next_boss"])


def test_v3_is_registered():
    assert ARCHITECTURES[0] == "velocity-flow" and "velocity-flow-v3" in ARCHITECTURES
    assert "velocity-flow-v3" in GPU_ARCHITECTURES
    assert CRITIC_ARCHITECTURES == ("velocity-flow-v3",)
    assert hyperparameters("velocity-flow-v3") == hyperparameters("velocity-flow-v2")
    assert policy_class("velocity-flow-v3") is VelocityFlowV3Policy
    assert architecture_from_kwargs(policy_kwargs("velocity-flow-v3")) == "velocity-flow-v3"
    assert EXTRACTOR_ARCHITECTURE["VelocityFlowV3Extractor"] == "velocity-flow-v3"


def test_unknown_flags_are_refused():
    with pytest.raises(TypeError):
        policy_kwargs("velocity-flow-v3", carry_velocty=False)
    with pytest.raises(TypeError):
        policy_kwargs("velocity-flow-v2", carry_velocity=False)


def _model(**flags):
    from dodge.env import DodgeVecEnv
    from dodge.training import DashboardPPO
    env = DodgeVecEnv(n_envs=2, max_episode_frames=200, critic_features=True)
    model = DashboardPPO(policy_class("velocity-flow-v3"), env, n_steps=32,
                         batch_size=32, n_epochs=2, seed=0, device="cpu",
                         policy_kwargs=policy_kwargs("velocity-flow-v3", **flags))
    return model, env


def test_a_saved_ablation_reloads_as_itself(tmp_path):
    from dodge.training import DashboardPPO
    model, env = _model(graduated_holds=False, centred_logits=False)
    env.close()
    model.save(tmp_path / "v3.zip")
    loaded = DashboardPPO.load(tmp_path / "v3.zip", device="cpu")
    extractor = loaded.policy.features_extractor
    assert extractor.carry_velocity and not extractor.graduated_holds
    assert not loaded.policy.centred_logits


def test_training_commits_the_scale_once_per_update_and_saves_it(tmp_path):
    from dodge.training import DashboardPPO
    model, env = _model()
    try:
        commits = []
        end_update = model.policy.end_update
        model.policy.end_update = lambda: (commits.append(1), end_update())
        model.learn(total_timesteps=128)  # two updates of 2 x 32
        assert len(commits) == 2
        model.save(tmp_path / "v3.zip")
    finally:
        env.close()
    loaded = DashboardPPO.load(tmp_path / "v3.zip", device="cpu")
    assert torch.equal(loaded.policy.action_net.rms, model.policy.action_net.rms)
    assert float(loaded.policy.action_net.rms) not in (0.0, 1.0)


# --- wide and narrow observations, and the critic layout --------------------------

def test_actor_ignores_critic_columns():
    policy = _policy()
    width = policy.features_extractor.critic_width
    with torch.no_grad():  # make the zero-initialised critic columns live
        policy.features_extractor.value_net[0].weight[:, -width:].normal_()
    obs = _observations(16)
    other = obs.clone()
    other[:, OBSERVATION_SIZE:] = torch.rand(16, CRITIC_FEATURE_COUNT)
    a, va = policy._distribution_inputs(obs)
    b, vb = policy._distribution_inputs(other)
    assert torch.equal(a, b) and not torch.equal(va, vb)


def test_field_accepts_wide_and_narrow_observations():
    extractor = _policy().features_extractor
    narrow = torch.rand(2, OBSERVATION_SIZE) * 2 - 1
    wide = torch.cat((narrow, torch.rand(2, CRITIC_FEATURE_COUNT)), dim=1)
    assert torch.equal(extractor.field(narrow), extractor.field(wide))


def test_predict_accepts_actor_only_observations():
    policy = _policy()
    narrow = np.random.default_rng(0).uniform(-1, 1, (4, OBSERVATION_SIZE)).astype(np.float32)
    wide = np.concatenate([narrow, np.zeros((4, CRITIC_FEATURE_COUNT), np.float32)], 1)
    assert np.array_equal(policy.predict(narrow, deterministic=True)[0],
                          policy.predict(wide, deterministic=True)[0])
    one = policy.predict(narrow[0], deterministic=True)[0]
    assert np.array_equal(one, policy.predict(wide[0], deterministic=True)[0])


def test_v2_forward_equals_its_helper():
    extractor = VelocityFlowV2Extractor(NARROW)
    obs = torch.rand(4, OBSERVATION_SIZE) * 2 - 1
    with torch.no_grad():
        danger, inputs = extractor._danger_and_value_inputs(obs)
        expected = torch.cat((-danger, extractor.value_net(inputs)), dim=1)
        assert torch.equal(extractor(obs), expected)


def _layout():
    return policy_kwargs("velocity-flow-v3")["features_extractor_kwargs"]["critic_layout"]


def test_policy_kwargs_record_the_critic_layout():
    assert _layout() == {
        "version": CRITIC_FEATURE_VERSION, "count": CRITIC_FEATURE_COUNT,
        "groups": {name: list(span) for name, span in CRITIC_GROUPS.items()}}
    assert CRITIC_FEATURE_VERSION >= 1


def test_extractor_rejects_a_different_critic_layout():
    VelocityFlowV3Extractor(SPACE, critic_layout=_layout())
    for change in ({"version": CRITIC_FEATURE_VERSION + 1},
                   {"groups": {**_layout()["groups"], "clock": [1, 12]}}):
        with pytest.raises(ValueError, match="different critic feature layout"):
            VelocityFlowV3Extractor(SPACE, critic_layout={**_layout(), **change})


def test_extractor_rejects_a_narrow_observation_space():
    with pytest.raises(ValueError, match="critic_features=True"):
        VelocityFlowV3Extractor(NARROW)


def test_saved_v3_reloads_with_its_layout(tmp_path):
    from dodge.training import DashboardPPO
    model, env = _model()
    env.close()
    model.save(tmp_path / "v3.zip")
    loaded = DashboardPPO.load(tmp_path / "v3.zip", device="cpu")
    assert loaded.policy_kwargs["features_extractor_kwargs"]["critic_layout"] == _layout()


def test_v3_checkpoint_with_another_layout_fails_to_load(tmp_path):
    from dodge.training import DashboardPPO
    model, env = _model()
    env.close()
    model.policy_kwargs["features_extractor_kwargs"]["critic_layout"]["version"] = \
        CRITIC_FEATURE_VERSION + 1
    model.save(tmp_path / "v3.zip")
    with pytest.raises(ValueError, match="different critic feature layout"):
        DashboardPPO.load(tmp_path / "v3.zip", device="cpu")
