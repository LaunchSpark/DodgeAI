"""Segment boundaries make a checkpoint resume bit-exactly."""

import random

import numpy as np
import pytest
import torch

from dodge.env import native_available
from dodge.rewards import RewardConfig

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")

SEGMENT = 3
N_ENVS = 2
N_STEPS = 8
SEGMENT_STEPS = SEGMENT * N_ENVS * N_STEPS


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


def make_env(seed=3):
    from dodge.env import DodgeVecEnv
    return DodgeVecEnv(n_envs=N_ENVS, seed=seed, difficulty=3, patterns=True,
                       powerups=True, action_repeat=4)


def make_model(env):
    from stable_baselines3 import PPO
    from dodge.policies import policy_class, policy_kwargs
    return PPO(policy_class("mlp"), env, n_steps=N_STEPS, batch_size=8, n_epochs=2,
               policy_kwargs=policy_kwargs("mlp"), device="cpu", seed=0)


def scramble_rngs():
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)


def assert_same_parameters(a, b):
    left, right = a.policy.state_dict(), b.policy.state_dict()
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name
    left = a.policy.optimizer.state_dict()["state"]
    right = b.policy.optimizer.state_dict()["state"]
    for key in left:
        for name in left[key]:
            assert torch.equal(torch.as_tensor(left[key][name]),
                               torch.as_tensor(right[key][name])), (key, name)


def test_rng_state_round_trips():
    from dodge.resume import capture_rng_state, restore_rng_state

    model = make_model(make_env())
    state = capture_rng_state(model)
    expected = (random.random(), np.random.rand(), torch.rand(1).item(),
                model.action_space.sample())
    scramble_rngs()
    model.action_space.seed(1)
    restore_rng_state(state, model)
    assert (random.random(), np.random.rand(), torch.rand(1).item(),
            model.action_space.sample()) == expected


def test_boundaries_fall_every_segment_and_log_the_abandoned_games():
    from dodge.resume import SegmentCallback

    env = make_env()
    model = make_model(env)
    seen = []
    segment = SegmentCallback(SEGMENT, on_boundary=[lambda cb: seen.append(cb.updates)])
    model.learn(2 * SEGMENT_STEPS + N_ENVS * N_STEPS, callback=segment)
    assert segment.updates == 2 * SEGMENT + 1
    assert seen == [SEGMENT, 2 * SEGMENT]
    assert len(segment.abandoned) == 2 * N_ENVS
    first = segment.abandoned[:N_ENVS]
    assert [record["update"] for record in first] == [SEGMENT] * N_ENVS
    assert [record["slot"] for record in first] == list(range(N_ENVS))
    assert all({"seed", "frames", "score", "episode_index"} <= set(r) for r in first)
    # Every lane started a fresh game at each boundary.
    assert all(index >= 3 for index in env.next_episode_indices)


def test_segments_can_be_disabled():
    from dodge.resume import SegmentCallback

    model = make_model(make_env())
    segment = SegmentCallback(0, on_boundary=[lambda cb: pytest.fail("boundary")])
    model.learn(SEGMENT_STEPS, callback=segment)
    assert segment.updates == SEGMENT and segment.abandoned == []


def test_two_segments_straight_equal_one_saved_loaded_and_one_more(tmp_path):
    from dodge.resume import CheckpointWriter, SegmentCallback, load_checkpoint

    # Straight through.
    straight = make_model(make_env())
    straight_segments = SegmentCallback(SEGMENT)
    straight.learn(2 * SEGMENT_STEPS, callback=straight_segments)

    # One segment, checkpointed at its boundary...
    first = make_model(make_env())
    writer = CheckpointWriter()
    saved = []

    def save(callback):
        saved.append(writer.save(callback.model, tmp_path / "boundary",
                                 env=callback.training_env,
                                 ppo_updates=callback.updates, at_boundary=True,
                                 segment=callback.state(),
                                 abandoned=callback.abandoned))

    first.learn(SEGMENT_STEPS, callback=SegmentCallback(SEGMENT, on_boundary=[save]))
    assert len(saved) == 1
    del first

    # ...then loaded into fresh objects, with every global generator moved
    # on, and trained one more.
    scramble_rngs()
    env = make_env(seed=1234)
    resumed, state = load_checkpoint(saved[0], env, device="cpu")
    assert state["at_boundary"]
    assert list(env.next_episode_indices) == state["env"]["seeds"]["next_episode_indices"]
    resumed_segments = SegmentCallback.from_state(state, SEGMENT)
    resumed.learn(SEGMENT_STEPS, callback=resumed_segments, reset_num_timesteps=False)

    assert resumed.num_timesteps == straight.num_timesteps
    assert resumed_segments.updates == straight_segments.updates == 2 * SEGMENT
    assert_same_parameters(straight, resumed)
    assert resumed_segments.abandoned == straight_segments.abandoned


def test_a_mid_segment_checkpoint_restores_model_and_rng_on_new_games(tmp_path):
    from dodge.resume import CheckpointWriter, load_checkpoint

    env = make_env()
    model = make_model(env)
    model.learn(N_ENVS * N_STEPS)
    playing = list(env.game_seeds)
    path = CheckpointWriter().save(model, tmp_path / "mid", env=env, ppo_updates=1)
    torch_state = torch.get_rng_state()

    scramble_rngs()
    other = make_env(seed=5)
    loaded, state = load_checkpoint(path, other, device="cpu")
    assert not state["at_boundary"]
    assert torch.equal(torch.get_rng_state(), torch_state)
    # The games in flight cannot be resumed mid-way, so each lane starts
    # its next game from the stream.
    assert list(other.next_episode_indices) == [i + 1 for i in env.next_episode_indices]
    assert not set(other.game_seeds) & set(playing)
    assert loaded._last_obs is not None


def test_a_checkpoint_without_a_sidecar_loads_as_before(tmp_path):
    from dodge.resume import load_checkpoint

    model = make_model(make_env())
    model.save(tmp_path / "plain")
    loaded, state = load_checkpoint(tmp_path / "plain.zip", make_env(), device="cpu")
    assert state is None
    assert loaded._last_obs is None  # SB3 resets the env on learn, as ever


def test_the_budget_stops_at_the_next_boundary():
    from dodge.resume import SegmentCallback

    clock = iter([0.0] + [1_000.0] * 1_000)
    boundaries = []
    model = make_model(make_env())
    segment = SegmentCallback(SEGMENT, budget_seconds=10,
                              on_boundary=[lambda cb: boundaries.append(cb.updates)],
                              clock=lambda: next(clock))
    model.learn(10 * SEGMENT_STEPS, callback=segment)
    assert segment.stop_requested
    assert boundaries == [SEGMENT]
    assert segment.updates == SEGMENT
