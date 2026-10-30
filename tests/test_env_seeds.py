"""DodgeVecEnv draws its games from a seed stream, and can save and restore it."""

import numpy as np
import pytest

from dodge import seeds
from dodge.env import DodgeVecEnv, native_available
from dodge.rewards import RewardConfig
from dodge.seeds import SeedStream

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


def make(**kwargs):
    settings = dict(n_envs=4, seed=3, difficulty=3, patterns=True, powerups=True,
                    action_repeat=4)
    settings.update(kwargs)
    return DodgeVecEnv(**settings)


def idle(env):
    return np.zeros(env.num_envs, dtype=np.int64)


def test_reset_draws_each_slots_first_game_from_the_stream():
    env = make()
    env.reset()
    stream = SeedStream(run_seed=3, n_slots=4)
    assert list(env.game_seeds) == [stream.seed_for(slot, 0) for slot in range(4)]
    assert list(env.next_episode_indices) == [1, 1, 1, 1]
    assert [info["seed"] for info in env.reset_infos] == list(env.game_seeds)


def test_a_finished_game_restarts_on_that_slots_next_seed():
    env = make(max_episode_frames=8)
    env.reset()
    _, _, done, _ = env.step(idle(env))
    assert not done.any()
    _, _, done, infos = env.step(idle(env))
    assert done.all()
    stream = SeedStream(run_seed=3, n_slots=4)
    assert [info["run"]["seed"] for info in infos] == [stream.seed_for(s, 0) for s in range(4)]
    assert list(env.game_seeds) == [stream.seed_for(slot, 1) for slot in range(4)]
    assert list(env.next_episode_indices) == [2, 2, 2, 2]


def test_next_episode_indices_can_be_set():
    env = make()
    env.next_episode_indices = [5, 6, 7, 8]
    env.reset()
    stream = SeedStream(run_seed=3, n_slots=4)
    assert list(env.game_seeds) == [stream.seed_for(s, 5 + s) for s in range(4)]


def test_seeding_the_vecenv_restarts_the_stream_from_that_seed():
    env = make(seed=1)
    env.reset()
    env.seed(9)
    env.reset()
    assert list(env.game_seeds) == list(SeedStream(9, 4).draw(np.arange(4)))
    assert list(env.next_episode_indices) == [1, 1, 1, 1]


def test_training_never_plays_a_reserved_seed(monkeypatch):
    monkeypatch.setattr(seeds, "_RESERVED", {"bottom-half": (0, 2**31)})
    env = make(max_episode_frames=4)
    env.reset()
    played = list(env.game_seeds)
    for _ in range(3):
        env.step(idle(env))
        played.extend(env.game_seeds)
    assert min(played) >= 2**31


def test_explicit_seed_lists_are_played_in_order():
    env = make(seeds=[1_000_000 + i for i in range(8)], max_episode_frames=4)
    env.reset()
    assert list(env.game_seeds) == [1_000_000, 1_000_001, 1_000_002, 1_000_003]
    env.step(idle(env))
    assert list(env.game_seeds) == [1_000_004, 1_000_005, 1_000_006, 1_000_007]


def test_frames_are_counted_across_lanes():
    env = make()
    env.reset()
    env.step(idle(env))
    env.step(idle(env))
    assert env.total_frames == 4 * 2 * 4


def test_abandoning_resets_every_lane_onto_its_next_seed():
    env = make()
    env.reset()
    for _ in range(3):
        env.step(idle(env))
    played = list(env.game_seeds)
    observation, abandoned = env.abandon_episodes()
    assert [record["slot"] for record in abandoned] == [0, 1, 2, 3]
    assert [record["seed"] for record in abandoned] == played
    assert all(record["frames"] == 12 for record in abandoned)
    assert all(record["episode_index"] == 0 for record in abandoned)
    assert all("score" in record for record in abandoned)
    stream = SeedStream(run_seed=3, n_slots=4)
    assert list(env.game_seeds) == [stream.seed_for(s, 1) for s in range(4)]
    # Fresh games: the same observation a new env shows on those seeds.
    fresh = make(seeds=list(env.game_seeds))
    assert np.array_equal(observation, fresh.reset())


def test_seed_state_restores_the_games_in_flight_at_a_boundary():
    env = make()
    env.reset()
    env.step(idle(env))
    observation, _ = env.abandon_episodes()
    state = env.seed_state()

    other = make(seed=77)
    restored = other.restore_seed_state(state)
    assert np.array_equal(restored, observation)
    assert list(other.game_seeds) == list(env.game_seeds)
    assert list(other.next_episode_indices) == list(env.next_episode_indices)
    assert other.total_frames == env.total_frames
    # And both carry on identically.
    a = env.step(idle(env))
    b = other.step(idle(other))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    a, b = env.step(idle(env)), other.step(idle(other))
    assert np.array_equal(a[0], b[0])
