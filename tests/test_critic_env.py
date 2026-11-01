"""DodgeVecEnv(critic_features=True) appends the critic's hidden state."""

import numpy as np
import pytest

from dodge.critic import WIDE_OBSERVATION_SIZE
from dodge.env import DodgeVecEnv, make_engine, native_available
from dodge.observe import OBSERVATION_SIZE

pytestmark = pytest.mark.skipif(not native_available(), reason="needs dodge_native")

SETTINGS = dict(difficulty=3, patterns=True, powerups=True, action_repeat=1)


def test_actor_columns_match_the_plain_env():
    plain = DodgeVecEnv(n_envs=4, seed=7, **SETTINGS)
    wide = DodgeVecEnv(n_envs=4, seed=7, critic_features=True, **SETTINGS)
    a, b = plain.reset(), wide.reset()
    rng = np.random.default_rng(0)
    for _ in range(1500):
        assert np.array_equal(b[:, :OBSERVATION_SIZE], a)
        actions = rng.integers(0, 9, 4)
        a, ra, da, _ = plain.step(actions)
        b, rb, db, ib = wide.step(actions)
        assert np.array_equal(ra, rb) and np.array_equal(da, db)
        for info in ib:
            if "terminal_observation" in info:
                assert info["terminal_observation"].shape == (WIDE_OBSERVATION_SIZE,)


@pytest.mark.parametrize("mode", ["nursery", "batch"])
def test_restarted_lane_carries_its_new_games_features(mode):
    env = DodgeVecEnv(n_envs=4, seed=3, critic_features=True, **SETTINGS)
    env._restart_observe = mode
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(5000):
        observation, _, done, _ = env.step(rng.integers(0, 9, 4))
        if done.any():
            break
    else:
        pytest.fail("no lane died")
    lane = int(np.flatnonzero(done)[0])
    reference = make_engine(**{**env._engine_settings}, execution="serial")
    reference.reset_ml_batch(np.full(4, env.game_seeds[lane], dtype=np.uint32))
    expected = reference.observe_velocity(critic=True)
    assert np.array_equal(observation[lane, OBSERVATION_SIZE:], expected["critic"][0])


def test_observation_space_is_wide():
    env = DodgeVecEnv(n_envs=2, critic_features=True, **SETTINGS)
    assert env.observation_space.shape == (WIDE_OBSERVATION_SIZE,)
    assert env.game_settings["critic_features"] is True
