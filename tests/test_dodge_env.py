import faulthandler
from pathlib import Path

import numpy as np
import pytest
import pygame

gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("lupa.luajit21")

from gymnasium.utils.env_checker import check_env

from dodge.env import DEATH_PENALTY, DodgeEnv, center_survival_scalar
from dodge.observe import FRAME_OBSERVATION_SIZE, OBSERVATION_SIZE

pytestmark = pytest.mark.skipif(not Path("logic.js").exists(), reason="logic.js not present")


@pytest.fixture(scope="module", autouse=True)
def _quiet_luajit_seh():
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


def test_environment_passes_gymnasium_checker():
    env = DodgeEnv(max_episode_frames=40)
    check_env(env, skip_render_check=True)
    env.close()


def test_frame_cap_truncates_the_episode():
    env = DodgeEnv(action_repeat=2, max_episode_frames=6)
    env.reset(seed=11)
    truncated = False
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(0)
        if terminated:
            break
    assert truncated and info["frames"] == 6
    env.close()


def test_no_frame_cap_never_truncates():
    """Watching a policy runs until the agent dies, however long that takes."""
    env = DodgeEnv(action_repeat=2, max_episode_frames=None)
    env.reset(seed=11)
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(0)
        assert not truncated
        if terminated:
            break
    assert info["frames"] > 6, "should have run well past the usual training cap"
    env.close()


def test_non_positive_frame_cap_is_rejected():
    with pytest.raises(ValueError):
        DodgeEnv(max_episode_frames=0)


def test_environment_step_returns_expected_contract():
    env = DodgeEnv(action_repeat=2, max_episode_frames=20)
    observation, info = env.reset(seed=11)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert info["seed"] == 11
    np.testing.assert_array_equal(
        observation[:FRAME_OBSERVATION_SIZE],
        observation[FRAME_OBSERVATION_SIZE:],
    )

    observation, reward, terminated, truncated, info = env.step(2)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["frames"] == 2
    env.close()


def test_observation_contains_previous_then_current_game_frame():
    env = DodgeEnv(action_repeat=2, max_episode_frames=20)
    initial, _ = env.reset(seed=11)
    initial_current = initial[FRAME_OBSERVATION_SIZE:].copy()

    observation, *_ = env.step(2)
    previous = observation[:FRAME_OBSERVATION_SIZE]
    current = observation[FRAME_OBSERVATION_SIZE:]

    np.testing.assert_array_equal(previous, initial_current)
    assert previous[0] < current[0]
    assert not np.array_equal(previous, current)

    next_observation, *_ = env.step(2)
    np.testing.assert_array_equal(
        next_observation[:FRAME_OBSERVATION_SIZE],
        current,
    )
    env.close()


def test_same_seed_and_actions_are_deterministic():
    def run():
        env = DodgeEnv(action_repeat=2, max_episode_frames=100)
        observation, _ = env.reset(seed=29)
        trace = [observation]
        for action in [0, 2, 2, 3, 3, 1, 4] * 3:
            observation, reward, terminated, truncated, _ = env.step(action)
            trace.append(np.append(observation, reward))
            if terminated or truncated:
                break
        env.close()
        return trace

    left, right = run(), run()
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)


def test_near_miss_means_within_one_cell_without_collision():
    env = DodgeEnv()
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=69,y=64,s=4,p=0}}")
    assert env._is_near_miss()
    env.machine.lua.execute("enemies={{x=66,y=64,s=4,p=0}}")
    assert not env._is_near_miss()
    env.close()


def test_center_survival_scalar_uses_euclidean_distance():
    assert center_survival_scalar(5, 5, screen_width=10) == pytest.approx(1.5)
    assert center_survival_scalar(10, 5, screen_width=10) == pytest.approx(0.5)
    assert center_survival_scalar(10, 10, screen_width=10) == pytest.approx(
        1.5 - np.sqrt(2)
    )


def test_center_survival_reward_is_fifty_percent_higher():
    env = DodgeEnv(action_repeat=1)
    env.reset(seed=31)

    _, reward, terminated, _, info = env.step(0)

    assert not terminated
    assert info["survived_frames"] == 1
    assert info["survival_reward"] == pytest.approx(0.015)
    assert reward == pytest.approx(0.015)
    env.close()


def test_death_penalty_is_two_points():
    env = DodgeEnv(action_repeat=4)
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert info["survived_frames"] == 0
    assert info["score_delta"] == 0
    assert reward == -DEATH_PENALTY == -2.0
    env.close()


def test_human_render_backend_is_reused_across_episode_resets(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    env = DodgeEnv(render_mode="human", scale=1, caption="Persistent watcher")
    env.reset(seed=1)
    backend = env.machine.backend
    window = backend.window
    env.reset(seed=2)
    assert env.machine.backend is backend
    assert env.machine.backend.window is window
    assert pygame.display.get_caption()[0] == "Persistent watcher"
    env.close()
