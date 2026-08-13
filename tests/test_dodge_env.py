import faulthandler
from pathlib import Path

import numpy as np
import pytest
import pygame

gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("lupa.luajit21")

from gymnasium.utils.env_checker import check_env

from dodge.env import DodgeEnv, center_survival_scalar
from dodge.rewards import RewardConfig
from dodge.observe import FRAME_OBSERVATION_SIZE, OBSERVATION_SIZE

pytestmark = pytest.mark.skipif(not Path("dodge.p8.png").exists(), reason="dodge.p8.png not present")


@pytest.fixture(scope="module", autouse=True)
def _quiet_luajit_seh():
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    """Keep these tests off whatever weights the developer is currently trying.

    Reward weights are tuned live from the dashboard, so reading the repository's
    own rewards.json would make the suite fail whenever someone moved a slider.
    """
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


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


def test_headless_drawing_changes_nothing_about_the_game():
    """Nulling the draw builtins must not move the shared RNG stream.

    drawgame() calls spawntrail(), which draws two random values per powerup
    enemy per frame, so anything that skipped the draw traversal would change
    which enemies spawn later. Binding the builtins as Lua no-ops leaves the
    traversal and its arguments intact, which is what this pins down.
    """
    from pico8 import cart
    from pico8.backends import NullBackend
    from pico8.machine import Machine

    class CountingBackend:
        """Draws for real, as far as the machine can tell.

        Deliberately not a Backend subclass: __getattr__ only fires for names
        that normal lookup misses, and Backend defines every drawing method.
        """

        draws = True

        def __init__(self):
            self.calls = 0

        def __getattr__(self, name):
            def record(*args):
                self.calls += 1
            return record

    rom, _ = cart.read_rom(Path("dodge.p8.png"))
    code = cart.decompress_code(rom)

    def run(backend):
        machine = Machine(rom, code, backend, seed=77)
        machine.start_game(difficulty=1, patterns=True, powerups=True)
        trace = []
        for frame in range(400):
            machine.set_buttons(1 << (frame // 31) % 4)
            machine.step()
            g = machine.globals
            trace.append((float(g.px), float(g.py), float(g.score),
                          [(float(g.enemies[i].x), float(g.enemies[i].y),
                            float(g.enemies[i].p))
                           for i in range(1, len(g.enemies) + 1)]))
            if g.isdead:
                break
        return trace

    drawing = CountingBackend()
    assert run(drawing) == run(NullBackend())
    assert drawing.calls > 0, "the drawing backend must actually have been used"


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


def test_death_penalty_is_one_point():
    env = DodgeEnv(action_repeat=4)
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert info["survived_frames"] == 0
    assert info["score_delta"] == 0
    assert reward == -RewardConfig().death_penalty == -1.0
    env.close()


def test_weights_are_reread_when_an_episode_starts(tmp_path):
    """The dashboard's sliders reach worker processes only through the file.

    Environments live in separate processes during training, so a weight
    change must arrive by re-reading rather than by any in-memory update.
    """
    from dodge.rewards import RewardConfig, save

    path = tmp_path / "rewards.json"
    save(RewardConfig(death_penalty=1.0), path)
    env = DodgeEnv(action_repeat=4, reward_config_path=path)
    env.reset(seed=31)
    assert env.reward_config.death_penalty == 1.0

    save(RewardConfig(death_penalty=4.0), path)
    assert env.reward_config.death_penalty == 1.0, "not until the episode ends"

    env.reset(seed=31)
    assert env.reward_config.death_penalty == 4.0
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == pytest.approx(-4.0)
    env.close()


def test_luck_score_weight_restores_the_carts_own_scoring(tmp_path):
    from dodge.rewards import RewardConfig, save

    path = tmp_path / "rewards.json"
    save(RewardConfig(uncontrolled_score_weight=1.0), path)
    env = DodgeEnv(action_repeat=1, reward_config_path=path)
    env.reset(seed=31)
    env.machine.lua.execute("score=score+0.5 difficultycurve(true)")

    _, reward, _, _, info = env.step(0)

    assert info["uncontrolled_score"] == pytest.approx(0.5)
    assert reward == pytest.approx(info["survival_reward"] + 0.5)
    env.close()


def test_score_from_enemies_colliding_is_not_rewarded():
    """Enemies destroying each other is luck, not play.

    The cart credits 0.5 whenever an enemy dies to another enemy or a pattern,
    which the agent does not cause. Paying for it injects variance the critic
    cannot explain, so only agent-caused score reaches the reward.
    """
    env = DodgeEnv(action_repeat=1)
    env.reset(seed=31)
    # Drive the cart's own enemy-death path: score rises and the hook counts it.
    env.machine.lua.execute("score=score+0.5 difficultycurve(true)")

    _, reward, terminated, _, info = env.step(0)

    assert not terminated
    assert info["score_delta"] == pytest.approx(0.5)
    assert info["training_events"]["enemies_destroyed"] == 1
    assert info["uncontrolled_score"] == pytest.approx(0.5)
    # Only the survival credit survives into the reward.
    assert reward == pytest.approx(info["survival_reward"])
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
