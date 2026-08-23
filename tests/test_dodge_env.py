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
from dodge.observe import (CELL_COUNT, OBSERVATION_SIZE, PLAYER_VALUES,
                           VELOCITY_SCALE)

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

    observation, reward, terminated, truncated, info = env.step(2)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["frames"] == 2
    env.close()


def test_the_observation_carries_velocity_not_history():
    """One frame. Motion is given per cell rather than inferred from a stack.

    Everything in this game moves under one 4px cell per decision, so a frame
    difference usually showed nothing at all.
    """
    env = DodgeEnv(action_repeat=4, max_episode_frames=None, powerups=True)
    observation, _ = env.reset(seed=11)
    assert observation.shape == (OBSERVATION_SIZE,)

    for _ in range(40):
        observation, *_ = env.step(2)     # hold right
    categories = observation[PLAYER_VALUES:PLAYER_VALUES + CELL_COUNT]
    vx = observation[PLAYER_VALUES + CELL_COUNT:PLAYER_VALUES + 2 * CELL_COUNT]
    occupied = categories > 0

    assert occupied.any(), "the arena should have entities by now"
    assert np.all(vx[~occupied] == 0.0), "empty cells carry no velocity"
    env.close()


def test_the_player_reports_its_own_velocity():
    """Without a second frame there is nothing else to derive momentum from."""
    env = DodgeEnv(action_repeat=4, max_episode_frames=None)
    observation, _ = env.reset(seed=11)
    assert observation[2] == 0.0 and observation[3] == 0.0

    for _ in range(20):
        observation, *_ = env.step(2)     # hold right
    # Terminal speed is 2.5px a frame, scaled by VELOCITY_SCALE.
    assert observation[2] == pytest.approx(2.5 / VELOCITY_SCALE, abs=0.02)
    assert observation[3] == 0.0

    for _ in range(20):
        observation, *_ = env.step(1)     # hold left
    assert observation[2] < 0.0, "velocity is signed"
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


def test_survival_credit_is_flat_across_the_interior():
    """The centre must not out-pay its surroundings.

    A peak at the centre paid the agent to stand still, because that is where
    it spawns and any movement reduced its income. Flat interior, no attractor.
    """
    centre = center_survival_scalar(5, 5, screen_width=10)
    assert centre == pytest.approx(1.0)
    for x, y in ((5, 6), (6, 5), (6.5, 6.5), (7, 5), (3, 5)):
        assert center_survival_scalar(x, y, screen_width=10) == pytest.approx(centre)


def test_survival_credit_tapers_only_near_the_wall():
    scalar = center_survival_scalar
    # Plateau ends at 0.6 of the half-arena: 5 +/- 3 on a 10-wide screen.
    assert scalar(8, 5, screen_width=10) == pytest.approx(1.0)
    assert scalar(9, 5, screen_width=10) == pytest.approx(0.75)
    assert scalar(10, 5, screen_width=10) == pytest.approx(0.5)
    # Corners lie beyond a cardinal edge; the taper clamps instead of inverting.
    assert scalar(10, 10, screen_width=10) == pytest.approx(0.5)
    assert scalar(0, 0, screen_width=10) == pytest.approx(0.5)


def test_edge_penalty_of_zero_pays_every_position_equally():
    for x, y in ((5, 5), (10, 5), (10, 10)):
        assert center_survival_scalar(x, y, screen_width=10,
                                      edge_penalty=0.0) == pytest.approx(1.0)


def test_survival_reward_at_the_spawn_point_is_the_flat_rate():
    env = DodgeEnv(action_repeat=1)
    env.reset(seed=31)

    _, reward, terminated, _, info = env.step(0)

    assert not terminated
    assert info["survived_frames"] == 1
    assert info["survival_reward"] == pytest.approx(0.01)
    assert reward == pytest.approx(0.01)
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


def test_an_absorbed_hit_ends_the_episode_but_leaves_the_game_running():
    """The agent's episode stops at the hit; the cart does not.

    Ending the episode is what stops the agent learning to spend lives: a
    terminal bootstraps V=0, so nothing after the hit is ever credited to the
    actions before it. Leaving the cart running is what buys the state variety.
    """
    env = DodgeEnv(action_repeat=4, lives=1)
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated, "the RL episode ends at the hit"
    assert not truncated, "terminated, not truncated: truncation would bootstrap V(s')"
    assert reward == pytest.approx(-RewardConfig().death_penalty)
    assert not bool(env.machine.globals.isdead), "the cart is still alive"
    env.close()


def test_reset_after_an_absorbed_hit_continues_the_same_game():
    """The next episode starts mid-game, which is the point of the device.

    Rebuilding the cartridge here would put the agent back at the spawn point
    and throw away the very state variety the life was spent reaching. No frame
    is advanced either, so the new episode opens on the state the old one ended
    on.
    """
    env = DodgeEnv(action_repeat=4, lives=1, max_episode_frames=None)
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    final, _, terminated, _, _ = env.step(0)
    assert terminated
    machine = env.machine

    observation, info = env.reset()

    assert env.machine is machine, "the cartridge run must not be rebuilt"
    np.testing.assert_array_equal(observation, final)
    env.close()


def test_the_cartridge_is_rebuilt_once_the_lives_are_gone():
    """Resuming a finished run would hand the agent a corpse to play."""
    env = DodgeEnv(action_repeat=4, lives=1, max_episode_frames=None)
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    env.step(0)                                   # spends the only life
    env.reset()
    dead_run = env.machine
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    _, _, terminated, _, info = env.step(0)
    assert terminated and bool(env.machine.globals.isdead), "no lives left, so a real death"
    assert info["training_events"]["deaths"] == 1

    env.reset()

    assert env.machine is not dead_run, "a real death starts a new run"
    assert not bool(env.machine.globals.isdead)
    env.close()


def test_run_frames_accumulate_across_episodes_while_frames_restart():
    """Two clocks: one the agent's episode owns, one the cartridge run owns."""
    env = DodgeEnv(action_repeat=2, lives=2, max_episode_frames=None)
    env.reset(seed=31)
    env.step(0)                                   # frames 1-2
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    _, _, terminated, _, info = env.step(0)       # frame 3 collides
    assert terminated
    assert info["frames"] == 3 and info["run_frames"] == 3

    env.reset()
    _, _, _, _, info = env.step(0)                # frames 4-5

    assert info["frames"] == 2, "the episode's own clock restarts"
    assert info["run_frames"] == 5, "the run's clock does not"
    env.close()


def test_the_frame_cap_bounds_the_run_not_the_episode():
    """A run that keeps absorbing hits would otherwise never turn over, holding
    one seed and one difficulty ramp in the buffer indefinitely."""
    env = DodgeEnv(action_repeat=2, lives=3, max_episode_frames=6)
    env.reset(seed=31)
    env.step(0)                                   # frames 1-2
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["run_frames"] == 3

    env.reset()
    _, _, _, truncated, _ = env.step(0)           # run 5, episode 2
    assert not truncated
    _, _, _, truncated, info = env.step(0)        # run 7, episode 4

    assert truncated, "the run passed its cap"
    assert info["frames"] == 4, "this episode is well short of the cap on its own"
    assert info["run_frames"] == 7
    env.close()


def test_a_pattern_crush_kills_outright_however_many_lives_remain():
    """Patterns call die() straight from collisioncheck(), not through collide().

    That distinction is the whole reason the hook tracks which path it is on:
    the player stays inside a pattern rectangle after it is hit, so absorbing
    one would burn every life in consecutive frames without ever getting clear.
    Calling die() directly reproduces exactly that entry -- outside collide(),
    with lives in hand.
    """
    env = DodgeEnv(action_repeat=1, lives=3)
    env.reset(seed=31)
    env.machine.lua.execute("die()")

    _, reward, terminated, _, info = env.step(0)

    assert terminated and bool(env.machine.globals.isdead)
    assert info["training_events"]["lives_spent"] == 0
    assert info["training_events"]["deaths"] == 1
    env.close()


def test_lives_are_off_by_default_and_a_hit_ends_the_run():
    """The default path through the conditional reset must be the old one."""
    env = DodgeEnv(action_repeat=4)
    env.reset(seed=31)
    env.machine.lua.execute("enemies={{x=62,y=62,s=4,p=0}}")
    _, _, terminated, _, info = env.step(0)
    assert terminated and bool(env.machine.globals.isdead)
    assert info["training_events"]["lives_spent"] == 0
    machine = env.machine

    env.reset()

    assert env.machine is not machine
    env.close()


def test_negative_lives_are_rejected():
    with pytest.raises(ValueError):
        DodgeEnv(lives=-1)


def test_a_real_pattern_crush_ends_the_run_with_lives_in_hand():
    """The integration the flag exists for, driven through the cart itself.

    test_a_pattern_crush_kills_outright_however_many_lives_remain calls die()
    directly, which pins the hook's logic but not the path a pattern actually
    takes. Here the cart grows its own pattern to sh == 2 -- its "this kills
    now" state -- and the player is stood in it. Enemies are cleared each frame
    only so the idle player survives the ~500 frames a pattern needs to mature.
    """
    env = DodgeEnv(action_repeat=1, lives=3, patterns=True, max_episode_frames=None)
    env.reset(seed=5)
    g = env.machine.globals

    lethal = None
    for _ in range(4000):
        env.machine.lua.execute("enemies={}")
        env.step(0)
        if g.cp is not None:
            for i in range(1, len(g.cp.rects) + 1):
                r = g.cp.rects[i]
                if float(r.sh) >= 2 and float(r.w) > 8 and float(r.h) > 8:
                    lethal = (float(r.x), float(r.y), float(r.w), float(r.h))
                    break
        if lethal or bool(g.isdead):
            break
    assert lethal is not None and not bool(g.isdead), "no pattern turned lethal"

    x, y, w, h = lethal
    env.machine.lua.execute(f"px={x + w / 2} py={y + h / 2}")
    _, reward, terminated, _, info = env.step(0)

    assert terminated and bool(g.isdead), "a pattern is fatal, lives or not"
    assert int(g.training_lives) == 3, "and it does not spend one"
    assert info["training_events"]["lives_spent"] == 0
    assert reward == pytest.approx(-RewardConfig().death_penalty), "charged once"
    assert info["run_over"] is True

    machine = env.machine
    env.reset()
    assert env.machine is not machine, "the run restarts rather than dying again"
    assert (float(env.machine.globals.px), float(env.machine.globals.py)) == (64.0, 64.0)
    env.close()
