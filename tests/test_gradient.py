"""The analytic controller, and the forward model both bots rely on."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lupa.luajit21")

from dodge.actions import ACTION_MASKS, ACTION_NAMES
from dodge.gradient import (CAMP_FRAMES, ControllerConfig, GameState,
                            EMPTY_HAZARDS, action_scores, camp_penalty,
                            predict_enemies, read_state, simulate_player)

pytestmark = pytest.mark.skipif(not Path("dodge.p8.png").exists(),
                                reason="dodge.p8.png not present")


def _state(enemies, camp_timer=0.0, px=64.0, py=64.0):
    return GameState(px=px, py=py, dpx=0.0, dpy=0.0, size=4.0, speed=1.0,
                     freeze=1.0, camp_timer=camp_timer, camp_armed=True,
                     enemies=np.array(enemies, dtype=np.float64).reshape(-1, 7),
                     hazards=EMPTY_HAZARDS)


def test_player_model_matches_the_cart_exactly():
    """Every score depends on this, so it is pinned against the real game."""
    from dodge.env import DodgeEnv

    env = DodgeEnv(max_episode_frames=None, action_repeat=1)
    try:
        for action in (2, 4, 8):
            env.reset(seed=5)
            for _ in range(20):
                env.step(0)
            state = read_state(env.machine)
            predicted = simulate_player(state.px, state.py, state.dpx, state.dpy,
                                        ACTION_MASKS[action], 12, held=12)
            actual = []
            for _ in range(12):
                env.step(action)
                actual.append((float(env.machine.globals.px),
                               float(env.machine.globals.py)))
            np.testing.assert_allclose(predicted, actual, atol=1e-9)
    finally:
        env.close()


def test_enemy_model_matches_the_cart_exactly():
    from dodge.env import DodgeEnv

    env = DodgeEnv(max_episode_frames=None, action_repeat=1)
    try:
        env.reset(seed=5)
        for _ in range(90):
            env.step(0)
        state = read_state(env.machine)
        count = len(state.enemies)
        path = simulate_player(state.px, state.py, state.dpx, state.dpy,
                               ACTION_MASKS[2], 24, held=24)
        xs, ys = predict_enemies(state, path)

        for frame in range(24):
            env.step(2)
            live = env.machine.globals.enemies
            if len(live) != count:
                break            # one died; the model does not simulate that
            for index in range(count):
                enemy = live[index + 1]
                assert xs[frame][index] == pytest.approx(float(enemy.x), abs=1e-9)
                assert ys[frame][index] == pytest.approx(float(enemy.y), abs=1e-9)
    finally:
        env.close()


def test_the_button_is_released_after_the_decision_interval():
    """Holding it for the whole horizon drove every direction into a wall,
    which made standing still score best 83% of the time."""
    held = simulate_player(64, 64, 0, 0, ACTION_MASKS[2], 24, held=24)
    released = simulate_player(64, 64, 0, 0, ACTION_MASKS[2], 24, held=4)
    assert released[-1][0] < held[-1][0]
    assert released[-1][0] > 64.0, "it should still have travelled"


def test_it_moves_away_from_a_closing_threat():
    # Away from the arena centre, so the anti-camp penalty plays no part: at
    # the centre, idling is charged for camping and that decides the ordering.
    scores = action_scores(_state([[16, 38, 1.0, 0.0, 4.0, 0.0, 1.0]],
                                  px=40.0, py=40.0))
    best, worst = ACTION_NAMES[int(np.argmin(scores))], ACTION_NAMES[int(np.argmax(scores))]
    assert "right" in best, f"threat is on the left, went {best}"
    assert worst == "left", f"straight into it should be worst, got {worst}"


def test_the_camp_timer_is_charged_before_it_trips():
    """The cart spawns a wave after CAMP_FRAMES inside the centre box.

    A penalty charged only on the trip is invisible to a horizon shorter than
    CAMP_FRAMES, so the bot camped for most of a timer before reacting.
    """
    config = ControllerConfig()
    inside = np.full((24, 2), 64.0)
    fresh = camp_penalty(inside, _state([], camp_timer=0.0), config)
    nearly = camp_penalty(inside, _state([], camp_timer=CAMP_FRAMES - 10), config)
    assert 0.0 < fresh < nearly, "cost must grow as the timer runs"

    outside = np.full((24, 2), 20.0)
    assert camp_penalty(outside, _state([], camp_timer=CAMP_FRAMES - 5), config) == 0.0
