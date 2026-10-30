"""The analytic controller, and the forward model both bots rely on."""

import numpy as np
import pytest

from dodge.actions import ACTION_MASKS, ACTION_NAMES
from dodge.gradient import (CAMP_FRAMES, ControllerConfig, GameState,
                            EMPTY_HAZARDS, action_scores, camp_penalty,
                            predict_enemies, read_state, simulate_player)

from dodge.env import native_available

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native not installed")


def _state(enemies, camp_timer=0.0, px=64.0, py=64.0):
    return GameState(px=px, py=py, dpx=0.0, dpy=0.0, size=4.0, speed=1.0,
                     freeze=1.0, camp_timer=camp_timer, camp_armed=True,
                     enemies=np.array(enemies, dtype=np.float64).reshape(-1, 7),
                     hazards=EMPTY_HAZARDS)


def _engine():
    from dodge.env import make_engine
    engine = make_engine(difficulty=1)
    engine.reset_ml_batch(np.array([5], dtype=np.uint32))
    return engine


def _step(engine, action, frames=1):
    for _ in range(frames):
        engine.step_ml_batch(np.array([action], dtype=np.uint8))
    return engine.lane_state(0)


def test_player_model_matches_the_engine():
    """Every score depends on this, so it is pinned against the real game.

    The engine runs PICO-8's 16.16 fixed point and the model runs floats, so
    they agree to fixed-point resolution rather than bit for bit.
    """
    for action in (2, 4, 8):
        engine = _engine()
        state = read_state(_step(engine, 0, 20))
        predicted = simulate_player(state.px, state.py, state.dpx, state.dpy,
                                    ACTION_MASKS[action], 12, held=12)
        actual = []
        for _ in range(12):
            lane = _step(engine, action)
            actual.append((lane["px"], lane["py"]))
        np.testing.assert_allclose(predicted, actual, atol=1e-3)


def test_enemy_model_matches_the_engine():
    engine = _engine()
    state = read_state(_step(engine, 0, 90))
    count = len(state.enemies)
    assert count > 0
    path = simulate_player(state.px, state.py, state.dpx, state.dpy,
                           ACTION_MASKS[2], 24, held=24)
    xs, ys = predict_enemies(state, path)

    for frame in range(24):
        live = _step(engine, 2)["enemies"]
        if len(live) != count:
            break            # one died; the model does not simulate that
        for index in range(count):
            assert xs[frame][index] == pytest.approx(live[index]["x"], abs=1e-2)
            assert ys[frame][index] == pytest.approx(live[index]["y"], abs=1e-2)


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
