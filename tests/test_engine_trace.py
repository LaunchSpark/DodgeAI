"""The engine's gameplay trace (Taylor Turner's recorder), read from Python."""

import numpy as np
import pytest

pytest.importorskip("dodge_native")

from dodge.env import DEATH_CAUSES, make_engine  # noqa: E402

# The trace names causes as Taylor's DeathCause does; the env as Telemetry does.
TRACE_CAUSES = {
    "normal_enemy": "normal",
    "kamikaze_enemy": "kamikaze",
    "kamikaze_explosion": "kamikaze_blast",
    "static_pattern_rect": "static_rect",
    "moving_pattern_rect": "moving_rect",
}


def _engine():
    return make_engine(step_frames=1, difficulty=3, patterns=True, powerups=True,
                       execution="serial")


def test_trace_is_off_by_default_and_on_after_enabling():
    engine = _engine()
    engine.reset_ml_batch(np.arange(2, dtype=np.uint32))
    engine.step_ml_batch(np.zeros(2, dtype=np.uint8))
    assert engine.lane_trace(0) is None
    assert engine.lane_sample(0) is None

    engine.set_trace_enabled(True)
    engine.reset_ml_batch(np.arange(2, dtype=np.uint32))
    engine.step_ml_batch(np.zeros(2, dtype=np.uint8))
    trace = engine.lane_trace(0)
    assert trace["frame"] == 1
    assert trace["death"] is None
    assert trace["patterns"] == []
    assert trace["contacts"] == [0] * 6
    sample = engine.lane_sample(0)
    assert sample["frame"] == 1
    assert sample["pattern_id"] is None
    assert 0 < sample["x"] < 128 and 0 < sample["y"] < 128


def test_death_trace_matches_existing_telemetry():
    engine = _engine()
    engine.set_trace_enabled(True)
    lanes = 16
    engine.reset_ml_batch(np.arange(lanes, dtype=np.uint32))
    rng = np.random.default_rng(3)
    seen = {}
    filler = 1_000
    for _ in range(6_000):
        stepped = engine.step_ml_batch(rng.integers(0, 9, lanes, dtype=np.uint8))
        state = engine.observe_velocity()
        done = np.flatnonzero(stepped["done"])
        for lane in done:
            if lane in seen:
                continue
            trace = engine.lane_trace(int(lane))
            seen[lane] = trace
            death = trace["death"]
            assert TRACE_CAUSES[death["cause"]] == DEATH_CAUSES[int(state["death_cause"][lane])]
            active = death["active_pattern"]
            assert (active["id"] if active else -1) == int(state["death_pattern"][lane])
            assert death["frame"] == trace["frame"]
            assert sum(trace["contacts"]) >= 1
            if death["killer"] is not None:
                assert death["killer"]["fyou_spawned"] in (True, False)
                assert death["killer_path"][-1][0] == 1
        if len(seen) == lanes:
            break
        if len(done):
            engine.reset_ml_lanes(done.astype(np.uint32),
                                  np.arange(filler, filler + len(done), dtype=np.uint32))
            filler += len(done)
    assert len(seen) == lanes


def test_lane_sample_frame_advances_one_per_frame():
    engine = _engine()
    engine.set_trace_enabled(True)
    engine.reset_ml_batch(np.array([5], dtype=np.uint32))
    frames = []
    for _ in range(20):
        engine.step_ml_batch(np.array([1], dtype=np.uint8))
        frames.append(engine.lane_sample(0)["frame"])
    assert frames == list(range(1, 21))
