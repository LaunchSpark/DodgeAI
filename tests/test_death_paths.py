"""The sampling run: every pattern appearance, from the engine trace."""

import numpy as np
import pytest

pytest.importorskip("dodge_native")

from dodge.death_paths import Recording, record  # noqa: E402
from tests.dodger import dodge  # noqa: E402

SEEDS = range(1_000_000, 1_000_048)


@pytest.fixture(scope="module")
def recording():
    return record(dodge, SEEDS, step_frames=1, difficulty=1, patterns=True, powerups=True)


@pytest.fixture(scope="module")
def capped():
    return record(dodge, SEEDS[:16], frame_cap=600, step_frames=1, difficulty=1,
                  patterns=True, powerups=True)


def _in_patterns(recording):
    return [a for a in recording.appearances if a["pid"] >= 0]


def test_every_pattern_appearance_is_recorded(recording):
    assert isinstance(recording, Recording)
    assert recording.games == len(SEEDS)
    assert [a["id"] for a in recording.appearances] == list(range(len(recording.appearances)))
    summary = recording.summary()
    assert summary["run_id"] == recording.run_id
    counted = sum(p["encounters"] for p in summary["patterns"] if p["pid"] >= 0)
    assert counted == len(_in_patterns(recording)) > 0
    deaths = sum(p["deaths"] for p in summary["patterns"] if p["pid"] >= 0)
    assert deaths == sum(1 for a in _in_patterns(recording) if a["outcome"] == "died")
    # Every game ends in exactly one death (no game reaches the cap here).
    died = [a for a in recording.appearances if a["outcome"] == "died"]
    assert sorted(a["seed"] for a in died) == list(SEEDS)


def test_died_appearance_ends_at_death(recording):
    died = [a for a in _in_patterns(recording) if a["outcome"] == "died"]
    assert died
    for appearance in died:
        assert appearance["end"] == appearance["death"]["frame"]
        assert len(appearance["x"]) == appearance["end"] - appearance["start"] + 1
        assert appearance["cause"] in ("rect", "enemy")
        assert appearance["death"]["since"] == appearance["end"] - appearance["start"]
    for appearance in _in_patterns(recording):
        assert len(appearance["x"]) == len(appearance["y"])
        assert len(appearance["x"]) == appearance["end"] - appearance["start"] + 1


def test_no_pattern_death_keeps_last_120_frames(recording):
    loose = [a for a in recording.appearances if a["pid"] == -1]
    assert loose
    for appearance in loose:
        assert appearance["outcome"] == "died"
        assert appearance["type"] is None
        assert 0 < len(appearance["x"]) <= 120
        assert appearance["start"] == appearance["end"] - len(appearance["x"]) + 1
    patterns = recording.summary()["patterns"]
    assert patterns[-1]["pid"] == -1
    assert patterns[-1]["deaths"] == len(loose)


def test_capped_pattern_is_unfinished(capped):
    unfinished = [a for a in capped.appearances if a["outcome"] == "unfinished"]
    assert unfinished
    for appearance in unfinished:
        assert appearance["death"] is None and appearance["cause"] is None
        assert appearance["end"] >= 600
        assert len(appearance["x"]) == appearance["end"] - appearance["start"] + 1


def test_survived_appearance_keeps_rectangles(recording, capped):
    survived = [a for r in (recording, capped) for a in r.appearances
                if a["outcome"] == "survived"]
    assert survived
    for appearance in survived:
        assert appearance["rects"], appearance["pid"]
        assert appearance["end"] is not None


def test_pattern_view_lists_its_appearances(recording):
    pid = recording.summary()["patterns"][0]["pid"]
    view = recording.pattern(pid)
    assert view["run_id"] == recording.run_id and view["pid"] == pid
    assert [a["id"] for a in view["appearances"]] == [
        a["id"] for a in recording.appearances if a["pid"] == pid]
    layouts = {tuple(map(tuple, a["rects"])) for a in view["appearances"]
               if any(rect[4] for rect in a["rects"])}
    assert (view["layout"] is not None) == (len(layouts) == 1)


def test_actions_are_recorded_per_seed(recording):
    assert set(recording.actions) == set(SEEDS)
    for appearance in recording.appearances:
        if appearance["outcome"] == "died":
            actions = recording.actions[appearance["seed"]]
            assert actions.dtype == np.uint8
            # One decision per frame at action repeat 1.
            assert len(actions) == appearance["end"]
    assert recording.settings["step_frames"] == 1
