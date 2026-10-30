"""Replaying one recorded appearance as real rendered frames."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("dodge_native")

from dodge.death_paths import record  # noqa: E402
from dodge.pattern_replay import Replay, replay  # noqa: E402
from tests.dodger import dodge  # noqa: E402

SEEDS = range(1_000_000, 1_000_064)


@pytest.fixture(scope="module")
def recording():
    return record(dodge, SEEDS, step_frames=1, difficulty=1, patterns=True, powerups=True)


def _window(appearance):
    if appearance["pid"] == -1:
        return min(300, appearance["end"])
    return appearance["end"] - appearance["start"] + 1


def test_replay_reproduces_every_death(recording):
    died = [a for a in recording.appearances if a["outcome"] == "died"]
    assert len(died) == len(SEEDS)
    for appearance in died:
        result = replay(recording, appearance["id"])
        assert isinstance(result, Replay)
        assert result.matched, (appearance["seed"], result.reason)
        assert result.reason is None
        assert result.frames.shape == (_window(appearance), 128, 128)
        assert result.frames.dtype == np.uint8
        assert len(result.samples) == len(result.frames)
        assert {"st", "enemies", "enemy_gap", "rect_gap"} <= set(result.samples[0])


def test_replay_of_a_survived_appearance_matches(recording):
    survived = [a for a in recording.appearances if a["outcome"] == "survived"][:5]
    assert survived
    for appearance in survived:
        result = replay(recording, appearance["id"])
        assert result.matched, result.reason
        assert len(result.frames) == _window(appearance)
        # The pattern is on screen until the frame it ends on.
        assert all(sample["st"] > 0 for sample in result.samples[:-1])
        assert result.samples[-1]["st"] == 0


def test_replay_matches_at_action_repeat_4():
    repeated = record(dodge, SEEDS[:16], step_frames=4, difficulty=1, patterns=True,
                      powerups=True)
    died = [a for a in repeated.appearances if a["outcome"] == "died"]
    assert died
    for appearance in died:
        result = replay(repeated, appearance["id"])
        assert result.matched, (appearance["seed"], result.reason)
        assert len(result.frames) == _window(appearance)


def test_replay_flags_a_tampered_action(recording):
    appearance = next(a for a in recording.appearances
                      if a["outcome"] == "died" and a["pid"] >= 0 and a["start"] > 200)
    actions = recording.actions[appearance["seed"]].copy()
    # Steer the other way for two seconds before the pattern begins.
    span = slice(appearance["start"] - 160, appearance["start"] - 40)
    actions[span] = np.where(actions[span] == 1, 2, 1)
    tampered = dataclasses.replace(
        recording, actions={**recording.actions, appearance["seed"]: actions})
    result = replay(tampered, appearance["id"])
    assert not result.matched
    assert result.reason


def test_no_pattern_replay_is_last_300_frames(recording):
    loose = [a for a in recording.appearances if a["pid"] == -1]
    assert loose
    for appearance in loose:
        result = replay(recording, appearance["id"])
        assert result.matched, result.reason
        assert len(result.frames) == min(300, appearance["end"])


def test_unfinished_appearance_replays_to_the_cap():
    capped = record(dodge, SEEDS[:16], frame_cap=600, step_frames=1, difficulty=1,
                    patterns=True, powerups=True)
    unfinished = [a for a in capped.appearances if a["outcome"] == "unfinished"]
    assert unfinished
    for appearance in unfinished:
        result = replay(capped, appearance["id"])
        assert result.matched, result.reason
        assert len(result.frames) == _window(appearance)
        assert result.samples[-1]["frame"] == appearance["end"]
