"""How a finished game's trace becomes appearances, without the engine."""

import numpy as np

from dodge.death_paths import Recording, _appearances

LETHAL = [(0.0, 0.0, 16.0, 16.0, 1)]
WARNING = [(0.0, 2.0, 16.0, 12.0, 0)]


def _span(start, end, rects, pid=28):
    count = (end if end is not None else start + 4) - start + 1
    return {"id": pid, "type": 0, "start": start, "end": end,
            "x": [60.0 + i / 3 for i in range(count)], "y": [64.0] * count,
            "rects": rects, "rects_frame": start if rects else None}


def _died_in_warning(start):
    """A game killed by an enemy before its pattern's rectangles turned lethal."""
    return {
        "frame": start + 4, "fyou_triggers": [], "contacts": [1, 0, 0, 0, 0, 0],
        "patterns": [_span(start - 50, start - 40, LETHAL), _span(start, None, [])],
        "death": {"cause": "normal_enemy", "frame": start + 4, "killer": None,
                  "enemies_alive": [1, 0, 0, 0], "rects": WARNING,
                  "player": {"x": 61.0, "y": 64.0, "vx": 0.0, "vy": 0.0, "size": 3.0},
                  "active_pattern": {"id": 28, "type": 0, "start": start, "since": 4},
                  "killer_path": [], "recent": {}},
    }


def test_paths_are_stored_compactly():
    found = _appearances(1_000_000, _died_in_warning(500), 12.0)
    assert len(found) == 2
    for appearance in found:
        assert isinstance(appearance["x"], np.ndarray) and appearance["x"].dtype == np.float32
        assert isinstance(appearance["y"], np.ndarray) and appearance["y"].dtype == np.float32
    for index, appearance in enumerate(found):
        appearance["id"] = index
    recording = Recording(run_id="r", settings={}, games=1, actions={}, appearances=found)
    view = recording.pattern(28)
    # What the API sends is plain JSON lists, rounded to the hundredth.
    first = view["appearances"][0]
    assert isinstance(first["x"], list) and isinstance(first["x"][0], float)
    assert first["x"][:2] == [60.0, 60.33]


def test_a_death_during_the_warning_keeps_the_shared_layout():
    found = _appearances(1_000_000, _died_in_warning(500), 12.0)
    # The early death still draws the rectangles as they stood when it died.
    assert [a["outcome"] for a in found] == ["survived", "died"]
    assert found[1]["rects"] == [list(WARNING[0])]
    for index, appearance in enumerate(found):
        appearance["id"] = index
    recording = Recording(run_id="r", settings={}, games=1, actions={}, appearances=found)
    assert recording.pattern(28)["layout"] == [list(LETHAL[0])]
