"""The reference VelocityFlow encoder, dodge/observe.py.

The engine computes the observation natively; this module is the
specification it is checked against (tests/test_dodge_env.py). These tests pin
the specification itself on hand-built game states.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from dodge.observe import (
    CELL_COUNT,
    CELL_TYPES,
    ENEMY_BLAST,
    ENEMY_NORMAL,
    GRID_SIZE,
    HAZARD_LETHAL,
    HAZARD_WARNING,
    OBSERVATION_SIZE,
    PLAYER_VALUES,
    VELOCITY_SCALE,
    cell_grid,
    observe,
)


class LuaList:
    """1-based indexing, as the cartridge's tables are read."""

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index - 1]


ENEMY_DEFAULTS = dict(vx=0.0, vy=0.0, spd=1.0)
RECT_DEFAULTS = dict(sh=2, spd=0.0, dx=0, dy=0)


def world(enemies=(), cp=None, **values):
    """Game globals with the given enemies and active pattern."""
    pattern = None
    if cp is not None:
        rects = []
        for rect in cp["rects"]:
            rect = {**RECT_DEFAULTS, **rect}
            if "warnings" in rect:
                rect["warnings"] = LuaList(LuaList(line) for line in rect["warnings"])
            rects.append(SimpleNamespace(**rect))
        pattern = SimpleNamespace(tpe=cp["tpe"], timer=cp.get("timer", 200),
                                  rects=LuaList(rects))
    defaults = dict(px=64.0, py=64.0, dpx=0.0, dpy=0.0, spd=1.0, frz=1.0)
    return SimpleNamespace(
        **{**defaults, **values},
        enemies=LuaList(SimpleNamespace(**{**ENEMY_DEFAULTS, **e}) for e in enemies),
        cp=pattern)


def test_grid_has_1024_cells():
    assert GRID_SIZE == 32
    assert CELL_COUNT == 1024
    assert PLAYER_VALUES == 4
    assert OBSERVATION_SIZE == PLAYER_VALUES + 3 * CELL_COUNT == 3076


def test_observation_starts_with_player_position_then_velocity():
    state = world(px=31.75, py=95.25, dpx=-2.0, dpy=1.0)
    result = observe(SimpleNamespace(globals=state))
    assert result.shape == (OBSERVATION_SIZE,)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(0.25)
    assert result[1] == pytest.approx(0.75)
    assert result[2] == pytest.approx(-2.0 / VELOCITY_SCALE)
    assert result[3] == pytest.approx(1.0 / VELOCITY_SCALE)
    # Positions are in [0, 1]; velocities are signed, so the box is [-1, 1].
    assert np.all((-1 <= result) & (result <= 1))


def test_velocity_planes_follow_the_entity_that_owns_the_cell():
    state = world([dict(x=8, y=12, s=4, p=0, vx=0.5, vy=-0.25)], spd=2.0)
    grid, vx, vy = cell_grid(state, with_velocity=True)

    # The cart moves enemies by vx * spd * frz * e.spd, not by vx alone.
    assert vx[3, 2] == pytest.approx(0.5 * 2.0)
    assert vy[3, 2] == pytest.approx(-0.25 * 2.0)
    assert np.count_nonzero(grid) == 1
    assert np.count_nonzero(vx) == 1, "only the occupied cell carries velocity"


def test_enemy_rectangle_paints_touched_cell():
    grid = cell_grid(world([dict(x=8, y=12, s=4, p=0)]))
    assert grid[3, 2] == 1
    assert np.count_nonzero(grid) == 1


def test_lethal_pattern_wins_when_cells_overlap_enemy():
    grid = cell_grid(world([dict(x=8, y=12, s=4, p=0)],
                           cp=dict(tpe=0, rects=[dict(x=8, y=12, w=4, h=4, sh=2)])))
    assert grid[3, 2] == HAZARD_LETHAL


def test_paint_priority_covers_every_entity():
    """A category with no rank would raise the moment it was painted."""
    from dodge.observe import ENEMY_CELL, PAINT_PRIORITY
    for value in ENEMY_CELL.values():
        assert value in PAINT_PRIORITY
    assert HAZARD_WARNING in PAINT_PRIORITY
    assert HAZARD_LETHAL in PAINT_PRIORITY
    assert max(PAINT_PRIORITY) == CELL_TYPES


def test_lethal_outranks_harmless_in_every_overlap():
    """Ranking by raw ID hid a lethal blast behind a pickup, and a lethal
    enemy behind a harmless warning telegraph."""
    from dodge.observe import (ENEMY_KAMIKAZE, PAINT_PRIORITY, PICKUP_EXPLODE,
                               PICKUP_FREEZE, PICKUP_SHRINK)
    lethal = (HAZARD_LETHAL, ENEMY_BLAST, ENEMY_KAMIKAZE, ENEMY_NORMAL)
    harmless = (HAZARD_WARNING, PICKUP_EXPLODE, PICKUP_FREEZE, PICKUP_SHRINK)
    for danger in lethal:
        for safe in harmless:
            assert PAINT_PRIORITY[danger] > PAINT_PRIORITY[safe], (
                f"{danger} must not be hidden behind {safe}")


@pytest.mark.parametrize("enemies,cp,expected", [
    # A blast passes through other entities, so it is one of only two
    # overlaps the cart can actually produce.
    ([dict(x=60, y=60, s=20, p=-1), dict(x=64, y=64, s=8, p=3)], None, ENEMY_BLAST),
    # A pattern only destroys enemies once lethal, so enemies sit inside its
    # warning telegraph -- the other reachable overlap.
    ([dict(x=64, y=64, s=6, p=0)],
     dict(tpe=0, rects=[dict(x=48, y=48, w=32, h=32, sh=1)]), ENEMY_NORMAL),
    ([dict(x=64, y=64, s=6, p=0)],
     dict(tpe=0, rects=[dict(x=48, y=48, w=32, h=32, sh=2)]), HAZARD_LETHAL),
])
def test_overlapping_cells_report_the_dangerous_entity(enemies, cp, expected):
    assert int(cell_grid(world(enemies, cp))[16, 16]) == expected


INCOMING = dict(x=32, y=131, w=64, h=24, sh=0.5, spd=0.7, dx=0, dy=-1)


def test_an_incoming_pattern_telegraph_reaches_the_observation():
    """A side-entering bar sits off-screen for its whole telegraph.

    The cart grows `sh` from 0 to 2 before a rect starts moving, and a tpe==1
    rect spawns outside the arena -- so painting the rect alone left the agent
    an empty arena for the 105 frames the warning was on screen. The warning
    lines are drawn inside the arena edge, which is what has to be encoded.
    """
    grid = cell_grid(world(cp=dict(tpe=1, timer=1, rects=[
        dict(INCOMING, warnings=[(40, 125, 40, 125), (88, 125, 88, 125)])])))
    warned = np.argwhere(grid == HAZARD_WARNING)

    assert len(warned) > 0, "the telegraph is invisible to the agent"
    assert {int(row) for row, _ in warned} == {31}, "125 // 4 -- the bottom row"
    assert {int(col) for _, col in warned} == {10, 22}, "40 // 4 and 88 // 4"
    assert not (grid == HAZARD_LETHAL).any(), "nothing lethal is on screen yet"


def test_a_telegraph_carries_the_direction_it_will_arrive_from():
    """Where is half the cue; which way it is coming is the other half."""
    grid, vx, vy = cell_grid(world(cp=dict(tpe=1, timer=1, rects=[
        dict(INCOMING, warnings=[(40, 125, 88, 125)])])), with_velocity=True)
    cells = grid == HAZARD_WARNING

    assert cells.any()
    assert np.all(vy[cells] == pytest.approx(-0.7)), "sweeping upward"
    assert np.all(vx[cells] == 0.0)


def test_the_telegraph_stops_being_encoded_when_the_cart_stops_drawing_it():
    """The cart draws warnings only while cp.timer < 125. The agent is given
    what is on screen, for as long as it is on screen -- no more."""
    grid = cell_grid(world(cp=dict(tpe=1, timer=125, rects=[
        dict(INCOMING, warnings=[(40, 125, 88, 125)])])))
    assert not (grid == HAZARD_WARNING).any()


def test_a_real_side_entering_pattern_is_no_longer_a_blind_window():
    """The regression, driven through the engine rather than a synthetic state.

    Enemies are switched off so the idle player survives the ~420 frames it
    takes a pattern to appear.
    """
    from dodge.env import make_engine, native_available
    if not native_available():
        pytest.skip("dodge_native not installed")

    engine = make_engine(difficulty=1, patterns=True, powerups=False,
                         normal_enemies=False, kamikaze_enemies=False)
    engine.reset_ml_batch(np.array([3], dtype=np.uint32))
    # Wander so the anti-camping check never fires either.
    for frame in range(8000):
        engine.step_ml_batch(np.array([1 if frame % 120 < 60 else 2], dtype=np.uint8))
        state = engine.lane_state(0)
        pattern = state["cp"]
        if pattern is None or pattern["tpe"] != 1 or pattern["timer"] >= 125:
            continue
        rect = pattern["rects"][0]
        if rect["sh"] >= 2 or not rect["warnings"]:
            continue
        assert not (0 <= rect["x"] < 128 and 0 <= rect["y"] < 128), \
            "the setup under test: the rect itself is still off-screen"
        categories = engine.observe_velocity()["observation"][0][
            PLAYER_VALUES:PLAYER_VALUES + CELL_COUNT] * CELL_TYPES
        assert np.isclose(categories, HAZARD_WARNING).any(), \
            "a telegraph the player can see must reach the observation"
        return
    pytest.fail("no side-entering pattern reached its telegraph in 8000 frames")
