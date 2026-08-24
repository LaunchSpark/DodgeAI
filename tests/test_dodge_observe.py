import faulthandler
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lupa.luajit21")

from dodge.observe import (
    ENEMY_BLAST,
    ENEMY_NORMAL,
    CELL_COUNT,
    PLAYER_VALUES,
    VELOCITY_SCALE,
    GRID_SIZE,
    HAZARD_LETHAL,
    OBSERVATION_SIZE,
    cell_grid,
    observe,
)
from pico8.backends import NullBackend
from pico8.machine import boot

pytestmark = pytest.mark.skipif(not Path("dodge.p8.png").exists(), reason="dodge.p8.png not present")


@pytest.fixture(scope="module", autouse=True)
def _quiet_luajit_seh():
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


@pytest.fixture
def game():
    machine = boot(backend=NullBackend(), seed=3)
    machine.start_game(patterns=True)
    return machine


def test_grid_has_1024_cells():
    assert GRID_SIZE == 32
    assert CELL_COUNT == 1024
    assert PLAYER_VALUES == 4
    assert OBSERVATION_SIZE == PLAYER_VALUES + 3 * CELL_COUNT == 3076


def test_observation_starts_with_player_position_then_velocity(game):
    game.globals.px = 31.75
    game.globals.py = 95.25
    game.globals.dpx = -2.0
    game.globals.dpy = 1.0
    result = observe(game)
    assert result.shape == (OBSERVATION_SIZE,)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(0.25)
    assert result[1] == pytest.approx(0.75)
    assert result[2] == pytest.approx(-2.0 / VELOCITY_SCALE)
    assert result[3] == pytest.approx(1.0 / VELOCITY_SCALE)
    # Positions are in [0, 1]; velocities are signed, so the box is [-1, 1].
    assert np.all((-1 <= result) & (result <= 1))


def test_velocity_planes_follow_the_entity_that_owns_the_cell(game):
    game.lua.execute("enemies={{x=8,y=12,s=4,p=0,vx=0.5,vy=-0.25,spd=1}} cp=nil")
    game.globals.spd = 2.0
    game.globals.frz = 1.0
    grid, vx, vy = cell_grid(game.globals, with_velocity=True)

    # The cart moves enemies by vx * spd * frz * e.spd, not by vx alone.
    assert vx[3, 2] == pytest.approx(0.5 * 2.0)
    assert vy[3, 2] == pytest.approx(-0.25 * 2.0)
    assert np.count_nonzero(grid) == 1
    assert np.count_nonzero(vx) == 1, "only the occupied cell carries velocity"


def test_enemy_rectangle_paints_touched_cell(game):
    game.lua.execute("enemies={{x=8,y=12,s=4,p=0}} cp=nil")
    grid = cell_grid(game.globals)
    assert grid[3, 2] == 1
    assert np.count_nonzero(grid) == 1


def test_lethal_pattern_wins_when_cells_overlap_enemy(game):
    game.lua.execute("""
        enemies={{x=8,y=12,s=4,p=0}}
        cp={tpe=0,rects={{x=8,y=12,w=4,h=4,sh=2}}}
    """)
    grid = cell_grid(game.globals)
    assert grid[3, 2] == HAZARD_LETHAL


def test_paint_priority_covers_every_entity():
    """A category with no rank would raise the moment it was painted."""
    from dodge.observe import (ENEMY_CELL, HAZARD_LETHAL, HAZARD_WARNING,
                               PAINT_PRIORITY)
    for value in ENEMY_CELL.values():
        assert value in PAINT_PRIORITY
    assert HAZARD_WARNING in PAINT_PRIORITY
    assert HAZARD_LETHAL in PAINT_PRIORITY


def test_lethal_outranks_harmless_in_every_overlap():
    """Ranking by raw ID hid a lethal blast behind a pickup, and a lethal
    enemy behind a harmless warning telegraph."""
    from dodge.observe import (ENEMY_BLAST, ENEMY_KAMIKAZE, ENEMY_NORMAL,
                               HAZARD_LETHAL, HAZARD_WARNING, PAINT_PRIORITY,
                               PICKUP_EXPLODE, PICKUP_FREEZE, PICKUP_SHRINK)
    lethal = (HAZARD_LETHAL, ENEMY_BLAST, ENEMY_KAMIKAZE, ENEMY_NORMAL)
    harmless = (HAZARD_WARNING, PICKUP_EXPLODE, PICKUP_FREEZE, PICKUP_SHRINK)
    for danger in lethal:
        for safe in harmless:
            assert PAINT_PRIORITY[danger] > PAINT_PRIORITY[safe], (
                f"{danger} must not be hidden behind {safe}")


@pytest.mark.skipif(not Path("dodge.p8.png").exists(), reason="cartridge absent")
@pytest.mark.parametrize("lua,expected", [
    # A blast passes through other entities (dodge.lua:1017), so it is one of
    # only two overlaps the cart can actually produce.
    ("enemies={{x=60,y=60,s=20,p=-1},{x=64,y=64,s=8,p=3}}", ENEMY_BLAST),
    # A pattern only destroys enemies once lethal (dodge.lua:2099), so enemies
    # sit inside its warning telegraph -- the other reachable overlap.
    ("enemies={{x=64,y=64,s=6,p=0}} cp={tpe=0,rects={{x=48,y=48,w=32,h=32,sh=1}}}",
     ENEMY_NORMAL),
    ("enemies={{x=64,y=64,s=6,p=0}} cp={tpe=0,rects={{x=48,y=48,w=32,h=32,sh=2}}}",
     HAZARD_LETHAL),
])
def test_overlapping_cells_report_the_dangerous_entity(lua, expected):
    from dodge.env import DodgeEnv
    from dodge.observe import cell_grid

    env = DodgeEnv(action_repeat=1)
    env.reset(seed=3)
    env.machine.lua.execute(lua)
    value = int(cell_grid(env.machine.globals)[16, 16])
    env.close()
    assert value == expected


def test_an_incoming_pattern_telegraph_reaches_the_observation(game):
    """A side-entering bar sits off-screen for its whole telegraph.

    The cart grows `sh` from 0 to 2 before a rect starts moving, and a tpe==1
    rect spawns outside the arena -- so _paint clamps it away and the agent saw
    an empty arena for the 105 frames the warning was on screen. The warning
    lines are drawn inside the arena edge, which is what has to be encoded.
    """
    from dodge.observe import HAZARD_WARNING

    game.lua.execute("""
        enemies={}
        cp={tpe=1,timer=1,rects={{
            x=32,y=131,w=64,h=24,sh=0.5,spd=0.7,dx=0,dy=-1,
            warnings={{40,125,40,125},{88,125,88,125}}
        }}}
    """)

    grid = cell_grid(game.globals)
    warned = np.argwhere(grid == HAZARD_WARNING)

    assert len(warned) > 0, "the telegraph is invisible to the agent"
    assert {int(row) for row, _ in warned} == {31}, "125 // 4 -- the bottom row"
    assert {int(col) for _, col in warned} == {10, 22}, "40 // 4 and 88 // 4"
    assert not (grid == HAZARD_LETHAL).any(), "nothing lethal is on screen yet"


def test_a_telegraph_carries_the_direction_it_will_arrive_from(game):
    """Where is half the cue; which way it is coming is the other half."""
    from dodge.observe import HAZARD_WARNING

    game.lua.execute("""
        enemies={}
        spd=1 frz=1
        cp={tpe=1,timer=1,rects={{
            x=32,y=131,w=64,h=24,sh=0.5,spd=0.7,dx=0,dy=-1,
            warnings={{40,125,88,125}}
        }}}
    """)

    grid, vx, vy = cell_grid(game.globals, with_velocity=True)
    cells = grid == HAZARD_WARNING

    assert cells.any()
    assert np.all(vy[cells] == pytest.approx(-0.7)), "sweeping upward"
    assert np.all(vx[cells] == 0.0)


def test_the_telegraph_stops_being_encoded_when_the_cart_stops_drawing_it(game):
    """The cart draws warnings only while cp.timer < 125. The agent is given
    what is on screen, for as long as it is on screen -- no more."""
    from dodge.observe import HAZARD_WARNING

    game.lua.execute("""
        enemies={}
        cp={tpe=1,timer=125,rects={{
            x=32,y=131,w=64,h=24,sh=0.5,spd=0.7,dx=0,dy=-1,
            warnings={{40,125,88,125}}
        }}}
    """)

    grid = cell_grid(game.globals)

    assert not (grid == HAZARD_WARNING).any()


def test_a_real_side_entering_pattern_is_no_longer_a_blind_window(game):
    """The regression, driven through the cart rather than a synthetic table.

    Enemies are cleared each frame only so the idle player survives the ~420
    frames it takes a pattern to appear.
    """
    from dodge.observe import HAZARD_WARNING

    for _ in range(4000):
        game.lua.execute("enemies={}")
        game.step()
        pattern = game.globals.cp
        if pattern is None or int(pattern.tpe) != 1:
            continue
        rect = pattern.rects[1]
        if float(rect.sh) >= 2 or float(pattern.timer) >= 125:
            continue
        warnings = getattr(rect, "warnings", None)
        if warnings is None or len(warnings) == 0:
            continue
        assert not (0 <= float(rect.x) < 128 and 0 <= float(rect.y) < 128), \
            "the setup under test: the rect itself is still off-screen"
        assert (cell_grid(game.globals) == HAZARD_WARNING).any(), \
            "a telegraph the player can see must reach the observation"
        return
    pytest.fail("no side-entering pattern reached its telegraph in 4000 frames")
