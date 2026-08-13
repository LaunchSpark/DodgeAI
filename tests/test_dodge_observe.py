import faulthandler
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lupa.luajit21")

from dodge.observe import (
    ENEMY_BLAST,
    ENEMY_NORMAL,
    CELL_COUNT,
    FRAME_OBSERVATION_SIZE,
    FRAME_STACK_SIZE,
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
    assert FRAME_OBSERVATION_SIZE == 1026
    assert FRAME_STACK_SIZE == 2
    assert OBSERVATION_SIZE == 2052


def test_observation_starts_with_normalized_player_coordinates(game):
    game.globals.px = 31.75
    game.globals.py = 95.25
    result = observe(game)
    assert result.shape == (2052,)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(0.25)
    assert result[1] == pytest.approx(0.75)
    np.testing.assert_array_equal(result[:1026], result[1026:])
    assert np.all((0 <= result) & (result <= 1))


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
