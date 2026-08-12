import faulthandler
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lupa.luajit21")

from dodge.observe import (
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

pytestmark = pytest.mark.skipif(not Path("logic.js").exists(), reason="logic.js not present")


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
