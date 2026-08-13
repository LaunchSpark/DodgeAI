"""Convert Dodge's Lua state into a small, fixed PPO observation."""

import math

import numpy as np


ARENA_SIZE = 128
CELL_SIZE = 4
GRID_SIZE = ARENA_SIZE // CELL_SIZE
CELL_COUNT = GRID_SIZE * GRID_SIZE
CELL_TYPES = 8
FRAME_OBSERVATION_SIZE = 2 + CELL_COUNT
FRAME_STACK_SIZE = 2

# How many decisions back the older of the two observed frames is taken from.
#
# The observation is still two frames; this is the gap between them. At one
# decision (4 game frames) every moving thing in the game covers less than a
# single 4px cell -- enemies 0.54-0.74 cells, pattern bars 0.70, the player
# 0.80 -- so the categorical grid usually shows no change at all and hazard
# motion is quantised out of the observation entirely. At two decisions
# everything lands at 1.1-1.6 cells, which is unambiguous, while the worst case
# in the game (the player at p99 on hard) is only 3.9 cells, far short of any
# correspondence problem.
#
# The player's own motion is exact either way, since its coordinates are floats.
# This is entirely about enemies and pattern bars, whose movement is visible
# only through the grid.
DECISION_LOOKBACK = 2

OBSERVATION_SIZE = FRAME_OBSERVATION_SIZE * FRAME_STACK_SIZE

EMPTY = 0
ENEMY_NORMAL = 1
ENEMY_KAMIKAZE = 2
ENEMY_BLAST = 3
PICKUP_EXPLODE = 4
PICKUP_FREEZE = 5
PICKUP_SHRINK = 6
HAZARD_WARNING = 7
HAZARD_LETHAL = 8

# Which entity survives when two want the same cell, ranked by how much it
# matters to staying alive. Kept separate from the IDs above so the channel a
# category occupies never moves: reordering the IDs would silently repoint every
# trained filter at a different entity type.
#
# Only two overlaps can actually persist in the cart. A kamikaze blast is
# excluded from enemy-vs-enemy collision (dodge.lua:1017) so it passes through
# other entities, and a pattern only destroys enemies once it turns lethal
# (dodge.lua:2099) so enemies sit inside its warning telegraph. Everything else
# annihilates on contact and cannot overlap. Ranking by raw ID got both of those
# backwards: a pickup outranked a lethal blast, and a harmless warning outranked
# a lethal enemy.
PAINT_PRIORITY = {
    HAZARD_LETHAL: 6,
    ENEMY_BLAST: 5,
    ENEMY_KAMIKAZE: 4,
    ENEMY_NORMAL: 3,
    HAZARD_WARNING: 2,
    PICKUP_EXPLODE: 1,
    PICKUP_FREEZE: 1,
    PICKUP_SHRINK: 1,
}

ENEMY_CELL = {
    0: ENEMY_NORMAL,
    1: ENEMY_KAMIKAZE,
    -1: ENEMY_BLAST,
    2: PICKUP_EXPLODE,
    3: PICKUP_FREEZE,
    4: PICKUP_SHRINK,
}


def _paint(grid, ranks, x, y, width, height, value):
    """Paint every 4px cell touched by an axis-aligned rectangle.

    A cell keeps whichever entity ranks highest in PAINT_PRIORITY, so a lethal
    threat is never hidden behind something harmless.
    """
    if width <= 0 or height <= 0:
        return
    left = max(0, math.floor(x / CELL_SIZE))
    top = max(0, math.floor(y / CELL_SIZE))
    right = min(GRID_SIZE - 1, math.floor((x + width - 1e-9) / CELL_SIZE))
    bottom = min(GRID_SIZE - 1, math.floor((y + height - 1e-9) / CELL_SIZE))
    if left > right or top > bottom:
        return
    rank = PAINT_PRIORITY[value]
    view = grid[top:bottom + 1, left:right + 1]
    rank_view = ranks[top:bottom + 1, left:right + 1]
    replace = rank_view < rank
    view[replace] = value
    rank_view[replace] = rank


def cell_grid(globals_):
    """Return the 32x32 categorical map before normalization."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    ranks = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    enemies = globals_.enemies
    for index in range(1, len(enemies) + 1):
        enemy = enemies[index]
        personality = int(enemy.p)
        value = ENEMY_CELL.get(personality, ENEMY_NORMAL)
        if personality >= 2:
            _paint(grid, ranks, float(enemy.x) - 4, float(enemy.y) - 4,
                   8, 8, value)
        else:
            size = float(enemy.s)
            _paint(grid, ranks, float(enemy.x), float(enemy.y),
                   size, size, value)

    pattern = globals_.cp
    if pattern is not None:
        lethal_pattern = int(pattern.tpe) == 1
        rects = pattern.rects
        for index in range(1, len(rects) + 1):
            rect = rects[index]
            lethal = lethal_pattern or float(rect.sh) == 2
            value = HAZARD_LETHAL if lethal else HAZARD_WARNING
            _paint(grid, ranks, float(rect.x), float(rect.y),
                   float(rect.w), float(rect.h), value)
    return grid


def observe_frame(machine):
    """Return [player_x, player_y, 1024 normalized cell values]."""
    globals_ = machine.globals
    observation = np.empty(FRAME_OBSERVATION_SIZE, dtype=np.float32)
    observation[0] = np.clip(float(globals_.px) / 127.0, 0.0, 1.0)
    observation[1] = np.clip(float(globals_.py) / 127.0, 0.0, 1.0)
    observation[2:] = cell_grid(globals_).reshape(-1).astype(np.float32) / CELL_TYPES
    return observation


def observe(machine):
    """Return a reset-safe stack with the current game frame duplicated."""
    frame = observe_frame(machine)
    return np.concatenate((frame, frame))
