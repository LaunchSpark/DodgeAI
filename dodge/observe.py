"""The VelocityFlow observation, as a reference encoder.

The engine computes this observation natively (`observe_velocity`); this module
is the specification it is tested against, reading game state under the
cartridge's own names.

One frame, not a stack. Motion is supplied directly: every occupied cell also
carries the velocity of whatever occupies it, read from the game rather than
inferred.

Inference was the problem. A two-frame stack asks the network to recover motion
from the difference between two categorical grids, and everything in this game
moves less than one 4px cell per decision -- enemies 0.54-0.74, pattern bars
0.70, the player 0.80 -- so the usual outcome was no visible change at all.
Widening the gap to two decisions helped but did not fix it: differencing still
conflates everything in a region, so two enemies crossing near each other
cannot keep separate velocities, and sub-cell motion is still lost.

A velocity raster has neither failure. It is exact, it is per entity, and it
costs about 120us a step against the ~2000us a physics forecast would need.
Empty cells hold zero, which is unambiguous because the category channels say
separately whether a cell is occupied at all.
"""

import math

import numpy as np


ARENA_SIZE = 128
CELL_SIZE = 4
GRID_SIZE = ARENA_SIZE // CELL_SIZE
CELL_COUNT = GRID_SIZE * GRID_SIZE
CELL_TYPES = 8

# px, py, dpx, dpy. The player's own velocity has to be given explicitly now
# that there is no second frame to derive it from -- without it the agent would
# not know its own momentum, which matters in a game with friction.
PLAYER_VALUES = 4

# Divisor putting velocities in roughly [-1, 1]. The fastest thing in the game
# is the player at its terminal 2.5px a frame; enemies reach about 3 on hard.
VELOCITY_SCALE = 4.0

# player values, then the categorical grid, then one velocity plane per axis.
OBSERVATION_SIZE = PLAYER_VALUES + 3 * CELL_COUNT

EMPTY = 0
ENEMY_NORMAL = 1
ENEMY_KAMIKAZE = 2
ENEMY_BLAST = 3
PICKUP_EXPLODE = 4
PICKUP_FREEZE = 5
PICKUP_SHRINK = 6
HAZARD_WARNING = 7
HAZARD_LETHAL = 8

# How long the cart draws a side-entering pattern's warning lines
# (dodge.lua:1240). The observation mirrors that window exactly: the agent is
# given the cue for as long as it is on screen and no longer.
WARNING_FRAMES = 125

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


def _paint(grid, ranks, x, y, width, height, value, velocity=None,
           planes=None):
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
    if planes is not None:
        # The winner of the cell supplies its velocity too, so the two always
        # describe the same entity.
        planes[0][top:bottom + 1, left:right + 1][replace] = velocity[0]
        planes[1][top:bottom + 1, left:right + 1][replace] = velocity[1]


def _paint_warnings(grid, ranks, rect, velocity, planes):
    """Paint the warning segments the cart draws for a rect still off-screen.

    A side-entering rect spends its whole telegraph parked outside the arena --
    the cart grows `sh` from 0 to 2 before a rect moves at all, and tpe==1
    rects spawn beyond the edge -- so _paint clamps it away and the agent saw
    an empty arena for the ~105 frames the player spent looking at a warning.
    The segments sit just inside the edge and grow outward as `sh` rises, so
    they carry where the bar will enter and roughly how soon.
    """
    warnings = getattr(rect, "warnings", None)
    if warnings is None:
        return
    for index in range(1, len(warnings) + 1):
        line = warnings[index]
        x1, y1 = float(line[1]), float(line[2])
        x2, y2 = float(line[3]), float(line[4])
        # Axis-aligned, and zero-length at the start: a fresh telegraph is one
        # dot that stretches toward the width of what is coming.
        _paint(grid, ranks, min(x1, x2), min(y1, y2),
               max(abs(x2 - x1), 1.0), max(abs(y2 - y1), 1.0),
               HAZARD_WARNING, velocity, planes)


def cell_grid(globals_, with_velocity=False):
    """The 32x32 categorical map, optionally with a velocity plane per axis.

    Velocities are the entity's real displacement per frame -- the cart moves
    enemies by ``vx * spd * frz * e.spd`` (dodge.lua:1043), not by vx alone --
    so the number means pixels, whatever the difficulty is doing to the global
    speed.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    ranks = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    planes = ((np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32),
               np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32))
              if with_velocity else None)
    step = float(globals_.spd) * float(globals_.frz) if with_velocity else 0.0

    enemies = globals_.enemies
    for index in range(1, len(enemies) + 1):
        enemy = enemies[index]
        personality = int(enemy.p)
        value = ENEMY_CELL.get(personality, ENEMY_NORMAL)
        velocity = None
        if with_velocity:
            own = step * float(enemy.spd)
            velocity = (float(enemy.vx) * own, float(enemy.vy) * own)
        if personality >= 2:
            _paint(grid, ranks, float(enemy.x) - 4, float(enemy.y) - 4,
                   8, 8, value, velocity, planes)
        else:
            size = float(enemy.s)
            _paint(grid, ranks, float(enemy.x), float(enemy.y),
                   size, size, value, velocity, planes)

    pattern = globals_.cp
    if pattern is not None:
        lethal_pattern = int(pattern.tpe) == 1
        telegraphed = float(pattern.timer or 0) < WARNING_FRAMES
        rects = pattern.rects
        for index in range(1, len(rects) + 1):
            rect = rects[index]
            lethal = lethal_pattern or float(rect.sh) == 2
            value = HAZARD_LETHAL if lethal else HAZARD_WARNING
            velocity = None
            if with_velocity:
                # Pattern rects carry dx/dy as a direction and spd as a rate.
                rate = float(rect.spd) if rect.spd is not None else 0.0
                velocity = (float(rect.dx or 0) * rate * float(globals_.frz),
                            float(rect.dy or 0) * rate * float(globals_.frz))
            _paint(grid, ranks, float(rect.x), float(rect.y),
                   float(rect.w), float(rect.h), value, velocity, planes)
            if telegraphed:
                _paint_warnings(grid, ranks, rect, velocity, planes)
    if with_velocity:
        return grid, planes[0], planes[1]
    return grid


def observe(machine):
    """The whole observation: player state, categories, then the two velocity planes.

    Layout, in order:
        0..3        px, py, dpx, dpy  -- position normalised, velocity scaled
        4..1027     categorical cell values, normalised by CELL_TYPES
        1028..2051  vx of whatever occupies each cell
        2052..3075  vy of the same
    """
    globals_ = machine.globals
    observation = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    observation[0] = np.clip(float(globals_.px) / 127.0, 0.0, 1.0)
    observation[1] = np.clip(float(globals_.py) / 127.0, 0.0, 1.0)
    observation[2] = np.clip(float(globals_.dpx) / VELOCITY_SCALE, -1.0, 1.0)
    observation[3] = np.clip(float(globals_.dpy) / VELOCITY_SCALE, -1.0, 1.0)

    grid, vx, vy = cell_grid(globals_, with_velocity=True)
    start = PLAYER_VALUES
    observation[start:start + CELL_COUNT] =         grid.reshape(-1).astype(np.float32) / CELL_TYPES
    observation[start + CELL_COUNT:start + 2 * CELL_COUNT] =         np.clip(vx.reshape(-1) / VELOCITY_SCALE, -1.0, 1.0)
    observation[start + 2 * CELL_COUNT:] =         np.clip(vy.reshape(-1) / VELOCITY_SCALE, -1.0, 1.0)
    return observation
