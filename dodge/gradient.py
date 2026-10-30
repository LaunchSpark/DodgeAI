"""An analytic danger field over the arena, and a controller that follows it.

This is the hand-written half of the potential-field idea. The field assigns a
danger score to every point in the arena; the controller simulates where each
of the nine actions would take the player and picks the safest path. Nothing
here is learned.

It exists to answer two questions before anything is trained. Can a potential
field play this game at all -- or does it walk into local minima and die? And
what does a good score look like, which nothing in this project currently
knows: 6.67s of survival might be poor or might be near the ceiling, and every
architecture comparison is unreadable until that is pinned down.

The shape is deliberately the one a network could take over. ``potential_field``
returns a 32x32 grid on the same lattice as the CNN observation, and
``action_scores`` only ever reads the field at candidate positions. Replacing
the analytic field with a convolutional one that emits the same 32x32 output
leaves the controller untouched.

Physics constants are the cart's own, from dodge.lua:
  player  dpx *= 0.8 then += 0.5 per held direction, px += dpx, clamped to 2..125
  enemy   vx  *= 0.99, += 0.01 toward the player, x += vx * spd * frz * e.spd
"""

from dataclasses import dataclass, field as dataclass_field

import numpy as np

from .actions import ACTION_MASKS
from .observe import CELL_SIZE, GRID_SIZE

ARENA_MIN, ARENA_MAX = 2.0, 125.0

PLAYER_FRICTION = 0.8
PLAYER_ACCELERATION = 0.5
ENEMY_FRICTION = 0.99
ENEMY_HOMING = 0.01

# Button order is LEFT, RIGHT, UP, DOWN -- dirx/diry in dodge.lua:58.
BUTTON_DIRECTIONS = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))

NORMAL, KAMIKAZE, BLAST = 0, 1, -1

# The cart punishes camping (dodge.lua:711). Linger with the player box
# overlapping 60..68 on both axes for CAMP_FRAMES frames and it spawns a fresh
# enemy at every spawn point, then resets the timer and starts again.
#
# This is the whole game's answer to standing still, and it is invisible in the
# observation: nothing about four enemies appearing says they were summoned.
# Without it the field's safest point is the middle of the arena, which is
# precisely the square the cart is watching.
CAMP_BOX_MIN, CAMP_BOX_MAX = 60.0, 68.0
CAMP_FRAMES = 50


@dataclass(frozen=True)
class ControllerConfig:
    """Everything tunable, in one place, so a sweep can vary it."""

    horizon: int = 24              # frames of lookahead
    decay: float = 0.94            # per-frame discount on future danger
    margin: float = 5.0            # padding on the lethal contact radius
    wall_weight: float = 2.0
    wall_scale: float = 14.0       # px over which wall danger falls away
    camp_weight: float = 12.0      # charged once per anti-camp trigger
    pickup_weight: float = -0.35   # negative: pickups attract
    threat_weights: dict = dataclass_field(default_factory=lambda: {
        NORMAL: 1.0,
        KAMIKAZE: 1.4,     # accelerates in, and explodes when it lands
        BLAST: 1.8,        # already lethal, and passes through other enemies
    })


@dataclass(frozen=True)
class GameState:
    px: float
    py: float
    dpx: float
    dpy: float
    size: float
    speed: float                   # the cart's global difficulty speed
    freeze: float
    camp_timer: float              # frames already spent inside the camp box
    camp_armed: bool               # patterns disarm the check while they run
    # One row per enemy: x, y, vx, vy, s, p, per-enemy speed.
    enemies: np.ndarray
    # One row per lethal pattern rectangle: x, y, w, h.
    hazards: np.ndarray


EMPTY_ENEMIES = np.zeros((0, 7), dtype=np.float64)
EMPTY_HAZARDS = np.zeros((0, 4), dtype=np.float64)


def read_state(state):
    """The exact game state the field needs, from the engine's `lane_state`."""
    rows = [(enemy["x"], enemy["y"], enemy["vx"], enemy["vy"],
             enemy["s"], enemy["p"], enemy["spd"]) for enemy in state["enemies"]]
    enemies = np.array(rows, dtype=np.float64) if rows else EMPTY_ENEMIES

    hazards = EMPTY_HAZARDS
    pattern = state["cp"]
    if pattern is not None:
        lethal_pattern = int(pattern["tpe"]) == 1
        lethal = [(r["x"], r["y"], r["w"], r["h"]) for r in pattern["rects"]
                  if lethal_pattern or r["sh"] == 2]
        if lethal:
            hazards = np.array(lethal, dtype=np.float64)

    return GameState(
        px=state["px"], py=state["py"],
        dpx=state["dpx"], dpy=state["dpy"],
        size=state["size"],
        speed=state["spd"], freeze=state["frz"],
        camp_timer=float(state["ftimer"]), camp_armed=bool(state["fy"]),
        enemies=enemies, hazards=hazards,
    )


def camp_penalty(path, state, config):
    """Cost of the anti-camp spawns this path would trigger.

    The timer is carried forward from the game's own, so a path is judged by
    what it does to a count that is already running rather than from zero.
    """
    if not state.camp_armed or not config.camp_weight:
        return 0.0
    low = CAMP_BOX_MIN - state.size + 2.0
    high = CAMP_BOX_MAX + state.size - 2.0
    inside = ((path[:, 0] > low) & (path[:, 0] < high)
              & (path[:, 1] > low) & (path[:, 1] < high))

    timer, penalty = state.camp_timer, 0.0
    for frame, is_inside in enumerate(inside):
        timer = timer + 1 if is_inside else 0
        # Graded by how far the timer has run, not charged only when it trips.
        # A horizon shorter than CAMP_FRAMES can only ever see the trip from
        # inside the last few frames, by which point the bot has already spent
        # most of a timer's worth of frames in the box.
        penalty += (config.camp_weight * (timer / CAMP_FRAMES) ** 2
                    * config.decay ** frame)
        if timer > CAMP_FRAMES:
            timer = 0
    return penalty


def predict_enemies(state, player_path):
    """Where each enemy will be, given where the player is going.

    Homing must track the *candidate* path, not the player's position at
    decision time. Freezing it there quietly rewards running directly away
    from a pursuer: the predicted enemy carries on toward the spot the player
    has left, so fleeing along its heading scores as an escape. It is not one
    -- the real enemy re-aims every frame, so the player stays parked in front
    of it and is caught. Against a homing chaser the escape is lateral, and a
    frozen-target model cannot see that.

    This makes the prediction depend on the action being scored, which is why
    ``action_scores`` simulates enemies once per action rather than once.
    """
    enemies = state.enemies
    horizon = len(player_path)
    if not len(enemies):
        return np.zeros((horizon, 0)), np.zeros((horizon, 0))

    x, y = enemies[:, 0].copy(), enemies[:, 1].copy()
    vx, vy = enemies[:, 2].copy(), enemies[:, 3].copy()
    size, personality = enemies[:, 4], enemies[:, 5]
    step = state.speed * state.freeze * enemies[:, 6]
    # Only p >= 0 chase; a kamikaze blast (p == -1) keeps its heading.
    homes = personality >= 0

    xs = np.empty((horizon, len(enemies)))
    ys = np.empty((horizon, len(enemies)))
    for frame in range(horizon):
        target_x, target_y = player_path[frame]
        vx *= ENEMY_FRICTION
        vy *= ENEMY_FRICTION
        vx += np.where(homes, np.sign(target_x - x) * ENEMY_HOMING, 0.0)
        vy += np.where(homes, np.sign(target_y - y) * ENEMY_HOMING, 0.0)
        # Enemies bounce off the arena edge once they are inside it.
        vx = np.where(x >= 129 - size, -np.abs(vx),
                      np.where(x <= -2, np.abs(vx), vx))
        vy = np.where(y >= 129 - size, -np.abs(vy),
                      np.where(y <= -2, np.abs(vy), vy))
        x = x + vx * step
        y = y + vy * step
        xs[frame], ys[frame] = x, y
    return xs, ys


def _threat_weights(personalities, config):
    weights = np.full(len(personalities), config.pickup_weight)
    for kind, weight in config.threat_weights.items():
        weights[personalities == kind] = weight
    return weights


def danger_along(player_path, state, config=ControllerConfig(), enemy_tracks=None):
    """Danger the player meets at each frame of ``player_path``.

    Frame ``t`` of the path is judged against where the threats will be at
    frame ``t``, so a route is scored against a moving world rather than a
    snapshot of it.
    """
    points = np.asarray(player_path, dtype=np.float64).reshape(-1, 2)
    total = np.zeros(len(points))
    frames = np.arange(len(points))

    if len(state.enemies):
        xs, ys = (predict_enemies(state, points) if enemy_tracks is None
                  else enemy_tracks)
        weights = _threat_weights(state.enemies[:, 5], config)
        # Contact is rectangle overlap, so the radius that matters is the sum
        # of the half-extents plus a margin for reaction time.
        radius = state.enemies[:, 4] / 2.0 + state.size + config.margin
        dx = points[:, 0:1] - (xs + state.enemies[:, 4] / 2.0)
        dy = points[:, 1:2] - (ys + state.enemies[:, 4] / 2.0)
        squared = dx * dx + dy * dy
        influence = np.exp(-squared / (2.0 * radius * radius))
        total += (influence * weights * (config.decay ** frames[:, None])).sum(axis=1)

    for hx, hy, hw, hh in state.hazards:
        # A rectangle is dangerous inside and just outside it, not at a centre.
        gap_x = np.maximum(np.maximum(hx - points[:, 0], points[:, 0] - (hx + hw)), 0.0)
        gap_y = np.maximum(np.maximum(hy - points[:, 1], points[:, 1] - (hy + hh)), 0.0)
        gap = np.hypot(gap_x, gap_y)
        total += 2.0 * np.exp(-gap / max(config.margin, 1e-6))

    if config.wall_weight:
        edge = np.minimum(
            np.minimum(points[:, 0] - ARENA_MIN, ARENA_MAX - points[:, 0]),
            np.minimum(points[:, 1] - ARENA_MIN, ARENA_MAX - points[:, 1]),
        )
        total += config.wall_weight * np.exp(-np.maximum(edge, 0.0)
                                             / config.wall_scale)
    return total


def potential_field(state, config=ControllerConfig()):
    """The danger of standing at each cell, on the CNN's own 32x32 lattice.

    For rendering and as the target a convolutional field would have to
    produce. Note what it cannot express: every cell is scored as though the
    player stayed there, so the field says nothing about pursuit. Two threats
    at equal distance look equally bad whether they are closing or leaving,
    and "step sideways, because it is chasing you" has no representation in a
    scalar over positions at all. The controller does not read this -- it
    scores whole trajectories -- and any learned field would inherit the same
    blind spot.
    """
    centres = (np.arange(GRID_SIZE) + 0.5) * CELL_SIZE
    grid_x, grid_y = np.meshgrid(centres, centres)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    stationary = np.repeat([[state.px, state.py]], config.horizon, axis=0)
    tracks = predict_enemies(state, stationary)
    danger = np.array([
        danger_along(np.repeat([point], config.horizon, axis=0), state, config,
                     enemy_tracks=tracks).sum()
        for point in points
    ])
    return danger.reshape(GRID_SIZE, GRID_SIZE)


def simulate_player(px, py, dpx, dpy, mask, frames, held=None):
    """Replay the cart's own movement, holding the mask then coasting.

    The button is held for ``held`` frames and released for the rest, because
    that is what actually happens: the environment commits an action for the
    decision interval and then chooses again. Holding it for the whole horizon
    instead makes every direction end against a wall, which quietly rules out
    moving at all -- the first version of this scored idle best 83% of the time
    for exactly that reason.

    Returns a position per frame. The clamp matches dodge.lua:316, which pins
    the position without clearing the velocity, so a player held against a wall
    keeps the speed it would need to leave.
    """
    if held is None:
        held = frames
    ax = ay = 0.0
    for button, (dx, dy) in enumerate(BUTTON_DIRECTIONS):
        if mask & (1 << button):
            ax += dx * PLAYER_ACCELERATION
            ay += dy * PLAYER_ACCELERATION

    path = np.empty((frames, 2))
    for frame in range(frames):
        accel_x, accel_y = (ax, ay) if frame < held else (0.0, 0.0)
        dpx = dpx * PLAYER_FRICTION + accel_x
        dpy = dpy * PLAYER_FRICTION + accel_y
        px = min(max(px + dpx, ARENA_MIN), ARENA_MAX)
        py = min(max(py + dpy, ARENA_MIN), ARENA_MAX)
        path[frame] = (px, py)
    return path


def action_scores(state, config=ControllerConfig(), action_repeat=4):
    """Danger along the path each action takes, lowest is best.

    The whole path is scored rather than only where it ends, so an action
    cannot win by passing through a threat and coming out the far side. Every
    action is simulated past the decision interval as well, since a move that
    is safe for four frames and cornered by frame twenty is not safe.
    """
    frames = max(action_repeat, config.horizon)
    scores = np.empty(len(ACTION_MASKS))
    for action, mask in enumerate(ACTION_MASKS):
        path = simulate_player(state.px, state.py, state.dpx, state.dpy,
                               mask, frames, held=action_repeat)
        # Enemies are re-simulated per action, since they chase this path.
        along = danger_along(path, state, config)
        scores[action] = along.sum() + camp_penalty(path, state, config)
    return scores


def choose_action(state, config=ControllerConfig(), action_repeat=4):
    """The safest of the nine actions. Ties go to the lowest index, which is idle."""
    return int(np.argmin(action_scores(state, config, action_repeat)))


class GradientBot:
    """Plays Dodge from the danger field. No learning, no observation stack."""

    def __init__(self, config=ControllerConfig(), action_repeat=4):
        self.config = config
        self.action_repeat = action_repeat

    def act(self, state):
        """The action for one game, given its engine `lane_state`."""
        return choose_action(read_state(state), self.config, self.action_repeat)

    def policy(self, observations, lane_state):
        """Act for every lane of a batch; the shape `play_games` expects."""
        return [self.act(lane_state(lane)) for lane in range(len(observations))]
