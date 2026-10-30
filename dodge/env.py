"""Dodge on the native Rust engine: batched games, observation and reward.

Every environment is one lane of a single Rust batch (TaylorTurnerIT's
dodge-ai-3 engine, built from the velocity-flow branch), stepped in one call at
a few hundred thousand frames a second. The engine also computes the
VelocityFlow observation; the reward is computed here.
"""

from dataclasses import asdict
import math
import secrets
import time

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from . import rewards
from .actions import ACTION_MASKS
from .seeds import ExplicitSeeds, SeedStream
from .observe import ARENA_SIZE, OBSERVATION_SIZE

try:
    import dodge_native
except ImportError:  # pragma: no cover - depends on the local build
    dodge_native = None


# How much of the half-arena pays full survival credit.
CENTRE_PLATEAU = 0.6

SPAWN_RATE_PERCENT_MIN = 50
SPAWN_RATE_PERCENT_MAX = 150

# The engine's codes for the first lethal contact of a game.
DEATH_CAUSES = {
    0: None,
    1: "normal",
    2: "kamikaze",
    3: "kamikaze_blast",
    4: "static_rect",
    5: "moving_rect",
}

INSTALL_HINT = ("dodge_native is not installed: build the velocity-flow branch of "
                "dodge-ai-3 with maturin (see README)")


def native_available():
    return dodge_native is not None and hasattr(dodge_native.NativeBatchEnv,
                                                "observe_velocity")


def validate_spawn_rate_percent(value):
    """Return an integer spawn percentage in the supported curriculum range."""
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("spawn rate must be an integer percentage") from None
    if numeric != integer:
        raise ValueError("spawn rate must be an integer percentage")
    if not SPAWN_RATE_PERCENT_MIN <= integer <= SPAWN_RATE_PERCENT_MAX:
        raise ValueError(f"spawn rate must be between {SPAWN_RATE_PERCENT_MIN}% "
                         f"and {SPAWN_RATE_PERCENT_MAX}%")
    return integer


def center_survival_scalar(x, y, screen_width=ARENA_SIZE, edge_penalty=0.5):
    """Full survival credit across the interior, tapering only near the walls.

    This used to peak at the exact centre and fall away in every direction,
    which made standing perfectly still the best-paying action available: the
    agent spawns at the centre, so any movement at all reduced its income. That
    is a reward for doing nothing, in a game about moving.

    A plateau keeps the part that was wanted -- do not cower against a wall,
    where there is nowhere left to dodge -- without paying the agent to freeze
    on one pixel. Inside the plateau every position pays the same, so position
    only matters once the agent nears an edge.
    """
    center = screen_width / 2.0
    distance = math.hypot(float(x) - center, float(y) - center) / center
    if distance <= CENTRE_PLATEAU:
        return 1.0
    # Corners sit past 1.0, so the taper is clamped rather than going negative.
    taper = min((distance - CENTRE_PLATEAU) / (1.0 - CENTRE_PLATEAU), 1.0)
    return max(0.0, 1.0 - edge_penalty * taper)


# Measured crossover of center_survival_scalars' two paths.
SCALARS_LOOP_BELOW = 24


def center_survival_scalars(x, y, screen_width=ARENA_SIZE, edge_penalty=0.5):
    """`center_survival_scalar` over arrays of positions, bit for bit.

    The distance goes through the same `math.hypot` the scalar version calls
    -- numpy's hypot is a different routine that may round the last bit
    differently -- and every other operation is the same IEEE double
    operation in the same order, so each lane gets exactly the scalar's value
    and the reward does not move by a bit.

    Below a few dozen lanes numpy's per-call overhead outweighs its speed, so
    a small batch simply calls the scalar function on each lane.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < SCALARS_LOOP_BELOW:
        return np.array([center_survival_scalar(a, b, screen_width, edge_penalty)
                         for a, b in zip(x.tolist(), y.tolist())], dtype=np.float64)
    center = screen_width / 2.0
    dx = (x - center).tolist()
    dy = (y - center).tolist()
    distance = np.fromiter(map(math.hypot, dx, dy), dtype=np.float64,
                           count=len(dx)) / center
    taper = np.minimum((distance - CENTRE_PLATEAU) / (1.0 - CENTRE_PLATEAU), 1.0)
    tapered = np.maximum(0.0, 1.0 - edge_penalty * taper)
    return np.where(distance <= CENTRE_PLATEAU, 1.0, tapered)


MAX_ACTION_REPEAT = 5


def make_engine(*, step_frames=1, difficulty=1, patterns=False, powerups=False,
                normal_enemies=True, kamikaze_enemies=True, spawn_rate_percent=100,
                lives=0, pixels=False, execution="parallel", ml=None):
    """A Rust batch with the game settings applied.

    Each step holds the action for `step_frames` frames, stopping a lane early
    at the frame it dies. `ml` (default: not `pixels`) starts games through the
    ML reset; with `pixels` too, rendered steps replay the ML engine's games.
    """
    if not native_available():
        raise RuntimeError(INSTALL_HINT)
    if int(difficulty) not in (1, 2, 3):
        raise ValueError("difficulty must be 1 (easy), 2 (normal), or 3 (hard)")
    if not 1 <= step_frames <= MAX_ACTION_REPEAT:
        raise ValueError(f"action_repeat must be between 1 and {MAX_ACTION_REPEAT}")
    if lives < 0:
        raise ValueError("lives cannot be negative")
    return dodge_native.NativeBatchEnv(
        step_frames=int(step_frames),
        execution=execution,
        board=False,
        pixels=pixels,
        ml=(not pixels) if ml is None else bool(ml),
        difficulty=int(difficulty),
        patterns_enabled=bool(patterns),
        powerups_enabled=bool(powerups),
        normal_enemies=bool(normal_enemies),
        kamikaze_enemies=bool(kamikaze_enemies),
        spawn_rate_percent=validate_spawn_rate_percent(spawn_rate_percent),
        lives=int(lives),
    )


# The keys of info["training_events"], in the order they appear there.
EVENT_FIELDS = ("enemies_destroyed", "explosion_pickups", "freeze_pickups",
                "shrink_pickups", "lives_spent", "survival_frames",
                "patterns_survived", "deaths")


class StepBatch:
    """Every lane's per-step info fields for one step, as arrays.

    Each attribute is the array of what `info[<same name>]` holds per lane.
    `event_counts` holds `training_events`, one column per key of
    EVENT_FIELDS, so a consumer can total every event in one reduction;
    `events` maps each key to its column. The arrays belong to this step
    alone: the env never writes to them afterwards.
    """

    __slots__ = ("score", "score_delta", "frames", "run_frames", "run_over",
                 "survived_frames", "survival_reward", "uncontrolled_score",
                 "event_counts", "done")

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)

    @property
    def events(self):
        """training_events key -> that event's per-lane counts."""
        return {key: self.event_counts[:, column]
                for column, key in enumerate(EVENT_FIELDS)}

    def lane_info(self, lane):
        """Lane `lane`'s info dict, without the end-of-episode entries."""
        return {
            "score": float(self.score[lane]),
            "score_delta": float(self.score_delta[lane]),
            "frames": int(self.frames[lane]),
            "run_frames": int(self.run_frames[lane]),
            # The episode always ended; this says whether the game did
            # too, which is what the dashboard's survival metric follows.
            "run_over": bool(self.run_over[lane]),
            "survived_frames": int(self.survived_frames[lane]),
            "survival_reward": float(self.survival_reward[lane]),
            "uncontrolled_score": float(self.uncontrolled_score[lane]),
            "training_events": dict(zip(EVENT_FIELDS,
                                        self.event_counts[lane].tolist())),
        }


class StepInfos(list):
    """One step's infos: a list of one dict per lane, as a VecEnv returns.

    Building a full dict for every lane on every step cost more than the
    engine's own step. So a lane whose episode ended gets its full dict --
    everything SB3 reads (terminal_observation, TimeLimit.truncated, episode)
    and death and run, which only ever appear there -- and every other lane
    an empty dict. Every lane's per-step fields are on `batch` as arrays,
    which is where the dashboard reads them; `lane(i)` rebuilds lane i's
    full dict from them, identical to what the env used to put in the list.

    A wrapper that copies the list into a plain one drops `batch`.
    """

    def __init__(self, infos, batch):
        super().__init__(infos)
        self.batch = batch

    def lane(self, index):
        """Lane `index`'s full info dict."""
        return self[index] if self.batch.done[index] else self.batch.lane_info(index)


class DodgeVecEnv(VecEnv):
    """N games as one VecEnv, with the VelocityFlow observation and reward.

    `lives` is a training-only device. With it set, the first `lives` lethal
    collisions leave the player alive, so one game supplies several episodes
    starting from mid-game states: an episode ends (terminated) at each one
    while the game carries on underneath it. `max_episode_frames` caps the
    game, not the episode, and ends it as a truncation.

    Game seeds come from a `SeedStream` keyed by `seed`: each slot's n-th game
    plays a hash of (stream, seed, slot, n) that avoids every reserved range
    (see dodge.seeds), so training cannot land on an evaluation cohort.
    `seeds` instead deals an explicit list in order, reserved or not.

    `step` returns its infos as a `StepInfos`: the full info dict only for a
    lane whose episode ended, every lane's per-step fields as arrays on
    `infos.batch`, and `infos.lane(i)` for lane i's full dict.
    """

    def __init__(
        self,
        n_envs=8,
        seed=None,
        difficulty=1,
        patterns=False,
        powerups=False,
        normal_enemies=True,
        kamikaze_enemies=True,
        spawn_rate_percent=100,
        action_repeat=1,
        lives=0,
        max_episode_frames=None,
        reward_config_path=None,
        execution="parallel",
        seeds=None,
        seed_stream="train",
    ):
        if max_episode_frames is not None and max_episode_frames < 1:
            raise ValueError("max_episode_frames must be positive or None")
        self.render_mode = None
        super().__init__(
            n_envs,
            # Velocities are signed, so the box is not the unit interval.
            spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32),
            spaces.Discrete(len(ACTION_MASKS)),
        )
        self._engine_settings = dict(
            step_frames=action_repeat,
            difficulty=difficulty, patterns=patterns, powerups=powerups,
            normal_enemies=normal_enemies, kamikaze_enemies=kamikaze_enemies,
            spawn_rate_percent=spawn_rate_percent, lives=lives,
        )
        self.engine = make_engine(**self._engine_settings, execution=execution)
        # A second, small engine that starts restarted games on their own, so
        # their first observation costs a few lanes' encoding rather than the
        # whole batch's. See _fresh_games.
        self._nursery = None
        # "auto" picks whichever of the two is cheaper, from the timings
        # below; "nursery" and "batch" force one, for the parity tests.
        self._restart_observe = "auto"
        self._observe_seconds = None
        self._nursery_lane_seconds = None
        self.action_repeat = int(action_repeat)
        self.max_episode_frames = max_episode_frames
        # Resolved here rather than as a default argument so the location can
        # be redirected at runtime, which keeps tests off the developer's file.
        self.reward_config_path = (reward_config_path if reward_config_path is not None
                                   else rewards.CONFIG_PATH)
        self.reward_config = rewards.load(self.reward_config_path)
        # What the env was built with, for a checkpoint's provenance manifest
        # and for evaluating on the same game.
        self.game_settings = {
            "difficulty": int(difficulty), "patterns": bool(patterns),
            "powerups": bool(powerups), "normal_enemies": bool(normal_enemies),
            "kamikaze_enemies": bool(kamikaze_enemies),
            "spawn_rate_percent": int(spawn_rate_percent),
            "action_repeat": int(action_repeat), "lives": int(lives),
            "max_episode_frames": max_episode_frames, "execution": execution,
        }
        if seeds is not None:
            self._stream = ExplicitSeeds(seeds, n_slots=n_envs)
        else:
            # An unseeded env still gets a definite run seed, so its stream
            # can be recorded and resumed like any other.
            run_seed = seed if seed is not None else secrets.randbits(63)
            self._stream = SeedStream(run_seed, n_envs, stream=seed_stream)
        # Raw game frames advanced across every lane, for the run's counters.
        self.total_frames = 0
        self._actions = None
        lanes = np.zeros(n_envs)
        self._episode_started = lanes.copy()
        self._episode_reward = lanes.copy()
        self._episode_frames = np.zeros(n_envs, dtype=np.int64)
        self._run_frames = np.zeros(n_envs, dtype=np.int64)
        self._game_seeds = np.zeros(n_envs, dtype=np.uint32)
        # Each lane's game as the stream counts them: its seed is the hash of
        # this index, which is what an abandoned-episode record reports.
        self._game_indices = np.zeros(n_envs, dtype=np.int64)
        # Per game: frames lived with a pattern active, and how many times each
        # pattern was met -- what the per-pattern and hazard metrics need.
        self._pattern_frames = np.zeros(n_envs, dtype=np.int64)
        self._encounters = [{} for _ in range(n_envs)]
        self._previous = None

    def _draw_seeds(self, lanes):
        """The next game seed for each of `lanes`, recorded as their games."""
        lanes = np.asarray(lanes, dtype=np.int64)
        seeds = self._stream.draw(lanes)
        self._game_seeds[lanes] = seeds
        self._game_indices[lanes] = self._stream.next_episode_indices[lanes] - 1
        return seeds

    @property
    def next_episode_indices(self):
        """Per slot, the index of the next game it will start."""
        return self._stream.next_episode_indices.copy()

    @next_episode_indices.setter
    def next_episode_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} episode indices, got {len(indices)}")
        self._stream.next_episode_indices = indices.copy()

    @property
    def game_seeds(self):
        """The seed of the game each lane is playing."""
        return self._game_seeds.copy()

    @property
    def seed_mode(self):
        return self._stream.kind

    def seed_state(self):
        """Everything needed to put the same games back in the same lanes."""
        return {
            "seeds": self._stream.state_dict(),
            "game_seeds": [int(seed) for seed in self._game_seeds],
            "game_indices": [int(index) for index in self._game_indices],
            "total_frames": int(self.total_frames),
        }

    def restore_seed_state(self, state):
        """Restart every lane on the game `seed_state` recorded, from frame 0.

        Exact when the state was taken just after `abandon_episodes` (a
        segment boundary), where every lane was at the start of its game.
        Returns the observation.
        """
        # SB3's load seeds the env; a restored stream supersedes that.
        self._reset_seeds()
        saved = state["seeds"]
        if saved.get("kind") == SeedStream.kind:
            self._stream = SeedStream(saved["run_seed"], self.num_envs,
                                      stream=saved["stream"])
        self._stream.load_state_dict(saved)
        self._game_indices[:] = state["game_indices"]
        self.total_frames = int(state["total_frames"])
        return self._start_all(np.asarray(state["game_seeds"], dtype=np.uint32))

    def abandon_episodes(self):
        """Give up every game in flight; each lane starts on its next seed.

        The segment boundary: what Turner's teacher does every
        `segment_updates` PPO updates, so that a checkpoint taken here can be
        resumed exactly. Returns the new observation and one record per lane
        of the game it gave up.
        """
        score = (self._previous["score"] if self._previous is not None
                 else np.zeros(self.num_envs))
        abandoned = [{
            "slot": lane,
            "seed": int(self._game_seeds[lane]),
            "episode_index": int(self._game_indices[lane]),
            "frames": int(self._run_frames[lane]),
            "episode_frames": int(self._episode_frames[lane]),
            "score": float(score[lane]),
        } for lane in range(self.num_envs)]
        lanes = np.arange(self.num_envs)
        return self._start_all(self._draw_seeds(lanes)), abandoned

    def config(self):
        """The env's full configuration, for a provenance manifest."""
        stream = self._stream.state_dict()
        return {
            "n_envs": self.num_envs,
            **self.game_settings,
            "seed_mode": self._stream.kind,
            "run_seed": stream.get("run_seed"),
            "seed_stream": stream.get("stream"),
            "reserved_seed_ranges": {name: list(bounds) for name, bounds in
                                     getattr(self._stream, "reserved", {}).items()},
            "reward_config_path": str(self.reward_config_path),
            "reward_config": asdict(self.reward_config),
        }

    def _counters(self, state):
        # copy=False: observe_velocity returns fresh arrays, so the ones
        # already of the right type need no copy of their own.
        return {
            "score": state["score"].astype(np.float64, copy=False),
            "kills": state["uncontrolled_kills"].astype(np.int64),
            "lives": state["lives_spent"].astype(np.int64),
            "pickups": state["pickups"].astype(np.int64),
            "pattern": state["pattern_active"],
            "pattern_id": state["pattern_id"].astype(np.int64),
        }

    def _start_runs(self, lanes, fresh):
        """Bookkeeping for lanes whose game was just reset.

        `fresh` holds those lanes' counters, in the order of `lanes`.
        """
        self._run_frames[lanes] = 0
        self._pattern_frames[lanes] = 0
        for lane in lanes:
            self._encounters[lane] = {}
        self._start_episodes(lanes)
        if self._previous is None:
            self._previous = fresh
        else:
            for key, values in fresh.items():
                self._previous[key][lanes] = values

    def _start_episodes(self, lanes):
        self._episode_started[lanes] = time.monotonic()
        self._episode_reward[lanes] = 0.0
        self._episode_frames[lanes] = 0

    def reset(self):
        if (self._seeds and self._seeds[0] is not None
                and isinstance(self._stream, SeedStream)):
            # VecEnv.seed(), which SB3 calls with the model's seed: the stream
            # restarts from that run seed, as the old rng was re-seeded.
            self._stream = SeedStream(self._seeds[0], self.num_envs,
                                      stream=self._stream.stream)
        self._reset_seeds()
        return self._start_all(self._draw_seeds(np.arange(self.num_envs)))

    def _start_all(self, seeds):
        """Start every lane on a fresh game of `seeds`."""
        # Picked up per reset so a dashboard edit reaches the next episodes.
        self.reward_config = rewards.load(self.reward_config_path)
        self._game_seeds[:] = seeds
        self.engine.reset_ml_batch(self._game_seeds)
        state = self.engine.observe_velocity()
        self._previous = None
        self._start_runs(np.arange(self.num_envs), self._counters(state))
        self.reset_infos = [{"seed": int(seed), "score": 0.0, "frames": 0, "run_frames": 0}
                            for seed in self._game_seeds]
        return state["observation"]

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.uint8).reshape(self.num_envs)

    def _step_engine(self, actions):
        # The same step as step_ml_batch, without encoding the 225-value ML
        # observation this env never reads.
        step = getattr(self.engine, "step_ml_positions_batch", None)
        return (step or self.engine.step_ml_batch)(actions)

    def _observe(self):
        """observe_velocity, timed for _fresh_games' choice."""
        started = time.perf_counter()
        state = self.engine.observe_velocity()
        elapsed = time.perf_counter() - started
        self._observe_seconds = (elapsed if self._observe_seconds is None
                                 else 0.9 * self._observe_seconds + 0.1 * elapsed)
        return state

    def _fresh_games(self, lanes, seeds):
        """Start `lanes` on `seeds`; their first observations and counters.

        Observing the whole batch again to read a few restarted lanes costs
        every lane's encoding. A game's start depends only on the settings
        and its seed, so the nursery -- a second engine on the same settings
        -- can start the same games on their own and encode only them. That
        costs a reset per lane on top of the batch's own, so it is used only
        while the measured costs say it is the cheaper of the two.
        """
        started = time.perf_counter()
        self.engine.reset_ml_lanes(lanes, seeds)
        reset_lane = (time.perf_counter() - started) / len(lanes)
        mode = self._restart_observe
        if mode == "auto":
            per_lane = (self._nursery_lane_seconds if self._nursery_lane_seconds is not None
                        else reset_lane)
            mode = ("nursery" if self._observe_seconds is not None
                    and len(lanes) * per_lane < self._observe_seconds else "batch")
        if mode == "batch":
            if self._nursery_lane_seconds is not None:
                # Drift back towards the batch's own reset cost, so one slow
                # nursery call cannot rule the nursery out for good.
                self._nursery_lane_seconds = (0.9 * self._nursery_lane_seconds
                                              + 0.1 * reset_lane)
            fresh = self._observe()
            observation = fresh["observation"][lanes]
            counters = {key: values[lanes] for key, values in self._counters(fresh).items()}
            return observation, counters
        if self._nursery is None:
            # Serial: it holds a handful of lanes, too few to split.
            self._nursery = make_engine(**self._engine_settings, execution="serial")
        started = time.perf_counter()
        self._nursery.reset_ml_batch(seeds)
        fresh = self._nursery.observe_velocity()
        elapsed = (time.perf_counter() - started) / len(lanes)
        self._nursery_lane_seconds = (elapsed if self._nursery_lane_seconds is None
                                      else 0.9 * self._nursery_lane_seconds + 0.1 * elapsed)
        return fresh["observation"], self._counters(fresh)

    def step_wait(self):
        lanes = self.num_envs
        weights = self.reward_config
        # The engine holds each action for action_repeat frames and stops a
        # lane early at the frame it dies.
        stepped = self._step_engine(self._actions)
        frames = stepped["frames_advanced"].astype(np.int64)
        state = self._observe()
        self._run_frames += frames
        self._episode_frames += frames
        self.total_frames += int(frames.sum())

        now = self._counters(state)
        previous = self._previous
        dead = state["dead"]
        lives_spent = now["lives"] - previous["lives"]
        # A spent life ends the episode too. terminated, never truncated: SB3
        # bootstraps V=0 at a terminal and V(s') at a truncation, and it is the
        # zero that keeps the value of surviving the hit from being credited
        # back to the actions that led into it.
        terminated = dead | (lives_spent > 0)
        # The frame of a hit earns nothing, as a death frame never has.
        survived = np.maximum(frames - terminated, 0)
        # Credited at the decision's end position: exact at one frame per
        # decision, and the plateau makes it nearly so at a few.
        credit = survived * center_survival_scalars(state["player_x"], state["player_y"],
                                                    edge_penalty=weights.edge_penalty)
        patterns_survived = previous["pattern"] & ~(now["pattern"] | dead)
        self._pattern_frames += frames * now["pattern"]
        met = (now["pattern_id"] >= 0) & (now["pattern_id"] != previous["pattern_id"])
        for lane in np.flatnonzero(met).tolist():
            pattern = int(now["pattern_id"][lane])
            self._encounters[lane][pattern] = self._encounters[lane].get(pattern, 0) + 1
        score_delta = now["score"] - previous["score"]
        kills = now["kills"] - previous["kills"]
        pickups = now["pickups"] - previous["pickups"]
        uncontrolled = rewards.UNCONTROLLED_SCORE_PER_ENEMY * kills
        survival_reward = weights.survival_per_frame * credit
        reward = (survival_reward
                  + weights.score_weight * (score_delta - uncontrolled)
                  + weights.uncontrolled_score_weight * uncontrolled
                  - weights.death_penalty * (lives_spent + dead))
        # The cap bounds the game, not the episode, so a run that keeps
        # absorbing hits cannot hold one seed forever.
        if self.max_episode_frames is None:
            truncated = np.zeros(lanes, dtype=bool)
            run_over, done = dead, terminated
        else:
            truncated = (self._run_frames >= self.max_episode_frames) & ~terminated
            run_over = dead | truncated
            done = terminated | truncated
        self._episode_reward += reward
        self._previous = now

        # training_events, columns in EVENT_FIELDS order.
        event_counts = np.empty((lanes, len(EVENT_FIELDS)), dtype=np.int64)
        event_counts[:, 0] = kills
        event_counts[:, 1:4] = pickups
        event_counts[:, 4] = lives_spent
        event_counts[:, 5] = survived
        event_counts[:, 6] = patterns_survived
        event_counts[:, 7] = dead
        batch = StepBatch(
            # Copies of what the restarts below overwrite in place.
            score=now["score"].copy(),
            score_delta=score_delta,
            frames=self._episode_frames.copy(),
            run_frames=self._run_frames.copy(),
            run_over=run_over,
            survived_frames=survived,
            survival_reward=survival_reward,
            uncontrolled_score=uncontrolled,
            event_counts=event_counts,
            done=done,
        )
        infos = StepInfos([{} for _ in range(lanes)], batch)
        observation = state["observation"]
        ended = np.flatnonzero(done)
        for lane in ended.tolist():
            info = batch.lane_info(lane)
            if dead[lane]:
                info["death"] = {
                    "cause": DEATH_CAUSES.get(int(state["death_cause"][lane])),
                    "pattern": int(state["death_pattern"][lane]),
                    "fyou_triggers": int(state["fyou_triggers"][lane]),
                }
            if run_over[lane]:
                # The whole game, for the dashboard's evaluation figures.
                info["run"] = {
                    "seed": int(self._game_seeds[lane]),
                    "frames": int(self._run_frames[lane]),
                    "score": float(now["score"][lane]),
                    "died": bool(dead[lane]),
                    "cause": DEATH_CAUSES.get(int(state["death_cause"][lane])),
                    "pattern": int(state["death_pattern"][lane]),
                    "fyou_triggers": int(state["fyou_triggers"][lane]),
                    "pattern_frames": int(self._pattern_frames[lane]),
                    "encounters": dict(self._encounters[lane]),
                    "lives_spent": int(now["lives"][lane]),
                }
            info["terminal_observation"] = observation[lane].copy()
            info["TimeLimit.truncated"] = bool(truncated[lane])
            info["episode"] = {
                "r": float(self._episode_reward[lane]),
                "l": int(math.ceil(self._episode_frames[lane] / self.action_repeat)),
                "t": round(time.monotonic() - self._episode_started[lane], 6),
            }
            infos[lane] = info

        if len(ended):
            # A spent life opens a new episode on the running game; only a
            # death or the cap starts a new game.
            spent = ended[~run_over[ended]]
            if len(spent):
                self._start_episodes(spent)
            restart = ended[run_over[ended]].astype(np.uint32)
            if len(restart):
                seeds = self._draw_seeds(restart)
                fresh_observation, fresh = self._fresh_games(restart, seeds)
                self._start_runs(restart, fresh)
                # Every terminal observation was copied out above, so the
                # step's own array can take the restarted games' rows.
                observation[restart] = fresh_observation
        return observation, reward.astype(np.float32), done, infos

    def lane_state(self, lane):
        """Lane `lane`'s raw game state under the cartridge's own names."""
        return self.engine.lane_state(int(lane))

    def close(self):
        self.engine = None
        self._nursery = None

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("DodgeVecEnv's lanes are not separate env objects")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def play_games(policy, seeds, *, frame_cap=None, progress=None, **game_settings):
    """Play one game per seed, all at once as lanes of one batch.

    `policy(observations, lane_state)` returns an action per lane, where
    `lane_state(lane)` reads that lane's raw game state for a scripted
    controller. Each game runs until it dies, or reaches `frame_cap`; lanes
    that have finished keep running throwaway games so the batch stays whole.
    `progress(finished, total)` is called as games end. Returns one record per
    seed, in the same shape DodgeVecEnv reports as info["run"].
    """
    seeds = np.asarray(seeds, dtype=np.uint32)
    engine = make_engine(**game_settings)
    engine.reset_ml_batch(seeds)
    count = len(seeds)
    results = [None] * count
    frames = np.zeros(count, dtype=np.int64)
    pattern_frames = np.zeros(count, dtype=np.int64)
    encounters = [{} for _ in range(count)]
    previous_pattern = np.full(count, -1, dtype=np.int64)
    filler = 2**31
    finished_count = 0
    while finished_count < count:
        state = engine.observe_velocity()
        actions = np.asarray(policy(state["observation"], engine.lane_state),
                             dtype=np.uint8).reshape(count)
        stepped = engine.step_ml_batch(actions)
        advanced = stepped["frames_advanced"].astype(np.int64)
        frames += advanced
        state = engine.observe_velocity()
        pattern = state["pattern_id"].astype(np.int64)
        pattern_frames += advanced * (pattern >= 0)
        for lane in np.flatnonzero((pattern >= 0) & (pattern != previous_pattern)):
            met = encounters[lane]
            met[int(pattern[lane])] = met.get(int(pattern[lane]), 0) + 1
        previous_pattern = pattern
        capped = frames >= frame_cap if frame_cap is not None else np.zeros(count, bool)
        ended = np.flatnonzero(state["dead"] | capped)
        for lane in ended:
            if results[lane] is None:
                results[lane] = {
                    "seed": int(seeds[lane]),
                    "frames": int(frames[lane]),
                    "score": float(state["score"][lane]),
                    "died": bool(state["dead"][lane]),
                    "cause": DEATH_CAUSES.get(int(state["death_cause"][lane])),
                    "pattern": int(state["death_pattern"][lane]),
                    "fyou_triggers": int(state["fyou_triggers"][lane]),
                    "pattern_frames": int(pattern_frames[lane]),
                    "encounters": dict(encounters[lane]),
                }
                finished_count += 1
                if progress is not None:
                    progress(finished_count, count)
        if len(ended):
            restart = ended.astype(np.uint32)
            engine.reset_ml_lanes(restart, np.arange(filler, filler + len(restart),
                                                     dtype=np.uint32))
            filler += len(restart)
            previous_pattern[restart] = -1
    return results
