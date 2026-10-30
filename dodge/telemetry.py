"""Data shapes shared by training and the dashboard."""

import random
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from .metrics import summary
from .series import RunSeries


EVENT_KEYS = (
    "survival_frames",
    "enemies_destroyed",
    "explosion_pickups",
    "freeze_pickups",
    "shrink_pickups",
    "patterns_survived",
    "deaths",
    "lives_spent",
)

# Training lives top out at 5, so a cartridge run holds at most six episodes.
LIFE_BANDS = 6


@dataclass
class EventCount:
    """Per-rollout counters. The dashboard shows these as recent history,
    deliberately short — they answer 'what just happened', not 'am I improving'."""
    rollout: int = 0
    total: int = 0
    history: deque[int] = field(default_factory=lambda: deque(maxlen=18))


class GameLog:
    """Finished games for the evaluation figures, and quarter means per update.

    Games are kept in a bounded window, so the figures describe the policy as
    it plays now. The per-update rows span the whole run: when they fill, every
    other row is dropped, so old history loses resolution instead of falling
    off the left edge.
    """

    def __init__(self, capacity=2048, window=256, rows=480):
        self.games = deque(maxlen=capacity)
        self.window = window
        self.rows_capacity = rows
        self.quarters = []
        self._new_games = 0

    def add(self, game, update):
        self.games.append({**game, "update": update})
        self._new_games += 1

    def recent(self, count=None):
        games = list(self.games)
        return games[-(count or self.window):]

    def close_update(self, update):
        """Record the quarter means of the latest window, if games ended since."""
        if not self._new_games:
            return
        self._new_games = 0
        stats = summary(self.recent())
        self.quarters.append({"update": update, "q1": stats["q1"], "mean": stats["mean"],
                              "q4": stats["q4"], "games": stats["games"]})
        if len(self.quarters) > self.rows_capacity:
            self.quarters = self.quarters[::2]

    def copy(self):
        clone = GameLog(self.games.maxlen, self.window, self.rows_capacity)
        clone.games = deque(self.games, maxlen=self.games.maxlen)
        clone.quarters = list(self.quarters)
        clone._new_games = self._new_games
        return clone


@dataclass
class DashboardSnapshot:
    status: str = "starting"
    # Empty until a session explicitly starts or loads a model.
    model_name: str = ""
    update: int = 0
    seed: int = 42
    steps_per_second: float = 0.0
    best_score: float = 0.0
    elapsed_seconds: float = 0.0
    progress_phase: str = "rollout"
    rollout_progress: float = 0.0
    learning_progress: float = 0.0
    # Share of the policy trunk that emitted zero for an entire rollout. Rises
    # toward 1.0 when units die; at 1.0 the policy is a constant distribution
    # and training is doing nothing, which nothing else here reveals.
    dead_units: float = 0.0
    # Every series spans the whole run, so the charts can show improvement
    # rather than only the present moment.
    reward: RunSeries = field(default_factory=RunSeries)
    entropy: RunSeries = field(default_factory=RunSeries)
    episode_length: RunSeries = field(default_factory=RunSeries)
    # Survival Time, split by life. Band i holds how long the run's (i+1)th
    # life lasted, and zero for a life the run never spent, so the bands stack
    # to exactly the run total drawn over them. Every band gets one append per
    # run, which keeps them compacting in lockstep and readable positionally.
    life_lengths: tuple = field(
        default_factory=lambda: tuple(RunSeries() for _ in range(LIFE_BANDS))
    )
    # One entry per finished episode, holding that episode's enemy kills.
    # info["training_events"] reports kills as a per-step delta, so the
    # trainer accumulates them across the episode.
    enemies_killed: RunSeries = field(default_factory=RunSeries)
    events: dict[str, EventCount] = field(
        default_factory=lambda: {key: EventCount() for key in EVENT_KEYS}
    )
    # Whole games, as the environment reports them when they end.
    games: GameLog = field(default_factory=GameLog)

    def copy(self):
        """A detached copy the dashboard can render from outside the lock.

        The trainer writes this structure from the PPO callback on every step.
        Rendering directly from it means holding that lock for the length of a
        frame, which stalls collection; copying costs microseconds because
        every series here is capacity-bounded.
        """
        return replace(
            self,
            reward=self.reward.copy(),
            entropy=self.entropy.copy(),
            episode_length=self.episode_length.copy(),
            life_lengths=tuple(band.copy() for band in self.life_lengths),
            enemies_killed=self.enemies_killed.copy(),
            games=self.games.copy(),
            events={
                key: EventCount(count.rollout, count.total,
                                deque(count.history, maxlen=count.history.maxlen))
                for key, count in self.events.items()
            },
        )


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: Path
    kind: str


# Metrics shown in the Progress panel, in display order.
# "rising" says whether a larger number means the agent got better; entropy
# falls as the policy converges, which is expected rather than an improvement.
PROGRESS_ROWS = (
    ("reward", "Episode reward", "", True),
    ("episode_length", "Survival time", "s", True),
    ("enemies_killed", "Enemies killed", "", True),
    ("entropy", "Policy entropy", "", False),
)


def sample_snapshot(seed=42, updates=180):
    """Build a populated snapshot for renderer tests and --screenshot.

    Deterministic for a given seed. This exists so the dashboard can be drawn
    and tested without launching a trainer; it is not a simulation of one.
    """
    rng = random.Random(seed)
    data = DashboardSnapshot(status="training", model_name="velocity-flow",
                             seed=seed, update=updates,
                             steps_per_second=4820.0, elapsed_seconds=2 * 3600 + 930,
                             rollout_progress=0.33)
    for i in range(updates):
        progress = i / max(1, updates - 1)
        data.reward.append(17 + progress * 34 + rng.uniform(-8, 8))
        data.entropy.append(2.2 - progress * 0.62 + rng.uniform(-0.03, 0.03))
        length = 6.6 + progress * 12 + rng.uniform(-1.4, 1.4)
        data.episode_length.append(length)
        # Split across a plausible number of lives so --screenshot and the
        # renderer tests exercise the stacked form, not just a single band.
        lives = rng.randint(1, 4)
        weights = [rng.uniform(0.5, 1.5) for _ in range(lives)]
        scale = length / sum(weights)
        for band in range(LIFE_BANDS):
            data.life_lengths[band].append(
                weights[band] * scale if band < lives else 0.0)
        data.enemies_killed.append(max(0.0, 1.4 + progress * 7 + rng.uniform(-1.1, 1.1)))
    # best_score is the game's own score, not the reward, so it is not derived
    # from the reward series.
    data.best_score = 12.0

    for key in EVENT_KEYS:
        event = data.events[key]
        scale = {"survival_frames": 900, "deaths": 12}.get(key, 6)
        for _ in range(18):
            event.history.append(rng.randint(0, scale))
        event.rollout = event.history[-1]
        event.total = sum(event.history) * 7
    return data
