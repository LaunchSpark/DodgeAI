"""Data shapes shared by training and the dashboard."""

import random
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

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


@dataclass
class DashboardSnapshot:
    status: str = "starting"
    # Placeholder only: the session overwrites this from the running
    # policy name. Kept a literal so this module stays free of torch.
    model_name: str = "velocity-flow"
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
    # One entry per finished episode, holding that episode's enemy kills.
    # info["training_events"] reports kills as a per-step delta, so the
    # trainer accumulates them across the episode.
    enemies_killed: RunSeries = field(default_factory=RunSeries)
    events: dict[str, EventCount] = field(
        default_factory=lambda: {key: EventCount() for key in EVENT_KEYS}
    )

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
            enemies_killed=self.enemies_killed.copy(),
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
    data = DashboardSnapshot(status="training", seed=seed, update=updates,
                             steps_per_second=4820.0, elapsed_seconds=2 * 3600 + 930,
                             rollout_progress=0.33)
    for i in range(updates):
        progress = i / max(1, updates - 1)
        data.reward.append(17 + progress * 34 + rng.uniform(-8, 8))
        data.entropy.append(2.2 - progress * 0.62 + rng.uniform(-0.03, 0.03))
        data.episode_length.append(6.6 + progress * 12 + rng.uniform(-1.4, 1.4))
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
