"""No-replacement evaluation: every seed is played exactly once, to the end.

Each lane of one Rust batch plays one seed. A lane whose game ends is masked
out of the batch -- never refilled with another game -- and the policy runs
batched over the lanes still alive. The result is one game per seed whatever
the games' lengths, which an evaluator that stops after N finished episodes
of an auto-resetting VecEnv does not give: short games finish first and get
counted again before the long ones end, biasing every statistic low.
"""

from dataclasses import dataclass, field
import random

import numpy as np

from .env import DEATH_CAUSES, make_engine


def quarter_means(frames):
    """(Q1, Q4): mean of the worst and of the best 25% of games.

    The paper's definition (docs/paper/eval/analyze.py): a quarter is
    max(1, n // 4) games, and Q1 is a mean, not the 25th percentile.
    """
    ordered = np.sort(np.asarray(frames, dtype=float))
    quarter = max(1, len(ordered) // 4)
    return float(ordered[:quarter].mean()), float(ordered[-quarter:].mean())


@dataclass
class EvalResult:
    seeds: list
    frames: np.ndarray
    scores: np.ndarray
    died: list
    causes: list = field(default_factory=list)

    @property
    def mean(self):
        return float(np.mean(self.frames))

    @property
    def q1(self):
        return quarter_means(self.frames)[0]

    @property
    def q4(self):
        return quarter_means(self.frames)[1]

    def summary(self):
        frames = np.asarray(self.frames, dtype=float)
        return {
            "n": len(frames),
            "mean": self.mean,
            "q1": self.q1,
            "q4": self.q4,
            "median": float(np.median(frames)),
            "worst": float(frames.min()),
            "best": float(frames.max()),
            "died": int(sum(self.died)),
        }

    def to_dict(self):
        return {
            **self.summary(),
            "games": [{"seed": int(seed), "frames": int(frames), "score": float(score),
                       "died": bool(died)}
                      for seed, frames, score, died in zip(self.seeds, self.frames,
                                                           self.scores, self.died)],
        }


def engine_settings(game_settings):
    """make_engine arguments for evaluating on a training env's game.

    Training lives and the training frame cap are dropped: evaluation plays
    whole games, capped only by the evaluator's own frame_cap.
    """
    return {
        "step_frames": int(game_settings["action_repeat"]),
        "difficulty": int(game_settings["difficulty"]),
        "patterns": bool(game_settings["patterns"]),
        "powerups": bool(game_settings["powerups"]),
        "normal_enemies": bool(game_settings["normal_enemies"]),
        "kamikaze_enemies": bool(game_settings["kamikaze_enemies"]),
        "spawn_rate_percent": int(game_settings["spawn_rate_percent"]),
        "execution": game_settings.get("execution", "parallel"),
    }


def evaluate_policy(policy, seeds, *, frame_cap=None, stop_after=None, progress=None,
                    on_step=None, **game_settings):
    """Play one game per seed with `policy(observations) -> actions`.

    `game_settings` are make_engine's arguments. A game ends at its death or
    once it has run `frame_cap` frames. With `stop_after`, the evaluation
    itself ends once that many games have: every lane steps together, so the
    games still playing all stop at the same length, alive, and are counted
    at it. `progress(finished, count)`, if given, is called as games end,
    `count` being the games waited for, and `on_step(decisions)` after every
    step with the number of games it moved. Results come back in seed order.
    """
    seeds = np.asarray(list(seeds), dtype=np.uint32)
    count = len(seeds)
    if count == 0:
        raise ValueError("no seeds to evaluate")
    wait_for = count if stop_after is None else min(count, int(stop_after))
    engine = make_engine(**game_settings)
    engine.reset_ml_batch(seeds)
    alive = np.ones(count, dtype=bool)
    frames = np.zeros(count, dtype=np.int64)
    scores = np.zeros(count, dtype=float)
    died = np.zeros(count, dtype=bool)
    causes = [None] * count
    state = engine.observe_velocity()
    while alive.any():
        lanes = np.flatnonzero(alive)
        actions = np.zeros(count, dtype=np.uint8)
        actions[lanes] = np.asarray(policy(state["observation"][lanes]),
                                    dtype=np.uint8).reshape(len(lanes))
        stepped = engine.step_batch_active(actions, alive)
        frames[np.asarray(stepped["lane_ids"], dtype=np.int64)] += \
            np.asarray(stepped["frames_advanced"], dtype=np.int64)
        if on_step is not None:
            on_step(len(lanes))
        state = engine.observe_velocity()
        ended = state["dead"].astype(bool)
        if frame_cap is not None:
            ended = ended | (frames >= frame_cap)
        finished = alive & ended
        for lane in np.flatnonzero(finished):
            scores[lane] = float(state["score"][lane])
            died[lane] = bool(state["dead"][lane])
            causes[lane] = (DEATH_CAUSES.get(int(state["death_cause"][lane]))
                            if died[lane] else None)
        alive &= ~finished
        ended = count - int(alive.sum())
        if progress is not None and finished.any():
            progress(min(ended, wait_for), wait_for)
        if ended >= wait_for:
            for lane in np.flatnonzero(alive):
                scores[lane] = float(state["score"][lane])
            break
    return EvalResult(seeds=[int(seed) for seed in seeds], frames=frames,
                      scores=scores, died=[bool(d) for d in died], causes=causes)


class preserved_rng:
    """Leave python, numpy and torch's global generators as they were.

    Argmax evaluation draws nothing, but a retention check runs in the
    middle of training, and a resumed run must not depend on whether one ran.
    """

    def __enter__(self):
        import torch
        self._python = random.getstate()
        self._numpy = np.random.get_state()
        self._torch = torch.get_rng_state()
        # Only once initialized: before that nothing can have drawn from it.
        self._cuda = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() and torch.cuda.is_initialized()
                      else None)
        return self

    def __exit__(self, *exc):
        import torch
        random.setstate(self._python)
        np.random.set_state(self._numpy)
        torch.set_rng_state(self._torch)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


# The smallest batch argmax_policy pads to. Below it a forward costs about the
# same whatever the batch, so smaller sizes would only cost more graphs.
MIN_PADDED_BATCH = 16


def padded_batch(n):
    """The batch argmax_policy runs `n` observations at: a power of two, >= 16."""
    return max(MIN_PADDED_BATCH, 1 << (int(n) - 1).bit_length())


def argmax_policy(model):
    """`model`'s argmax actions, through its policy's forward at padded batches.

    SB3's predict goes through get_distribution, never the policy's forward,
    so it misses the velocity policies' CUDA graphs, and the batch shrinks by
    one every time a game ends, so every size would be new to cuDNN's
    autotuner too. Padding to a power of two -- by repeating the last
    observation; no layer mixes samples -- leaves a cohort of 128 four sizes,
    each captured once. The actions are predict's: the argmax of the same
    probabilities.
    """
    import torch

    network = model.policy

    def policy(observations):
        n = len(observations)
        rows = np.minimum(np.arange(padded_batch(n)), n - 1)
        network.set_training_mode(False)
        with torch.no_grad():
            inputs = torch.as_tensor(np.asarray(observations)[rows], device=network.device)
            actions, _, _ = network(inputs, deterministic=True)
        return actions[:n].cpu().numpy()

    return policy


def evaluate_model(model, seeds, *, deterministic=True, frame_cap=None, stop_after=None,
                   progress=None, on_step=None, **game_settings):
    """Evaluate an SB3 model, argmax by default, without touching its RNG."""
    if deterministic:
        policy = argmax_policy(model)
    else:
        def policy(observations):
            actions, _ = model.predict(observations, deterministic=False)
            return actions

    with preserved_rng():
        return evaluate_policy(policy, seeds, frame_cap=frame_cap, stop_after=stop_after,
                               progress=progress, on_step=on_step, **game_settings)
