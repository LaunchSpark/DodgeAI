"""Segment boundaries, state sidecars, and exact resume.

Following Turner's teacher: every `segment_updates` PPO updates the run
abandons the games in flight and restarts every lane on the next seeds of its
stream, while the model, Adam and every random generator carry on. Nothing at
that moment depends on the games abandoned, so a checkpoint taken there --
the SB3 zip plus a state sidecar holding the generators, the seed stream and
the counters -- resumes to exactly the run that never stopped.

A checkpoint taken anywhere else still restores the model, Adam and the
generators; its lanes start their next games from the stream, since a game
cannot be resumed mid-way.
"""

import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from .lineage import (checkpoint_path, model_config, parent_reference,
                      read_manifest, sidecar_paths, write_manifest)

STATE_FORMAT = 1
DEFAULT_SEGMENT_UPDATES = 60


# --- random generators -----------------------------------------------------

def _space_state(space):
    generator = getattr(space, "_np_random", None)
    return None if generator is None else generator.bit_generator.state


def _restore_space(space, state):
    if state is None or space is None:
        return
    space.seed(0)  # creates the generator, whose state is then replaced
    space._np_random.bit_generator.state = state


def capture_rng_state(model=None):
    """Every generator a PPO run draws from.

    CUDA's is recorded only once CUDA is initialized: before that nothing can
    have drawn from it, and asking would start a CUDA context on a CPU run.
    """
    cuda = (torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": cuda,
    }
    if model is not None:
        state["action_space"] = _space_state(model.action_space)
        state["observation_space"] = _space_state(model.observation_space)
    return state


def restore_rng_state(state, model=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if model is not None:
        _restore_space(model.action_space, state.get("action_space"))
        _restore_space(model.observation_space, state.get("observation_space"))


# --- counters --------------------------------------------------------------

def optimizer_steps(model):
    """Adam's own step count: one per minibatch ever taken."""
    try:
        per_parameter = model.policy.optimizer.state_dict()["state"]
    except AttributeError:
        return 0
    for values in per_parameter.values():
        if "step" in values:
            return int(torch.as_tensor(values["step"]).item())
    return 0


def counters(model, env, ppo_updates):
    return {
        "ppo_updates": int(ppo_updates),
        "optimizer_steps": optimizer_steps(model),
        "decisions": int(model.num_timesteps),
        "raw_frames": int(getattr(env, "total_frames", 0)),
    }


# --- the sidecar -----------------------------------------------------------

def read_state(checkpoint):
    """A checkpoint's state sidecar, or None when it has none."""
    path = sidecar_paths(checkpoint)[0]
    if not path.exists():
        return None
    # Our own file, holding python and numpy generator states as well as
    # tensors, so not loadable as weights-only.
    return torch.load(path, map_location="cpu", weights_only=False)


def _training_state(model, env, *, ppo_updates, at_boundary, segment, abandoned):
    return {
        "format": STATE_FORMAT,
        "at_boundary": bool(at_boundary),
        "rng": capture_rng_state(model),
        "env": env.seed_state() if hasattr(env, "seed_state") else None,
        "sb3": {
            "num_timesteps": int(model.num_timesteps),
            "_n_updates": int(model._n_updates),
            "_episode_num": int(model._episode_num),
            "_current_progress_remaining": float(model._current_progress_remaining),
        },
        "counters": counters(model, env, ppo_updates),
        "segment": dict(segment or {}),
        "abandoned": list(abandoned),
    }


class CheckpointWriter:
    """Save SB3 checkpoints with their state sidecar and manifest.

    Successive saves chain: each manifest names the previous checkpoint as its
    parent, the first one names `parent` (the checkpoint a run resumed from),
    and wall-clock seconds accumulate from `base_seconds`.
    """

    def __init__(self, *, parent=None, base_seconds=0.0, extra_config=None,
                 clock=time.monotonic):
        self.parent = parent_reference(parent) if parent is not None else None
        self.base_seconds = float(base_seconds)
        self.extra_config = dict(extra_config or {})
        self.clock = clock
        self.started = clock()
        self.last_path = None

    @classmethod
    def resuming(cls, checkpoint, **kwargs):
        manifest = read_manifest(checkpoint) or {}
        return cls(parent=checkpoint,
                   base_seconds=manifest.get("wall_clock_seconds", 0.0), **kwargs)

    def wall_clock_seconds(self):
        return self.base_seconds + (self.clock() - self.started)

    def save(self, model, path, *, env=None, ppo_updates, at_boundary=False,
             segment=None, abandoned=()):
        """model.save(path), then its sidecars. Returns the zip's path."""
        model.save(path)
        output = checkpoint_path(path)
        self.write_sidecars(model, output, env=env, ppo_updates=ppo_updates,
                            at_boundary=at_boundary, segment=segment,
                            abandoned=abandoned)
        return output

    def write_sidecars(self, model, checkpoint, *, env=None, ppo_updates,
                       at_boundary=False, segment=None, abandoned=()):
        """Write the state sidecar and manifest for a zip already on disk.

        Only an SB3 algorithm has the state to record; anything else (a test
        double, say) is left with its zip alone.
        """
        if not isinstance(model, BaseAlgorithm):
            return None
        checkpoint = checkpoint_path(checkpoint)
        env = env if env is not None else model.get_env()
        state = _training_state(model, env, ppo_updates=ppo_updates,
                                at_boundary=at_boundary, segment=segment,
                                abandoned=abandoned)
        state_path = sidecar_paths(checkpoint)[0]
        temporary = state_path.with_name(f".{state_path.name}.tmp")
        torch.save(state, temporary)
        temporary.replace(state_path)
        config = {**self.extra_config, "model": model_config(model),
                  "env": env.config() if hasattr(env, "config") else None}
        manifest = write_manifest(
            checkpoint, config=config, counters=state["counters"],
            parent=self.parent, wall_clock_seconds=self.wall_clock_seconds(),
            extra={"at_boundary": bool(at_boundary),
                   "segment": state["segment"],
                   "abandoned_episodes": len(state["abandoned"])},
        )
        self.parent = parent_reference(checkpoint)
        self.last_path = checkpoint
        return manifest


# --- resuming --------------------------------------------------------------

def restore_training_state(model, env, state, *, strict=True):
    """Put a loaded model's env and generators back where the sidecar says.

    At a boundary the lanes restart on exactly the games recorded; elsewhere
    they start their next games from the restored stream. With strict=False
    an env whose lane count no longer matches keeps its own stream.
    """
    saved = state.get("env")
    if saved is not None and hasattr(env, "restore_seed_state"):
        slots = len(saved["seeds"]["next_episode_indices"])
        if slots != env.num_envs:
            if strict:
                raise ValueError(f"checkpoint was taken with {slots} envs, "
                                 f"this env has {env.num_envs}")
        else:
            observation = env.restore_seed_state(saved)
            if not state.get("at_boundary"):
                observation, _ = env.abandon_episodes()
            model._last_obs = observation
            model._last_episode_starts = np.ones(env.num_envs, dtype=bool)
    # Last: loading the model re-seeded every generator from its seed.
    restore_rng_state(state["rng"], model)


def load_checkpoint(path, env, *, model_cls=PPO, strict=True, **load_kwargs):
    """Load a checkpoint and, if it has a state sidecar, everything else.

    Returns (model, state); state is None for a checkpoint without one, which
    then loads exactly as SB3 always has.
    """
    path = checkpoint_path(path)
    model = model_cls.load(path, env=env, **load_kwargs)
    state = read_state(path)
    if state is not None:
        restore_training_state(model, env, state, strict=strict)
    return model, state


# --- the segment boundary --------------------------------------------------

class SegmentCallback(BaseCallback):
    """Abandon every game each `segment_updates` PPO updates.

    The boundary falls after an update and before the next rollout's first
    step -- at the start of that rollout, or at the end of learn() when the
    run stops on one -- so a run cut there and resumed meets it at the same
    point. `on_boundary` hooks run there (checkpoint, retention gate); then,
    if `budget_seconds` of wall clock have passed since training started,
    training stops before taking another step.
    """

    def __init__(self, segment_updates=DEFAULT_SEGMENT_UPDATES, *, updates=0,
                 last_boundary=None, abandoned=None, on_boundary=(),
                 budget_seconds=None, clock=time.monotonic):
        super().__init__(verbose=0)
        if segment_updates < 0:
            raise ValueError("segment_updates cannot be negative")
        if budget_seconds is not None and not segment_updates:
            raise ValueError("a wall-clock budget stops at a segment boundary, "
                             "so it needs segment_updates > 0")
        self.segment_updates = int(segment_updates)
        self.updates = int(updates)
        self.last_boundary = last_boundary
        self.abandoned = list(abandoned or [])
        self.on_boundary = list(on_boundary)
        self.budget_seconds = budget_seconds
        self.clock = clock
        self.started = None
        self.stop_requested = False

    @classmethod
    def from_state(cls, state, segment_updates=DEFAULT_SEGMENT_UPDATES, **kwargs):
        """A callback that carries on from a checkpoint's sidecar (or None)."""
        if state is None:
            return cls(segment_updates, **kwargs)
        return cls(segment_updates,
                   updates=state["counters"]["ppo_updates"],
                   last_boundary=state.get("segment", {}).get("last_boundary"),
                   abandoned=state.get("abandoned", ()), **kwargs)

    def state(self):
        return {"segment_updates": self.segment_updates,
                "last_boundary": self.last_boundary,
                "ppo_updates": self.updates}

    def elapsed(self):
        return 0.0 if self.started is None else self.clock() - self.started

    def _on_training_start(self):
        if self.started is None:
            self.started = self.clock()

    def _on_rollout_start(self):
        self._maybe_boundary()

    def _on_step(self):
        return not self.stop_requested

    def _on_rollout_end(self):
        # train() always follows a rollout that reached its end.
        self.updates += 1

    def _on_training_end(self):
        self._maybe_boundary()

    def due(self):
        return (self.segment_updates > 0 and self.updates > 0
                and self.updates % self.segment_updates == 0
                and self.last_boundary != self.updates)

    def _maybe_boundary(self):
        if not self.due():
            return
        env = self.training_env
        observation, abandoned = env.abandon_episodes()
        self.model._last_obs = observation
        self.model._last_episode_starts = np.ones(env.num_envs, dtype=bool)
        self.last_boundary = self.updates
        self.abandoned.extend({**record, "update": self.updates} for record in abandoned)
        for hook in self.on_boundary:
            hook(self)
        if self.budget_seconds is not None and self.elapsed() >= self.budget_seconds:
            self.stop_requested = True
