"""Background Stable-Baselines3 session used by the Pygame dashboard."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
import warnings

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import DodgeEnv, validate_spawn_rate_percent
from .policies import (ARCHITECTURES, COORDINATE_INPUTS, DEFAULT_ARCHITECTURE,
                       architecture_from_kwargs,
                       HYBRID_INPUT_CHANNELS, architecture_name,
                       dead_unit_fraction, policy_class, policy_kwargs)
from .telemetry import (DashboardSnapshot, EVENT_KEYS, LIFE_BANDS,
                        ModelEntry)


RATE_WINDOW_SECONDS = 0.5

# Observations drawn from the rollout buffer to judge unit death against.
DEAD_UNIT_SAMPLES = 512

# Only the convolutional policy is worth moving to a GPU. Measured on a GTX
# 1070: the hybrid-cnn update runs 5.1x faster there (7.20s -> 1.40s), while the
# MLP is unchanged (0.686s -> 0.693s) because its layers are too small to cover
# the transfer and launch overhead. Stable-Baselines3 warns about exactly this.
GPU_ARCHITECTURES = ("hybrid-cnn", "velocity-flow", "velocity-flow-v2")


def resolve_device(device, architecture):
    """Pick the torch device, defaulting on measured benefit per architecture.

    "auto" means "use a GPU where it actually helps" rather than "use a GPU
    whenever one exists". An explicit "cuda" or "cpu" is always honoured.
    """
    if device != "auto":
        return get_device(device)
    if architecture in GPU_ARCHITECTURES:
        return get_device("auto")
    return get_device("cpu")


MINIBATCHES_PER_EPOCH = 16
MIN_MINIBATCH = 64


# PPO settings that vary by architecture. Keyed rather than global so
# velocity-flow keeps the values every curve recorded so far was trained
# under: v2 changes the network and the optimizer schedule together, and
# without a fixed baseline there is no way to tell which one moved a result.
DEFAULT_HYPERPARAMETERS = {
    "gamma": 0.995,
    "n_epochs": 10,
    "vf_coef": 0.5,
    "target_kl": None,
    "ent_coef": 0.01,
    "minibatches_per_epoch": MINIBATCHES_PER_EPOCH,
}

ARCHITECTURE_HYPERPARAMETERS = {
    "velocity-flow-v2": {
        # 0.995 is an effective horizon of ~200 decisions, about 13 seconds.
        # Most of that is unpredictable from the current frame -- which enemies
        # spawn, at what size, which pattern comes next -- so it is noise
        # sitting in the value target, capping explained variance no matter how
        # good the critic is. 0.99 is ~6.7s, still far longer than a dodge.
        "gamma": 0.99,
        # Learning is about five seconds of a six-second cycle, and ten epochs
        # is the multiplier. The late ones run on stale data and are mostly
        # clipped anyway; target_kl makes the cut adaptive rather than flat, so
        # an update that has not moved the policy still gets its passes.
        "n_epochs": 4,
        "target_kl": 0.02,
        # The critic is the weak half, so its loss gets equal weight.
        "vf_coef": 1.0,
        # 256-sample minibatches: profiled on a GTX 1070, per-sample cost is
        # flat from 128 to 256 and then degrades badly (512 is 0.41ms/sample
        # against 0.32, and 2048 is 2.88). 256 is the floor of that curve.
        "minibatches_per_epoch": 8,
    },
}


def hyperparameters(architecture=None):
    """PPO settings for an architecture, falling back to the shared defaults."""
    return {**DEFAULT_HYPERPARAMETERS,
            **ARCHITECTURE_HYPERPARAMETERS.get(architecture, {})}


def minibatch_size(buffer_samples, architecture=None):
    """Split a rollout buffer into a workable number of optimizer passes."""
    per_epoch = hyperparameters(architecture)["minibatches_per_epoch"]
    return min(max(MIN_MINIBATCH, buffer_samples // per_epoch), buffer_samples)


# Watchers must never read the mutable *-latest.zip autosave, so snapshots go
# to their own directory under names that never change once visible.
SNAPSHOT_RETENTION = 3


def snapshot_name(update):
    """Zero-padded so lexicographic order is numeric order."""
    return f"update-{int(update):08d}.zip"


def newest_snapshot(directory, after=-1):
    """The highest completed update in `directory` above `after`.

    Dotfiles are skipped: a snapshot is written to a hidden temporary name and
    only becomes visible on its atomic rename, so anything starting with a dot
    is a partial write.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best = None
    for path in directory.glob("update-*.zip"):
        if path.name.startswith("."):
            continue
        try:
            update = int(path.stem.removeprefix("update-"))
        except ValueError:
            continue
        if update > after and (best is None or update > best[1]):
            best = (path, update)
    return best


@dataclass
class _SaveRequest:
    path: Path
    finished: threading.Event
    error: Exception | None = None
    completed_update: int = 0
    kind: str = "save"
    notify: bool = False
    # The path the checkpoint actually lands at once SB3 has applied its own
    # "append .zip only when the suffix is empty" rule to `path`. Carried
    # explicitly rather than reconstructed afterwards -- a name with a dot in
    # it (e.g. "run2.5") already looks like it has a suffix to Path, so
    # stripping and re-adding ".zip" produces the wrong stem.
    output_path: Path | None = None

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = (self.path if self.path.suffix
                                else self.path.with_suffix(".zip"))


@dataclass(frozen=True)
class SessionEvent:
    """Something the trainer finished, for the GUI to turn into a toast.

    The event loop cannot block on the trainer, so results come back here
    instead of as return values.
    """
    kind: str
    ok: bool
    message: str
    path: Path | None = None
    update: int = 0


class DashboardPPO(PPO):
    """PPO variant that exposes optimizer minibatch progress to the dashboard."""

    def __init__(self, *args, learning_tracker=None, post_update=None, **kwargs):
        # Counted rather than derived. _n_updates // n_epochs was the old
        # answer, and target_kl breaks the epoch loop early, so for any
        # architecture that sets it the quotient is short. None is a sentinel
        # meaning "never counted" (pre-counter checkpoint); load() will preserve
        # any int value from the checkpoint.
        self.completed_updates = None
        self._learning_tracker = learning_tracker
        self._post_update = post_update
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_learning_tracker",
                                                  "_post_update"]

    def set_learning_tracker(self, tracker):
        self._learning_tracker = tracker

    def set_post_update(self, callback):
        self._post_update = callback

    def train(self):
        sample_count = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        minibatches_per_epoch = math.ceil(sample_count / self.batch_size)
        total_minibatches = max(1, self.n_epochs * minibatches_per_epoch)
        completed_minibatches = 0

        if self._learning_tracker:
            self._learning_tracker(completed_minibatches, total_minibatches)

        optimizer = self.policy.optimizer
        original_step = optimizer.step

        def tracked_step(*args, **kwargs):
            nonlocal completed_minibatches
            result = original_step(*args, **kwargs)
            completed_minibatches += 1
            if self._learning_tracker:
                self._learning_tracker(completed_minibatches, total_minibatches)
            return result

        optimizer.step = tracked_step
        try:
            result = super().train()
        finally:
            optimizer.step = original_step
        # Only after train() returned and the hook is off: a raised exception
        # is not a completed update.
        self.completed_updates = (self.completed_updates or 0) + 1
        if self._post_update:
            # The one moment the model is provably not mid-update: train() has
            # returned and the optimizer hook is back to the real step.
            self._post_update(self)
        return result


GRID_CONV_WEIGHT = "features_extractor.grid_net.0.weight"
COORDINATE_WEIGHT = "features_extractor.coordinate_net.0.weight"


def needs_input_migration(params):
    """True when a checkpoint predates inputs the current policy expects."""
    policy = params.get("policy", {})
    grid = policy.get(GRID_CONV_WEIGHT)
    coordinates = policy.get(COORDINATE_WEIGHT)
    return ((grid is not None and grid.shape[1] != HYBRID_INPUT_CHANNELS)
            or (coordinates is not None
                and coordinates.shape[1] != COORDINATE_INPUTS))


def migrate_widened_inputs(source_params, target, path, device="cpu"):
    """Load a checkpoint whose first layers take fewer inputs than they now do.

    New inputs are always appended -- player channels after the hazard
    channels, derived senses after the raw coordinates -- so an older
    checkpoint's weights occupy the leading slice of each widened layer.
    Everything else is copied by name and shape; the new connections start from
    the current policy's own initialization, as if those inputs had just been
    added to a network that was otherwise mid-training.
    """
    target_state = target.policy.state_dict()
    for name, target_value in target_state.items():
        source_value = source_params.get(name)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
        elif (source_value.dim() == target_value.dim()
                and source_value.dim() in (2, 4)
                and source_value.shape[0] == target_value.shape[0]
                and source_value.shape[2:] == target_value.shape[2:]
                and source_value.shape[1] < target_value.shape[1]):
            # Widened input dimension: convolution channels or linear columns.
            target_value[:, :source_value.shape[1]].copy_(source_value)
    target.policy.load_state_dict(target_state)

    # Keep the run's progress; the optimizer restarts, since its state refers
    # to a tensor that changed shape.
    data, _, _ = load_from_zip_file(path, device=device)
    target.num_timesteps = data.get("num_timesteps", 0)
    target._n_updates = data.get("_n_updates", 0)
    target._episode_num = data.get("_episode_num", 0)
    return target


class DashboardCallback(BaseCallback):
    def __init__(self, session):
        super().__init__(verbose=0)
        self.session = session
        self.rollout_events = {key: 0 for key in EVENT_KEYS}
        # Reward and kills both arrive per step and per env, so they are banked
        # here until that env's episode ends and the episode total is known.
        # Accumulating rather than reading Monitor's episode payload keeps this
        # working whichever VecEnv wrappers are in play.
        self.episode_kills = {}
        self.episode_reward = {}
        # Each env's finished-episode lengths for the run in progress, banked
        # until the run ends and the whole split is known.
        self.run_segments = {}
        self.last_step_time = time.monotonic()
        self.last_timesteps = 0
        # Throughput is measured over a window rather than per callback. A
        # single call can span less than the clock's resolution, and dividing
        # by that produces millions of steps per second -- worse with several
        # envs, since each call advances the step count by n_envs.
        self._rate_started = time.monotonic()
        self._rate_steps = 0

    def _on_training_start(self):
        self.last_timesteps = self.num_timesteps
        self.last_step_time = time.monotonic()
        with self.session.lock:
            self.session.data.status = "training"

    def _on_rollout_start(self):
        with self.session.lock:
            data = self.session.data
            data.update = self.session._completed_updates(self.model)
            data.progress_phase = "rollout"
            data.rollout_progress = 0.0
            data.learning_progress = 0.0

    def _on_step(self):
        wait_started = time.monotonic()
        if self.session._should_stop(self.model):
            return False
        if time.monotonic() - wait_started > 0.01:
            self.last_step_time = time.monotonic()

        now = time.monotonic()
        elapsed = max(now - self.last_step_time, 1e-6)
        step_delta = self.num_timesteps - self.last_timesteps
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        rewards = np.atleast_1d(self.locals.get("rewards", 0.0))

        self._rate_steps += step_delta
        rate_window = now - self._rate_started

        with self.session.lock:
            data = self.session.data
            data.elapsed_seconds += elapsed
            if rate_window >= RATE_WINDOW_SECONDS:
                measured = self._rate_steps / rate_window
                data.steps_per_second = (
                    measured if data.steps_per_second == 0
                    else data.steps_per_second * 0.7 + measured * 0.3
                )
                self._rate_started = now
                self._rate_steps = 0
            data.rollout_progress = (self.n_calls % self.session.rollout_steps) / self.session.rollout_steps

            for index, info in enumerate(infos):
                events = info.get("training_events", {})
                for key, count in events.items():
                    if key in self.rollout_events:
                        self.rollout_events[key] += int(count)
                        data.events[key].total += int(count)

                self.episode_kills[index] = (
                    self.episode_kills.get(index, 0)
                    + int(events.get("enemies_destroyed", 0))
                )
                if index < len(rewards):
                    self.episode_reward[index] = (
                        self.episode_reward.get(index, 0.0) + float(rewards[index])
                    )

                episode = info.get("episode")
                finished = episode is not None or (
                    # Monitor normally supplies episode, but retain the Dodge
                    # frame duration if another VecEnv wrapper is used.
                    index < len(dones) and bool(dones[index])
                )
                if finished:
                    if episode is not None:
                        data.best_score = max(data.best_score,
                                              float(info.get("score", 0.0)))
                    data.reward.append(self.episode_reward.pop(index, 0.0))
                    data.enemies_killed.append(float(self.episode_kills.pop(index, 0)))
                    # Survival time follows the cartridge run, not the episode.
                    # With training lives an episode ends at every absorbed hit,
                    # and charting those would quietly redefine the dashboard's
                    # headline metric as time between hits, against curves
                    # recorded when it meant time alive. Without lives the two
                    # numbers are identical, so an ordinary run is unaffected.
                    segments = self.run_segments.setdefault(index, [])
                    segments.append(float(info.get("frames", 0)) / 60.0)
                    if info.get("run_over", True):
                        self.run_segments.pop(index, None)
                        for band in range(LIFE_BANDS):
                            # Anything past the last band is folded into it, so
                            # the stack still totals the run however the lives
                            # setting was changed mid-run.
                            value = (sum(segments[band:]) if band == LIFE_BANDS - 1
                                     else (segments[band] if band < len(segments) else 0.0))
                            data.life_lengths[band].append(value)
                        frames = info.get("run_frames", info.get("frames", 0))
                        data.episode_length.append(float(frames) / 60.0)

        self.last_step_time = now
        self.last_timesteps = self.num_timesteps
        return True

    def _on_rollout_end(self):
        try:
            observation = self.model._last_obs
            obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(obs_tensor)
            entropy = float(distribution.distribution.entropy().mean().detach().cpu())
        except (AttributeError, RuntimeError, TypeError):
            entropy = 0.0

        dead_units = self._dead_unit_fraction()

        with self.session.lock:
            data = self.session.data
            data.rollout_progress = 1.0
            data.dead_units = dead_units
            data.entropy.append(entropy)
            for key, count in self.rollout_events.items():
                event = data.events[key]
                event.rollout = count
                event.history.append(count)
                self.rollout_events[key] = 0

    def _dead_unit_fraction(self):
        """Measure policy-trunk death against the rollout the update will use.

        Judged over the whole buffer rather than the last step, since a unit is
        only dead if nothing in the batch wakes it.
        """
        try:
            observations = self.model.rollout_buffer.observations
            flat = observations.reshape(-1, observations.shape[-1])
            sample = flat[::max(1, len(flat) // DEAD_UNIT_SAMPLES)]
            tensor, _ = self.model.policy.obs_to_tensor(sample)
            return dead_unit_fraction(self.model.policy, tensor)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0.0

    def _on_training_end(self):
        with self.session.lock:
            self.session.data.update = self.session._completed_updates(self.model)


class PPOTrainingSession:
    """Own a PPO model and train it without blocking the Pygame event loop."""

    def __init__(
        self,
        seed=42,
        model_dir=Path("artifacts/models"),
        logs_dir=Path("artifacts/tensorboard"),
        total_steps=250_000,
        n_envs=8,
        rollout_steps=256,
        difficulty=1,
        patterns=False,
        powerups=True,
        normal_enemies=True,
        kamikaze_enemies=True,
        spawn_rate_percent=100,
        action_repeat=4,
        lives=0,
        max_episode_frames=None,
        architecture=DEFAULT_ARCHITECTURE,
        device="auto",
        autostart=True,
    ):
        if architecture is not None and architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.seed = seed
        self.model_dir = Path(model_dir)
        self.logs_dir = Path(logs_dir)
        if total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        self.total_steps = total_steps
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        spawn_rate_percent = validate_spawn_rate_percent(spawn_rate_percent)
        # PPO's buffer is n_envs * rollout_steps samples per update, held at
        # 2048 by the defaults.
        #
        # Eight envs rather than more steps: collection makes one forward pass
        # per step regardless of batch size, and at these sizes that pass is
        # dominated by kernel launches, so envs cut the number of passes
        # without changing the simulation work.
        #
        # 256 steps because rollout length is close to free once the minibatch
        # scales with the buffer -- collection and the update both grow with
        # it, so samples/second stays flat and mainly the cadence changes. The
        # longer rollout cuts gradient noise by drawing each update from four
        # times as much fresh experience, and cuts how much of each advantage
        # rests on the bootstrap at the rollout boundary from 5.5% to 1.4%.
        # Both matter here because the critic is the weak part.
        #
        # It used to also mean a rollout spanned a whole episode, back when an
        # episode was about 100 decisions. It no longer does: episodes run far
        # longer than that and there is no frame cap, so a rollout is a slice
        # of one and most buffers hold no terminal at all. GAE bootstraps at
        # the rollout boundary, so this is sound -- just not the reason to
        # pick 256.
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.architecture = architecture
        self.model_architecture = architecture
        self.device = resolve_device(device, architecture)
        self.env_kwargs = {
            "difficulty": difficulty,
            "patterns": patterns,
            "powerups": powerups,
            "normal_enemies": bool(normal_enemies),
            "kamikaze_enemies": bool(kamikaze_enemies),
            "spawn_rate_percent": spawn_rate_percent,
            "action_repeat": action_repeat,
            "lives": lives,
            # No cap by default. It existed to keep episodes turning over,
            # which training lives now do, and it was costing more than it
            # bought: difficulty ramps with score, so a cap the agent reliably
            # reached was a ceiling on how hard a game it ever saw. The trade
            # is that a run only ends on a real death, so an agent that stops
            # being hit can hold one game seed for a long time.
            "max_episode_frames": max_episode_frames,
        }
        self.lock = threading.RLock()
        self.data = DashboardSnapshot(
            status="idle" if architecture is None else "starting",
            model_name=architecture or "",
            seed=seed,
        )
        self.pause_event = threading.Event()
        self.restart_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.save_requests = queue.Queue()
        self.events = queue.Queue()
        self.pending_load = None
        # Staged by configure_game and committed by _commit_pending_config
        # once its preservation checkpoint lands, so a save that fails leaves
        # the running game untouched.
        self.pending_config = None
        self.model = None
        self.active_model_name = architecture or ""
        self.last_error = None
        self.last_watch_label = None
        self.worker = None
        # Private to this session: two dashboards must not publish into the
        # same stream, and a watcher should only ever see this trainer's work.
        self.live_watch_dir = (self.model_dir / "watch"
                               / f"live-{os.getpid()}-{id(self):x}")
        self.watchers = []
        self.watch_pending = False
        if autostart and architecture is not None:
            self.start()

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        with self.lock:
            if self.architecture is None and self.pending_load is None:
                return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(target=self._run, name="dodge-ppo", daemon=True)
        self.worker.start()

    def _new_model(self, env, architecture=None):
        architecture = architecture or self.architecture
        with warnings.catch_warnings():
            # Stable-Baselines3 inspects the policy class, which is
            # ActorCriticPolicy for every architecture here, so it cannot see
            # that hybrid-cnn's features extractor is convolutional. It then
            # warns that a GPU "might take longer than on CPU". Measured on this
            # project that update is 5.1x faster on a GPU, so the warning is
            # wrong for us and only obscures real ones.
            warnings.filterwarnings(
                "ignore",
                message=".*primarily intended to run on the CPU.*",
                category=UserWarning,
            )
            return self._build_model(env, architecture)

    def _build_model(self, env, architecture):
        settings = hyperparameters(architecture)
        return DashboardPPO(
            # velocity-flow replaces the action head, so it needs its own policy
            # class rather than only a features extractor.
            policy_class(architecture),
            env,
            learning_rate=3e-4,
            n_steps=self.rollout_steps,
            # Sixteen minibatches an epoch, floored at 64 samples and capped by
            # the buffer itself.
            #
            # The minibatch has to track the buffer, not sit at a constant: the
            # buffer is n_envs * rollout_steps, and a fixed 64 would have put 32
            # minibatches in every epoch once the buffer reached 2048, doubling
            # the optimizer passes when the update is already most of the cycle.
            # The floor keeps small buffers from being split into gradients too
            # noisy to be worth taking.
            batch_size=minibatch_size(self.n_envs * self.rollout_steps,
                                      architecture),
            n_epochs=settings["n_epochs"],
            gamma=settings["gamma"],
            vf_coef=settings["vf_coef"],
            target_kl=settings["target_kl"],
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=settings["ent_coef"],
            verbose=0,
            tensorboard_log=str(self.logs_dir),
            seed=self.seed,
            device=self.device,
            policy_kwargs=policy_kwargs(architecture),
            learning_tracker=self._track_learning,
            post_update=self._on_post_update,
        )

    def _load_model(self, path, env):
        # Inspected before construction: a checkpoint predating the player
        # channels or the derived senses has narrower first layers, and building
        # the policy around it raises a shape error rather than something
        # recoverable.
        data, params, _ = load_from_zip_file(path, device=self.device)
        if needs_input_migration(params):
            model = self._new_model(env, architecture="hybrid-cnn")
            return migrate_widened_inputs(params["policy"], model, path,
                                          device=self.device)

        # Rebound by loading again with the env rather than by set_env, which
        # asserts the checkpoint was trained on the same number of environments.
        # Without this, changing --envs makes every existing checkpoint
        # unloadable. The second read costs well under a second.
        #
        # n_steps comes from the session rather than the checkpoint so the
        # buffer stays at n_envs * rollout_steps. Inheriting a checkpoint's own
        # value multiplies it by the env count -- a model saved at 512 would
        # collect 2048 samples per update across four envs, quadrupling the
        # wait between learning steps.
        # Every hyperparameter comes from the architecture's table, not from
        # the checkpoint. The table used to reach only _new_model -- and the
        # Models panel is the only way to select an architecture, so every real
        # run loaded a checkpoint and silently kept whatever schedule it was
        # seeded with. A live velocity-flow-v2 run was found at ten epochs and
        # batch 64: 320 optimizer steps an update where the table says 32, with
        # no entropy bonus and no tensorboard log.
        architecture = architecture_from_kwargs(data.get("policy_kwargs"))
        settings = hyperparameters(architecture)
        model = DashboardPPO.load(
            path, env=env, device=self.device,
            custom_objects={
                "n_steps": self.rollout_steps,
                "batch_size": minibatch_size(self.n_envs * self.rollout_steps,
                                             architecture),
                "n_epochs": settings["n_epochs"],
                "gamma": settings["gamma"],
                "vf_coef": settings["vf_coef"],
                "target_kl": settings["target_kl"],
                "ent_coef": settings["ent_coef"],
                "tensorboard_log": str(self.logs_dir),
            },
        )
        model.set_learning_tracker(self._track_learning)
        model.set_post_update(self._on_post_update)
        return model

    def _clear_snapshot(self, status="starting"):
        with self.lock:
            self.data = DashboardSnapshot(
                status=status,
                model_name=self.active_model_name,
                seed=self.seed,
            )

    def _run(self):
        while not self.shutdown_event.is_set():
            self.restart_event.clear()
            with self.lock:
                load_path = self.pending_load
                self.pending_load = None
                env_kwargs = dict(self.env_kwargs)
            self._clear_snapshot("loading" if load_path else "starting")
            env = make_vec_env(
                DodgeEnv,
                n_envs=self.n_envs,
                seed=self.seed,
                env_kwargs=env_kwargs,
                # DummyVecEnv steps envs sequentially in this process, which
                # gives varied experience but no CPU parallelism. Each Dodge env
                # runs its own LuaJIT machine, so real processes are what make
                # extra envs faster rather than merely more diverse.
                vec_env_cls=SubprocVecEnv if self.n_envs > 1 else DummyVecEnv,
            )
            try:
                self.model = self._load_model(load_path, env) if load_path else self._new_model(env)
                self.rollout_steps = self.model.n_steps
                self.model_architecture = architecture_name(self.model)
                callback = DashboardCallback(self)
                reset_num_timesteps = load_path is None
                while not self.shutdown_event.is_set() and not self.restart_event.is_set():
                    self.model.learn(
                        total_timesteps=self.total_steps,
                        callback=callback,
                        reset_num_timesteps=reset_num_timesteps,
                        tb_log_name=self.model_architecture,
                    )
                    reset_num_timesteps = False
                    if self.shutdown_event.is_set() or self.restart_event.is_set():
                        break

                    # Training is open-ended. Each completed chunk is only an
                    # autosave boundary; the same model and optimizer continue
                    # immediately into the next chunk.
                    latest = self.model_dir / f"{self.model_architecture}-latest"
                    self.model.save(latest)
            except Exception as error:
                # Assigned before the status flips to "error": request_save
                # reads status under self.lock and then last_error outside
                # it, so a request landing in the gap between these two
                # would otherwise see status == "error" but last_error still
                # None (or stale), and report "Training failed: None".
                self.last_error = error
                with self.lock:
                    self.data.status = "error"
                self._fail_save_requests(error)
                while not self.shutdown_event.is_set() and not self.restart_event.is_set():
                    time.sleep(0.1)
            finally:
                env.close()
        self._fail_save_requests(RuntimeError("training session closed"))

    def _process_save_requests(self, model):
        while True:
            try:
                request = self.save_requests.get_nowait()
            except queue.Empty:
                return
            try:
                model.save(request.path)
                request.completed_update = self._completed_updates(model)
            except Exception as error:
                request.error = error
            finally:
                request.finished.set()
            if request.kind == "save" and request.error is None:
                # request_save is the only path the GUI's Save button reaches,
                # so a checkpoint written under a new name has to become the
                # one everything else -- the header, the watch title -- calls
                # the active model, the same way save_model used to.
                with self.lock:
                    self.active_model_name = request.output_path.stem
                    self.data.model_name = request.output_path.stem
            if request.kind == "config":
                self._commit_pending_config(request)
            if request.notify:
                self._emit(SessionEvent(
                    request.kind, request.error is None,
                    (str(request.error) if request.error
                     else f"Saved {request.output_path.name}"),
                    request.output_path, request.completed_update,
                ))

    def _commit_pending_config(self, request):
        """Apply a staged game configuration once its checkpoint is on disk.

        A failed save leaves the running game exactly as it was, which is the
        whole point of staging it rather than applying it at the click.
        """
        if self.pending_config is None:
            return
        if request.error is None:
            with self.lock:
                self.env_kwargs = self.pending_config
                self.pending_load = request.output_path
                self.data.status = "reconfiguring"
            self.restart_event.set()
        self.pending_config = None

    def _watchers_alive(self):
        """Prune exited Popen handles and report whether any remain.

        A watcher's poll() returns None while still running and its exit code
        (typically 0) once it has exited. Maintaining this list allows the
        trainer to stop publishing snapshots once the last window closes.
        """
        with self.lock:
            self.watchers = [w for w in self.watchers if w.poll() is None]
            return bool(self.watchers)

    def _on_post_update(self, model):
        """The safe point. Saves first, then a snapshot if anyone needs one."""
        self._process_save_requests(model)
        self._service_pending_watch(model, publish_for_live_watchers=True)

    def _service_pending_watch(self, model, *, publish_for_live_watchers):
        """Publish a snapshot and open a waiting watcher, if either is due.

        Shared by the post-update safe point and the pause loop in
        `_should_stop` -- the spec lists both as service points, and a watch
        request made while paused must not have to wait for a resume that may
        never come.

        `publish_for_live_watchers` is False from the pause loop: that loop
        spins roughly every 50ms while the update counter is frozen, and an
        already-running watcher has nothing new to see until training
        resumes. Publishing there anyway would rewrite the same
        update-XXXXXXXX.zip twenty times a second, and every rename that
        loses its race with the watcher's own open handle emits a
        SessionEvent -- flooding the toast queue at that same rate. A
        pending watch is different: it has never had a snapshot at all, so
        it still needs exactly one publish to start from, paused or not.
        """
        if self.watch_pending or (publish_for_live_watchers and self._watchers_alive()):
            self._publish_snapshot(model)
        if self.watch_pending:
            self.watch_pending = False
            found = newest_snapshot(self.live_watch_dir)
            if found:
                self._launch_watcher(*found)

    def _publish_snapshot(self, model):
        """Write one immutable snapshot, then prune to the newest few.

        Saved to a hidden temporary name and renamed, so a watcher scanning
        the directory can never open a half-written zip.
        """
        update = self._completed_updates(model)
        directory = self.live_watch_dir
        directory.mkdir(parents=True, exist_ok=True)
        # SB3 only appends ".zip" when the path it's given has an empty
        # suffix, so the stem must carry no dot beyond the leading one that
        # hides it -- embedding "update-...zip" in the name would already
        # look like a suffix to SB3, which would then write the path
        # unchanged and leave the rename below with nothing to find.
        stem = directory / f".pending-{int(update):08d}-{secrets.token_hex(4)}"
        try:
            model.save(stem)
            temporary = Path(f"{stem}.zip")
            temporary.replace(directory / snapshot_name(update))
        except Exception as error:
            self._emit(SessionEvent("watch", False,
                                    f"Could not publish snapshot: {error}"))
            return
        self._prune_snapshots(directory)

    @staticmethod
    def _prune_snapshots(directory):
        """Best-effort. A watcher holding an old snapshot open locks it on
        Windows, and losing that race must not interrupt training."""
        snapshots = sorted(directory.glob("update-*.zip"))
        for path in snapshots[:-SNAPSHOT_RETENTION]:
            try:
                path.unlink()
            except OSError:
                pass

    def _track_learning(self, completed, total):
        with self.lock:
            self.data.progress_phase = "learning"
            self.data.learning_progress = min(1.0, completed / max(1, total))

    def _fail_save_requests(self, error):
        while True:
            try:
                request = self.save_requests.get_nowait()
            except queue.Empty:
                return
            request.error = error
            request.finished.set()
            if request.notify:
                self._emit(SessionEvent(request.kind, False, str(error)))

    def request_save(self, path, kind="save"):
        """Queue a checkpoint and return. Results arrive via drain_events()."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            # The caller is the Pygame event loop -- an exception here would
            # close the window and take training down with it.
            self._emit(SessionEvent(kind, False, str(error)))
            return
        if self.model is None:
            self._emit(SessionEvent(kind, False, "No model to save yet"))
            return
        if self.shutdown_event.is_set():
            self._emit(SessionEvent(kind, False, "Training session is closed"))
            return
        with self.lock:
            status = self.data.status
        if status == "error":
            # _run has already failed every request queued before this one
            # and gone to sleep in its error loop -- nothing will ever drain
            # this one, so it must be rejected here rather than stranded.
            self._emit(SessionEvent(kind, False, f"Training failed: {self.last_error}"))
            return
        # `path` is handed to model.save() unchanged; _SaveRequest derives
        # output_path from it using SB3's own append rule (see __post_init__).
        self.save_requests.put(_SaveRequest(
            path=path, finished=threading.Event(), kind=kind, notify=True,
        ))

    def drain_events(self):
        drained = []
        while True:
            try:
                drained.append(self.events.get_nowait())
            except queue.Empty:
                return drained

    def _emit(self, event):
        self.events.put(event)

    def _should_stop(self, model):
        self._process_save_requests(model)
        while self.pause_event.is_set() and not self.shutdown_event.is_set() \
                and not self.restart_event.is_set():
            self._process_save_requests(model)
            self._service_pending_watch(model, publish_for_live_watchers=False)
            with self.lock:
                self.data.status = "paused"
                self.data.steps_per_second = 0.0
            time.sleep(0.05)
        if not self.pause_event.is_set():
            with self.lock:
                if self.data.status == "paused":
                    self.data.status = "training"
        return self.shutdown_event.is_set() or self.restart_event.is_set()

    def toggle_pause(self):
        if self.model is None and not (self.worker and self.worker.is_alive()):
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            with self.lock:
                self.data.status = "training"
        else:
            self.pause_event.set()
            with self.lock:
                self.data.status = "paused"
                self.data.steps_per_second = 0.0

    def reset(self):
        if self.architecture is None:
            return
        self.pause_event.clear()
        with self.lock:
            self.pending_load = None
            # A game-config save may still be queued from before this click.
            # Without dropping it here, _commit_pending_config would land
            # after this reset and overwrite pending_load with the old
            # checkpoint, quietly discarding the reset.
            self.pending_config = None
            self.active_model_name = self.architecture
            self.data.status = "resetting"
            self.data.model_name = self.active_model_name
        self.restart_event.set()

    def available_models(self):
        if not self.model_dir.exists():
            return []
        # User-organised subdirectories are valid model libraries. Runtime and
        # Watch hold ephemeral implementation checkpoints and would otherwise
        # flood the picker with internal snapshots.
        internal = {"runtime", "watch", "recovery"}
        paths = (
            path for path in self.model_dir.rglob("*.zip")
            if path.relative_to(self.model_dir).parts[0].lower() not in internal
        )
        entries = [ModelEntry(path.stem, path, "PPO") for path in paths]
        return sorted(entries, key=lambda entry: entry.path.stat().st_mtime, reverse=True)

    @staticmethod
    def _safe_name(name):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-")
        if not safe_name:
            raise ValueError("Enter a model name")
        return safe_name

    @staticmethod
    def _completed_updates(model):
        counter = getattr(model, "completed_updates", None)
        if counter is not None:
            return int(counter)
        # Checkpoints written before the counter existed.
        epochs = max(int(getattr(model, "n_epochs", 1)), 1)
        return int(getattr(model, "_n_updates", 0)) // epochs

    def save_model(self, name):
        safe_name = self._safe_name(name)
        path = self.model_dir / f"{safe_name}.zip"
        saved = self._save_to_path(path)
        with self.lock:
            self.active_model_name = safe_name
            self.data.model_name = safe_name
        return saved

    def _save_to_path(self, path, return_update=False):
        with self.lock:
            if self.data.status == "error":
                raise ValueError(f"Training failed: {self.last_error}")
        if self.model is None:
            message = ("No model selected" if self.architecture is None
                       and self.pending_load is None
                       else "The PPO model is still starting")
            raise ValueError(message)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        request = _SaveRequest(path=path.with_suffix(""), finished=threading.Event())
        self.save_requests.put(request)
        if not request.finished.wait(timeout=15):
            raise ValueError("Timed out waiting for PPO to save")
        if request.error:
            raise ValueError(f"Could not save model: {request.error}")
        return (path, request.completed_update) if return_update else path

    def load_model(self, entry):
        if entry.kind != "PPO":
            raise ValueError("Select a Stable-Baselines3 PPO model")
        if not entry.path.exists():
            raise ValueError("That model no longer exists")
        self.pause_event.clear()
        with self.lock:
            self.pending_load = entry.path
            # Same reasoning as reset(): a queued config commit must not be
            # allowed to overwrite this choice once its checkpoint lands.
            self.pending_config = None
            self.active_model_name = entry.name
            self.data.status = "loading"
            self.data.model_name = entry.name
        self.restart_event.set()
        self.start()
        return entry.path

    def game_config(self):
        with self.lock:
            return dict(self.env_kwargs)

    def configure_game(self, *, difficulty, patterns, powerups,
                       normal_enemies=True, kamikaze_enemies=True,
                       spawn_rate_percent=100, action_repeat, lives=0):
        """Rebuild the games around the current policy with new mechanics."""
        difficulty = int(difficulty)
        action_repeat = int(action_repeat)
        lives = int(lives)
        spawn_rate_percent = validate_spawn_rate_percent(spawn_rate_percent)
        if difficulty not in (1, 2, 3):
            raise ValueError("Difficulty must be easy, normal, or hard")
        if not 1 <= action_repeat <= 8:
            raise ValueError("Decision interval must be between 1 and 8 frames")
        if not 0 <= lives <= 5:
            raise ValueError("Training lives must be between 0 and 5")
        changes = {
            "difficulty": difficulty,
            "patterns": bool(patterns),
            "powerups": bool(powerups),
            "normal_enemies": bool(normal_enemies),
            "kamikaze_enemies": bool(kamikaze_enemies),
            "spawn_rate_percent": spawn_rate_percent,
            "action_repeat": action_repeat,
            "lives": lives,
        }
        with self.lock:
            # Merged rather than rebuilt, so a setting this panel does not own
            # survives the comparison below instead of reading as a change and
            # restarting training on every apply.
            config = {**self.env_kwargs, **changes}
            if config == self.env_kwargs:
                return False

        if self.model is None:
            with self.lock:
                self.env_kwargs = config
                idle = self.architecture is None and self.model is None
                self.data.status = "idle" if idle else "reconfiguring"
            if not idle:
                self.restart_event.set()
            return True

        # Queued, not taken here: this runs on the event loop, and the trainer
        # may be mid-update. The config is applied once the checkpoint lands
        # on disk, so a failed save leaves the running game exactly as it was.
        self.pending_config = config
        self.pause_event.clear()
        self.request_save(self.model_dir / "runtime" / "game-config.zip",
                          kind="config")
        return "pending"

    def watch_agent(self):
        """Open a watcher without blocking the event loop.

        Returns True if a window was launched now, False if one is waiting for
        the next completed update. An older same-session snapshot is preferred
        over waiting: the watcher reloads to the newest policy at its first
        death anyway.
        """
        if self.model is None:
            self._emit(SessionEvent("watch", False, "No model to watch yet"))
            return False
        existing = newest_snapshot(self.live_watch_dir)
        if existing:
            return self._launch_watcher(*existing)
        self.watch_pending = True
        return False

    def _launch_watcher(self, path, update):
        with self.lock:
            model_name = self.active_model_name
            game_config = dict(self.env_kwargs)
            self.data.update = update
        project = Path(__file__).resolve().parents[1]
        difficulty = {1: "easy", 2: "normal", 3: "hard"}[game_config["difficulty"]]
        command = [
            sys.executable, str(project / "watch_ppo.py"), str(path),
            "--difficulty", difficulty,
            "--action-repeat", str(game_config["action_repeat"]),
            "--spawn-rate-percent", str(game_config["spawn_rate_percent"]),
            "--update", str(update),
            "--current-update", str(update),
            "--live-dir", str(self.live_watch_dir),
            "--seed", str(secrets.randbelow(2**31 - 1)),
            "--model-name", model_name,
        ]
        if game_config["patterns"]:
            command.append("--patterns")
        command.append("--powerups" if game_config["powerups"] else "--no-powerups")
        command.append("--normal-enemies" if game_config["normal_enemies"]
                       else "--no-normal-enemies")
        command.append("--kamikaze-enemies" if game_config["kamikaze_enemies"]
                       else "--no-kamikaze-enemies")
        try:
            process = subprocess.Popen(command, cwd=project)
        except OSError as error:
            self._emit(SessionEvent("watch", False, f"Could not open window: {error}"))
            return False
        with self.lock:
            self.watchers.append(process)
        self.last_watch_label = f"{model_name} | Update {update}"
        self._emit(SessionEvent("watch", True, f"Opening {self.last_watch_label}",
                                path, update))
        return True

    def tick(self, dt):
        return None

    def close(self):
        self.shutdown_event.set()
        self.restart_event.set()
        self.pause_event.clear()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=10)
