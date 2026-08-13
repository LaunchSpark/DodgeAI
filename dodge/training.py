"""Background Stable-Baselines3 session used by the Pygame dashboard."""

from dataclasses import dataclass
import math
from pathlib import Path
import queue
import re
import secrets
import subprocess
import sys
import threading
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import DodgeEnv
from .observe import FRAME_OBSERVATION_SIZE, FRAME_STACK_SIZE
from .policies import ARCHITECTURES, architecture_name, policy_kwargs
from .telemetry import DashboardSnapshot, EVENT_KEYS, ModelEntry


RATE_WINDOW_SECONDS = 0.5


@dataclass
class _SaveRequest:
    path: Path
    finished: threading.Event
    error: Exception | None = None
    completed_update: int = 0


class DashboardPPO(PPO):
    """PPO variant that exposes optimizer minibatch progress to the dashboard."""

    def __init__(self, *args, learning_tracker=None, **kwargs):
        self._learning_tracker = learning_tracker
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_learning_tracker"]

    def set_learning_tracker(self, tracker):
        self._learning_tracker = tracker

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
            return super().train()
        finally:
            optimizer.step = original_step


def migrate_single_frame_policy(source, target):
    """Copy a one-frame PPO policy into the current half of a two-frame policy."""
    source_width = int(np.prod(source.observation_space.shape))
    target_width = int(np.prod(target.observation_space.shape))
    if source_width != FRAME_OBSERVATION_SIZE or \
            target_width != FRAME_OBSERVATION_SIZE * FRAME_STACK_SIZE:
        raise ValueError(
            f"Cannot migrate observation size {source_width} to {target_width}"
        )
    if source.action_space != target.action_space:
        raise ValueError("Cannot migrate a model with a different action space")

    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()
    first_layers = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    }
    current_frame_start = target_width - source_width
    for name, target_value in target_state.items():
        source_value = source_state.get(name)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target_value.copy_(source_value)
        elif name in first_layers and \
                source_value.shape == target_value[:, current_frame_start:].shape:
            # Previous-frame columns retain the new policy's normal initialization.
            target_value[:, current_frame_start:].copy_(source_value)
    target.policy.load_state_dict(target_state)

    # Keep progress metadata, but use a fresh optimizer for the enlarged network.
    target.num_timesteps = source.num_timesteps
    target._n_updates = source._n_updates
    target._episode_num = source._episode_num
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
                    data.episode_length.append(float(info.get("frames", 0)) / 60.0)
                    data.enemies_killed.append(float(self.episode_kills.pop(index, 0)))

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

        with self.session.lock:
            data = self.session.data
            data.rollout_progress = 1.0
            data.entropy.append(entropy)
            for key, count in self.rollout_events.items():
                event = data.events[key]
                event.rollout = count
                event.history.append(count)
                self.rollout_events[key] = 0

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
        n_envs=4,
        rollout_steps=128,
        difficulty=1,
        patterns=False,
        powerups=False,
        action_repeat=4,
        architecture="hybrid-cnn",
        device="auto",
        autostart=True,
    ):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        self.seed = seed
        self.model_dir = Path(model_dir)
        self.logs_dir = Path(logs_dir)
        self.total_steps = total_steps
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        # PPO's buffer is n_envs * rollout_steps samples per update. The
        # defaults keep that at 512 while collecting it from four independently
        # seeded games, so each game only advances 128 decisions before the
        # policy learns and openings stay varied.
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        self.architecture = architecture
        self.model_architecture = architecture
        # The simulation is CPU-bound and stays there, but the policy update is
        # plain torch. With the hybrid CNN that update is roughly 80% of
        # wall-clock, so it is the part worth accelerating. "auto" falls back to
        # CPU when the installed torch has no CUDA support.
        self.device = get_device(device)
        self.env_kwargs = {
            "difficulty": difficulty,
            "patterns": patterns,
            "powerups": powerups,
            "action_repeat": action_repeat,
        }
        self.lock = threading.RLock()
        self.data = DashboardSnapshot(seed=seed)
        self.pause_event = threading.Event()
        self.restart_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.save_requests = queue.Queue()
        self.pending_load = None
        self.model = None
        self.active_model_name = architecture
        self.last_error = None
        self.last_watch_label = None
        self.worker = None
        if autostart:
            self.start()

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.worker = threading.Thread(target=self._run, name="dodge-ppo", daemon=True)
        self.worker.start()

    def _new_model(self, env, architecture=None):
        architecture = architecture or self.architecture
        return DashboardPPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=self.rollout_steps,
            batch_size=min(64, self.rollout_steps),
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            tensorboard_log=str(self.logs_dir),
            seed=self.seed,
            device=self.device,
            policy_kwargs=policy_kwargs(architecture),
            learning_tracker=self._track_learning,
        )

    def _load_model(self, path, env):
        # Loaded without an env first, because a legacy checkpoint's observation
        # space may not match and load(path, env=...) rejects that outright.
        source = DashboardPPO.load(path, device=self.device)
        if source.observation_space != env.observation_space:
            source.set_learning_tracker(self._track_learning)
            return migrate_single_frame_policy(
                source, self._new_model(env, architecture="mlp")
            )

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
        model = DashboardPPO.load(
            path, env=env, device=self.device,
            custom_objects={"n_steps": self.rollout_steps},
        )
        model.set_learning_tracker(self._track_learning)
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
            self._clear_snapshot("loading" if load_path else "starting")
            env = make_vec_env(
                DodgeEnv,
                n_envs=self.n_envs,
                seed=self.seed,
                env_kwargs=self.env_kwargs,
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
                self.model.learn(
                    total_timesteps=self.total_steps,
                    callback=callback,
                    reset_num_timesteps=load_path is None,
                    tb_log_name=self.model_architecture,
                )
                if self.shutdown_event.is_set():
                    break
                if self.restart_event.is_set():
                    continue

                latest = self.model_dir / f"{self.model_architecture}-latest"
                self.model.save(latest)
                with self.lock:
                    self.data.status = "complete"
                    self.data.progress_phase = "complete"
                    self.data.rollout_progress = 1.0
                while not self.shutdown_event.is_set() and not self.restart_event.is_set():
                    self._process_save_requests(self.model)
                    time.sleep(0.05)
            except Exception as error:
                with self.lock:
                    self.data.status = "error"
                self._fail_save_requests(error)
                self.last_error = error
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

    def _should_stop(self, model):
        self._process_save_requests(model)
        while self.pause_event.is_set() and not self.shutdown_event.is_set() \
                and not self.restart_event.is_set():
            self._process_save_requests(model)
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
        self.pause_event.clear()
        with self.lock:
            self.pending_load = None
            self.active_model_name = self.architecture
            self.data.status = "resetting"
            self.data.model_name = self.active_model_name
        self.restart_event.set()

    def available_models(self):
        if not self.model_dir.exists():
            return []
        entries = [ModelEntry(path.stem, path, "PPO") for path in self.model_dir.glob("*.zip")]
        return sorted(entries, key=lambda entry: entry.path.stat().st_mtime, reverse=True)

    @staticmethod
    def _safe_name(name):
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-")
        if not safe_name:
            raise ValueError("Enter a model name")
        return safe_name

    @staticmethod
    def _completed_updates(model):
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
            raise ValueError("The PPO model is still starting")
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
            self.active_model_name = entry.name
            self.data.status = "loading"
            self.data.model_name = entry.name
        self.restart_event.set()
        return entry.path

    def watch_agent(self):
        with self.lock:
            model_name = self.active_model_name
        seed = secrets.randbelow(2**31 - 1)
        temporary = self.model_dir / "watch" / f".pending-seed-{seed}.zip"
        try:
            temporary, update = self._save_to_path(temporary, return_update=True)
        except ValueError:
            return False
        safe_model_name = self._safe_name(model_name)
        path = temporary.with_name(
            f"{safe_model_name}-update-{update:06d}-seed-{seed}.zip"
        )
        temporary.replace(path)
        with self.lock:
            self.data.update = update
        project = Path(__file__).resolve().parents[1]
        difficulty = {1: "easy", 2: "normal", 3: "hard"}[self.env_kwargs["difficulty"]]
        command = [
            sys.executable,
            str(project / "watch_ppo.py"),
            str(path),
            "--difficulty",
            difficulty,
            "--action-repeat",
            str(self.env_kwargs["action_repeat"]),
            "--update",
            str(update),
            "--seed",
            str(seed),
            "--model-name",
            model_name,
        ]
        if self.env_kwargs["patterns"]:
            command.append("--patterns")
        if self.env_kwargs["powerups"]:
            command.append("--powerups")
        subprocess.Popen(command, cwd=project)
        self.last_watch_label = f"{model_name} | Update {update} | Seed {seed}"
        return True

    def tick(self, dt):
        return None

    def close(self):
        self.shutdown_event.set()
        self.restart_event.set()
        self.pause_event.clear()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=10)
