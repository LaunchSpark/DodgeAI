import faulthandler
from pathlib import Path
import sys
import time

import pytest
import numpy as np

pytest.importorskip("stable_baselines3")
pytest.importorskip("lupa.luajit21")

from dodge.telemetry import ModelEntry
from dodge.observe import OBSERVATION_SIZE
from dodge.policies import ARCHITECTURES
from dodge.training import DashboardPPO, PPOTrainingSession
from watch_ppo import window_title

pytestmark = pytest.mark.skipif(not Path("dodge.p8.png").exists(), reason="dodge.p8.png not present")


class _Alive:
    """A Popen stand-in that never exits."""
    def poll(self):
        return None


@pytest.fixture(scope="module", autouse=True)
def _quiet_luajit_seh():
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


def _wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for background PPO session")


def test_model_list_only_contains_ppo_checkpoints(tmp_path):
    (tmp_path / "one.zip").write_bytes(b"placeholder")
    (tmp_path / "old.demo.json").write_text("{}", encoding="utf-8")
    library = tmp_path / "experiments" / "v2"
    library.mkdir(parents=True)
    (library / "vector-flow-v2.zip").write_bytes(b"placeholder")
    internal = tmp_path / "watch"
    internal.mkdir()
    (internal / "temporary.zip").write_bytes(b"placeholder")
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert {entry.name for entry in session.available_models()} == {
        "one", "vector-flow-v2"
    }


def test_training_enables_powerups_by_default(tmp_path):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert session.env_kwargs["powerups"] is True


def test_session_can_wait_for_an_explicit_model_selection(tmp_path, monkeypatch):
    session = PPOTrainingSession(model_dir=tmp_path, architecture=None,
                                 autostart=True, device="cpu")
    assert session.worker is None
    assert session.model is None
    assert session.data.status == "idle"
    assert session.data.model_name == ""

    checkpoint = tmp_path / "vector-flow-v2.zip"
    checkpoint.write_bytes(b"checkpoint")
    started = []
    monkeypatch.setattr(session, "start", lambda: started.append(True))

    session.load_model(ModelEntry("vector-flow-v2", checkpoint, "PPO"))

    assert started == [True]
    assert session.pending_load == checkpoint
    assert session.data.model_name == "vector-flow-v2"


def test_dashboard_ppo_reports_each_learning_minibatch():
    import gymnasium as gym
    from gymnasium import spaces

    class TinyEnv(gym.Env):
        observation_space = spaces.Box(0.0, 1.0, (4,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    progress = []
    model = DashboardPPO(
        "MlpPolicy",
        TinyEnv(),
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        device="cpu",
        learning_tracker=lambda completed, total: progress.append((completed, total)),
    )
    model.learn(total_timesteps=8)

    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def test_live_session_pauses_saves_and_resumes(tmp_path):
    session = PPOTrainingSession(
        seed=19, device="cpu",
        model_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        total_steps=512,
        rollout_steps=64,
        # This exercises the session lifecycle, not a policy shape. mlp is the
        # cheapest, and on CPU the default velocity-flow takes ~4x longer per
        # forward because it never downsamples -- enough to time this out.
        architecture="mlp",
    )
    try:
        # Reward is recorded per finished episode, not per rollout, so wait for
        # an episode to end rather than for the first rollout to close.
        _wait_for(lambda: (session.data.update >= 1 and session.data.reward)
                  or session.data.status == "error")
        assert session.data.status != "error", repr(session.last_error)
        assert session.data.reward
        assert len(session.data.episode_length) == len(session.data.reward)
        assert session.data.events["survival_frames"].total > 0

        session.toggle_pause()
        _wait_for(lambda: session.data.status == "paused")
        path = session.save_model("paused policy")
        assert path.exists()
        exact_path, exact_update = session._save_to_path(
            tmp_path / "models" / "exact-update.zip", return_update=True
        )
        assert exact_path.exists()
        assert exact_update == session.data.update
        assert exact_update > 0

        session.toggle_pause()
        _wait_for(lambda: session.data.status == "training")
    finally:
        session.close()
    assert not session.worker.is_alive()


def test_training_continues_after_autosave_boundary(tmp_path, monkeypatch):
    from dodge import training

    class StubEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class StubModel:
        n_steps = 64

        def __init__(self, session):
            self.session = session
            self.learn_calls = []
            self.saved = []

        def learn(self, **kwargs):
            self.learn_calls.append(kwargs)
            if len(self.learn_calls) == 2:
                self.session.shutdown_event.set()
            return self

        def save(self, path):
            self.saved.append(path)

    env = StubEnv()
    session = training.PPOTrainingSession(
        model_dir=tmp_path / "models", logs_dir=tmp_path / "logs",
        total_steps=128, autostart=False, device="cpu",
    )
    model = StubModel(session)
    monkeypatch.setattr(training, "make_vec_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(training, "architecture_name", lambda value: "hybrid-cnn")
    monkeypatch.setattr(session, "_new_model", lambda value: model)

    session._run()

    assert len(model.learn_calls) == 2
    assert [call["reset_num_timesteps"] for call in model.learn_calls] == [True, False]
    assert model.saved == [tmp_path / "models" / "hybrid-cnn-latest"]
    assert session.data.status != "complete"
    assert env.closed


def test_watch_launches_immediately_from_an_existing_snapshot(tmp_path, monkeypatch):
    from dodge.training import PPOTrainingSession, snapshot_name

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.live_watch_dir.mkdir(parents=True)
    (session.live_watch_dir / snapshot_name(12)).write_bytes(b"checkpoint")

    commands = []
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command) or _Alive())

    assert session.watch_agent() is True
    assert not session.watch_pending
    assert commands and commands[0][commands[0].index("--current-update") + 1] == "12"
    assert commands[0][commands[0].index("--live-dir") + 1] == str(session.live_watch_dir)


def test_watch_with_no_snapshot_waits_for_the_next_safe_point(tmp_path, monkeypatch):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    commands = []
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command) or _Alive())

    assert session.watch_agent() is False, "nothing to open yet"
    assert session.watch_pending
    assert commands == []

    assert session.watch_agent() is False, "a second click does not queue twice"
    assert session.save_requests.empty()

    class Stub:
        completed_updates = 3
        def save(self, path):
            path = Path(path)
            if path.suffix == "":
                path = path.with_suffix(".zip")
            path.write_bytes(b"checkpoint")

    session._on_post_update(Stub())
    assert commands, "the pending watch launched at the safe point"
    assert not session.watch_pending


def test_watch_uses_unique_update_snapshot_and_random_seed(tmp_path, monkeypatch):
    from dodge.training import snapshot_name

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.env_kwargs.update(
        difficulty=3, patterns=True, powerups=False, action_repeat=7
    )
    session.model = object()
    session.live_watch_dir.mkdir(parents=True)
    (session.live_watch_dir / snapshot_name(37)).write_bytes(b"checkpoint")

    commands = []
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command) or _Alive())

    assert session.watch_agent() is True
    architecture = session.architecture
    assert (session.live_watch_dir / snapshot_name(37)).exists()
    command = commands[0]
    assert command[command.index("--live-dir") + 1] == str(session.live_watch_dir)
    assert command[command.index("--current-update") + 1] == "37"
    assert command[command.index("--model-name") + 1] == architecture
    assert command[command.index("--difficulty") + 1] == "hard"
    assert command[command.index("--action-repeat") + 1] == "7"
    assert "--patterns" in command
    assert "--no-powerups" in command


def test_game_config_restarts_with_the_live_policy(tmp_path, monkeypatch):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    checkpoint = tmp_path / "runtime" / "game-config.zip"
    saved = []

    def fake_save(path, return_update=False):
        saved.append(path)
        return checkpoint

    monkeypatch.setattr(session, "_save_to_path", fake_save)

    assert session.configure_game(
        difficulty=3, patterns=True, powerups=False, action_repeat=7
    )
    assert saved == [checkpoint]
    assert session.pending_load == checkpoint
    assert session.game_config() == {
        "difficulty": 3,
        "patterns": True,
        "powerups": False,
        "action_repeat": 7,
        "lives": 0,
        "max_episode_frames": None,
    }
    assert session.data.status == "reconfiguring"
    assert session.restart_event.is_set()


@pytest.mark.parametrize("n_envs,expected", [(1, "DummyVecEnv"), (4, "SubprocVecEnv")])
def test_parallel_envs_use_separate_processes(tmp_path, monkeypatch, n_envs, expected):
    """Extra envs are only faster if they are actually stepped in parallel.

    DummyVecEnv would step each Dodge machine sequentially in this process,
    which diversifies experience but adds no throughput.
    """
    from dodge import training

    captured = {}

    def fake_make_vec_env(env_cls, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop before spawning")

    monkeypatch.setattr(training, "make_vec_env", fake_make_vec_env)
    session = training.PPOTrainingSession(
        model_dir=tmp_path, logs_dir=tmp_path, n_envs=n_envs, autostart=False,
        device="cpu",
    )
    # make_vec_env is called before the loop's try block, so the sentinel
    # exception escapes _run instead of being recorded as a session error.
    with pytest.raises(RuntimeError, match="stop before spawning"):
        session._run()
    assert captured["n_envs"] == n_envs
    assert captured["vec_env_cls"].__name__ == expected


def _matching_vec_env(n_envs):
    """A VecEnv with Dodge's spaces but no PICO-8 machine, for load tests."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    class SpacesOnlyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(0.0, 1.0, (OBSERVATION_SIZE,),
                                                dtype=np.float32)
            self.action_space = spaces.Discrete(9)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32), {}

        def step(self, action):
            return (np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                    0.0, False, False, {})

    return DummyVecEnv([SpacesOnlyEnv for _ in range(n_envs)])


@pytest.mark.parametrize("saved_envs,loaded_envs", [(1, 4), (4, 1), (4, 4)])
def test_checkpoint_loads_into_a_different_env_count(tmp_path, saved_envs, loaded_envs):
    """A checkpoint must not be locked to the env count that produced it.

    set_env asserts the counts match, so changing --envs would otherwise make
    every existing checkpoint unloadable.
    """
    from dodge.training import DashboardPPO, PPOTrainingSession

    saved = DashboardPPO("MlpPolicy", _matching_vec_env(saved_envs),
                         n_steps=8, batch_size=8, n_epochs=1, device="cpu")
    path = tmp_path / "checkpoint.zip"
    saved.save(path)

    session = PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path,
                                 n_envs=loaded_envs, autostart=False,
                                 device="cpu")
    model = session._load_model(path, _matching_vec_env(loaded_envs))
    assert model.n_envs == loaded_envs
    assert model.get_env().num_envs == loaded_envs


def test_loaded_checkpoint_adopts_the_session_buffer_size(tmp_path):
    """Resuming must not inherit the checkpoint's own rollout length.

    A model saved at n_steps=512 loaded into four envs would collect 2048
    samples per update -- four times the intended buffer, and four times the
    wait between learning steps.
    """
    from dodge.training import DashboardPPO, PPOTrainingSession

    saved = DashboardPPO("MlpPolicy", _matching_vec_env(1),
                         n_steps=512, batch_size=64, n_epochs=1, device="cpu")
    path = tmp_path / "wide.zip"
    saved.save(path)
    assert saved.n_steps == 512

    session = PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path,
                                 n_envs=4, rollout_steps=128, autostart=False,
                                 device="cpu")
    model = session._load_model(path, _matching_vec_env(4))
    assert model.n_steps == 128
    assert model.rollout_buffer.buffer_size == 128
    assert model.n_envs * model.n_steps == 512


def test_auto_device_only_reaches_for_a_gpu_where_it_helps(monkeypatch):
    """The MLP update measured no faster on a GPU, so auto keeps it on the CPU.

    An explicit choice is always honoured; only "auto" is opinionated.
    """
    from dodge import training

    monkeypatch.setattr(training, "get_device", lambda name: f"resolved:{name}")
    assert training.resolve_device("auto", "hybrid-cnn") == "resolved:auto"
    assert training.resolve_device("auto", "mlp") == "resolved:cpu"
    assert training.resolve_device("cuda", "mlp") == "resolved:cuda"
    assert training.resolve_device("cpu", "hybrid-cnn") == "resolved:cpu"


def test_rejects_a_non_positive_env_count(tmp_path):
    from dodge.training import PPOTrainingSession
    with pytest.raises(ValueError):
        PPOTrainingSession(model_dir=tmp_path, n_envs=0, autostart=False, device="cpu")


def test_rejects_a_non_positive_training_chunk(tmp_path):
    from dodge.training import PPOTrainingSession
    with pytest.raises(ValueError):
        PPOTrainingSession(model_dir=tmp_path, total_steps=0,
                           autostart=False, device="cpu")


def test_watch_runs_episodes_without_a_frame_cap(tmp_path, monkeypatch):
    """Watching is not training, so an episode ends only when the agent dies."""
    import watch_ppo

    built = {}

    class StubEnv:
        def __init__(self, **kwargs):
            built.update(kwargs)

        def reset(self, seed=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return (np.zeros(1, dtype=np.float32), 0.0, True, False,
                    {"survived_frames": 4})

        def close(self):
            return None

    class StubModel:
        def predict(self, observation, deterministic=True):
            return 0, None

    model_path = tmp_path / "policy.zip"
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(watch_ppo, "DodgeEnv", StubEnv)
    monkeypatch.setattr(watch_ppo.PPO, "load", staticmethod(lambda *a, **k: StubModel()))
    monkeypatch.setattr(watch_ppo.pygame.event, "get", lambda: [])
    monkeypatch.setattr(watch_ppo.pygame.display, "set_caption", lambda title: None)
    monkeypatch.setattr(sys, "argv",
                        ["watch_ppo.py", str(model_path), "--frames", "4"])

    assert watch_ppo.main() == 0
    assert "max_episode_frames" in built
    assert built["max_episode_frames"] is None


def test_watch_window_title_identifies_policy_and_run():
    assert window_title("patient-dodger", 37, 1849201, episode=4) == (
        "patient-dodger | Update 37 | Episode 4 | Seed 1849201"
    )


def test_save_and_load_set_the_active_model_name(tmp_path, monkeypatch):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    monkeypatch.setattr(session, "_save_to_path", lambda path: path)
    session.save_model("Patient Dodger")
    assert session.active_model_name == "Patient-Dodger"
    assert session.data.model_name == "Patient-Dodger"

    loaded = ModelEntry("aggressive-v2", tmp_path / "aggressive-v2.zip", "PPO")
    loaded.path.write_bytes(b"placeholder")
    session.load_model(loaded)
    assert session.active_model_name == "aggressive-v2"
    assert session.data.model_name == "aggressive-v2"


def test_checkpoint_without_player_channels_migrates(tmp_path):
    """A hybrid checkpoint from before the player channels must still load.

    Its grid convolution has 16 input channels against the current 18, which
    is a hard shape error rather than something SB3 can reconcile.
    """
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    from dodge.policies import HAZARD_CHANNELS, HYBRID_INPUT_CHANNELS, policy_kwargs
    from dodge.training import GRID_CONV_WEIGHT, DashboardPPO, PPOTrainingSession

    saved = DashboardPPO("MlpPolicy", _matching_vec_env(1), n_steps=8,
                         batch_size=8, n_epochs=1, device="cpu",
                         policy_kwargs=policy_kwargs("hybrid-cnn"))
    # Narrow the grid convolution back to the pre-player shape.
    state = saved.policy.state_dict()
    for name in list(state):
        if name.endswith(GRID_CONV_WEIGHT.split(".", 1)[-1]) and state[name].dim() == 4:
            state[name] = state[name][:, :HAZARD_CHANNELS].clone()
    path = tmp_path / "legacy.zip"
    saved.save(path)
    data, params, variables = load_from_zip_file(path, device="cpu")
    params["policy"] = {name: value for name, value in state.items()}
    data["num_timesteps"] = 4242
    from stable_baselines3.common.save_util import save_to_zip_file
    save_to_zip_file(path, data=data, params=params, pytorch_variables=variables)

    session = PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path,
                                 n_envs=1, autostart=False, device="cpu")
    model = session._load_model(path, _matching_vec_env(1))

    weight = model.policy.features_extractor.grid_net[0].weight
    assert weight.shape[1] == HYBRID_INPUT_CHANNELS
    torch.testing.assert_close(weight[:, :HAZARD_CHANNELS],
                               state[GRID_CONV_WEIGHT])
    assert weight[:, HAZARD_CHANNELS:].abs().sum() > 0, "player weights initialised"
    assert model.num_timesteps == 4242


@pytest.mark.parametrize("n_envs,rollout_steps,expected", [
    (8, 256, 128),   # the defaults: a 2048-sample buffer, 16 minibatches
    (8, 64, 64),     # buffer 512 comfortably exceeds the target minibatch
    (16, 32, 64),    # buffer is still 512 even though each env is short
    (1, 8, 8),       # genuinely small buffer must clamp
])
def test_minibatch_size_follows_the_whole_buffer(tmp_path, monkeypatch,
                                                 n_envs, rollout_steps, expected):
    """The buffer is n_envs * rollout_steps, so clamping to steps alone was
    shrinking the minibatch whenever envs were traded for steps."""
    from dodge import training

    captured = {}

    class Recorder(training.DashboardPPO):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(training, "DashboardPPO", Recorder)
    session = training.PPOTrainingSession(
        model_dir=tmp_path, logs_dir=tmp_path, n_envs=n_envs,
        rollout_steps=rollout_steps, autostart=False, device="cpu",
    )
    session._new_model(_matching_vec_env(n_envs))
    assert captured["n_steps"] == rollout_steps
    assert captured["batch_size"] == expected


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_saves_and_loads_through_the_session(tmp_path, architecture):
    """Loading must not special-case an architecture.

    velocity-flow swaps the action head for a policy subclass, so it is the one
    most likely to need a bespoke path. It does not: Stable-Baselines3 records
    the policy class in the checkpoint, and the migration check keys off
    hybrid-cnn's layer names, which a field policy simply does not have.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    from dodge.env import DodgeEnv
    from dodge.policies import architecture_name, policy_class, policy_kwargs
    from dodge.training import PPOTrainingSession

    saved = PPO(policy_class(architecture), DodgeEnv(max_episode_frames=20),
                n_steps=16, batch_size=8, seed=0, device="cpu",
                policy_kwargs=policy_kwargs(architecture))
    saved.save(tmp_path / architecture)

    session = PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path,
                                 autostart=False, device="cpu", n_envs=1,
                                 rollout_steps=16)
    assert architecture in [entry.name for entry in session.available_models()]

    env = make_vec_env(DodgeEnv, n_envs=1,
                       env_kwargs={"max_episode_frames": 20})
    try:
        loaded = session._load_model(tmp_path / f"{architecture}.zip", env)
        assert architecture_name(loaded) == architecture
        assert loaded.n_steps == session.rollout_steps
        observation = np.zeros((1, loaded.observation_space.shape[0]),
                               dtype=np.float32)
        action, _ = loaded.predict(observation, deterministic=True)
        assert 0 <= int(action[0]) < 9
    finally:
        env.close()


def test_the_dashboard_command_line_accepts_every_architecture(tmp_path):
    """Registering an architecture should be enough; the CLI reads the list."""
    import os

    import training_gui
    from dodge.policies import ARCHITECTURES

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for architecture in ARCHITECTURES:
        # --screenshot renders one frame of sample telemetry and exits, so this
        # exercises argument parsing without starting a trainer.
        assert training_gui.main([
            "--architecture", architecture,
            "--screenshot", str(tmp_path / f"{architecture}.png"),
        ]) == 0


@pytest.mark.parametrize("architecture", ["velocity-flow", "mlp"])
def test_new_policy_tool_makes_an_architecture_selectable(tmp_path, architecture):
    """The Models panel lists files, and loading one switches architecture.

    So an architecture with no checkpoint on disk is unreachable from the
    dashboard -- there is nothing to click. This writes the starting point.
    """
    from stable_baselines3.common.env_util import make_vec_env

    from dodge.env import DodgeEnv
    from dodge.policies import architecture_name
    from dodge.training import PPOTrainingSession
    from tools.new_policy import main

    assert main([architecture, "--models-dir", str(tmp_path)]) == 0

    session = PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path,
                                 autostart=False, device="cpu", n_envs=1,
                                 rollout_steps=16)
    assert architecture in [entry.name for entry in session.available_models()]

    env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
    try:
        loaded = session._load_model(tmp_path / f"{architecture}.zip", env)
        assert architecture_name(loaded) == architecture
    finally:
        env.close()


def test_lives_are_configurable_but_never_reach_the_watch_window(tmp_path, monkeypatch):
    """Lives are a training device only.

    Watching has to show what the policy actually does, and a policy that gets
    three free hits is not the one being shipped, so the watch command is built
    without them however the trainer is configured.
    """
    from dodge.training import snapshot_name

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert session.game_config()["lives"] == 0, "off unless asked for"

    assert session.configure_game(difficulty=1, patterns=False, powerups=True,
                                  action_repeat=4, lives=3)
    assert session.game_config()["lives"] == 3

    session.model = object()
    session.live_watch_dir.mkdir(parents=True)
    (session.live_watch_dir / snapshot_name(1)).write_bytes(b"checkpoint")

    commands = []
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command) or _Alive())

    assert session.watch_agent()

    assert "--lives" not in commands[0]


def test_negative_lives_are_rejected_by_the_session(tmp_path):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    with pytest.raises(ValueError):
        session.configure_game(difficulty=1, patterns=False, powerups=True,
                               action_repeat=4, lives=-1)


def test_survival_time_measures_the_run_not_the_episode(tmp_path, monkeypatch):
    """With lives on, an episode ends at every absorbed hit.

    Charting those as survival time would silently redefine the dashboard's
    headline metric as 'time between hits' and break comparability with every
    curve recorded before lives existed.
    """
    from dodge.training import DashboardCallback

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    monkeypatch.setattr(session, "_should_stop", lambda model: False)
    callback = DashboardCallback(session)
    callback.model = None
    callback.num_timesteps = 1
    callback.n_calls = 1

    def finish_episode(frames, run_frames, run_over, deaths, lives_spent):
        callback.locals = {
            "infos": [{"frames": frames, "run_frames": run_frames,
                       "run_over": run_over, "score": 0.0,
                       "training_events": {"deaths": deaths,
                                           "lives_spent": lives_spent}}],
            "dones": [True],
            "rewards": np.array([1.0]),
        }
        callback._on_step()

    finish_episode(frames=120, run_frames=120, run_over=False, deaths=0, lives_spent=1)
    assert len(session.data.episode_length) == 0, "an absorbed hit does not end the run"
    assert len(session.data.reward) == 1, "but it does end an episode"

    finish_episode(frames=60, run_frames=180, run_over=True, deaths=1, lives_spent=0)

    assert session.data.episode_length.as_list() == [pytest.approx(3.0)], "180 run frames"
    assert session.data.events["lives_spent"].total == 1


def test_training_runs_without_a_frame_cap(tmp_path):
    """The cap capped difficulty, not just time.

    Difficulty ramps with score, so an agent that reliably reached 60 seconds
    never saw a harder game than 60 seconds reaches. Episode turnover, which is
    what the cap was for, now comes from training lives instead.
    """
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert session.env_kwargs["max_episode_frames"] is None


def test_game_config_preserves_settings_it_does_not_manage(tmp_path):
    """configure_game compares its dict against env_kwargs to decide whether to
    restart, so a key it does not set must still survive the comparison."""
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False,
                                 max_episode_frames=1200, device="cpu")

    assert session.configure_game(difficulty=2, patterns=False, powerups=True,
                                  action_repeat=4, lives=1)

    assert session.game_config()["max_episode_frames"] == 1200
    assert not session.configure_game(difficulty=2, patterns=False, powerups=True,
                                      action_repeat=4, lives=1), "no change, no restart"


def test_survival_time_is_banded_by_life(tmp_path, monkeypatch):
    """Each run contributes one value per band, so the chart can stack them.

    The bands have to sum to the run total, or the stack would say something
    different from the trend line drawn over it.
    """
    from dodge.training import DashboardCallback

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    monkeypatch.setattr(session, "_should_stop", lambda model: False)
    callback = DashboardCallback(session)
    callback.model = None
    callback.num_timesteps = 1
    callback.n_calls = 1

    def finish_episode(frames, run_frames, run_over):
        callback.locals = {
            "infos": [{"frames": frames, "run_frames": run_frames,
                       "run_over": run_over, "score": 0.0,
                       "training_events": {}}],
            "dones": [True],
            "rewards": np.array([1.0]),
        }
        callback._on_step()

    finish_episode(frames=120, run_frames=120, run_over=False)    # first life, 2s
    finish_episode(frames=60, run_frames=180, run_over=False)     # second life, 1s
    finish_episode(frames=240, run_frames=420, run_over=True)     # last life, 4s

    bands = [series.as_list() for series in session.data.life_lengths]
    assert len(bands) == 6, "lives top out at 5, so a run has at most six"
    assert bands[0] == [pytest.approx(2.0)]
    assert bands[1] == [pytest.approx(1.0)]
    assert bands[2] == [pytest.approx(4.0)]
    assert bands[3] == [0.0], "a life never spent is a zero-height band"
    assert sum(band[0] for band in bands) == pytest.approx(
        session.data.episode_length.as_list()[0]), "the stack must equal the total"


def test_v2_carries_its_own_hyperparameters():
    """v1 has to keep today's settings, or it stops being a baseline for v2."""
    from dodge.training import hyperparameters

    v1, v2 = hyperparameters("velocity-flow"), hyperparameters("velocity-flow-v2")

    assert (v1["gamma"], v1["n_epochs"], v1["vf_coef"], v1["target_kl"]) \
        == (0.995, 10, 0.5, None)
    assert (v2["gamma"], v2["n_epochs"], v2["vf_coef"], v2["target_kl"]) \
        == (0.99, 4, 1.0, 0.02)


def test_v2_splits_the_buffer_into_larger_minibatches():
    """Profiled on a 1070: per-sample cost is flat from 128 to 256 and then
    degrades hard -- 512 makes the learning phase slower, not faster. 256 is
    the floor of that curve."""
    from dodge.training import minibatch_size

    assert minibatch_size(2048, "velocity-flow") == 128
    assert minibatch_size(2048, "velocity-flow-v2") == 256
    assert minibatch_size(2048) == 128, "the default stays v1's"


def test_completed_updates_survive_kl_early_stopping(tmp_path):
    """target_kl breaks the epoch loop early, so _n_updates // n_epochs
    under-counts. v2 sets it, and the number names every watch snapshot."""
    from stable_baselines3.common.env_util import make_vec_env
    from dodge.env import DodgeEnv
    from dodge.training import DashboardPPO
    from dodge.policies import policy_class, policy_kwargs

    env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
    model = DashboardPPO(policy_class("mlp"), env, n_steps=16, batch_size=8,
                         n_epochs=10, target_kl=0.0, device="cpu", seed=0,
                         policy_kwargs=policy_kwargs("mlp"))
    assert model.completed_updates is None, "sentinel meaning never counted"

    model.learn(total_timesteps=32)

    # target_kl=0.0 stops after the first epoch every time, so the old
    # derivation would report 0 updates however many actually completed.
    assert model.completed_updates == 2, "one per train() call"
    assert model._n_updates < model.completed_updates * model.n_epochs
    env.close()


def test_the_session_reads_the_explicit_counter(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class Stub:
        completed_updates = 7
        _n_updates = 3
        n_epochs = 10

    assert session._completed_updates(Stub()) == 7


def test_completed_updates_fallback_when_attribute_missing(tmp_path):
    """A pre-counter model has no completed_updates attribute at all."""
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class OldCheckpoint:
        _n_updates = 120  # 12 epochs * 10 updates
        n_epochs = 10

    # Should fall back to derivation: 120 // 10 == 12
    assert session._completed_updates(OldCheckpoint()) == 12


def test_completed_updates_fallback_when_none(tmp_path):
    """A fresh model or one explicitly set to None uses the derivation."""
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class FreshModel:
        completed_updates = None  # Sentinel value
        _n_updates = 80   # 8 epochs * 10 updates
        n_epochs = 10

    # Should fall back to derivation: 80 // 10 == 8
    assert session._completed_updates(FreshModel()) == 8


def test_completed_updates_survives_save_and_load(tmp_path):
    """A checkpoint preserves completed_updates through save and load."""
    from dodge.training import DashboardPPO

    # Create a model and train it briefly
    saved = DashboardPPO("MlpPolicy", _matching_vec_env(1),
                         n_steps=8, batch_size=8, n_epochs=1, device="cpu")
    assert saved.completed_updates is None, "starts as sentinel"

    saved.learn(total_timesteps=16)
    assert saved.completed_updates == 2, "incremented twice (two updates)"

    # Save and load
    path = tmp_path / "checkpoint.zip"
    saved.save(path)

    loaded = DashboardPPO.load(path, env=_matching_vec_env(1), device="cpu")
    # The loaded model preserves its counter from the checkpoint
    assert loaded.completed_updates == 2, "counter preserved after load"

    # Continue training
    loaded.learn(total_timesteps=8)
    assert loaded.completed_updates == 3, "counter continues incrementing"


def test_the_post_update_hook_runs_after_the_optimizer_hook_is_off():
    """A save taken while the optimizer step is patched would capture a model
    mid-update, so the hook has to fire after train() has fully unwound."""
    from stable_baselines3.common.env_util import make_vec_env
    from dodge.env import DodgeEnv
    from dodge.training import DashboardPPO
    from dodge.policies import policy_class, policy_kwargs

    env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
    seen = []
    model = DashboardPPO(policy_class("mlp"), env, n_steps=16, batch_size=8,
                         n_epochs=1, device="cpu", seed=0,
                         policy_kwargs=policy_kwargs("mlp"))
    model.set_post_update(lambda m: seen.append(
        (m.completed_updates, m.policy.optimizer.step.__name__)))

    model.learn(total_timesteps=32)

    assert seen, "the hook never ran"
    assert all(name == "step" for _, name in seen), "optimizer still patched"
    assert [update for update, _ in seen] == [1, 2], "once per update, in order"
    env.close()


def test_a_failing_train_is_not_counted_and_does_not_reach_the_hook(monkeypatch):
    """An update that raised is not a completed update, and a save taken from
    the hook must never see a model whose train() blew up halfway."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from dodge.env import DodgeEnv
    from dodge.training import DashboardPPO
    from dodge.policies import policy_class, policy_kwargs

    env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
    calls = []
    model = DashboardPPO(policy_class("mlp"), env, n_steps=16, batch_size=8,
                         n_epochs=1, device="cpu", seed=0,
                         policy_kwargs=policy_kwargs("mlp"))
    model.set_post_update(lambda m: calls.append(m))

    def explode(self):
        raise RuntimeError("optimizer blew up")

    monkeypatch.setattr(PPO, "train", explode)

    with pytest.raises(RuntimeError, match="optimizer blew up"):
        model.train()

    assert calls == [], "the hook must not fire for a failed update"
    assert model.completed_updates is None, "and a failure is not counted"
    assert model.policy.optimizer.step.__name__ == "step", "hook restored"
    env.close()


def test_request_save_returns_before_the_save_happens(tmp_path):
    """The caller is the Pygame event loop. It must not wait on the trainer."""
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()          # non-None so the request is accepted

    session.request_save(tmp_path / "async.zip", kind="save")

    assert session.drain_events() == [], "nothing has been served yet"
    assert session.save_requests.qsize() == 1


def test_serving_a_request_emits_a_success_event(tmp_path):
    from dodge.training import PPOTrainingSession, SessionEvent

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.request_save(tmp_path / "async.zip", kind="save")

    class Stub:
        completed_updates = 4
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    session._process_save_requests(Stub())
    events = session.drain_events()

    assert len(events) == 1
    assert events[0].ok and events[0].kind == "save" and events[0].update == 4
    assert events[0].path == tmp_path / "async.zip"
    assert session.drain_events() == [], "draining clears"


def test_a_failed_save_emits_an_error_event_rather_than_raising(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.request_save(tmp_path / "async.zip", kind="save")

    class Boom:
        completed_updates = 0
        def save(self, path):
            raise OSError("disk full")

    session._process_save_requests(Boom())
    events = session.drain_events()

    assert len(events) == 1 and not events[0].ok
    assert "disk full" in events[0].message


def test_shutdown_completes_pending_requests_with_an_error(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.request_save(tmp_path / "orphan.zip", kind="watch")

    session._fail_save_requests(RuntimeError("training session closed"))
    events = session.drain_events()

    assert len(events) == 1 and not events[0].ok and events[0].kind == "watch"


def test_request_save_mkdir_failure_emits_an_error_event_rather_than_raising(tmp_path):
    """The caller is the Pygame event loop; mkdir runs synchronously in it
    before any other error handling, so a bad path must not raise there."""
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"a file, not a directory")

    session.request_save(blocker / "sub" / "async.zip", kind="save")

    events = session.drain_events()
    assert len(events) == 1 and not events[0].ok and events[0].kind == "save"
    assert session.save_requests.qsize() == 0, "a path that can't even mkdir must not be queued"


def test_request_save_after_close_emits_failure_event(tmp_path):
    """A request_save after close() must not be orphaned but emit a failure event."""
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()  # non-None so the request would normally be accepted
    session.close()

    session.request_save(tmp_path / "after_close.zip", kind="save")

    events = session.drain_events()
    assert len(events) == 1 and not events[0].ok and events[0].kind == "save"
    assert "closed" in events[0].message
    assert session.save_requests.qsize() == 0, "request must not be queued"


def test_snapshot_names_sort_lexicographically_with_the_update(tmp_path):
    from dodge.training import snapshot_name

    assert snapshot_name(7) == "update-00000007.zip"
    assert sorted([snapshot_name(9), snapshot_name(10)]) == [
        snapshot_name(9), snapshot_name(10)]


def test_newest_snapshot_ignores_partial_and_malformed_files(tmp_path):
    from dodge.training import newest_snapshot, snapshot_name

    (tmp_path / snapshot_name(3)).write_bytes(b"x")
    (tmp_path / snapshot_name(11)).write_bytes(b"x")
    (tmp_path / ".update-00000099.zip.tmp").write_bytes(b"partial")
    (tmp_path / "velocity-flow-latest.zip").write_bytes(b"mutable")

    assert newest_snapshot(tmp_path) == (tmp_path / snapshot_name(11), 11)
    assert newest_snapshot(tmp_path, after=11) is None
    assert newest_snapshot(tmp_path, after=3) == (tmp_path / snapshot_name(11), 11)
    assert newest_snapshot(tmp_path / "missing") is None


def test_publishing_renames_atomically_and_keeps_three(tmp_path):
    from dodge.training import PPOTrainingSession, newest_snapshot

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    live = session.live_watch_dir

    class Stub:
        completed_updates = 0
        def save(self, path):
            # SB3 only appends ".zip" when the given path has no suffix of
            # its own; a stub that appended unconditionally would hide a bug
            # where the real code hands SB3 a name that already looks like
            # it has one.
            path = Path(path)
            if path.suffix == "":
                path = path.with_suffix(".zip")
            path.write_bytes(b"checkpoint")

    model = Stub()
    for update in range(1, 6):
        model.completed_updates = update
        session._publish_snapshot(model)

    names = sorted(p.name for p in live.glob("update-*.zip"))
    assert names == ["update-00000003.zip", "update-00000004.zip",
                     "update-00000005.zip"], "newest three retained"
    assert not list(live.glob(".*")), "no temporary files left behind"
    assert newest_snapshot(live)[1] == 5


def test_retention_survives_a_locked_file(tmp_path, monkeypatch):
    """A watcher reading an old snapshot locks it on Windows. Failing to
    delete it must not take training down."""
    from dodge.training import PPOTrainingSession
    import pathlib

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class Stub:
        completed_updates = 0
        def save(self, path):
            # SB3 only appends ".zip" when the given path has no suffix of
            # its own; a stub that appended unconditionally would hide a bug
            # where the real code hands SB3 a name that already looks like
            # it has one.
            path = Path(path)
            if path.suffix == "":
                path = path.with_suffix(".zip")
            path.write_bytes(b"checkpoint")

    real_unlink = pathlib.Path.unlink

    def locked(self, *a, **k):
        if self.name.startswith("update-"):
            raise PermissionError("file in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", locked)
    model = Stub()
    for update in range(1, 6):
        model.completed_updates = update
        session._publish_snapshot(model)     # must not raise

    assert len(list(session.live_watch_dir.glob("update-*.zip"))) == 5


def test_snapshots_are_published_only_while_a_watcher_is_alive(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class Stub:
        completed_updates = 1
        def save(self, path):
            path = Path(path)
            if path.suffix == "":
                path = path.with_suffix(".zip")
            path.write_bytes(b"checkpoint")

    model = Stub()
    session._on_post_update(model)
    assert not list(session.live_watch_dir.glob("update-*.zip")) \
        if session.live_watch_dir.exists() else True, "no watcher, no snapshot"

    class Alive:
        def poll(self):
            return None

    class Exited:
        def poll(self):
            return 0

    session.watchers = [Alive()]
    model.completed_updates = 2
    session._on_post_update(model)
    assert len(list(session.live_watch_dir.glob("update-*.zip"))) == 1

    session.watchers = [Exited()]
    assert session._watchers_alive() is False
    assert session.watchers == [], "exited handles are pruned"
    model.completed_updates = 3
    session._on_post_update(model)
    assert len(list(session.live_watch_dir.glob("update-*.zip"))) == 1, \
        "publishing stops once the last watcher is gone"


def test_the_watcher_adopts_a_newer_snapshot(tmp_path):
    import watch_ppo
    from dodge.training import snapshot_name

    class Old:
        pass

    loaded = []
    real = watch_ppo.PPO.load
    watch_ppo.PPO.load = staticmethod(lambda path, device: loaded.append(path) or "new")
    try:
        (tmp_path / snapshot_name(9)).write_bytes(b"checkpoint")
        model, update, error = watch_ppo.reload_if_newer(Old(), tmp_path, 4)
        assert model == "new" and update == 9 and error is None
        assert loaded == [tmp_path / snapshot_name(9)]

        again, update, error = watch_ppo.reload_if_newer(model, tmp_path, 9)
        assert again is model and update == 9, "nothing newer, nothing loaded"
    finally:
        watch_ppo.PPO.load = real


def test_a_broken_snapshot_leaves_the_watcher_on_its_model(tmp_path):
    import watch_ppo
    from dodge.training import snapshot_name

    class Old:
        pass

    old = Old()
    real = watch_ppo.PPO.load

    def explode(path, device):
        raise ValueError("bad zip")

    watch_ppo.PPO.load = staticmethod(explode)
    try:
        (tmp_path / snapshot_name(9)).write_bytes(b"garbage")
        model, update, error = watch_ppo.reload_if_newer(old, tmp_path, 4)
        assert model is old, "the in-memory policy has to survive"
        assert update == 4, "and stay on its own update, so it retries later"
        assert "bad zip" in error
    finally:
        watch_ppo.PPO.load = real


def test_no_live_directory_is_not_an_error(tmp_path):
    import watch_ppo

    class Old:
        pass

    old = Old()
    model, update, error = watch_ppo.reload_if_newer(old, None, 3)
    assert model is old and update == 3 and error is None
