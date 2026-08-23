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
from dodge.training import DashboardPPO, PPOTrainingSession
from watch_ppo import window_title

pytestmark = pytest.mark.skipif(not Path("dodge.p8.png").exists(), reason="dodge.p8.png not present")


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
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert session.available_models() == [ModelEntry("one", tmp_path / "one.zip", "PPO")]


def test_training_enables_powerups_by_default(tmp_path):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert session.env_kwargs["powerups"] is True


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


def test_watch_uses_unique_update_snapshot_and_random_seed(tmp_path, monkeypatch):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.env_kwargs.update(
        difficulty=3, patterns=True, powerups=False, action_repeat=7
    )
    saved_paths = []
    commands = []

    def fake_save(path, return_update=False):
        saved_paths.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"snapshot")
        return (path, 37) if return_update else path

    monkeypatch.setattr(session, "_save_to_path", fake_save)
    monkeypatch.setattr("dodge.training.secrets.randbelow", lambda upper: 1849201)
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append((command, cwd)))

    assert session.watch_agent()
    assert saved_paths[0].name == ".pending-seed-1849201.zip"
    assert saved_paths[0].parent.name == "watch"
    architecture = session.architecture
    final = (tmp_path / "watch"
             / f"{architecture}-update-000037-seed-1849201.zip")
    assert final.exists()
    command = commands[0][0]
    assert command[command.index("--update") + 1] == "37"
    assert command[command.index("--seed") + 1] == "1849201"
    assert command[command.index("--model-name") + 1] == architecture
    assert command[command.index("--difficulty") + 1] == "hard"
    assert command[command.index("--action-repeat") + 1] == "7"
    assert "--patterns" in command
    assert "--no-powerups" in command
    assert session.last_watch_label == f"{architecture} | Update 37 | Seed 1849201"


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


@pytest.mark.parametrize("architecture", ["velocity-flow", "hybrid-cnn", "mlp"])
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
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    assert session.game_config()["lives"] == 0, "off unless asked for"

    assert session.configure_game(difficulty=1, patterns=False, powerups=True,
                                  action_repeat=4, lives=3)
    assert session.game_config()["lives"] == 3

    commands = []
    def fake_save(path, return_update=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"snapshot")
        return (path, 1) if return_update else path

    monkeypatch.setattr(session, "_save_to_path", fake_save)
    monkeypatch.setattr("dodge.training.secrets.randbelow", lambda upper: 7)
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command))

    assert session.watch_agent()

    assert "--lives" not in commands[0]


def test_negative_lives_are_rejected_by_the_session(tmp_path):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    with pytest.raises(ValueError):
        session.configure_game(difficulty=1, patterns=False, powerups=True,
                               action_repeat=4, lives=-1)


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


