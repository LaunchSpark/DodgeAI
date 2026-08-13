import faulthandler
from pathlib import Path
import sys
import time

import pytest
import numpy as np

pytest.importorskip("stable_baselines3")
pytest.importorskip("lupa.luajit21")

from dodge.telemetry import ModelEntry
from dodge.observe import FRAME_OBSERVATION_SIZE, OBSERVATION_SIZE
from dodge.training import DashboardPPO, PPOTrainingSession, migrate_single_frame_policy
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


def test_watch_uses_unique_update_snapshot_and_random_seed(tmp_path, monkeypatch):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
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
    final = tmp_path / "watch" / "hybrid-cnn-update-000037-seed-1849201.zip"
    assert final.exists()
    command = commands[0][0]
    assert command[command.index("--update") + 1] == "37"
    assert command[command.index("--seed") + 1] == "1849201"
    assert command[command.index("--model-name") + 1] == "hybrid-cnn"
    assert session.last_watch_label == "hybrid-cnn | Update 37 | Seed 1849201"


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


def test_legacy_policy_weights_migrate_to_current_frame_inputs():
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class ObservationEnv(gym.Env):
        def __init__(self, width):
            self.observation_space = spaces.Box(0.0, 1.0, (width,), dtype=np.float32)
            self.action_space = spaces.Discrete(9)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            return observation, 0.0, False, False, {}

    def model(width, seed):
        return PPO(
            "MlpPolicy",
            ObservationEnv(width),
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=seed,
            device="cpu",
        )

    legacy = model(FRAME_OBSERVATION_SIZE, 3)
    target = model(OBSERVATION_SIZE, 7)
    initialized = {
        name: value.clone()
        for name, value in target.policy.state_dict().items()
    }
    legacy.num_timesteps = 1234
    legacy._n_updates = 50

    migrate_single_frame_policy(legacy, target)

    legacy_state = legacy.policy.state_dict()
    migrated = target.policy.state_dict()
    for name in (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.value_net.0.weight",
    ):
        np.testing.assert_array_equal(
            migrated[name][:, :FRAME_OBSERVATION_SIZE].cpu(),
            initialized[name][:, :FRAME_OBSERVATION_SIZE].cpu(),
        )
        np.testing.assert_array_equal(
            migrated[name][:, FRAME_OBSERVATION_SIZE:].cpu(),
            legacy_state[name].cpu(),
        )
    np.testing.assert_array_equal(
        migrated["action_net.weight"].cpu(),
        legacy_state["action_net.weight"].cpu(),
    )
    assert target.num_timesteps == 1234
    assert target._n_updates == 50


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
