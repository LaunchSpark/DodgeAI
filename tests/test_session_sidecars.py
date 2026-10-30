"""The dashboard session's saves carry sidecars, and its loads restore them."""

import json
import threading

import pytest
import torch

from dodge.env import native_available
from dodge.lineage import sha256_file, sidecar_paths
from dodge.rewards import RewardConfig

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


def session_and_model(tmp_path, n_envs=2):
    from dodge.env import DodgeVecEnv
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path / "models", logs_dir=tmp_path / "tb",
                                 n_envs=n_envs, rollout_steps=8, architecture="mlp",
                                 autostart=False, device="cpu")
    env = DodgeVecEnv(n_envs=n_envs, seed=3, action_repeat=4)
    model = session._new_model(env)
    model.learn(16)
    return session, model, env


def test_a_session_save_writes_the_sidecars(tmp_path):
    from dodge.training import _SaveRequest

    session, model, env = session_and_model(tmp_path)
    request = _SaveRequest(path=tmp_path / "saved", finished=threading.Event())
    session.save_requests.put(request)
    session._process_save_requests(model)
    assert request.error is None

    manifest = json.loads(sidecar_paths(tmp_path / "saved.zip")[1].read_text())
    assert manifest["model_sha256"] == sha256_file(tmp_path / "saved.zip")
    assert manifest["counters"]["ppo_updates"] == model.completed_updates == 1
    assert manifest["config"]["env"]["n_envs"] == 2
    assert manifest["at_boundary"] is False


def test_a_session_load_restores_the_seed_stream_and_generators(tmp_path):
    from dodge.env import DodgeVecEnv
    from dodge.resume import CheckpointWriter

    session, model, env = session_and_model(tmp_path)
    path = CheckpointWriter().save(model, tmp_path / "ckpt", env=env, ppo_updates=1)
    expected_rng = torch.get_rng_state()
    indices = env.next_episode_indices

    torch.manual_seed(77)
    fresh = DodgeVecEnv(n_envs=2, seed=3, action_repeat=4)
    loaded = session._load_model(path, fresh)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    # Mid-segment: the lanes go on to their next games rather than replaying
    # the stream from its start.
    assert list(fresh.next_episode_indices) == [i + 1 for i in indices]
    assert loaded._last_obs is not None


def test_a_session_load_into_a_different_env_count_still_loads(tmp_path):
    from dodge.env import DodgeVecEnv
    from dodge.resume import CheckpointWriter

    session, model, env = session_and_model(tmp_path)
    path = CheckpointWriter().save(model, tmp_path / "ckpt", env=env, ppo_updates=1)
    wider = DodgeVecEnv(n_envs=4, seed=3, action_repeat=4)
    loaded = session._load_model(path, wider)
    assert loaded.n_envs == 4


def test_the_autosave_chains_to_the_checkpoint_it_was_loaded_from(tmp_path):
    from dodge.resume import CheckpointWriter

    session, model, env = session_and_model(tmp_path)
    parent = CheckpointWriter().save(model, tmp_path / "parent", env=env, ppo_updates=1)
    session._begin_lineage(parent)
    model.save(tmp_path / "child.zip")
    session._write_sidecars(model, tmp_path / "child.zip")
    manifest = json.loads(sidecar_paths(tmp_path / "child.zip")[1].read_text())
    assert manifest["parent"]["checkpoint_sha256"] == sha256_file(parent)
