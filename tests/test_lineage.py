"""Every checkpoint carries a state sidecar and a provenance manifest."""

from dataclasses import asdict
import hashlib
import json
import re

import pytest

from dodge.env import native_available
from dodge.rewards import RewardConfig

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def small_model(n_envs=2):
    from stable_baselines3 import PPO
    from dodge.env import DodgeVecEnv
    from dodge.policies import policy_class, policy_kwargs

    env = DodgeVecEnv(n_envs=n_envs, seed=3, action_repeat=4)
    model = PPO(policy_class("mlp"), env, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs=policy_kwargs("mlp"), device="cpu", seed=0)
    return model, env


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def test_a_checkpoint_gets_a_state_sidecar_and_a_manifest(tmp_path):
    import dodge_native
    from dodge.lineage import sidecar_paths
    from dodge.resume import CheckpointWriter

    model, env = small_model()
    model.learn(16)
    clock = Clock()
    writer = CheckpointWriter(extra_config={"architecture": "mlp"}, clock=clock)
    clock.now += 12.5
    path = writer.save(model, tmp_path / "run", env=env, ppo_updates=1)

    assert path == tmp_path / "run.zip" and path.exists()
    state_path, manifest_path = sidecar_paths(path)
    assert state_path == tmp_path / "run.state.pt" and state_path.exists()
    assert manifest_path == tmp_path / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["model_sha256"] == sha(path)
    assert manifest["state_sha256"] == sha(state_path)
    config = manifest["config"]
    assert config["architecture"] == "mlp"
    assert config["model"]["n_steps"] == 8 and config["model"]["batch_size"] == 8
    assert config["model"]["gamma"] == model.gamma
    assert config["model"]["learning_rate"] == 3e-4
    assert config["model"]["clip_range"] == 0.2
    assert config["model"]["n_envs"] == 2
    assert config["env"]["action_repeat"] == 4
    assert config["env"]["run_seed"] == 0  # PPO's seed restarts the stream
    assert config["env"]["reward_config"] == asdict(RewardConfig())
    assert "paper-eval" in config["env"]["reserved_seed_ranges"]

    versions = manifest["versions"]
    assert set(versions) >= {"python", "numpy", "torch", "stable_baselines3"}
    native = manifest["dodge_native"]
    from pathlib import Path
    binary = Path(native["path"])
    assert binary.suffix in (".pyd", ".so")
    assert native["sha256"] == sha(binary)
    assert Path(dodge_native.dodge_native.__file__).resolve() == binary.resolve()

    git = manifest["git"]
    assert re.fullmatch(r"[0-9a-f]{40}", git["commit"])
    assert isinstance(git["dirty"], bool)

    assert manifest["parent"] is None
    counters = manifest["counters"]
    assert counters["ppo_updates"] == 1
    assert counters["decisions"] == model.num_timesteps == 16
    assert counters["raw_frames"] == env.total_frames == 16 * 4
    assert counters["optimizer_steps"] == 2  # one epoch of two minibatches
    assert manifest["wall_clock_seconds"] == pytest.approx(12.5)


def test_checkpoints_chain_to_their_parent(tmp_path):
    from dodge.lineage import sidecar_paths
    from dodge.resume import CheckpointWriter

    model, env = small_model()
    clock = Clock()
    writer = CheckpointWriter(clock=clock)
    first = writer.save(model, tmp_path / "a", env=env, ppo_updates=0)
    clock.now += 3
    second = writer.save(model, tmp_path / "b.zip", env=env, ppo_updates=0)
    manifest = json.loads(sidecar_paths(second)[1].read_text())
    assert manifest["parent"]["checkpoint_sha256"] == sha(first)
    assert manifest["parent"]["manifest_sha256"] == sha(sidecar_paths(first)[1])
    assert manifest["wall_clock_seconds"] == pytest.approx(3)

    # A writer resuming from `second` continues its lineage and its clock.
    clock.now += 100
    resumed = CheckpointWriter.resuming(second, clock=clock)
    clock.now += 4
    third = resumed.save(model, tmp_path / "c", env=env, ppo_updates=0)
    manifest = json.loads(sidecar_paths(third)[1].read_text())
    assert manifest["parent"]["checkpoint_sha256"] == sha(second)
    assert manifest["wall_clock_seconds"] == pytest.approx(3 + 4)


def test_a_parent_without_a_manifest_is_still_referenced(tmp_path):
    from dodge.lineage import parent_reference

    legacy = tmp_path / "legacy.zip"
    legacy.write_bytes(b"not really a zip")
    reference = parent_reference(legacy)
    assert reference["checkpoint_sha256"] == sha(legacy)
    assert reference["manifest_sha256"] is None


def test_non_sb3_models_are_saved_without_sidecars(tmp_path):
    from dodge.resume import CheckpointWriter

    saved = []

    class Stub:
        def save(self, path):
            saved.append(path)

    path = CheckpointWriter().save(Stub(), tmp_path / "stub", ppo_updates=0)
    assert saved == [tmp_path / "stub"]
    assert path == tmp_path / "stub.zip"
    assert not (tmp_path / "stub.manifest.json").exists()
