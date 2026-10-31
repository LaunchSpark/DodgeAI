"""train_ppo.py: segments, resume, retention and the wall-clock budget."""

import json

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


def train(tmp_path, *extra):
    import train_ppo
    return train_ppo.main([
        "--architecture", "mlp", "--device", "cpu", "--envs", "2",
        "--rollout-steps", "8", "--difficulty", "hard", "--patterns",
        "--segment-updates", "2",
        "--models-dir", str(tmp_path / "models"), "--logs-dir", str(tmp_path / "tb"),
        *extra,
    ])


def parameters(path):
    from stable_baselines3 import PPO
    return PPO.load(path, device="cpu").policy.state_dict()


def test_a_resumed_cli_run_matches_one_that_never_stopped(tmp_path):
    models = tmp_path / "models"
    train(tmp_path, "--steps", "64", "--name", "straight")
    train(tmp_path, "--steps", "32", "--name", "first")
    boundary = models / "first-u0000002.zip"
    assert boundary.exists()
    torch.manual_seed(1)
    train(tmp_path, "--steps", "32", "--name", "second", "--resume", str(boundary))

    straight, resumed = parameters(models / "straight.zip"), parameters(models / "second.zip")
    for name in straight:
        assert torch.equal(straight[name], resumed[name]), name
    manifest = json.loads(sidecar_paths(models / "second.zip")[1].read_text())
    assert manifest["counters"]["ppo_updates"] == 4
    # The resumed run's first checkpoint descends from the one it resumed.
    child = json.loads(sidecar_paths(models / "second-u0000004.zip")[1].read_text())
    assert child["parent"]["checkpoint_sha256"] == sha256_file(boundary)
    assert child["at_boundary"] is True


def test_the_budget_stops_at_a_boundary_with_a_resumable_checkpoint(tmp_path):
    from dodge.resume import read_state

    models = tmp_path / "models"
    train(tmp_path, "--steps", "100000", "--name", "budget", "--budget-seconds", "0")
    final = models / "budget.zip"
    assert final.read_bytes() == (models / "budget-u0000002.zip").read_bytes()
    state = read_state(final)
    assert state["at_boundary"] and state["counters"]["ppo_updates"] == 2
    assert not (models / "budget-u0000004.zip").exists()


def test_retention_keeps_the_best_and_training_restarts_from_it(tmp_path):
    models = tmp_path / "models"
    train(tmp_path, "--steps", "64", "--name", "kept", "--retain-every", "2",
          "--dev-seeds", "2", "--dev-frame-cap", "100")
    best = models / "retained" / "kept-best.zip"
    assert best.exists() and all(path.exists() for path in sidecar_paths(best))
    history = (models / "retained" / "kept-best.history.jsonl").read_text().splitlines()
    assert len(history) == 2  # updates 2 and 4
    record = json.loads((models / "retained" / "kept-best.retention.json").read_text())
    from pathlib import Path
    assert best.read_bytes() == Path(record["candidate"]).read_bytes()

    train(tmp_path, "--steps", "16", "--name", "kept", "--from-best")
    manifest = json.loads(sidecar_paths(models / "kept.zip")[1].read_text())
    assert manifest["counters"]["ppo_updates"] == record["update"] + 1


@pytest.mark.parametrize("flags", [
    ["--retain-every", "3"],
    ["--segment-updates", "0", "--budget-seconds", "10"],
    ["--segment-updates", "0", "--retain-every", "2"],
])
def test_incompatible_flags_are_refused(tmp_path, flags):
    import train_ppo
    with pytest.raises(SystemExit):
        train_ppo.main(["--architecture", "mlp", "--segment-updates", "2",
                        "--models-dir", str(tmp_path), "--logs-dir", str(tmp_path),
                        *flags])


def test_from_best_needs_a_retained_best(tmp_path):
    import train_ppo
    with pytest.raises(SystemExit):
        train_ppo.main(["--architecture", "mlp", "--from-best", "--name", "none",
                        "--models-dir", str(tmp_path), "--logs-dir", str(tmp_path)])


@pytest.mark.parametrize("flags,expected", [
    ([], (8, 256)),
    (["--preset", "gpu"], (64, 256)),
    (["--preset", "gpu", "--envs", "256", "--rollout-steps", "64"], (256, 64)),
    (["--envs", "32"], (32, 256)),
])
def test_a_preset_sets_the_env_shape_and_flags_override_it(flags, expected):
    import train_ppo
    _, args = train_ppo.parse_args(flags)
    assert (args.envs, args.rollout_steps) == expected


@pytest.mark.parametrize("flags", [["--envs", "0"], ["--rollout-steps", "0"],
                                   ["--checkpoint-updates", "-1"]])
def test_a_non_positive_shape_is_refused(flags):
    import train_ppo
    with pytest.raises(SystemExit):
        train_ppo.parse_args(flags)


def test_checkpoints_fall_every_n_updates_and_leave_training_untouched(tmp_path):
    """Counted in updates, not env steps, so --envs does not move them; the
    boundary updates are skipped, since the boundary already saved them; and
    taking a checkpoint must not perturb the run that takes it."""
    from dodge.resume import read_state

    models = tmp_path / "models"
    train(tmp_path, "--steps", "64", "--name", "every", "--checkpoint-updates", "1")
    train(tmp_path, "--steps", "64", "--name", "never", "--checkpoint-updates", "0")

    # 2 envs x 8 steps = 16 decisions an update; boundaries at updates 2 and 4.
    periodic = sorted(path.name for path in models.glob("every_*_steps.zip"))
    assert periodic == ["every_16_steps.zip", "every_48_steps.zip"]
    assert not list(models.glob("never_*_steps.zip"))
    state = read_state(models / "every_48_steps.zip")
    assert state["counters"]["ppo_updates"] == 3 and not state["at_boundary"]

    every, never = parameters(models / "every.zip"), parameters(models / "never.zip")
    for name in every:
        assert torch.equal(every[name], never[name]), name
