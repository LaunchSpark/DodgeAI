"""The retention gate keeps a candidate only if it beats the best on the dev cohort."""

import json

import numpy as np
import pytest

from dodge.env import native_available
from dodge.evaluate import EvalResult
from dodge.lineage import sidecar_paths
from dodge.rewards import RewardConfig

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")

GAME = {"difficulty": 3, "patterns": True, "powerups": True, "normal_enemies": True,
        "kamikaze_enemies": True, "spawn_rate_percent": 100, "action_repeat": 4,
        "lives": 0, "max_episode_frames": None, "execution": "parallel"}


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


def fake_candidate(directory, name, content):
    zip_path = directory / f"{name}.zip"
    zip_path.write_bytes(content)
    state, manifest = sidecar_paths(zip_path)
    state.write_bytes(content + b"-state")
    manifest.write_text(json.dumps({"model": content.decode()}))
    return zip_path


def scripted_scores(*means):
    """An evaluator that reports each candidate's preset mean in turn."""
    calls = []
    queue = list(means)

    def evaluate(model, seeds, **kwargs):
        calls.append((list(seeds), kwargs))
        frames = np.full(len(seeds), float(queue.pop(0)))
        return EvalResult(seeds=list(seeds), frames=frames,
                          scores=np.zeros(len(seeds)), died=[True] * len(seeds))

    return evaluate, calls


def test_the_first_candidate_is_retained_and_copied_whole(tmp_path):
    from dodge.retention import RetentionGate

    evaluate, calls = scripted_scores(100)
    gate = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                         evaluate=evaluate)
    candidate = fake_candidate(tmp_path, "run-u0000060", b"one")
    outcome = gate.consider(object(), candidate, update=60)

    assert outcome["promoted"] and outcome["dev"]["mean"] == 100
    best = tmp_path / "run-best.zip"
    assert best.read_bytes() == b"one"
    for source, copy in zip(sidecar_paths(candidate), sidecar_paths(best)):
        assert copy.read_bytes() == source.read_bytes()
    record = json.loads((tmp_path / "run-best.retention.json").read_text())
    assert record["candidate"] == str(candidate) and record["update"] == 60
    from dodge.seeds import DEV_COHORT_SIZE
    assert record["dev"]["mean"] == 100 and len(record["games"]) == DEV_COHORT_SIZE
    # Scored on the fixed dev cohort, argmax, with training lives dropped.
    seeds, kwargs = calls[0]
    from dodge.seeds import dev_cohort
    assert seeds == dev_cohort()
    assert kwargs["deterministic"] is True
    assert kwargs["step_frames"] == 4 and "lives" not in kwargs
    # Played until 90% of the cohort has ended, and no further.
    assert kwargs["stop_after"] == -(-9 * len(seeds) // 10)


def test_only_an_improvement_is_promoted(tmp_path):
    from dodge.retention import RetentionGate

    evaluate, _ = scripted_scores(100, 90, 100, 120)
    gate = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                         evaluate=evaluate)
    results = [gate.consider(object(), fake_candidate(tmp_path, f"c{i}", f"c{i}".encode()))
               for i in range(4)]
    assert [r["promoted"] for r in results] == [True, False, False, True]
    assert (tmp_path / "run-best.zip").read_bytes() == b"c3"
    assert gate.best_mean == 120
    history = (tmp_path / "run-best.history.jsonl").read_text().splitlines()
    assert [json.loads(line)["promoted"] for line in history] == [True, False, False, True]


def test_the_last_tenth_is_cut_at_the_length_of_the_ninetieth_percentile(tmp_path):
    from dodge.retention import RetentionGate

    calls = []

    def evaluate(model, seeds, **kwargs):
        calls.append(kwargs)
        return EvalResult(seeds=list(seeds), frames=np.arange(1.0, 11.0),
                          scores=np.zeros(10), died=[True] * 9 + [False])

    gate = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                         seeds=list(range(2_000_000, 2_000_010)), evaluate=evaluate)
    gate.consider(object(), fake_candidate(tmp_path, "a", b"a"))
    assert calls[0]["stop_after"] == 9
    record = json.loads((tmp_path / "run-best.retention.json").read_text())
    assert record["finish_fraction"] == 0.9


def test_a_best_scored_under_other_rules_is_rescored_before_it_is_compared(tmp_path):
    from dodge.retention import RetentionGate

    # A best kept under the old rules: 256 seeds, no cut, mean 9,000.
    evaluate, _ = scripted_scores(9_000)
    old = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                        seeds=list(range(2_000_000, 2_000_256)),
                        finish_fraction=None, evaluate=evaluate)
    old.consider(object(), fake_candidate(tmp_path, "old", b"old"), update=24_000)

    loaded = []

    def load(path):
        loaded.append(path.read_bytes())
        return object()

    # Rescored at 7,000 under the new rules; a 7,500 candidate now beats it.
    evaluate, calls = scripted_scores(7_000, 7_500)
    gate = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                         seeds=list(range(2_000_000, 2_000_128)),
                         evaluate=evaluate, load=load)
    outcome = gate.consider(object(), fake_candidate(tmp_path, "new", b"new"))
    assert loaded == [b"old"]
    assert [len(seeds) for seeds, _ in calls] == [128, 128]
    assert outcome["promoted"] and gate.best_mean == 7_500
    history = [json.loads(line) for line in
               (tmp_path / "run-best.history.jsonl").read_text().splitlines()]
    assert history[-1]["previous_best_mean"] == 7_000
    # Once rescored, the record matches and is not rescored again.
    evaluate, calls = scripted_scores(7_000)
    again = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                          seeds=list(range(2_000_000, 2_000_128)),
                          evaluate=evaluate, load=load)
    assert not again.consider(object(), fake_candidate(tmp_path, "x", b"x"))["promoted"]
    assert len(loaded) == 1 and len(calls) == 1


def test_the_best_so_far_survives_a_restart(tmp_path):
    from dodge.retention import RetentionGate

    evaluate, _ = scripted_scores(100)
    RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                  evaluate=evaluate).consider(object(), fake_candidate(tmp_path, "a", b"a"))
    evaluate, _ = scripted_scores(80)
    again = RetentionGate(tmp_path / "run-best.zip", game_settings=GAME,
                          evaluate=evaluate)
    assert again.best_mean == 100
    assert not again.consider(object(), fake_candidate(tmp_path, "b", b"b"))["promoted"]


def test_a_real_candidate_is_scored_and_restartable(tmp_path):
    from stable_baselines3 import PPO
    from dodge.env import DodgeVecEnv
    from dodge.policies import policy_class, policy_kwargs
    from dodge.resume import (CheckpointWriter, SegmentCallback, load_checkpoint,
                              read_state)
    from dodge.retention import RetentionGate

    env = DodgeVecEnv(n_envs=2, seed=3, **{k: v for k, v in GAME.items()
                                           if k != "execution"})
    model = PPO(policy_class("mlp"), env, n_steps=8, batch_size=8, n_epochs=1,
                policy_kwargs=policy_kwargs("mlp"), device="cpu", seed=0)
    writer = CheckpointWriter()
    gate = RetentionGate(tmp_path / "run-best.zip", game_settings=env.game_settings,
                         seeds=[2_000_000, 2_000_001, 2_000_002], frame_cap=200)
    outcomes = []

    def boundary(callback):
        candidate = writer.save(callback.model, tmp_path / f"run-u{callback.updates}",
                                env=env, ppo_updates=callback.updates,
                                at_boundary=True, segment=callback.state(),
                                abandoned=callback.abandoned)
        outcomes.append(gate.consider(callback.model, candidate, update=callback.updates))

    model.learn(32, callback=SegmentCallback(1, on_boundary=[boundary]))
    assert outcomes[0]["promoted"]
    assert outcomes[0]["dev"]["n"] == 3

    best = tmp_path / "run-best.zip"
    record = json.loads((tmp_path / "run-best.retention.json").read_text())
    candidate = record["candidate"]
    restarted, state = load_checkpoint(best, DodgeVecEnv(n_envs=2, seed=9),
                                       device="cpu")
    assert state["at_boundary"]
    import torch
    assert torch.equal(torch.get_rng_state(), read_state(candidate)["rng"]["torch"])
    reference = PPO.load(candidate, device="cpu")
    for name, value in reference.policy.state_dict().items():
        assert torch.equal(value, restarted.policy.state_dict()[name])


def test_scoring_candidates_does_not_change_training(tmp_path):
    import torch
    from stable_baselines3 import PPO
    from dodge.env import DodgeVecEnv
    from dodge.policies import policy_class, policy_kwargs
    from dodge.resume import CheckpointWriter, SegmentCallback
    from dodge.retention import RetentionGate

    def run(with_gate):
        env = DodgeVecEnv(n_envs=2, seed=3, action_repeat=4)
        model = PPO(policy_class("mlp"), env, n_steps=8, batch_size=8, n_epochs=1,
                    policy_kwargs=policy_kwargs("mlp"), device="cpu", seed=0)
        hooks = []
        if with_gate:
            writer = CheckpointWriter()
            gate = RetentionGate(tmp_path / "g-best.zip", game_settings=env.game_settings,
                                 seeds=[2_000_000, 2_000_001], frame_cap=100)

            def boundary(callback):
                path = writer.save(callback.model, tmp_path / f"g-{callback.updates}",
                                   env=env, ppo_updates=callback.updates,
                                   at_boundary=True)
                gate.consider(callback.model, path)
            hooks.append(boundary)
        model.learn(48, callback=SegmentCallback(1, on_boundary=hooks))
        return model.policy.state_dict()

    gated, plain = run(True), run(False)
    for name in plain:
        assert torch.equal(gated[name], plain[name]), name
