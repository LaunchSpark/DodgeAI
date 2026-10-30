"""The Training panel: a file-backed learning rate, discount and dev scoring."""

import json

import numpy as np
import pytest

from dodge import schedule
from dodge.schedule import TrainingConfig

pytest.importorskip("stable_baselines3")


def test_a_missing_or_malformed_file_yields_defaults(tmp_path):
    assert schedule.load(tmp_path / "absent.json") == TrainingConfig()
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert schedule.load(broken) == TrainingConfig()


def test_values_round_trip_and_are_clamped(tmp_path):
    path = tmp_path / "training.json"
    schedule.save(TrainingConfig(learning_rate=1e-4, gamma=0.997,
                                 retain_every=30, patience=2), path)
    assert schedule.load(path) == TrainingConfig(1e-4, 0.997, 30, 2)
    path.write_text(json.dumps({"learning_rate": 5.0, "gamma": 0.5,
                                "retain_every": 12.6, "unknown": 1}),
                    encoding="utf-8")
    loaded = schedule.load(path)
    assert loaded.learning_rate == 1e-3
    assert loaded.gamma == 0.98
    assert loaded.retain_every == 13 and isinstance(loaded.retain_every, int)


@pytest.mark.parametrize("rate,expected", [
    (3e-4, 1e-4), (1e-4, 3e-5), (3e-5, None), (2e-4, 1e-4), (5e-4, 3e-4),
])
def test_the_rate_steps_to_the_next_stage_below(rate, expected):
    assert schedule.next_stage(rate) == expected


def _session(tmp_path, monkeypatch):
    from dodge.env import native_available
    if not native_available():
        pytest.skip("dodge_native not installed")
    from dodge.training import PPOTrainingSession, make_training_env

    session = PPOTrainingSession(model_dir=tmp_path / "models", logs_dir=tmp_path / "logs",
                                 autostart=False, device="cpu", n_envs=2,
                                 rollout_steps=16, architecture="mlp")
    session.schedule_path = tmp_path / "training.json"
    session.score_in_background = False
    env = make_training_env(2, 0, session.env_kwargs)
    model = session._new_model(env)
    session.model_architecture = "mlp"
    session._game_settings = dict(env.game_settings)
    return session, model, env


def test_the_panel_sets_rate_and_discount_without_a_restart(tmp_path, monkeypatch):
    session, model, env = _session(tmp_path, monkeypatch)
    try:
        table_gamma = model.gamma
        # No file: the architecture's table stands.
        assert session._apply_schedule(model) is None
        assert model.gamma == table_gamma

        schedule.save(TrainingConfig(learning_rate=1e-4, gamma=0.997), session.schedule_path)
        session._apply_schedule(model)
        assert model.gamma == model.rollout_buffer.gamma == 0.997
        assert model.lr_schedule(1.0) == pytest.approx(1e-4)
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(None, []))
        model._update_learning_rate(model.policy.optimizer)
        assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert session.schedule_status["learning_rate"] == 1e-4
    finally:
        env.close()


def test_a_plateau_steps_the_rate_down_and_a_new_best_resets_it(tmp_path, monkeypatch):
    session, model, env = _session(tmp_path, monkeypatch)
    try:
        means = iter([500.0, 400.0, 450.0, 600.0, 300.0, 200.0])
        best = {"mean": None}

        class Gate:
            def consider(self, model, candidate, *, update=None):
                mean = next(means)
                promoted = best["mean"] is None or mean > best["mean"]
                if promoted:
                    best["mean"] = mean
                return {"promoted": promoted, "best_mean": best["mean"],
                        "dev": {"mean": mean, "q1": mean / 2}}

        monkeypatch.setattr(session, "_retention_gate", lambda: Gate())
        schedule.save(TrainingConfig(learning_rate=3e-4, retain_every=1, patience=2),
                      session.schedule_path)
        events = []
        monkeypatch.setattr(session, "_emit", events.append)
        rates = []
        for update in range(1, 7):
            model.completed_updates = update
            session._score_dev_cohort(model)
            rates.append(schedule.load(session.schedule_path).learning_rate)

        # 500 best; 400, 450 miss -> step to 1e-4; 600 best; 300, 200 miss -> 3e-5.
        assert rates == [3e-4, 3e-4, 1e-4, 1e-4, 1e-4, 3e-5]
        # The next boundary applies the stepped rate.
        session._apply_schedule(model)
        assert model.lr_schedule(1.0) == pytest.approx(3e-5)
        assert (tmp_path / "models" / "mlp-candidate.zip").exists()
        assert "steps down to 0.0001" in events[2].message
        assert session.schedule_status["last"]["mean"] == 200.0
    finally:
        env.close()


def test_scoring_is_off_between_boundaries_and_without_a_file(tmp_path, monkeypatch):
    session, model, env = _session(tmp_path, monkeypatch)
    try:
        monkeypatch.setattr(session, "_retention_gate",
                            lambda: pytest.fail("scored off schedule"))
        model.completed_updates = 60
        session._score_dev_cohort(model)  # no file
        schedule.save(TrainingConfig(retain_every=60), session.schedule_path)
        model.completed_updates = 59
        session._score_dev_cohort(model)
        schedule.save(TrainingConfig(retain_every=0), session.schedule_path)
        model.completed_updates = 60
        session._score_dev_cohort(model)
    finally:
        env.close()


def test_scoring_runs_beside_training_and_skips_while_busy(tmp_path, monkeypatch):
    import threading

    session, model, env = _session(tmp_path, monkeypatch)
    session.score_in_background = True
    release = threading.Event()
    calls = []

    class Gate:
        def consider(self, scored, candidate, *, update=None):
            calls.append(update)
            assert scored is not model  # a copy loaded off disk
            release.wait(10)
            return {"promoted": True, "best_mean": 1.0,
                    "dev": {"mean": 1.0, "q1": 1.0}}

    try:
        monkeypatch.setattr(session, "_retention_gate", lambda: Gate())
        monkeypatch.setattr(session, "_emit", lambda event: None)
        schedule.save(TrainingConfig(retain_every=1), session.schedule_path)
        model.completed_updates = 1
        session._score_dev_cohort(model)  # returns while the gate is blocked
        assert session.schedule_status["scoring"] == 1
        model.completed_updates = 2
        session._score_dev_cohort(model)  # still busy: skipped
        release.set()
        session._scoring.join(10)
        assert calls == [1]
        assert session.schedule_status["scoring"] is None
        assert session.schedule_status["last"]["update"] == 1
    finally:
        release.set()
        env.close()
