"""A loaded model plays a few games per lane before it trains, and the retention
gate keeps its files out of the model picker."""

import numpy as np
import pytest

from dodge import training
from dodge.retention import (RETAINED_DIR, best_path_for, candidate_path_for,
                             move_legacy_retention)
from dodge.training import DashboardCallback, PPOTrainingSession


class _LaneEnv:
    """Lane i's games last lengths[i] steps; the observation is the step count."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.num_envs = len(self.lengths)
        self.steps = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        dones = np.array([self.steps % length == 0 for length in self.lengths])
        infos = [{"frames": length, "run_over": True, "episode": {"r": 1.0}}
                 if done else {} for length, done in zip(self.lengths, dones)]
        observation = np.full((self.num_envs, 1), self.steps, dtype=np.float32)
        return observation, np.ones(self.num_envs), dones, infos


class _FrozenModel:
    completed_updates = 7

    def __init__(self):
        self._last_obs = None
        self._last_episode_starts = None
        self.deterministic = []

    def predict(self, observation, deterministic=False):
        self.deterministic.append(deterministic)
        return np.zeros(len(observation), dtype=np.int64), None


def _warm(tmp_path, lengths):
    session = PPOTrainingSession(model_dir=tmp_path / "models", logs_dir=tmp_path / "logs",
                                 autostart=False, architecture=None)
    env, model = _LaneEnv(lengths), _FrozenModel()
    session._warm_up(model, env, DashboardCallback(session))
    return session, env, model


def test_each_lane_charts_its_next_games_and_no_more(tmp_path):
    session, env, model = _warm(tmp_path, [3, 10])

    # Lane 1's fourth game ends at step 40; lane 0 has played thirteen by then,
    # and only its first four are charted.
    assert env.steps == 40
    lengths = sorted(round(value * 60) for value in session.data.episode_length.values)
    assert lengths == [3] * training.WARMUP_GAMES + [10] * training.WARMUP_GAMES
    assert len(session.data.reward.values) == 2 * training.WARMUP_GAMES
    # Played as training plays, and not counted as training.
    assert not any(model.deterministic)
    assert all(event.total == 0 for event in session.data.events.values())
    assert session.data.update == 7


def test_training_carries_on_from_the_warm_up_rather_than_a_reset(tmp_path):
    session, env, model = _warm(tmp_path, [3, 10])
    assert env.resets == 1
    assert model._last_obs[0, 0] == env.steps
    # Step 40 ended lane 1's game; lane 0 carries its game in progress over.
    assert model._last_episode_starts.tolist() == [False, True]


def test_a_lane_still_short_of_its_games_at_the_cap_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "WARMUP_MAX_STEPS", 15)
    session, env, _ = _warm(tmp_path, [3, 10])
    assert env.steps == 15
    lengths = sorted(round(value * 60) for value in session.data.episode_length.values)
    assert lengths == [3] * training.WARMUP_GAMES + [10]


def test_a_real_load_warms_up_and_charts_at_the_loaded_update(tmp_path, monkeypatch):
    from dodge.env import native_available
    if not native_available():
        pytest.skip("dodge_native not installed")
    from dodge.training import make_training_env

    monkeypatch.setattr(training, "WARMUP_GAMES", 2)
    session = PPOTrainingSession(model_dir=tmp_path / "models", logs_dir=tmp_path / "logs",
                                 autostart=False, device="cpu", n_envs=2, rollout_steps=16,
                                 architecture="mlp", max_episode_frames=120)
    env = make_training_env(2, 0, session.env_kwargs)
    try:
        model = session._new_model(env)
        model.completed_updates = 5
        before = [p.detach().clone() for p in model.policy.parameters()]
        session._warm_up(model, env, DashboardCallback(session))
        assert len(session.data.episode_length.values) == 4
        quarters = session.data.games.quarters
        assert [row["update"] for row in quarters] == [5]
        assert quarters[0]["games"] == 4
        assert model.num_timesteps == 0
        assert all(p.equal(q) for p, q in zip(model.policy.parameters(), before))
    finally:
        env.close()


def test_legacy_retention_files_move_into_the_hidden_folder(tmp_path):
    for name in ("run-best.zip", "run-best.state.pt", "run-best.retention.json",
                 "run-candidate.zip", "run.zip", "run-best-2.zip"):
        (tmp_path / name).write_text(name)
    (tmp_path / RETAINED_DIR).mkdir()
    (tmp_path / RETAINED_DIR / "run-best.state.pt").write_text("newer")

    move_legacy_retention(tmp_path, "run")

    assert best_path_for(tmp_path, "run").read_text() == "run-best.zip"
    assert (tmp_path / RETAINED_DIR / "run-best.retention.json").exists()
    assert candidate_path_for(tmp_path, "run").with_suffix(".zip").exists()
    # Never overwritten: the copy already in the folder wins, and stays put.
    assert (tmp_path / RETAINED_DIR / "run-best.state.pt").read_text() == "newer"
    assert (tmp_path / "run-best.state.pt").exists()
    # Other models are left where they are.
    assert (tmp_path / "run.zip").exists() and (tmp_path / "run-best-2.zip").exists()


def test_the_picker_does_not_offer_retention_files(tmp_path):
    session = PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path / "logs",
                                 autostart=False, architecture=None)
    (tmp_path / "run.zip").write_text("x")
    best = best_path_for(tmp_path, "run")
    best.parent.mkdir()
    best.write_text("x")
    assert [entry.name for entry in session.available_models()] == ["run"]
