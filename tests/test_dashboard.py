import pytest

pytest.importorskip("fastapi")
pytest.importorskip("altair")

from fastapi.testclient import TestClient

from dodge.dashboard import create_app
from dodge.rewards import RewardConfig
from dodge.telemetry import sample_snapshot
from dodge.training import PPOTrainingSession


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    from dodge import game_settings, rewards
    path = tmp_path / "rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)
    monkeypatch.setattr(game_settings, "CONFIG_PATH", tmp_path / "game-config.json")


def game(frames, cause="normal", pattern=-1, encounters=None):
    return {"frames": frames, "died": True, "cause": cause, "pattern": pattern,
            "pattern_frames": frames // 3, "encounters": encounters or {},
            "fyou_triggers": 0}


@pytest.fixture
def client(tmp_path):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, architecture=None,
                                 device="cpu")
    session.data = sample_snapshot()
    for index in range(300):
        session.data.games.add(game(200 + 37 * index, ["normal", "static_rect"][index % 2],
                                    pattern=28 if index % 2 else -1,
                                    encounters={28: 1}), update=index // 30)
        if index % 30 == 29:
            session.data.games.close_update(index // 30)
    return TestClient(create_app(session))


def test_state_carries_every_chart_once_games_have_finished(client):
    response = client.get("/api/state")
    assert response.status_code == 200
    state = response.json()
    assert {"reward", "episode_length", "quarters", "survival", "causes", "hazard",
            "patterns"} <= set(state["charts"])
    assert state["training_figures"]["summary"]["games"] == 256
    assert state["evaluation"]["status"] == "idle"
    assert {slider["name"] for slider in state["game_controls"]["sliders"]} >= {"lives"}


def test_reward_weights_are_saved_from_the_dashboard(client):
    assert client.post("/api/rewards", json={"death_penalty": 3.5}).status_code == 200
    assert client.get("/api/state").json()["rewards"]["death_penalty"] == 3.5


def test_game_config_accepts_the_whole_config_the_state_reports(client):
    config = client.get("/api/state").json()["game_config"]
    assert "max_episode_frames" in config  # a setting the panel has no control for

    response = client.post("/api/game-config", json={**config, "difficulty": 1})

    assert response.status_code == 200
    assert client.get("/api/state").json()["game_config"]["difficulty"] == 1


def test_evaluation_size_is_bounded(client):
    assert client.post("/api/evaluate", json={"games": 4}).status_code == 400


def test_the_live_socket_pushes_state(client):
    with client.websocket_connect("/api/live") as socket:
        assert socket.receive_json()["model_name"] == "velocity-flow"
