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


def _appearance(index, pid, outcome, seed=1_000_000):
    died = outcome == "died"
    return {"id": index, "seed": seed, "pid": pid, "type": "static", "start": 10, "end": 12,
            "outcome": outcome, "cause": "rect" if died else None,
            "kind": "static_rect" if died else None, "x": [60.0, 61.0, 62.0],
            "y": [64.0, 64.0, 64.0], "rects": [[0.0, 0.0, 16.0, 16.0, 1]],
            "death": ({"frame": 12, "score": 3.0, "since": 2, "alive": [1, 0, 0, 0],
                       "killer": None} if died else None)}


@pytest.fixture
def recorded(client):
    import numpy as np

    from dodge.death_paths import Recording
    recording = Recording(
        run_id="run-a", settings={"step_frames": 1}, games=2,
        actions={1_000_000: np.zeros(12, dtype=np.uint8)},
        appearances=[_appearance(0, 28, "died"), _appearance(1, 28, "survived"),
                     _appearance(2, 5, "survived")])
    job = client.app.state.dashboard.death_paths
    job.status, job.result = "done", recording
    return recording


def test_summary_lists_patterns(client, recorded):
    body = client.get("/api/death-paths").json()
    assert body["status"] == "done"
    assert body["data"]["run_id"] == "run-a"
    assert [p["pid"] for p in body["data"]["patterns"]] == [28, 5]
    assert body["data"]["patterns"][0] == {"pid": 28, "type": "static", "encounters": 2,
                                           "deaths": 1, "rate": 50.0}


def test_pattern_returns_its_appearances(client, recorded):
    body = client.get("/api/death-paths/pattern/28", params={"run": "run-a"}).json()
    assert [a["id"] for a in body["appearances"]] == [0, 1]
    assert body["layout"] == [[0.0, 0.0, 16.0, 16.0, 1]]


def test_replay_payload_decodes_to_frames(client, recorded, monkeypatch):
    import base64
    import zlib

    import numpy as np

    from dodge import pattern_replay
    frames = np.arange(3 * 128 * 128, dtype=np.uint32).astype(np.uint8).reshape(3, 128, 128)
    calls = []

    def fake(recording, appearance_id):
        calls.append((recording.run_id, appearance_id))
        samples = [{"frame": 10 + i, "st": 2, "enemies": 1, "enemy_gap": None,
                    "rect_gap": 1.5} for i in range(3)]
        return pattern_replay.Replay(frames=frames, samples=samples, matched=True, reason=None)

    monkeypatch.setattr(pattern_replay, "replay", fake)
    body = client.get("/api/death-paths/replay/0", params={"run": "run-a"}).json()
    assert calls == [("run-a", 0)]
    assert (body["fps"], body["count"], body["matched"], body["reason"]) == (60, 3, True, None)
    decoded = zlib.decompress(base64.b64decode(body["frames"]))
    assert decoded == frames.tobytes()
    assert len(body["samples"]) == 3
    assert body["appearance"]["id"] == 0


def test_replay_of_stale_run_is_rejected(client, recorded):
    for path in ("/api/death-paths/replay/0", "/api/death-paths/pattern/28"):
        response = client.get(path, params={"run": "run-b"})
        assert response.status_code == 409
        assert response.json()["detail"] == "That sampling run has been replaced; reopen the pattern."
    assert client.get("/api/death-paths/replay/99", params={"run": "run-a"}).status_code == 404


def test_the_last_run_stays_open_while_the_next_one_records(client, recorded):
    # The result is only replaced when the new run finishes.
    client.app.state.dashboard.death_paths.status = "running"
    response = client.get("/api/death-paths/pattern/28", params={"run": "run-a"})
    assert response.status_code == 200
    assert [a["id"] for a in response.json()["appearances"]] == [0, 1]


def test_death_paths_take_a_time_limit(client):
    dashboard = client.app.state.dashboard
    assert client.post("/api/death-paths", json={"games": 16, "minutes": 0}).status_code == 400
    assert client.post("/api/death-paths", json={"games": 16, "minutes": 2}).status_code == 200
    assert dashboard._pending["death-paths"]["options"] == {"time_limit": 120.0}


def test_training_is_paused_while_death_paths_record(client):
    dashboard = client.app.state.dashboard
    session = dashboard.session
    session.pause_event.clear()
    with dashboard.training_paused():
        assert session.pause_event.is_set()
    assert not session.pause_event.is_set()
    # A run that was already paused stays paused.
    session.pause_event.set()
    with dashboard.training_paused():
        pass
    assert session.pause_event.is_set()
