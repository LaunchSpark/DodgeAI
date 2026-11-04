"""Starting a fresh model: from the dashboard's Training panel, or the CLI."""

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("fastapi")
pytest.importorskip("altair")

from fastapi.testclient import TestClient

from dodge.critic import CRITIC_FEATURE_COUNT
from dodge.dashboard import create_app
from dodge.env import native_available
from dodge.new_policy import write_new_policy
from dodge.training import PPOTrainingSession

pytestmark = [
    pytest.mark.skipif(not native_available(), reason="dodge_native not installed"),
    pytest.mark.skipif(CRITIC_FEATURE_COUNT == 0,
                       reason="the native engine has no critic features"),
]

VELOCITY_ONLY = {"graduated_holds": False, "centred_logits": False}


def _flags(path):
    from dodge.training import DashboardPPO
    model = DashboardPPO.load(path, device="cpu")
    extractor = model.policy.features_extractor
    return model, {"carry_velocity": extractor.carry_velocity,
                   "graduated_holds": extractor.graduated_holds,
                   "centred_logits": model.policy.centred_logits}


def test_a_new_policy_carries_its_ablation_and_the_tables_schedule(tmp_path):
    path = write_new_policy("velocity-flow-v3", tmp_path, name="control", **VELOCITY_ONLY)
    assert path == tmp_path / "control.zip" and path.exists()
    model, flags = _flags(path)
    assert flags == {"carry_velocity": True, "graduated_holds": False,
                     "centred_logits": False}
    assert model.max_grad_norm == 2.0 and model.gamma == 0.997


def test_a_new_policy_never_overwrites_a_checkpoint(tmp_path):
    write_new_policy("velocity-flow-v3", tmp_path, name="run")
    with pytest.raises(ValueError, match="already exists"):
        write_new_policy("velocity-flow-v3", tmp_path, name="run")


def test_the_cli_takes_the_same_flags(tmp_path):
    from tools.new_policy import main
    assert main(["velocity-flow-v3", "--models-dir", str(tmp_path), "--name", "control",
                 "--no-graduated-holds", "--no-centred-logits"]) == 0
    _, flags = _flags(tmp_path / "control.zip")
    assert flags == {"carry_velocity": True, "graduated_holds": False,
                     "centred_logits": False}


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    from dodge import game_settings
    monkeypatch.setattr(game_settings, "CONFIG_PATH", tmp_path / "game-config.json")
    session = PPOTrainingSession(model_dir=tmp_path / "models", autostart=False,
                                 architecture=None, device="cpu")
    loaded = []
    monkeypatch.setattr(session, "load_model", loaded.append)
    return TestClient(create_app(session)), loaded, tmp_path / "models"


def test_the_training_panel_starts_a_new_model_and_trains_it(dashboard):
    client, loaded, models = dashboard
    response = client.post("/api/new-model", json={
        "architecture": "velocity-flow-v3", "name": "v3 velocity only",
        "flags": VELOCITY_ONLY})
    assert response.status_code == 200, response.text
    path = models / "v3-velocity-only.zip"
    assert [entry.path for entry in loaded] == [path]
    assert loaded[0].name == "v3-velocity-only"
    _, flags = _flags(path)
    assert flags["carry_velocity"] and not flags["graduated_holds"]


def test_the_training_panel_refuses_a_name_in_use(dashboard):
    client, loaded, _ = dashboard
    body = {"architecture": "velocity-flow-v3", "name": "run", "flags": {}}
    assert client.post("/api/new-model", json=body).status_code == 200
    response = client.post("/api/new-model", json=body)
    assert response.status_code == 400 and "already exists" in response.text
    assert len(loaded) == 1


def test_flags_are_refused_where_the_architecture_has_none(dashboard):
    client, loaded, _ = dashboard
    response = client.post("/api/new-model", json={
        "architecture": "velocity-flow-v2", "name": "v2", "flags": VELOCITY_ONLY})
    assert response.status_code == 400 and loaded == []


def test_the_state_lists_what_a_new_model_can_be(dashboard):
    client, _, _ = dashboard
    choices = client.get("/api/state").json()["new_model"]
    assert choices["architectures"][0] == "velocity-flow-v3"
    flags = {flag["name"]: flag for flag in choices["actor_flags"]["velocity-flow-v3"]}
    assert set(flags) == {"carry_velocity", "graduated_holds", "centred_logits"}
    assert all(flag["default"] is True and flag["label"] for flag in flags.values())
