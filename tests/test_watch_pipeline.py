"""Watch end to end, with the watcher process mocked and a real directory."""

import faulthandler
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("lupa.luajit21")

from dodge.training import PPOTrainingSession, newest_snapshot, snapshot_name

pytestmark = pytest.mark.skipif(not Path("dodge.p8.png").exists(),
                                reason="dodge.p8.png not present")


class _Watcher:
    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0


@pytest.fixture(autouse=True)
def _quiet_luajit_seh():
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


def test_watch_publishes_reloads_and_stops(tmp_path, monkeypatch):
    session = PPOTrainingSession(model_dir=tmp_path, autostart=False,
                                 architecture="mlp", device="cpu")
    session.model = object()
    watcher = _Watcher()
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: watcher)

    class Stub:
        completed_updates = 0
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    model = Stub()

    # No snapshot yet: the click queues rather than blocking.
    assert session.watch_agent() is False
    assert session.watch_pending

    model.completed_updates = 1
    session._on_post_update(model)
    assert not session.watch_pending
    assert newest_snapshot(session.live_watch_dir)[1] == 1
    assert session.watchers == [watcher]

    # While watched, every update publishes, retention keeps three.
    for update in range(2, 7):
        model.completed_updates = update
        session._on_post_update(model)
    names = sorted(p.name for p in session.live_watch_dir.glob("update-*.zip"))
    assert names == [snapshot_name(4), snapshot_name(5), snapshot_name(6)]
    assert not list(session.live_watch_dir.glob(".*")), "no partial files"

    # Once the window closes, publishing stops.
    watcher.alive = False
    model.completed_updates = 7
    session._on_post_update(model)
    assert newest_snapshot(session.live_watch_dir)[1] == 6

    # And a second click reuses the existing snapshot immediately.
    watcher.alive = True
    assert session.watch_agent() is True
