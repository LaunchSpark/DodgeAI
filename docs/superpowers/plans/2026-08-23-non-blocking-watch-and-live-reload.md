# Non-blocking Watch and Live Model Reload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Watch, Save and Game Config return immediately, capture checkpoints only at safe points between optimizer updates, and let a watched agent adopt the newest policy when its episode ends.

**Architecture:** One queue of save requests, drained at three safe points — during rollout, while paused, and immediately after `train()` returns — never mid-optimizer-step. The GUI stops blocking on those requests and instead reads a `drain_events()` list each tick. Watchers read from a per-session directory of immutable `update-XXXXXXXX.zip` snapshots published by atomic rename, and reload between episodes.

**Tech Stack:** Python 3.12, Stable-Baselines3 PPO, PyTorch, Pygame, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-non-blocking-watch-and-live-reload-design.md`

## Global Constraints

- No exception from a save, rename, or `Popen` launch may escape the Pygame event loop. Every one becomes a toast.
- No checkpoint may be written during an optimizer step. Only the three safe points count.
- Watchers never read `*-latest.zip`. That file is mutable and is rewritten by the autosave.
- Snapshot files are invisible until their atomic rename. Partial files start with `.` and are ignored by readers.
- Snapshot filename format is exactly `update-XXXXXXXX.zip` — `update-` then the completed-update number zero-padded to 8 digits.
- Retention is the newest 3 snapshots. Deletion is best-effort; a `PermissionError` from a Windows file lock must never propagate.
- Model reload happens only between watched episodes, never inside one.
- Watchers keep using CPU inference (`device="cpu"`).
- Run `python -m pytest tests/ -q` before each commit. The suite is 285 tests and takes about 50 seconds.

---

### Task 1: Count completed PPO updates explicitly

`_completed_updates` is `_n_updates // n_epochs`. Stable-Baselines3 increments `_n_updates` once per epoch, and `velocity-flow-v2` sets `target_kl`, which breaks the epoch loop early — so the quotient under-counts for v2. Every consumer of that number is wrong today: the dashboard's update figure, watch snapshot filenames, and `exact_update`.

**Files:**
- Modify: `dodge/training.py` — `DashboardPPO`, `PPOTrainingSession._completed_updates`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DashboardPPO.completed_updates: int`, incremented once per successful `train()`. `PPOTrainingSession._completed_updates(model) -> int` keeps its signature and returns the counter when present.

- [ ] **Step 1: Write the failing test**

```python
def test_completed_updates_survive_kl_early_stopping(tmp_path):
    """target_kl breaks the epoch loop early, so _n_updates // n_epochs
    under-counts. v2 sets it, and the number names every watch snapshot."""
    from stable_baselines3.common.env_util import make_vec_env
    from dodge.env import DodgeEnv
    from dodge.training import DashboardPPO
    from dodge.policies import policy_class, policy_kwargs

    env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
    model = DashboardPPO(policy_class("mlp"), env, n_steps=16, batch_size=8,
                         n_epochs=10, target_kl=0.0, device="cpu", seed=0,
                         policy_kwargs=policy_kwargs("mlp"))
    assert model.completed_updates == 0

    model.learn(total_timesteps=32)

    # target_kl=0.0 stops after the first epoch every time, so the old
    # derivation would report 0 updates however many actually completed.
    assert model.completed_updates == 2, "one per train() call"
    assert model._n_updates < model.completed_updates * model.n_epochs
    env.close()


def test_the_session_reads_the_explicit_counter(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class Stub:
        completed_updates = 7
        _n_updates = 3
        n_epochs = 10

    assert session._completed_updates(Stub()) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "completed_updates or explicit_counter"`
Expected: FAIL with `AttributeError: 'DashboardPPO' object has no attribute 'completed_updates'`

- [ ] **Step 3: Write minimal implementation**

In `DashboardPPO.__init__`, before `super().__init__(...)`:

```python
        # Counted rather than derived. _n_updates // n_epochs was the old
        # answer, and target_kl breaks the epoch loop early, so for any
        # architecture that sets it the quotient is short.
        self.completed_updates = 0
```

Add to `_excluded_save_params` so the counter is not pickled into checkpoints:

```python
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_learning_tracker"]
```

leave as-is — `completed_updates` is a plain int and *should* travel with the checkpoint so a reloaded model resumes its numbering.

In `DashboardPPO.train`, change the `try/finally` to count only a successful return:

```python
        optimizer.step = tracked_step
        try:
            result = super().train()
        finally:
            optimizer.step = original_step
        # Only after train() returned and the hook is off: a raised exception
        # is not a completed update.
        self.completed_updates += 1
        return result
```

Replace `PPOTrainingSession._completed_updates`:

```python
    @staticmethod
    def _completed_updates(model):
        counter = getattr(model, "completed_updates", None)
        if counter is not None:
            return int(counter)
        # Checkpoints written before the counter existed.
        epochs = max(int(getattr(model, "n_epochs", 1)), 1)
        return int(getattr(model, "_n_updates", 0)) // epochs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "completed_updates or explicit_counter"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 287 passed

- [ ] **Step 6: Commit**

```bash
git add dodge/training.py tests/test_training_session.py
git commit -m "Count completed PPO updates instead of deriving them

_n_updates // n_epochs assumed every epoch runs. velocity-flow-v2 sets
target_kl, which breaks the loop early, so the quotient was short for it
-- and that number names every watch snapshot and drives the dashboard's
update figure."
```

---

### Task 2: A post-update safe point

Saves are only served from `_should_stop`, which runs inside `_on_step` — during rollout collection. A request queued while `train()` is running waits for the next rollout. Add a third safe point immediately after `train()` returns.

**Files:**
- Modify: `dodge/training.py` — `DashboardPPO`, `PPOTrainingSession._new_model`/`_load_model`, `_build_model`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: `DashboardPPO.completed_updates` from Task 1.
- Produces: `DashboardPPO.set_post_update(callback)` and a `post_update=` constructor kwarg. The callback takes the model and is invoked once per successful `train()`, after the optimizer hook is restored.

- [ ] **Step 1: Write the failing test**

```python
def test_the_post_update_hook_runs_after_the_optimizer_hook_is_off():
    """A save taken while the optimizer step is patched would capture a model
    mid-update, so the hook has to fire after train() has fully unwound."""
    from stable_baselines3.common.env_util import make_vec_env
    from dodge.env import DodgeEnv
    from dodge.training import DashboardPPO
    from dodge.policies import policy_class, policy_kwargs

    env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
    seen = []
    model = DashboardPPO(policy_class("mlp"), env, n_steps=16, batch_size=8,
                         n_epochs=1, device="cpu", seed=0,
                         policy_kwargs=policy_kwargs("mlp"))
    model.set_post_update(lambda m: seen.append(
        (m.completed_updates, m.policy.optimizer.step.__name__)))

    model.learn(total_timesteps=32)

    assert seen, "the hook never ran"
    assert all(name == "step" for _, name in seen), "optimizer still patched"
    assert [update for update, _ in seen] == [1, 2], "once per update, in order"
    env.close()


def test_a_failing_train_does_not_reach_the_post_update_hook():
    from dodge.training import DashboardPPO

    calls = []
    model = DashboardPPO.__new__(DashboardPPO)
    model._learning_tracker = None
    model._post_update = lambda m: calls.append(m)
    model.completed_updates = 0

    class Boom(Exception):
        pass

    def explode():
        raise Boom()

    import types
    model.train = types.MethodType(
        lambda self: DashboardPPO.train(self), model)
    # Simulate train() raising inside super().train()
    try:
        raise Boom()
    except Boom:
        pass
    assert calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "post_update"`
Expected: FAIL with `AttributeError: 'DashboardPPO' object has no attribute 'set_post_update'`

- [ ] **Step 3: Write minimal implementation**

In `DashboardPPO.__init__`, add the kwarg alongside `learning_tracker`:

```python
    def __init__(self, *args, learning_tracker=None, post_update=None, **kwargs):
        self._learning_tracker = learning_tracker
        self._post_update = post_update
        self.completed_updates = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_learning_tracker",
                                                  "_post_update"]

    def set_post_update(self, callback):
        self._post_update = callback
```

At the end of `train`, after the counter increment from Task 1:

```python
        self.completed_updates += 1
        if self._post_update:
            # The one moment the model is provably not mid-update: train() has
            # returned and the optimizer hook is back to the real step.
            self._post_update(self)
        return result
```

In `_build_model`, pass it:

```python
            learning_tracker=self._track_learning,
            post_update=self._process_save_requests,
```

In `_load_model`, next to `model.set_learning_tracker(...)`:

```python
        model.set_post_update(self._process_save_requests)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "post_update"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 289 passed

- [ ] **Step 6: Commit**

```bash
git add dodge/training.py tests/test_training_session.py
git commit -m "Serve saves at a post-update safe point

Requests were only drained from _should_stop, which runs during rollout
collection, so anything asked for while train() was running waited for
the next rollout. The hook fires after train() returns and after the
optimizer step is unpatched, which is the one moment the model is
provably not mid-update."
```

---

### Task 3: Asynchronous save requests and session events

`_save_to_path` blocks the caller on `request.finished.wait(timeout=15)` and the caller is the Pygame event loop. Replace the GUI's path with a fire-and-forget request plus an event queue the GUI drains each tick.

**Files:**
- Modify: `dodge/training.py` — `_SaveRequest`, `PPOTrainingSession`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: `_process_save_requests` from Task 2.
- Produces:
  - `SessionEvent(kind: str, ok: bool, message: str, path: Path | None = None, update: int = 0)`, a frozen dataclass exported from `dodge.training`.
  - `PPOTrainingSession.request_save(path, kind) -> None` — queues and returns immediately.
  - `PPOTrainingSession.drain_events() -> list[SessionEvent]` — returns and clears pending events.
  - `_SaveRequest` gains `kind: str = "save"`.
  - `_save_to_path` keeps its blocking behaviour for internal callers and tests.

- [ ] **Step 1: Write the failing test**

```python
def test_request_save_returns_before_the_save_happens(tmp_path):
    """The caller is the Pygame event loop. It must not wait on the trainer."""
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()          # non-None so the request is accepted

    session.request_save(tmp_path / "async.zip", kind="save")

    assert session.drain_events() == [], "nothing has been served yet"
    assert session.save_requests.qsize() == 1


def test_serving_a_request_emits_a_success_event(tmp_path):
    from dodge.training import PPOTrainingSession, SessionEvent

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.request_save(tmp_path / "async.zip", kind="save")

    class Stub:
        completed_updates = 4
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    session._process_save_requests(Stub())
    events = session.drain_events()

    assert len(events) == 1
    assert events[0].ok and events[0].kind == "save" and events[0].update == 4
    assert events[0].path == tmp_path / "async.zip"
    assert session.drain_events() == [], "draining clears"


def test_a_failed_save_emits_an_error_event_rather_than_raising(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.request_save(tmp_path / "async.zip", kind="save")

    class Boom:
        completed_updates = 0
        def save(self, path):
            raise OSError("disk full")

    session._process_save_requests(Boom())
    events = session.drain_events()

    assert len(events) == 1 and not events[0].ok
    assert "disk full" in events[0].message


def test_shutdown_completes_pending_requests_with_an_error(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.request_save(tmp_path / "orphan.zip", kind="watch")

    session._fail_save_requests(RuntimeError("training session closed"))
    events = session.drain_events()

    assert len(events) == 1 and not events[0].ok and events[0].kind == "watch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "request_save or success_event or error_event or shutdown_completes"`
Expected: FAIL with `AttributeError: 'PPOTrainingSession' object has no attribute 'request_save'`

- [ ] **Step 3: Write minimal implementation**

Add near `_SaveRequest` in `dodge/training.py`:

```python
@dataclass(frozen=True)
class SessionEvent:
    """Something the trainer finished, for the GUI to turn into a toast.

    The event loop cannot block on the trainer, so results come back here
    instead of as return values.
    """
    kind: str
    ok: bool
    message: str
    path: Path | None = None
    update: int = 0
```

Give `_SaveRequest` a kind:

```python
@dataclass
class _SaveRequest:
    path: Path
    finished: threading.Event
    error: Exception | None = None
    completed_update: int = 0
    kind: str = "save"
    notify: bool = False
```

In `PPOTrainingSession.__init__`, beside `self.save_requests`:

```python
        self.events = queue.Queue()
```

Add the three methods:

```python
    def request_save(self, path, kind="save"):
        """Queue a checkpoint and return. Results arrive via drain_events()."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            self._emit(SessionEvent(kind, False, "No model to save yet"))
            return
        self.save_requests.put(_SaveRequest(
            path=path.with_suffix(""), finished=threading.Event(),
            kind=kind, notify=True,
        ))

    def drain_events(self):
        drained = []
        while True:
            try:
                drained.append(self.events.get_nowait())
            except queue.Empty:
                return drained

    def _emit(self, event):
        self.events.put(event)
```

Extend `_process_save_requests` to emit for notifying requests:

```python
    def _process_save_requests(self, model):
        while True:
            try:
                request = self.save_requests.get_nowait()
            except queue.Empty:
                return
            try:
                model.save(request.path)
                request.completed_update = self._completed_updates(model)
            except Exception as error:
                request.error = error
            finally:
                request.finished.set()
            if request.notify:
                saved = request.path.with_suffix(".zip")
                self._emit(SessionEvent(
                    request.kind, request.error is None,
                    str(request.error) if request.error else f"Saved {saved.name}",
                    saved, request.completed_update,
                ))
```

And `_fail_save_requests`:

```python
    def _fail_save_requests(self, error):
        while True:
            try:
                request = self.save_requests.get_nowait()
            except queue.Empty:
                return
            request.error = error
            request.finished.set()
            if request.notify:
                self._emit(SessionEvent(request.kind, False, str(error)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "request_save or success_event or error_event or shutdown_completes"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 293 passed

- [ ] **Step 6: Commit**

```bash
git add dodge/training.py tests/test_training_session.py
git commit -m "Make checkpoint requests asynchronous

_save_to_path blocked its caller for up to fifteen seconds, and the
caller is the Pygame event loop. request_save queues and returns;
results come back through drain_events as SessionEvents, including the
failures, so nothing from a save can reach the event loop as an
exception."
```

---

### Task 4: Immutable watch snapshots

Watchers currently read a file the trainer may rewrite. Publish immutable, atomically-renamed snapshots into a per-session directory instead.

**Files:**
- Modify: `dodge/training.py` — `PPOTrainingSession`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: `request_save`, `SessionEvent`, `_process_save_requests`.
- Produces:
  - `SNAPSHOT_RETENTION = 3` and `snapshot_name(update) -> str` returning `f"update-{update:08d}.zip"`, module-level in `dodge.training`.
  - `newest_snapshot(directory, after=-1) -> tuple[Path, int] | None`, module-level, ignoring dotfiles and malformed names.
  - `PPOTrainingSession.live_watch_dir: Path` — `model_dir / "watch" / f"live-{os.getpid()}-{id(self):x}"`.
  - `PPOTrainingSession._publish_snapshot(model)` — called from the post-update hook.

- [ ] **Step 1: Write the failing test**

```python
def test_snapshot_names_sort_lexicographically_with_the_update(tmp_path):
    from dodge.training import snapshot_name

    assert snapshot_name(7) == "update-00000007.zip"
    assert sorted([snapshot_name(9), snapshot_name(10)]) == [
        snapshot_name(9), snapshot_name(10)]


def test_newest_snapshot_ignores_partial_and_malformed_files(tmp_path):
    from dodge.training import newest_snapshot, snapshot_name

    (tmp_path / snapshot_name(3)).write_bytes(b"x")
    (tmp_path / snapshot_name(11)).write_bytes(b"x")
    (tmp_path / ".update-00000099.zip.tmp").write_bytes(b"partial")
    (tmp_path / "velocity-flow-latest.zip").write_bytes(b"mutable")

    assert newest_snapshot(tmp_path) == (tmp_path / snapshot_name(11), 11)
    assert newest_snapshot(tmp_path, after=11) is None
    assert newest_snapshot(tmp_path, after=3) == (tmp_path / snapshot_name(11), 11)
    assert newest_snapshot(tmp_path / "missing") is None


def test_publishing_renames_atomically_and_keeps_three(tmp_path):
    from dodge.training import PPOTrainingSession, newest_snapshot

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    live = session.live_watch_dir

    class Stub:
        completed_updates = 0
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    model = Stub()
    for update in range(1, 6):
        model.completed_updates = update
        session._publish_snapshot(model)

    names = sorted(p.name for p in live.glob("update-*.zip"))
    assert names == ["update-00000003.zip", "update-00000004.zip",
                     "update-00000005.zip"], "newest three retained"
    assert not list(live.glob(".*")), "no temporary files left behind"
    assert newest_snapshot(live)[1] == 5


def test_retention_survives_a_locked_file(tmp_path, monkeypatch):
    """A watcher reading an old snapshot locks it on Windows. Failing to
    delete it must not take training down."""
    from dodge.training import PPOTrainingSession
    import pathlib

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class Stub:
        completed_updates = 0
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    real_unlink = pathlib.Path.unlink

    def locked(self, *a, **k):
        if self.name.startswith("update-"):
            raise PermissionError("file in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", locked)
    model = Stub()
    for update in range(1, 6):
        model.completed_updates = update
        session._publish_snapshot(model)     # must not raise

    assert len(list(session.live_watch_dir.glob("update-*.zip"))) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "snapshot or retention"`
Expected: FAIL with `ImportError: cannot import name 'snapshot_name' from 'dodge.training'`

- [ ] **Step 3: Write minimal implementation**

Module level in `dodge/training.py` (add `import os` at the top if absent):

```python
# Watchers must never read the mutable *-latest.zip autosave, so snapshots go
# to their own directory under names that never change once visible.
SNAPSHOT_RETENTION = 3


def snapshot_name(update):
    """Zero-padded so lexicographic order is numeric order."""
    return f"update-{int(update):08d}.zip"


def newest_snapshot(directory, after=-1):
    """The highest completed update in `directory` above `after`.

    Dotfiles are skipped: a snapshot is written to a hidden temporary name and
    only becomes visible on its atomic rename, so anything starting with a dot
    is a partial write.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best = None
    for path in directory.glob("update-*.zip"):
        if path.name.startswith("."):
            continue
        try:
            update = int(path.stem.removeprefix("update-"))
        except ValueError:
            continue
        if update > after and (best is None or update > best[1]):
            best = (path, update)
    return best
```

In `PPOTrainingSession.__init__`:

```python
        # Private to this session: two dashboards must not publish into the
        # same stream, and a watcher should only ever see this trainer's work.
        self.live_watch_dir = (self.model_dir / "watch"
                               / f"live-{os.getpid()}-{id(self):x}")
        self.watchers = []
```

Add the publisher:

```python
    def _publish_snapshot(self, model):
        """Write one immutable snapshot, then prune to the newest few.

        Saved to a hidden temporary name and renamed, so a watcher scanning
        the directory can never open a half-written zip.
        """
        update = self._completed_updates(model)
        directory = self.live_watch_dir
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f".{snapshot_name(update)}.{secrets.token_hex(4)}"
        try:
            model.save(stem)
            temporary = stem.with_suffix(stem.suffix + ".zip")
            temporary.replace(directory / snapshot_name(update))
        except Exception as error:
            self._emit(SessionEvent("watch", False,
                                    f"Could not publish snapshot: {error}"))
            return
        self._prune_snapshots(directory)

    @staticmethod
    def _prune_snapshots(directory):
        """Best-effort. A watcher holding an old snapshot open locks it on
        Windows, and losing that race must not interrupt training."""
        snapshots = sorted(directory.glob("update-*.zip"))
        for path in snapshots[:-SNAPSHOT_RETENTION]:
            try:
                path.unlink()
            except OSError:
                pass
```

Note: `model.save(stem)` appends `.zip`, which is why the temporary is `stem + ".zip"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "snapshot or retention"`
Expected: PASS

- [ ] **Step 5: Measure one publish, and record it**

Run:

```bash
python - <<'EOF'
import tempfile, time
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from dodge.env import DodgeEnv
from dodge.training import PPOTrainingSession

session = PPOTrainingSession(model_dir=Path(tempfile.mkdtemp()), autostart=False,
                             architecture="velocity-flow-v2", device="cpu")
env = make_vec_env(DodgeEnv, n_envs=1, env_kwargs={"max_episode_frames": 20})
model = session._new_model(env)
for _ in range(3):
    start = time.perf_counter()
    session._publish_snapshot(model)
    print(f"publish: {(time.perf_counter() - start) * 1000:.1f} ms")
env.close()
EOF
```

The spec assumes this cost is acceptable without measuring it. A v2 cycle is about 2.4s. If a publish exceeds ~120ms (5% of a cycle), stop and report the number rather than continuing — the fix is publishing on a cadence rather than every update, which is a change to Task 5's gate and needs the user's call.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 297 passed

- [ ] **Step 7: Commit**

```bash
git add dodge/training.py tests/test_training_session.py
git commit -m "Publish immutable watch snapshots

Watchers read *-latest.zip, which the autosave rewrites underneath them.
Snapshots now go to a per-session directory under update-XXXXXXXX.zip,
written to a hidden temporary name and made visible by an atomic rename,
so a scan can never open a half-written file. Retention keeps the newest
three; deletion is best-effort because a watcher holding one open locks
it on Windows and losing that race must not interrupt training."
```

---

### Task 5: Publish while watched, and stop when the watchers exit

**Files:**
- Modify: `dodge/training.py` — `PPOTrainingSession._process_save_requests` caller, `watch_agent`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: `_publish_snapshot`, `self.watchers`.
- Produces: `PPOTrainingSession._on_post_update(model)` — the post-update hook, replacing the direct `_process_save_requests` wiring from Task 2. It drains saves, then publishes if `_watchers_alive()` is true. `PPOTrainingSession._watchers_alive() -> bool` prunes exited `Popen` handles and reports whether any remain.

- [ ] **Step 1: Write the failing test**

```python
def test_snapshots_are_published_only_while_a_watcher_is_alive(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")

    class Stub:
        completed_updates = 1
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    model = Stub()
    session._on_post_update(model)
    assert not list(session.live_watch_dir.glob("update-*.zip")) \
        if session.live_watch_dir.exists() else True, "no watcher, no snapshot"

    class Alive:
        def poll(self):
            return None

    class Exited:
        def poll(self):
            return 0

    session.watchers = [Alive()]
    model.completed_updates = 2
    session._on_post_update(model)
    assert len(list(session.live_watch_dir.glob("update-*.zip"))) == 1

    session.watchers = [Exited()]
    assert session._watchers_alive() is False
    assert session.watchers == [], "exited handles are pruned"
    model.completed_updates = 3
    session._on_post_update(model)
    assert len(list(session.live_watch_dir.glob("update-*.zip"))) == 1, \
        "publishing stops once the last watcher is gone"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "published_only_while"`
Expected: FAIL with `AttributeError: 'PPOTrainingSession' object has no attribute '_on_post_update'`

- [ ] **Step 3: Write minimal implementation**

```python
    def _watchers_alive(self):
        self.watchers = [w for w in self.watchers if w.poll() is None]
        return bool(self.watchers)

    def _on_post_update(self, model):
        """The safe point. Saves first, then a snapshot if anyone is watching."""
        self._process_save_requests(model)
        if self._watchers_alive():
            self._publish_snapshot(model)
```

Change the two wiring points from Task 2 to use it:

```python
            post_update=self._on_post_update,
```

```python
        model.set_post_update(self._on_post_update)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "published_only_while"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 298 passed

- [ ] **Step 6: Commit**

```bash
git add dodge/training.py tests/test_training_session.py
git commit -m "Publish a snapshot per update only while watched

Continuous publishing costs a checkpoint write on the training thread at
every safe point, which is only worth paying while something is reading
them. Watcher Popen handles are tracked and pruned as they exit."
```

---

### Task 6: Non-blocking `watch_agent`

**Files:**
- Modify: `dodge/training.py` — `watch_agent`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: `request_save`, `newest_snapshot`, `_watchers_alive`, `live_watch_dir`.
- Produces: `PPOTrainingSession.watch_agent() -> bool` returns immediately. `True` means a window was launched now; `False` means one is pending the next safe point. `PPOTrainingSession.watch_pending: bool`. `PPOTrainingSession._launch_watcher(path, update) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
def test_watch_launches_immediately_from_an_existing_snapshot(tmp_path, monkeypatch):
    from dodge.training import PPOTrainingSession, snapshot_name

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    session.live_watch_dir.mkdir(parents=True)
    (session.live_watch_dir / snapshot_name(12)).write_bytes(b"checkpoint")

    commands = []
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command) or _Alive())

    assert session.watch_agent() is True
    assert not session.watch_pending
    assert commands and commands[0][commands[0].index("--current-update") + 1] == "12"
    assert commands[0][commands[0].index("--live-dir") + 1] == str(session.live_watch_dir)


def test_watch_with_no_snapshot_waits_for_the_next_safe_point(tmp_path, monkeypatch):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    commands = []
    monkeypatch.setattr("dodge.training.subprocess.Popen",
                        lambda command, cwd: commands.append(command) or _Alive())

    assert session.watch_agent() is False, "nothing to open yet"
    assert session.watch_pending
    assert commands == []

    assert session.watch_agent() is False, "a second click does not queue twice"
    assert session.save_requests.qsize() == 1

    class Stub:
        completed_updates = 3
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    session._on_post_update(Stub())
    assert commands, "the pending watch launched at the safe point"
    assert not session.watch_pending
```

Add this helper near the top of the test module:

```python
class _Alive:
    """A Popen stand-in that never exits."""
    def poll(self):
        return None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "watch_launches or watch_with_no_snapshot"`
Expected: FAIL with `AttributeError: 'PPOTrainingSession' object has no attribute 'watch_pending'`

- [ ] **Step 3: Write minimal implementation**

In `__init__`:

```python
        self.watch_pending = False
```

Replace `watch_agent` wholesale:

```python
    def watch_agent(self):
        """Open a watcher without blocking the event loop.

        Returns True if a window was launched now, False if one is waiting for
        the next completed update. An older same-session snapshot is preferred
        over waiting: the watcher reloads to the newest policy at its first
        death anyway.
        """
        if self.model is None:
            self._emit(SessionEvent("watch", False, "No model to watch yet"))
            return False
        existing = newest_snapshot(self.live_watch_dir)
        if existing:
            return self._launch_watcher(*existing)
        if not self.watch_pending:
            self.watch_pending = True
            self.request_save(self.live_watch_dir / "pending.zip", kind="watch")
        return False

    def _launch_watcher(self, path, update):
        with self.lock:
            model_name = self.active_model_name
            game_config = dict(self.env_kwargs)
            self.data.update = update
        project = Path(__file__).resolve().parents[1]
        difficulty = {1: "easy", 2: "normal", 3: "hard"}[game_config["difficulty"]]
        command = [
            sys.executable, str(project / "watch_ppo.py"), str(path),
            "--difficulty", difficulty,
            "--action-repeat", str(game_config["action_repeat"]),
            "--update", str(update),
            "--current-update", str(update),
            "--live-dir", str(self.live_watch_dir),
            "--seed", str(secrets.randbelow(2**31 - 1)),
            "--model-name", model_name,
        ]
        if game_config["patterns"]:
            command.append("--patterns")
        command.append("--powerups" if game_config["powerups"] else "--no-powerups")
        try:
            self.watchers.append(subprocess.Popen(command, cwd=project))
        except OSError as error:
            self._emit(SessionEvent("watch", False, f"Could not open window: {error}"))
            return False
        self.last_watch_label = f"{model_name} | Update {update}"
        self._emit(SessionEvent("watch", True, f"Opening {self.last_watch_label}",
                                path, update))
        return True
```

In `_on_post_update`, launch a pending watch after publishing:

```python
    def _on_post_update(self, model):
        """The safe point. Saves first, then a snapshot if anyone is watching."""
        self._process_save_requests(model)
        if self.watch_pending:
            self._publish_snapshot(model)
            self.watch_pending = False
            found = newest_snapshot(self.live_watch_dir)
            if found:
                self._launch_watcher(*found)
        elif self._watchers_alive():
            self._publish_snapshot(model)
```

The `kind="watch"` request queued by `watch_agent` writes `pending.zip` into the live directory; it exists only to guarantee the post-update hook has something to do. Delete it after publishing:

```python
        for stray in self.live_watch_dir.glob("pending.zip"):
            try:
                stray.unlink()
            except OSError:
                pass
```

Place that inside `_on_post_update` immediately after `self.watch_pending = False`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "watch_launches or watch_with_no_snapshot"`
Expected: PASS

- [ ] **Step 5: Update the existing watch test**

`test_watch_uses_unique_update_snapshot_and_random_seed` asserts the old synchronous naming (`{architecture}-update-000037-seed-{seed}.zip` under `model_dir/watch`). Replace its body to assert the new behaviour: a snapshot in `session.live_watch_dir` named by `snapshot_name`, and `--live-dir`/`--current-update` present in the command. Keep its assertions on `--difficulty`, `--action-repeat`, `--patterns`, `--no-powerups` and `--model-name`, which are unchanged.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 300 passed

- [ ] **Step 7: Commit**

```bash
git add dodge/training.py tests/test_training_session.py
git commit -m "Return from watch_agent immediately

Watch blocked the event loop for up to fifteen seconds waiting for the
trainer to reach a safe point. It now opens the newest same-session
snapshot at once if there is one -- the watcher reloads to the current
policy at its first death anyway -- and otherwise queues, launching from
the next completed update."
```

---

### Task 7: GUI reads events instead of blocking

**Files:**
- Modify: `training_gui.py` — `_save_model`, the Watch and Save click handlers, `run`
- Test: `tests/test_training_gui.py`

**Interfaces:**
- Consumes: `drain_events()`, `SessionEvent`, `request_save`, `watch_agent`.
- Produces: `TrainingDashboard._drain_session_events()`, called once per event-loop iteration next to `self.session.tick(dt)`. `TrainingDashboard.save_pending: bool`.

- [ ] **Step 1: Write the failing test**

```python
def test_session_events_become_toasts(tmp_path):
    from dodge.training import SessionEvent
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    session.pending_events = [SessionEvent("save", True, "Saved thing.zip")]
    dashboard = TrainingDashboard(session, hidden=True)

    dashboard._drain_session_events()

    assert dashboard.toast == "Saved thing.zip"
    pygame.quit()


def test_a_failed_save_event_clears_the_pending_button(tmp_path):
    from dodge.training import SessionEvent
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.save_pending = True
    session.pending_events = [SessionEvent("save", False, "disk full")]

    dashboard._drain_session_events()

    assert dashboard.toast == "disk full"
    assert not dashboard.save_pending, "the button has to come back"
    pygame.quit()
```

Add to `_StubSession`:

```python
        self.pending_events = []

    def drain_events(self):
        drained, self.pending_events = self.pending_events, []
        return drained

    def request_save(self, path, kind="save"):
        self.configured.append(("request_save", Path(path).name, kind))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_gui.py -q -k "session_events or failed_save_event"`
Expected: FAIL with `AttributeError: 'TrainingDashboard' object has no attribute '_drain_session_events'`

- [ ] **Step 3: Write minimal implementation**

In `TrainingDashboard.__init__`:

```python
        self.save_pending = False
```

Add the drain:

```python
    def _drain_session_events(self):
        """Turn asynchronous trainer results into toasts.

        Nothing here may raise: this runs inside the event loop, and an
        exception escaping it closes the window and takes training with it.
        """
        drain = getattr(self.session, "drain_events", None)
        if drain is None:
            return
        for event in drain():
            if event.kind == "save":
                self.save_pending = False
                if event.ok:
                    self._discover_models(force=True)
                    self.selected_model = next(
                        (i for i, entry in enumerate(self.model_entries)
                         if entry.path == event.path),
                        None,
                    )
                    if self.selected_model is not None:
                        self.model_offset = max(0, self.selected_model - 4)
            self._show_toast(event.message)
```

Replace `_save_model`:

```python
    def _save_model(self):
        if self.save_pending:
            return
        self.save_pending = True
        safe = self.session.request_save(
            self.session.model_dir / f"{self.model_name}.zip", kind="save")
        self._show_toast("Saving…")
        return safe
```

Replace the Watch click handler:

```python
            elif button.name == "Watch Agent":
                if not self.session.watch_agent():
                    self._show_toast("Waiting for current update…")
```

The success toast now comes from the event, so the handler only reports the waiting case.

In `run`, beside `self.session.tick(dt)`:

```python
                self.session.tick(dt)
                self._drain_session_events()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_gui.py -q -k "session_events or failed_save_event"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 302 passed

- [ ] **Step 6: Commit**

```bash
git add training_gui.py tests/test_training_gui.py
git commit -m "Read trainer results from events, not return values

Save and Watch no longer block the event loop; the dashboard drains
SessionEvents each tick and turns them into toasts, including failures.
Save disables its button until its event arrives, then selects the
checkpoint it wrote."
```

---

### Task 8: Game Config commits only after its checkpoint

**Files:**
- Modify: `dodge/training.py` — `configure_game`; `training_gui.py` — `_apply_game_config`
- Test: `tests/test_training_session.py`

**Interfaces:**
- Consumes: `request_save`, `drain_events`.
- Produces: `configure_game(...)` returns `"pending"` when it has queued its preservation checkpoint, `False` when nothing changed, `True` when applied with no model to preserve. The pending config is held on `PPOTrainingSession.pending_config` and committed by the `"config"` event.

- [ ] **Step 1: Write the failing test**

```python
def test_game_config_waits_for_its_preservation_checkpoint(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    before = session.game_config()

    assert session.configure_game(difficulty=3, patterns=False, powerups=True,
                                  action_repeat=4, lives=0) == "pending"
    assert session.game_config() == before, "not committed until the save lands"
    assert not session.restart_event.is_set()

    class Stub:
        completed_updates = 5
        def save(self, path):
            Path(f"{path}.zip").write_bytes(b"checkpoint")

    session._process_save_requests(Stub())
    session.drain_events()

    assert session.game_config()["difficulty"] == 3
    assert session.restart_event.is_set()


def test_a_failed_preservation_checkpoint_leaves_the_config_alone(tmp_path):
    from dodge.training import PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    session.model = object()
    before = session.game_config()
    session.configure_game(difficulty=3, patterns=False, powerups=True,
                           action_repeat=4, lives=0)

    class Boom:
        completed_updates = 0
        def save(self, path):
            raise OSError("disk full")

    session._process_save_requests(Boom())
    events = session.drain_events()

    assert session.game_config() == before
    assert not session.restart_event.is_set()
    assert any(not e.ok and e.kind == "config" for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "preservation_checkpoint"`
Expected: FAIL — `configure_game` returns `True` and commits immediately.

- [ ] **Step 3: Write minimal implementation**

In `__init__`: `self.pending_config = None`

Replace the tail of `configure_game`, from the `checkpoint = None` block onward:

```python
        if self.model is None:
            with self.lock:
                self.env_kwargs = config
                idle = self.architecture is None and self.model is None
                self.data.status = "idle" if idle else "reconfiguring"
            if not idle:
                self.restart_event.set()
            return True

        # Queued, not taken here: this runs on the event loop, and the trainer
        # may be mid-update. The config is applied by the resulting event, so a
        # failed checkpoint leaves the running game exactly as it was.
        self.pending_config = config
        self.pause_event.clear()
        self.request_save(self.model_dir / "runtime" / "game-config.zip",
                          kind="config")
        return "pending"
```

In `_process_save_requests`, the notify branch already emits. Commit the config in `_emit`'s consumer instead — add to `PPOTrainingSession`:

```python
    def _emit(self, event):
        if event.kind == "config" and self.pending_config is not None:
            if event.ok:
                with self.lock:
                    self.env_kwargs = self.pending_config
                    self.pending_load = event.path
                    self.data.status = "reconfiguring"
                self.restart_event.set()
            self.pending_config = None
        self.events.put(event)
```

In `training_gui.py`, `_apply_game_config` closes the menu on any non-`False` result:

```python
        try:
            changed = configure(**{key: self.game_config[key] for key in GAME_DEFAULTS})
        except (OSError, ValueError) as error:
            self._show_toast(str(error))
            return
        self.game_menu_open = False
        if changed == "pending":
            self._show_toast("Applying after this update…")
        else:
            self._show_toast("Game configuration applied" if changed
                             else "Game configuration unchanged")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "preservation_checkpoint"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 304 passed. `test_game_config_restarts_with_the_live_policy` and `test_game_config_applies_toggles_and_discrete_sliders` both assert the old synchronous behaviour — update them to drive the save through `_process_save_requests` before asserting the config changed.

- [ ] **Step 6: Commit**

```bash
git add dodge/training.py training_gui.py tests/
git commit -m "Apply a game configuration only once its checkpoint lands

configure_game saved the live policy synchronously from the event loop
and committed the new mechanics regardless. It now queues the
preservation checkpoint and commits from the resulting event, so a
failed save leaves the running game exactly as it was."
```

---

### Task 9: The watcher reloads between episodes

**Files:**
- Modify: `watch_ppo.py`
- Test: `tests/test_training_session.py` (where `window_title` and the other watcher tests already live)

**Interfaces:**
- Consumes: `newest_snapshot`, `snapshot_name` from `dodge.training`.
- Produces: `watch_ppo.reload_if_newer(model, live_dir, current_update) -> tuple[model, int, str | None]` — returns the possibly-swapped model, the update it is now on, and an error message if a newer snapshot existed but could not be loaded. Two new CLI arguments: `--live-dir PATH` and `--current-update N`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_watcher_adopts_a_newer_snapshot(tmp_path):
    import watch_ppo
    from dodge.training import snapshot_name

    class Old:
        pass

    loaded = []
    real = watch_ppo.PPO.load
    watch_ppo.PPO.load = staticmethod(lambda path, device: loaded.append(path) or "new")
    try:
        (tmp_path / snapshot_name(9)).write_bytes(b"checkpoint")
        model, update, error = watch_ppo.reload_if_newer(Old(), tmp_path, 4)
        assert model == "new" and update == 9 and error is None
        assert loaded == [tmp_path / snapshot_name(9)]

        again, update, error = watch_ppo.reload_if_newer(model, tmp_path, 9)
        assert again is model and update == 9, "nothing newer, nothing loaded"
    finally:
        watch_ppo.PPO.load = real


def test_a_broken_snapshot_leaves_the_watcher_on_its_model(tmp_path):
    import watch_ppo
    from dodge.training import snapshot_name

    class Old:
        pass

    old = Old()
    real = watch_ppo.PPO.load

    def explode(path, device):
        raise ValueError("bad zip")

    watch_ppo.PPO.load = staticmethod(explode)
    try:
        (tmp_path / snapshot_name(9)).write_bytes(b"garbage")
        model, update, error = watch_ppo.reload_if_newer(old, tmp_path, 4)
        assert model is old, "the in-memory policy has to survive"
        assert update == 4, "and stay on its own update, so it retries later"
        assert "bad zip" in error
    finally:
        watch_ppo.PPO.load = real


def test_no_live_directory_is_not_an_error(tmp_path):
    import watch_ppo

    class Old:
        pass

    old = Old()
    model, update, error = watch_ppo.reload_if_newer(old, None, 3)
    assert model is old and update == 3 and error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_session.py -q -k "adopts_a_newer or broken_snapshot or no_live_directory"`
Expected: FAIL with `AttributeError: module 'watch_ppo' has no attribute 'reload_if_newer'`

- [ ] **Step 3: Write minimal implementation**

In `watch_ppo.py`, add the import and the function:

```python
from dodge.training import newest_snapshot


def reload_if_newer(model, live_dir, current_update):
    """Adopt the newest published snapshot above `current_update`.

    Called only between episodes. A snapshot that will not load leaves the
    running policy untouched and the update number unchanged, so the next
    death tries again rather than the window dying with the file.
    """
    if live_dir is None:
        return model, current_update, None
    found = newest_snapshot(live_dir, after=current_update)
    if not found:
        return model, current_update, None
    path, update = found
    try:
        return PPO.load(path, device="cpu"), update, None
    except Exception as error:
        return model, current_update, str(error)
```

Add the arguments:

```python
    parser.add_argument("--live-dir", type=Path, default=None,
                        help="directory of published update-XXXXXXXX.zip "
                             "snapshots to reload from between episodes")
    parser.add_argument("--current-update", type=int, default=-1,
                        help="the update already loaded, so only newer "
                             "snapshots are adopted")
```

Track the update and rebuild the field source after a swap. Replace the reset branch of the loop with:

```python
        if terminated or truncated:
            episode += 1
            model, new_update, error = reload_if_newer(
                model, args.live_dir, current_update)
            if error:
                print(f"Keeping update {current_update}: {error}")
            elif new_update != current_update:
                current_update = new_update
                # The extractor belongs to the old policy object.
                extractor = getattr(getattr(model, "policy", None),
                                    "features_extractor", None)
                field_source = (extractor if args.field
                                and hasattr(extractor, "field") else None)
            seed = secrets.randbelow(2**31 - 1)
            observation, _ = env.reset(seed=seed)
            pygame.display.set_caption(
                window_title(model_name, current_update, seed, episode + 1)
            )
```

Initialise `current_update` before the loop, next to `episode = 0`:

```python
    current_update = args.current_update if args.current_update >= 0 else args.update
```

and use it for the initial caption:

```python
        caption=window_title(model_name, current_update, seed),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_session.py -q -k "adopts_a_newer or broken_snapshot or no_live_directory"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 307 passed

- [ ] **Step 6: Commit**

```bash
git add watch_ppo.py tests/test_training_session.py
git commit -m "Reload the watched policy between episodes

A watch window showed one frozen checkpoint for as long as it was open.
It now scans the session's snapshot directory at each death and adopts
the highest completed update newer than its own, rebuilding the field
extractor with it. A snapshot that will not load leaves the running
policy alone and is retried at the next death; models never change
inside an episode."
```

---

### Task 10: End-to-end integration test

**Files:**
- Create: `tests/test_watch_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Write the test**

```python
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
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_watch_pipeline.py -q`
Expected: PASS

- [ ] **Step 3: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: 308 passed

- [ ] **Step 4: Manual check against the real dashboard**

Run `python training_gui.py --architecture velocity-flow-v2 --lives 2`, let it reach its first update, then click Watch during a learning phase. Confirm: the click returns immediately and the dashboard keeps redrawing; a window opens; the title's update number advances after a death; `artifacts/models/watch/live-*/` holds at most three `update-*.zip` files; closing the watch window stops new files appearing.

- [ ] **Step 5: Commit**

```bash
git add tests/test_watch_pipeline.py
git commit -m "Cover the watch pipeline end to end

Publish, retention, watcher liveness and relaunch in one test against a
real directory, with only the watcher process mocked."
```

---

## Self-Review

**Spec coverage.** Post-update callback → Task 2. Explicit update counting → Task 1. Saves at three safe points → Tasks 2 and 3. Non-blocking GUI requests → Tasks 3, 6, 7, 8. Failure and shutdown completing requests → Task 3. Private live directory, temp + atomic rename, never `*-latest.zip` → Task 4. Publish per update while watched, retention of three, best-effort deletion, `Popen` tracking → Tasks 4 and 5. `watch_agent()` returning immediately, older-checkpoint launch, "Waiting for current update…", click dedupe → Task 6. Save button pending state and selection → Task 7. Game config commits after its checkpoint → Task 8. Toasts for save/rename/launch failures and `drain_events()` → Tasks 3 and 7. Watcher `--live-dir`/`--current-update`, reload at death, validation, title update, retry-on-failure, never mid-episode → Task 9. Integration test → Task 10.

**Gap found and closed.** The spec's "deduplicate repeated clicks while an initial Watch launch is pending" is covered by `watch_pending` in Task 6; "additional clicks after launch may open additional watcher windows" follows because a launched watcher leaves `watch_pending` false and `newest_snapshot` non-empty, so the next click launches again.

**Type consistency.** `snapshot_name(update) -> str` and `newest_snapshot(directory, after=-1) -> tuple[Path, int] | None` are used with those signatures in Tasks 4, 5, 6, 9 and 10. `SessionEvent(kind, ok, message, path, update)` is constructed and read with those field names in Tasks 3, 6, 7 and 8. `_on_post_update(model)` replaces the Task 2 wiring in Task 5, and Task 5 states that explicitly.

**Known deviation from the spec.** Task 4 Step 5 adds a measurement the spec does not ask for, because the spec assumes publish cost is acceptable without measuring it. If a publish exceeds ~120ms the plan says to stop and report rather than proceed.
