"""The web dashboard's server: live state, controls and evaluation.

One process holds the training session, so the dashboard reads the trainer's
snapshot directly. State goes to the browser over a websocket once a second,
charts included as Vega-Lite specs; controls arrive as small POSTs.
"""

import asyncio
import contextlib
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import charts, game_settings, metrics, rewards, schedule
from .telemetry import EVENT_KEYS, PROGRESS_ROWS

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"
PUSH_SECONDS = 1.0
# Progress rows recorded once per PPO update rather than once per episode.
PER_UPDATE_ROWS = ("entropy", "explained_variance")
# The paper's protocol: fixed seeds from here, one game each, frame cap 18,000.
EVALUATION_FIRST_SEED = 1_000_000
EVALUATION_FRAME_CAP = 18_000

GAME_SLIDERS = (
    {"name": "difficulty", "label": "Difficulty", "min": 1, "max": 3, "step": 1},
    {"name": "spawn_rate_percent", "label": "Spawn rate", "min": 50, "max": 150, "step": 5},
    {"name": "action_repeat", "label": "Decision interval", "min": 1, "max": 5, "step": 1},
    {"name": "lives", "label": "Training lives", "min": 0, "max": 5, "step": 1},
)
GAME_TOGGLES = (
    {"name": "patterns", "label": "Patterns"},
    {"name": "powerups", "label": "Powerups"},
    {"name": "normal_enemies", "label": "Normal enemies"},
    {"name": "kamikaze_enemies", "label": "Kamikaze enemies"},
)


def _series_summary(series):
    if not series:
        return None
    return {"latest": series.latest, "baseline": series.baseline,
            "trend": series.smoothed, "delta": series.delta}


def _system():
    import psutil
    status = {"cpu": psutil.cpu_percent(interval=None),
              "memory": psutil.virtual_memory().percent, "gpu": None}
    with contextlib.suppress(Exception):
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            status["gpu"] = {"name": torch.cuda.get_device_name(0),
                             "memory": 100.0 * (total - free) / total}
    return status


def _game_settings(settings):
    return dict(step_frames=settings["action_repeat"], difficulty=settings["difficulty"],
                patterns=settings["patterns"], powerups=settings["powerups"],
                normal_enemies=settings["normal_enemies"],
                kamikaze_enemies=settings["kamikaze_enemies"],
                spawn_rate_percent=settings["spawn_rate_percent"])


def evaluate_games(policy, games, settings, progress):
    from .env import play_games
    seeds = range(EVALUATION_FIRST_SEED, EVALUATION_FIRST_SEED + games)
    return play_games(policy, seeds, frame_cap=EVALUATION_FRAME_CAP, progress=progress,
                      **_game_settings(settings))


def record_death_paths(policy, games, settings, progress):
    from .death_paths import record
    seeds = range(EVALUATION_FIRST_SEED, EVALUATION_FIRST_SEED + games)
    return record(policy, seeds, frame_cap=EVALUATION_FRAME_CAP, progress=progress,
                  **_game_settings(settings))


class PolicyJob:
    """A frozen checkpoint playing the paper's fixed seeds, in a thread.

    `work(policy, games, settings, progress)` does the playing; its return
    value is kept as the job's result.
    """

    def __init__(self, name, work):
        self.name = name
        self.work = work
        self.lock = threading.Lock()
        self.status = "idle"
        self.message = ""
        self.finished = 0
        self.total = 0
        self.result = None
        self.label = ""
        self.settings = {}

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "message": self.message,
                    "finished": self.finished, "total": self.total,
                    "label": self.label, "settings": dict(self.settings)}

    def running(self):
        with self.lock:
            return self.status == "running"

    def start(self, path, games, deterministic, settings, label):
        with self.lock:
            if self.status == "running":
                return False
            self.status, self.finished, self.total = "running", 0, games
            self.message, self.label, self.settings = "", label, dict(settings)
        thread = threading.Thread(target=self._run, name=f"dodge-{self.name}", daemon=True,
                                  args=(path, games, deterministic, settings))
        thread.start()
        return True

    def _progress(self, finished, total):
        with self.lock:
            self.finished = finished

    def _run(self, path, games, deterministic, settings):
        try:
            import torch
            from stable_baselines3 import PPO

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = PPO.load(path, device=device)
            Path(path).unlink(missing_ok=True)

            def policy(observations, _):
                actions, _ = model.predict(observations, deterministic=deterministic)
                return actions

            result = self.work(policy, games, settings, self._progress)
            with self.lock:
                self.result, self.status = result, "done"
        except Exception as error:  # surfaced in the dashboard, not raised
            with self.lock:
                self.status, self.message = "error", str(error)


class Dashboard:
    def __init__(self, session):
        self.session = session
        self.jobs = {"evaluate": PolicyJob("evaluate", evaluate_games),
                     "death-paths": PolicyJob("death-paths", record_death_paths)}
        self.evaluation = self.jobs["evaluate"]
        self.death_paths = self.jobs["death-paths"]
        self.toasts = deque(maxlen=20)
        self._pending = {}
        self._toast_id = 0

    # -- state ------------------------------------------------------------
    def _toast(self, ok, message):
        self._toast_id += 1
        self.toasts.append({"id": self._toast_id, "ok": ok, "message": message,
                            "time": time.time()})

    def drain_events(self):
        """Turn trainer results into toasts, and start a requested evaluation."""
        for event in self.session.drain_events():
            if event.kind in self.jobs:
                pending = self._pending.pop(event.kind, None)
                if event.ok and pending is not None:
                    # Each job plays its own frozen copy, so a later save
                    # cannot change the weights under it.
                    path = Path(event.path)
                    frozen = path.with_name(f"{event.kind}-{int(time.time())}.zip")
                    path.replace(frozen)
                    self.jobs[event.kind].start(frozen, **pending)
                    self._toast(True, f"Playing {pending['games']} fixed seeds ({event.kind})")
                else:
                    self._toast(False, f"{event.kind} could not start: {event.message}")
                continue
            self._toast(event.ok, event.message)

    def state(self):
        with self.session.lock:
            data = self.session.data.copy()
            schedule_status = dict(self.session.schedule_status)
        config = self.session.game_config()
        recent = data.games.recent()
        training_eval = metrics.evaluation(recent) if recent else None
        evaluation = self.evaluation.snapshot()
        evaluation_games = self.evaluation.result if evaluation["status"] == "done" else []
        evaluated = metrics.evaluation(evaluation_games) if evaluation_games else None

        curves = {}
        if training_eval:
            curves[f"Last {len(recent)} training games"] = training_eval["survival"]
        if evaluated:
            curves[f"Evaluation, {len(evaluation_games)} fixed seeds"] = evaluated["survival"]
        figures = evaluated or training_eval
        # Entropy and explained variance are logged once per update; the rest
        # once per episode.
        chart_specs = {
            key: charts.training_series(getattr(data, key), label, unit,
                                        per=("PPO updates" if key in PER_UPDATE_ROWS
                                             else "Episodes"))
            for key, label, unit, _ in PROGRESS_ROWS if getattr(data, key)
        }
        if data.games.quarters:
            chart_specs["quarters"] = charts.quarters_over_training(data.games.quarters)
        if curves:
            chart_specs["survival"] = charts.survival_curves(curves)
        if figures:
            chart_specs["causes"] = charts.causes_by_window(figures["causes"])
            chart_specs["hazard"] = charts.pattern_hazard(figures["hazard"])
            if figures["patterns"]:
                chart_specs["patterns"] = charts.pattern_lethality(figures["patterns"])

        return {
            "status": data.status,
            "model_name": data.model_name,
            "update": data.update,
            "seed": data.seed,
            "steps_per_second": data.steps_per_second,
            "best_score": data.best_score,
            "elapsed_seconds": data.elapsed_seconds,
            "progress_phase": data.progress_phase,
            "rollout_progress": data.rollout_progress,
            "learning_progress": data.learning_progress,
            "dead_units": data.dead_units,
            "progress": [{"key": key, "label": label, "unit": unit, "rising": rising,
                          **(_series_summary(getattr(data, key)) or {})}
                         for key, label, unit, rising in PROGRESS_ROWS],
            "events": {key: {"rollout": data.events[key].rollout,
                             "total": data.events[key].total,
                             "history": list(data.events[key].history)}
                       for key in EVENT_KEYS},
            "game_config": config,
            "game_controls": {"sliders": GAME_SLIDERS, "toggles": GAME_TOGGLES},
            "rewards": asdict(rewards.load()),
            "reward_controls": [{"name": spec.name, "label": spec.label,
                                 "min": spec.minimum, "max": spec.maximum,
                                 "step": spec.step, "hint": spec.hint}
                                for spec in rewards.SLIDERS],
            "training": asdict(schedule.load(self.session.schedule_path)),
            "training_applied": schedule.exists(self.session.schedule_path),
            "training_controls": [{"name": spec.name, "label": spec.label,
                                   "min": spec.minimum, "max": spec.maximum,
                                   "step": spec.step, "hint": spec.hint}
                                  for spec in schedule.SLIDERS],
            "training_status": schedule_status,
            "training_figures": training_eval,
            "evaluation": {**evaluation, "figures": evaluated},
            "death_paths": self.death_paths.snapshot(),
            "charts": chart_specs,
            "system": _system(),
            "toasts": list(self.toasts),
        }

    # -- controls ---------------------------------------------------------
    def models(self):
        return [{"name": entry.name, "path": str(entry.path)}
                for entry in self.session.available_models()]

    def load(self, path):
        entry = next((entry for entry in self.session.available_models()
                      if str(entry.path) == path), None)
        if entry is None:
            raise ValueError("No such model")
        self.session.load_model(entry)

    def save(self, name):
        safe = self.session._safe_name(name)
        self.session.request_save(self.session.model_dir / f"{safe}.zip", kind="save")

    def configure(self, values):
        # The panel posts back the whole game config, including settings it
        # has no control for (max_episode_frames); only its own are passed on.
        owned = {spec["name"] for spec in (*GAME_SLIDERS, *GAME_TOGGLES)}
        changed = self.session.configure_game(
            **{name: value for name, value in values.items() if name in owned})
        if changed:
            game_settings.save({**self.session.game_config()})
        return changed

    def set_rewards(self, values):
        config = rewards.RewardConfig(**{**asdict(rewards.load()), **values})
        rewards.save(config)

    def set_training(self, values):
        config = schedule.TrainingConfig(
            **{**asdict(schedule.load(self.session.schedule_path)), **values}).clamped()
        schedule.save(config, self.session.schedule_path)

    def run_job(self, kind, games, deterministic):
        """Freeze the live policy at the next safe point and play it in `kind`."""
        if self.jobs[kind].running() or kind in self._pending:
            raise ValueError(f"{kind} is already running")
        config = self.session.game_config()
        with self.session.lock:
            label = f"{self.session.data.model_name} @ update {self.session.data.update}"
        self._pending[kind] = {"games": games, "deterministic": deterministic,
                               "settings": config, "label": label}
        self.session.request_save(self.session.model_dir / "runtime" / f"{kind}.zip",
                                  kind=kind)

    def death_path_data(self):
        snapshot = self.death_paths.snapshot()
        if snapshot["status"] != "done":
            return {**snapshot, "data": None}
        return {**snapshot, "data": self.death_paths.result}


def create_app(session):
    from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    dashboard = Dashboard(session)
    app = FastAPI(title="DodgeAI dashboard")
    app.state.dashboard = dashboard

    def act(function, *args):
        try:
            result = function(*args)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        return {"ok": True, "result": result}

    @app.get("/api/state")
    def get_state():
        dashboard.drain_events()
        return dashboard.state()

    @app.get("/api/models")
    def get_models():
        return dashboard.models()

    @app.post("/api/pause")
    def pause():
        return act(session.toggle_pause)

    @app.post("/api/reset")
    def reset():
        return act(session.reset)

    @app.post("/api/watch")
    def watch():
        return act(session.watch_agent)

    @app.post("/api/save")
    def save(body: dict = Body(...)):
        return act(dashboard.save, str(body.get("name", "")))

    @app.post("/api/load")
    def load(body: dict = Body(...)):
        return act(dashboard.load, str(body.get("path", "")))

    @app.post("/api/game-config")
    def configure(body: dict = Body(...)):
        return act(dashboard.configure, body)

    @app.post("/api/rewards")
    def set_rewards(body: dict = Body(...)):
        return act(dashboard.set_rewards, body)

    @app.post("/api/training")
    def set_training(body: dict = Body(...)):
        return act(dashboard.set_training, body)

    def job(kind, body, most):
        games = int(body.get("games", 256))
        if not 16 <= games <= most:
            raise HTTPException(status_code=400, detail=f"games must be 16..{most}")
        return act(dashboard.run_job, kind, games, body.get("mode", "argmax") == "argmax")

    @app.post("/api/evaluate")
    def evaluate(body: dict = Body(...)):
        return job("evaluate", body, 2048)

    @app.post("/api/death-paths")
    def death_paths(body: dict = Body(...)):
        # Recording reads every lane's full state each frame, so it is slower.
        return job("death-paths", body, 1024)

    @app.get("/api/death-paths")
    def get_death_paths():
        return _jsonable(dashboard.death_path_data())

    @app.websocket("/api/live")
    async def live(socket: WebSocket):
        await socket.accept()
        try:
            while True:
                dashboard.drain_events()
                state = await asyncio.to_thread(dashboard.state)
                await socket.send_json(_jsonable(state))
                await asyncio.sleep(PUSH_SECONDS)
        except WebSocketDisconnect:
            return

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{path:path}")
        def index(path: str):
            target = STATIC_DIR / path
            if path and target.is_file():
                return FileResponse(target)
            return FileResponse(STATIC_DIR / "index.html")

    return app


def _jsonable(value):
    """numpy scalars and tuples to plain JSON types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
