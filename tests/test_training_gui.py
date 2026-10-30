from collections import deque
from pathlib import Path

import pytest

pygame = pytest.importorskip("pygame")

from dodge.telemetry import DashboardSnapshot, sample_snapshot
from training_gui import (
    BLUE, MIN_WINDOW_SIZE, WINDOW_SIZE, DashboardRenderer, GREEN, MUTED, RED,
    format_duration,
)


def test_format_duration():
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661.9) == "01:01:01"


def test_sample_snapshot_is_deterministic():
    left = sample_snapshot(seed=9)
    right = sample_snapshot(seed=9)
    assert left.reward.as_list() == right.reward.as_list()
    assert left.events["deaths"].total == right.events["deaths"].total


def test_sample_snapshot_populates_every_charted_series():
    data = sample_snapshot()
    for series in (data.reward, data.entropy, data.episode_length, data.enemies_killed):
        assert len(series) >= 2, "charts need at least two points to draw a line"
        assert series.count > len(series) or series.interval == 1


def test_sample_snapshot_shows_improvement():
    """The sample data must trend upward, or it cannot exercise the deltas."""
    data = sample_snapshot()
    assert data.reward.delta > 0
    assert data.episode_length.delta > 0
    assert data.entropy.delta < 0, "entropy is expected to fall as the policy converges"


@pytest.mark.parametrize("size", [WINDOW_SIZE, MIN_WINDOW_SIZE, (1100, 680)])
def test_dashboard_renders_at_supported_sizes(size):
    pygame.font.init()
    renderer = DashboardRenderer()
    surface = pygame.Surface(size)
    renderer.render(surface, sample_snapshot())
    assert renderer.buttons
    assert surface.get_at((0, 0))[:3] != (0, 0, 0)


def test_dashboard_renders_before_any_data_arrives():
    """A run's first frames have empty series; drawing must not raise."""
    pygame.font.init()
    surface = pygame.Surface(WINDOW_SIZE)
    DashboardRenderer().render(surface, DashboardSnapshot())


def test_empty_snapshot_has_no_default_model():
    assert DashboardSnapshot().model_name == ""


def test_training_bar_switches_to_blue_learning_progress():
    data = DashboardSnapshot(
        progress_phase="learning",
        rollout_progress=1.0,
        learning_progress=0.375,
    )
    assert DashboardRenderer._training_progress(data) == (
        "Learning", 0.375, BLUE
    )


def test_delta_style_marks_improvement_green_whichever_way_it_points():
    style = DashboardRenderer._delta_style
    assert style(5.0, rising=True)[1] == GREEN
    assert style(-5.0, rising=True)[1] == RED
    # Falling entropy is convergence, so it reads as an improvement.
    assert style(-0.6, rising=False)[1] == GREEN
    assert style(0.6, rising=False)[1] == RED
    assert style(0.0, rising=True)[1] == MUTED


def test_delta_arrow_follows_the_sign_not_the_verdict():
    style = DashboardRenderer._delta_style
    assert style(-0.6, rising=False)[0] == "▼"
    assert style(0.6, rising=False)[0] == "▲"


def test_span_label_reports_elapsed_time():
    assert DashboardRenderer._span_label(0) == "run start → now"
    assert "01:01:01" in DashboardRenderer._span_label(3661)


class _StubSession:
    """Enough session for the dashboard to draw and take clicks."""

    def __init__(self, reward_config_path):
        self.data = sample_snapshot()
        self.reward_config_path = reward_config_path
        # Mirrors PPOTrainingSession.model_dir: the checkpoint directory sits
        # beside the reward config in the same tmp_path fixture.
        self.model_dir = Path(reward_config_path).parent
        # Mirrors PPOTrainingSession.game_config_path, so a test never writes
        # the developer's own game-config.json.
        self.game_config_path = self.model_dir / "game-config.json"
        # Mirrors PPOTrainingSession.env_kwargs, including the settings the
        # Game Config panel does not own -- the panel must not hand those back.
        self.env_kwargs = {
            "difficulty": 1,
            "patterns": False,
            "powerups": True,
            "normal_enemies": True,
            "kamikaze_enemies": True,
            "spawn_rate_percent": 100,
            "action_repeat": 4,
            "lives": 0,
            "max_episode_frames": None,
        }
        self.configured = []
        self.pending_events = []
        self.watch_result = True

    def tick(self, dt):
        pass

    def close(self):
        pass

    def available_models(self):
        return []

    def drain_events(self):
        drained, self.pending_events = self.pending_events, []
        return drained

    def request_save(self, path, kind="save"):
        self.configured.append(("request_save", Path(path).name, kind))

    @staticmethod
    def _safe_name(name):
        # Delegate rather than reimplement, so this double can't drift from
        # what the real session considers a safe name.
        from dodge.training import PPOTrainingSession
        return PPOTrainingSession._safe_name(name)

    def watch_agent(self):
        self.configured.append(("watch_agent",))
        return self.watch_result

    def game_config(self):
        return dict(self.env_kwargs)

    def configure_game(self, *, difficulty, patterns, powerups,
                       normal_enemies=True, kamikaze_enemies=True,
                       spawn_rate_percent=100, action_repeat, lives=0):
        """Deliberately the real signature, not **config.

        A stub that swallows any keyword cannot catch the panel sending a key
        the session does not take, which is exactly how a crash on Apply got
        past this suite once already.
        """
        config = {"difficulty": difficulty, "patterns": patterns,
                  "powerups": powerups,
                  "normal_enemies": normal_enemies,
                  "kamikaze_enemies": kamikaze_enemies,
                  "spawn_rate_percent": spawn_rate_percent,
                  "action_repeat": action_repeat,
                  "lives": lives}
        self.env_kwargs = {**self.env_kwargs, **config}
        self.configured.append(dict(config))
        return True


def _dashboard(tmp_path):
    from training_gui import TrainingDashboard
    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_rewards()
    dashboard.draw()
    return dashboard


def test_reset_button_is_replaced_by_rewards():
    pygame.font.init()
    renderer = DashboardRenderer()
    renderer.render(pygame.Surface(WINDOW_SIZE), sample_snapshot())
    names = [button.name for button in renderer.buttons]
    assert "Rewards" in names
    assert "Reset" not in names


def test_game_config_button_is_directly_below_rewards():
    pygame.font.init()
    renderer = DashboardRenderer()
    renderer.render(pygame.Surface(WINDOW_SIZE), sample_snapshot())
    names = [button.name for button in renderer.buttons]
    assert names.index("Game Config") == names.index("Rewards") + 1


def test_game_config_applies_toggles_and_discrete_sliders(tmp_path):
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_game_config()
    dashboard.draw()

    dashboard._click(dashboard.game_regions["toggles"]["powerups"].center)
    dashboard._click(dashboard.game_regions["toggles"]["patterns"].center)
    dashboard._click(dashboard.game_regions["toggles"]["normal_enemies"].center)
    dashboard._click(dashboard.game_regions["toggles"]["kamikaze_enemies"].center)
    for spec, track in dashboard.game_regions["tracks"]:
        dashboard._click((track.right - 1, track.centery))
    dashboard._click(dashboard.game_regions["apply"].center)

    assert session.configured == [{
        "difficulty": 3,
        "patterns": True,
        "powerups": False,
        "normal_enemies": False,
        "kamikaze_enemies": False,
        "spawn_rate_percent": 150,
        "action_repeat": 5,
        "lives": 5,
    }]
    assert not dashboard.game_menu_open
    pygame.quit()


def test_game_config_spawn_rate_snaps_to_five_percent(tmp_path):
    from training_gui import TrainingDashboard

    dashboard = TrainingDashboard(
        _StubSession(tmp_path / "rewards.json"), hidden=True)
    dashboard._open_game_config()
    dashboard.draw()
    spec, track = next(
        (spec, track) for spec, track in dashboard.game_regions["tracks"]
        if spec.name == "spawn_rate_percent"
    )

    # 73% of the 50..150 range is 123; the five-point knob snaps to 125.
    dashboard._click((track.x + round(track.w * 0.73), track.centery))
    assert dashboard.game_config["spawn_rate_percent"] == 125
    assert DashboardRenderer._game_value(spec, 125) == "125%"
    pygame.quit()


def test_expanded_game_config_fits_the_minimum_window():
    from training_gui import GAME_DEFAULTS

    pygame.font.init()
    renderer = DashboardRenderer()
    surface = pygame.Surface(MIN_WINDOW_SIZE)
    regions = renderer.draw_game_menu(surface, dict(GAME_DEFAULTS))

    assert regions["modal"].top >= 0
    assert regions["modal"].bottom <= surface.get_height()
    assert set(regions["toggles"]) == {
        "powerups", "patterns", "normal_enemies", "kamikaze_enemies"
    }
    assert regions["apply"].top > max(track.bottom for _, track in regions["tracks"])


def test_reward_panel_offers_a_slider_for_every_weight(tmp_path):
    from dodge.rewards import SLIDERS

    dashboard = _dashboard(tmp_path)
    tracked = [spec.name for spec, _ in dashboard.reward_regions["tracks"]]
    assert tracked == [spec.name for spec in SLIDERS]
    pygame.quit()


def test_dragging_a_slider_writes_the_config_file(tmp_path):
    from dodge.rewards import load

    dashboard = _dashboard(tmp_path)
    spec, track = next((spec, track)
                       for spec, track in dashboard.reward_regions["tracks"]
                       if spec.name == "death_penalty")

    # right - 1 because Rect.collidepoint treats the right edge as exclusive.
    dashboard._click((track.right - 1, track.centery))
    assert load(tmp_path / "rewards.json").death_penalty == spec.maximum

    # Still held, so motion keeps updating it.
    dashboard._set_reward_from_position(spec, track, (track.x, track.centery))
    assert load(tmp_path / "rewards.json").death_penalty == spec.minimum
    pygame.quit()


def test_clicking_a_slider_starts_a_drag_and_releasing_ends_it(tmp_path):
    dashboard = _dashboard(tmp_path)
    _, track = dashboard.reward_regions["tracks"][0]
    dashboard._click((track.centerx, track.centery))
    assert dashboard.dragging_slider is not None
    dashboard.dragging_slider = None  # what MOUSEBUTTONUP does
    assert dashboard.dragging_slider is None
    pygame.quit()


def test_restore_defaults_rewrites_the_file(tmp_path):
    from dodge.rewards import RewardConfig, load, save

    save(RewardConfig(death_penalty=4.0), tmp_path / "rewards.json")
    dashboard = _dashboard(tmp_path)
    assert dashboard.reward_config.death_penalty == 4.0

    dashboard._click(dashboard.reward_regions["defaults"].center)
    assert load(tmp_path / "rewards.json") == RewardConfig()
    pygame.quit()


def test_panel_shows_values_edited_by_hand(tmp_path):
    """Opening the panel re-reads, so a text edit is not silently overwritten."""
    from dodge.rewards import RewardConfig, save

    dashboard = _dashboard(tmp_path)
    save(RewardConfig(score_weight=0.35), tmp_path / "rewards.json")
    dashboard._open_rewards()
    assert dashboard.reward_config.score_weight == 0.35
    pygame.quit()


def test_closing_the_panel(tmp_path):
    dashboard = _dashboard(tmp_path)
    dashboard._click(dashboard.reward_regions["close"].center)
    assert not dashboard.reward_menu_open
    pygame.quit()


def test_model_menu_starts_without_a_default_name_or_selection(tmp_path):
    from training_gui import TrainingDashboard

    dashboard = TrainingDashboard(_StubSession(tmp_path / "rewards.json"), hidden=True)
    dashboard._open_models()
    dashboard.draw()

    assert dashboard.model_name_focused
    assert dashboard.model_name == ""
    assert not dashboard.model_name_selected
    assert dashboard.selected_model is None
    dashboard._insert_model_text("patient-dodger")
    assert dashboard.model_name == "patient-dodger"
    assert dashboard.model_name_cursor == len("patient-dodger")
    pygame.quit()


def test_model_menu_discovers_checkpoints_added_while_it_is_open(tmp_path):
    from dodge.telemetry import ModelEntry
    from training_gui import TrainingDashboard

    class DiscoveringSession(_StubSession):
        def __init__(self, reward_config_path):
            super().__init__(reward_config_path)
            self.entries = []

        def available_models(self):
            return list(self.entries)

    session = DiscoveringSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_models()
    assert dashboard.model_entries == []

    path = tmp_path / "vector-flow-v2.zip"
    session.entries.append(ModelEntry("vector-flow-v2", path, "PPO"))
    dashboard.model_discovery_at = 0.0
    dashboard.draw()

    assert [entry.name for entry in dashboard.model_entries] == ["vector-flow-v2"]
    assert dashboard.selected_model is None, "discovery must not choose a default"
    pygame.quit()


def test_idle_dashboard_opens_the_model_picker(tmp_path):
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    session.data.status = "idle"
    session.data.model_name = ""

    dashboard = TrainingDashboard(session, hidden=True)

    assert dashboard.model_menu_open
    assert dashboard.selected_model is None
    pygame.quit()


def test_gui_command_line_has_no_default_architecture(tmp_path, monkeypatch):
    import dodge.training
    import training_gui

    captured = {}

    class DashboardStub:
        def __init__(self, session):
            captured["session"] = session

        def run(self):
            captured["ran"] = True

    def session_stub(**kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(dodge.training, "PPOTrainingSession", session_stub)
    monkeypatch.setattr(training_gui, "TrainingDashboard", DashboardStub)
    models = tmp_path / "my-models"

    assert training_gui.main([
        "--models-dir", str(models),
        "--no-normal-enemies", "--no-kamikaze-enemies",
        "--spawn-rate-percent", "125",
    ]) == 0
    assert captured["kwargs"]["architecture"] is None
    assert captured["kwargs"]["model_dir"] == models
    assert captured["kwargs"]["normal_enemies"] is False
    assert captured["kwargs"]["kamikaze_enemies"] is False
    assert captured["kwargs"]["spawn_rate_percent"] == 125
    assert captured["ran"]


def test_model_name_field_supports_caret_editing_and_click_focus(tmp_path):
    from training_gui import TrainingDashboard

    dashboard = TrainingDashboard(_StubSession(tmp_path / "rewards.json"), hidden=True)
    dashboard._open_models()
    dashboard.draw()
    dashboard.model_name = "bot-one"
    dashboard.model_name_cursor = 3
    dashboard.model_name_selected = False
    dashboard._insert_model_text("-new")
    assert dashboard.model_name == "bot-new-one"

    dashboard._blur_model_name()
    dashboard._click(dashboard.model_regions["name"].center)
    assert dashboard.model_name_focused
    assert dashboard.model_name_cursor == len(dashboard.model_name)
    pygame.quit()


def test_sparkline_accepts_empty_history():
    pygame.font.init()
    surface = pygame.Surface((100, 50))
    DashboardRenderer._sparkline(surface, pygame.Rect(0, 0, 100, 50), deque(), (1, 2, 3))


def test_achievement_heading_fits_long_model_names():
    pygame.font.init()
    renderer = DashboardRenderer()
    title = "an-extremely-long-saved-bot-name's Achievements"
    fitted = renderer._fit_text(renderer.font_title, title, 180)
    assert fitted.endswith("...")
    assert renderer.font_title.size(fitted)[0] <= 180


def test_rendering_happens_outside_the_telemetry_lock(tmp_path):
    """The PPO callback takes this lock on every step.

    Holding it for the length of a frame stalled the trainer, and behind it
    every environment subprocess, while charts were drawn.
    """
    import threading
    from training_gui import DashboardRenderer, TrainingDashboard

    class LockedSession(_StubSession):
        def __init__(self, path):
            super().__init__(path)
            self.lock = threading.Lock()

    session = LockedSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)

    held = []
    original = DashboardRenderer.render

    def spy(self, surface, data, toast=None):
        held.append(session.lock.locked())
        return original(self, surface, data, toast)

    dashboard.renderer.render = spy.__get__(dashboard.renderer)
    dashboard.draw()

    assert held == [False], "render ran while the trainer's lock was held"
    pygame.quit()


def test_dashboard_renders_from_a_copy_not_the_live_snapshot(tmp_path):
    """A copy is what makes releasing the lock safe."""
    from training_gui import DashboardRenderer, TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)

    seen = []
    original = DashboardRenderer.render

    def spy(self, surface, data, toast=None):
        seen.append(data)
        return original(self, surface, data, toast)

    dashboard.renderer.render = spy.__get__(dashboard.renderer)
    dashboard.draw()

    assert seen[0] is not session.data
    assert seen[0].reward.as_list() == session.data.reward.as_list()
    pygame.quit()


def test_system_readings_are_polled_about_once_a_second():
    """psutil.virtual_memory() measured 5.8ms -- far too slow for a frame."""
    from training_gui import DashboardRenderer

    pygame.font.init()   # an earlier test's pygame.quit() tears this down
    renderer = DashboardRenderer()
    first = renderer._system_reading_cached()
    assert renderer._system_reading_cached() is first, "cached between frames"

    renderer._system_read_at -= 2.0
    assert renderer._system_reading_cached() is not first, "refreshed after a second"


def test_the_game_menu_offers_training_lives():
    """Lives are staged with the other game mechanics, and read as lives.

    Every other slider in that panel is measured in frames, so the shared
    formatter would otherwise label three lives "3 frames".
    """
    from training_gui import GAME_DEFAULTS, GAME_SLIDERS

    assert GAME_DEFAULTS["lives"] == 0, "off unless asked for"
    spec = next((s for s in GAME_SLIDERS if s.name == "lives"), None)
    assert spec is not None, "the panel has no lives slider"
    assert DashboardRenderer._game_value(spec, 0) == "Off"
    assert DashboardRenderer._game_value(spec, 1) == "1 life"
    assert DashboardRenderer._game_value(spec, 3) == "3 lives"


def test_the_dashboard_command_line_accepts_training_lives(tmp_path):
    import os

    import training_gui

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    assert training_gui.main([
        "--lives", "2",
        "--screenshot", str(tmp_path / "lives.png"),
    ]) == 0


def test_sample_snapshot_bands_survival_time_by_life():
    """--screenshot draws from this, so the sample has to exercise the stack."""
    from dodge.telemetry import LIFE_BANDS

    data = sample_snapshot()
    bands = [band.values for band in data.life_lengths]
    assert len(bands) == LIFE_BANDS
    assert all(len(b) == len(data.episode_length.values) for b in bands), \
        "bands must compact in lockstep with the total, or the stack misaligns"
    for column, whole in zip(zip(*bands), data.episode_length.values):
        assert sum(column) == pytest.approx(whole), "the stack must equal the total"
    assert sum(1 for b in bands if any(v > 0 for v in b)) >= 2, \
        "a single-band sample would not show a stack at all"


def test_the_survival_chart_draws_its_life_bands():
    """Same totals, different split across lives, must draw differently.

    Drawn through _draw_chart rather than a whole dashboard frame: the System
    panel reports live CPU and memory, so comparing full renders would differ
    every time whatever the bands did.
    """
    from dodge.series import RunSeries
    from dodge.telemetry import LIFE_BANDS
    from training_gui import DashboardRenderer, YELLOW

    pygame.font.init()   # an earlier test's pygame.quit() tears this down
    totals = [4.0, 6.0, 5.0, 9.0, 8.0, 12.0]

    def series_of(values):
        series = RunSeries()
        for value in values:
            series.append(value)
        return series

    def bands_from(split):
        made = tuple(RunSeries() for _ in range(LIFE_BANDS))
        for total in totals:
            parts = split(total)
            for index, band in enumerate(made):
                band.append(parts[index] if index < len(parts) else 0.0)
        return made

    renderer = DashboardRenderer()
    rect = pygame.Rect(0, 0, 320, 220)

    def draw(bands):
        surface = pygame.Surface(rect.size)
        renderer._draw_chart(surface, rect, "Survival Time", series_of(totals),
                             YELLOW, " s", bands=bands)
        return pygame.image.tostring(surface, "RGB")

    one_life = bands_from(lambda total: [total])
    three_lives = bands_from(lambda total: [total / 2, total / 3, total / 6])

    assert draw(three_lives) != draw(one_life), "the split is not being drawn"
    assert draw(one_life) == draw(None),         "a single band must render exactly as an unbanded chart"


def test_the_game_panel_sends_only_the_settings_it_owns(tmp_path):
    """The session carries env settings this panel does not control.

    The frame cap is one. _apply_game_config splats its dict straight into
    configure_game, so anything extra picked up when the panel opened crashes
    Apply -- which is exactly what happened once the cap moved into env_kwargs.
    """
    from training_gui import GAME_DEFAULTS, TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    assert "max_episode_frames" in session.game_config(), "the setup under test"
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_game_config()
    dashboard.draw()

    dashboard._click(dashboard.game_regions["apply"].center)

    assert session.configured, "Apply never reached the session"
    assert set(session.configured[0]) == set(GAME_DEFAULTS)
    pygame.quit()


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


def test_a_successful_save_event_selects_the_checkpoint_it_wrote(tmp_path):
    """Drain must reach _discover_models and select by the event's path.

    Not just clear the pending flag: a stub that only cleared the flag would
    leave this green while the real selection logic silently broke.
    """
    from dodge.telemetry import ModelEntry
    from dodge.training import SessionEvent
    from training_gui import TrainingDashboard

    class DiscoveringSession(_StubSession):
        def __init__(self, reward_config_path):
            super().__init__(reward_config_path)
            self.entries = []

        def available_models(self):
            return list(self.entries)

    session = DiscoveringSession(tmp_path / "rewards.json")
    written = tmp_path / "fresh-bot.zip"
    session.entries.append(ModelEntry("fresh-bot", written, "PPO"))
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.save_pending = True
    session.pending_events = [SessionEvent("save", True, "Saved fresh-bot.zip", written)]

    dashboard._drain_session_events()

    assert not dashboard.save_pending
    assert dashboard.selected_model is not None
    assert dashboard.model_entries[dashboard.selected_model].path == written
    pygame.quit()


def test_drain_session_events_survives_a_session_with_none(tmp_path):
    """Nothing here may raise: an exception in the event loop closes the
    window and takes training with it. A None path or a stale checkpoint
    must be tolerated, not just a well-formed event."""
    from dodge.training import SessionEvent
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    session.pending_events = [
        SessionEvent("save", True, "Saved missing.zip", tmp_path / "missing.zip"),
        SessionEvent("watch", False, "Could not open window: boom"),
    ]

    dashboard._drain_session_events()

    assert dashboard.selected_model is None
    assert dashboard.toast == "Could not open window: boom"
    pygame.quit()


def test_drain_session_events_survives_available_models_raising(tmp_path):
    """A checkpoint can be deleted between _discover_models' rglob and the
    stat() available_models sorts on, raising FileNotFoundError straight into
    the event loop. An exception here closes the window and takes training
    with it, so the drain must swallow it -- and still clear save_pending and
    show the toast, not just avoid crashing."""
    from dodge.training import SessionEvent
    from training_gui import TrainingDashboard

    class FlakySession(_StubSession):
        def available_models(self):
            raise FileNotFoundError("checkpoint vanished mid-scan")

    session = FlakySession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.save_pending = True
    session.pending_events = [SessionEvent("save", True, "Saved thing.zip", tmp_path / "thing.zip")]

    dashboard._drain_session_events()

    assert not dashboard.save_pending, "the button has to come back even though the refresh failed"
    assert dashboard.toast == "Saved thing.zip"
    pygame.quit()


def test_drain_session_events_is_a_no_op_without_the_hook(tmp_path):
    """A session that predates drain_events must not crash the loop."""
    from training_gui import TrainingDashboard

    class NoEventsSession(_StubSession):
        drain_events = None

    session = NoEventsSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)

    dashboard._drain_session_events()
    pygame.quit()


def test_save_model_queues_a_request_and_disables_the_button(tmp_path):
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.model_name = "steady-bot"

    dashboard._save_model()

    assert session.configured == [
        ("request_save", "steady-bot.zip", "save"),
    ], "the GUI must reach request_save, not save_model, for this to be non-blocking"
    assert dashboard.save_pending
    assert dashboard.toast == "Saving…"

    dashboard._save_model()

    assert session.configured == [("request_save", "steady-bot.zip", "save")], \
        "a pending save must not queue a second request"
    pygame.quit()


def test_save_model_without_a_selected_name_does_not_queue(tmp_path):
    """model_name is empty until a checkpoint or new name is chosen; saving
    that unguarded would ask the session to write a file named '.zip'."""
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    assert dashboard.model_name == ""

    dashboard._save_model()

    assert session.configured == []
    assert not dashboard.save_pending
    assert dashboard.toast
    pygame.quit()


def test_save_model_sanitizes_a_name_with_path_separators(tmp_path):
    """_insert_model_text only filters non-printable characters, so a typed
    '/' or '\\' reaches _save_model unchanged. Unsanitized, it would turn a
    single filename into a subdirectory the session was never asked to make."""
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.model_name = "sub/dir/name"

    dashboard._save_model()

    assert session.configured == [("request_save", "sub-dir-name.zip", "save")]
    assert dashboard.save_pending
    pygame.quit()


def test_watch_click_only_toasts_the_waiting_case(tmp_path):
    """A launched watcher's toast now arrives as an event; the click handler
    reports nothing on success, only that a window is pending."""
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    session.watch_result = True
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard.draw()
    watch_button = next(b for b in dashboard.renderer.buttons if b.name == "Watch Agent")

    dashboard._click(watch_button.rect.center)

    assert session.configured == [("watch_agent",)]
    assert dashboard.toast is None

    session.watch_result = False
    dashboard._click(watch_button.rect.center)

    assert dashboard.toast == "Waiting for current update…"
    pygame.quit()


def test_applying_the_game_config_persists_it_for_the_next_session(tmp_path):
    """The panel is per-user tuning, not a per-launch decision.

    Before this, the settings lived only in the session's env kwargs, so the
    next process started from the command line's defaults and quietly threw
    away whatever curriculum had been dialled in.
    """
    from dodge import game_settings
    from training_gui import TrainingDashboard

    session = _StubSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_game_config()
    dashboard.game_config["difficulty"] = 3
    dashboard.game_config["lives"] = 2
    dashboard.game_config["powerups"] = False
    dashboard._apply_game_config()

    saved = game_settings.load(session.game_config_path)
    assert saved["difficulty"] == 3
    assert saved["lives"] == 2
    assert saved["powerups"] is False


def test_a_rejected_game_config_is_not_persisted(tmp_path):
    """The file records what training actually runs, not what was typed."""
    from training_gui import TrainingDashboard

    class RejectingSession(_StubSession):
        def configure_game(self, **kwargs):
            raise ValueError("Difficulty must be easy, normal, or hard")

    session = RejectingSession(tmp_path / "rewards.json")
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_game_config()
    dashboard.game_config["difficulty"] = 9
    dashboard._apply_game_config()

    assert not session.game_config_path.exists()
    assert dashboard.game_menu_open, "the panel stays up to show the error"


def test_the_panel_falls_back_to_the_file_not_the_shipped_defaults(tmp_path):
    """A live session is the truth about what is running, and normally agrees
    with the file because a launch is seeded from it. A session that cannot
    report its game settings at all leaves the file as the best answer."""
    from dodge import game_settings
    from training_gui import TrainingDashboard

    class SilentSession(_StubSession):
        def __init__(self, path):
            super().__init__(path)
            self.env_kwargs = {}
            self.game_config = None

    session = SilentSession(tmp_path / "rewards.json")
    game_settings.save({**game_settings.DEFAULTS, "difficulty": 2},
                       session.game_config_path)
    dashboard = TrainingDashboard(session, hidden=True)
    dashboard._open_game_config()

    assert dashboard.game_config["difficulty"] == 2


def test_the_dashboard_launches_with_the_saved_game_config(tmp_path, monkeypatch):
    from dodge import game_settings
    import dodge.training
    import training_gui

    captured = {}

    class DashboardStub:
        def __init__(self, session):
            pass

        def run(self):
            captured["ran"] = True

    def session_stub(**kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dodge.training, "PPOTrainingSession", session_stub)
    monkeypatch.setattr(training_gui, "TrainingDashboard", DashboardStub)
    game_settings.save({**game_settings.DEFAULTS, "difficulty": 3, "lives": 2,
                        "patterns": True, "spawn_rate_percent": 125,
                        "action_repeat": 2},
                       game_settings.CONFIG_PATH)

    assert training_gui.main([]) == 0
    assert captured["kwargs"]["difficulty"] == 3
    assert captured["kwargs"]["lives"] == 2
    assert captured["kwargs"]["patterns"] is True
    assert captured["kwargs"]["spawn_rate_percent"] == 125
    assert captured["kwargs"]["action_repeat"] == 2


def test_an_explicit_flag_still_beats_the_saved_game_config(tmp_path, monkeypatch):
    """The file is a remembered default, not a lock on the command line."""
    from dodge import game_settings
    import dodge.training
    import training_gui

    captured = {}

    class DashboardStub:
        def __init__(self, session):
            pass

        def run(self):
            captured["ran"] = True

    def session_stub(**kwargs):
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dodge.training, "PPOTrainingSession", session_stub)
    monkeypatch.setattr(training_gui, "TrainingDashboard", DashboardStub)
    game_settings.save({**game_settings.DEFAULTS, "difficulty": 3,
                        "patterns": True},
                       game_settings.CONFIG_PATH)

    assert training_gui.main(["--difficulty", "normal", "--no-patterns"]) == 0
    assert captured["kwargs"]["difficulty"] == 2
    assert captured["kwargs"]["patterns"] is False


def test_the_panels_defaults_are_the_persisted_modules_defaults():
    """One definition, so a shipped value cannot drift between the two."""
    from dodge import game_settings
    from training_gui import GAME_DEFAULTS

    assert GAME_DEFAULTS == game_settings.DEFAULTS
