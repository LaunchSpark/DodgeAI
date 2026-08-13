from collections import deque

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
        self.env_kwargs = {
            "difficulty": 1,
            "patterns": False,
            "powerups": True,
            "action_repeat": 4,
        }
        self.configured = []

    def tick(self, dt):
        pass

    def close(self):
        pass

    def available_models(self):
        return []

    def game_config(self):
        return dict(self.env_kwargs)

    def configure_game(self, **config):
        self.env_kwargs = dict(config)
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
    for spec, track in dashboard.game_regions["tracks"]:
        dashboard._click((track.right - 1, track.centery))
    dashboard._click(dashboard.game_regions["apply"].center)

    assert session.configured == [{
        "difficulty": 3,
        "patterns": True,
        "powerups": False,
        "action_repeat": 8,
    }]
    assert not dashboard.game_menu_open
    pygame.quit()


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


def test_model_name_field_is_focused_and_replaced_by_typing(tmp_path):
    from training_gui import TrainingDashboard

    dashboard = TrainingDashboard(_StubSession(tmp_path / "rewards.json"), hidden=True)
    dashboard._open_models()
    dashboard.draw()

    assert dashboard.model_name_focused
    assert dashboard.model_name_selected
    dashboard._insert_model_text("patient-dodger")
    assert dashboard.model_name == "patient-dodger"
    assert dashboard.model_name_cursor == len("patient-dodger")
    pygame.quit()


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
