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
