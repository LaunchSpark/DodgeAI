import pytest

from dodge.metrics import (causes_by_window, evaluation, pattern_hazard,
                           pattern_lethality, quarter_means, summary,
                           survival_curve, wilson)


def game(frames, cause="normal", pattern=-1, pattern_frames=0, encounters=None,
         died=True, fyou=0):
    return {"frames": frames, "died": died, "cause": cause, "pattern": pattern,
            "pattern_frames": pattern_frames, "encounters": encounters or {},
            "fyou_triggers": fyou}


def test_quarter_means_are_means_not_percentiles():
    q1, q4 = quarter_means([1, 2, 3, 4, 5, 6, 7, 100])
    assert q1 == pytest.approx(1.5)
    assert q4 == pytest.approx(53.5)


def test_summary_reports_the_papers_columns():
    result = summary([game(f) for f in (500, 1000, 2000, 4000)])
    assert result["worst"] == 500
    assert result["best"] == 4000
    assert result["mean"] == pytest.approx(1875)
    assert result["past_1000"] == pytest.approx(0.75)
    assert summary([]) is None


def test_survival_curve_starts_at_everyone_alive():
    curve = survival_curve([game(f) for f in (500, 1500, 2500)])
    assert curve[0] == {"frames": 0, "alive": 1.0}
    assert curve[1]["alive"] == pytest.approx(2 / 3)
    assert curve[-1]["alive"] == 0.0


def test_causes_are_grouped_by_the_window_the_game_ended_in():
    games = [game(500, "normal"), game(600, "kamikaze_blast"),
             game(1500, "static_rect"), game(9000, "moving_rect"),
             game(700, died=False)]
    rows = {row["window"]: row for row in causes_by_window(games)}
    assert rows["0-1k"]["games"] == 2
    assert rows["0-1k"]["causes"]["kamikaze"] == pytest.approx(0.5), "blasts are kamikaze"
    assert rows["1k-2k"]["causes"]["static_rect"] == 1.0
    assert rows["5k+"]["share_of_games"] == pytest.approx(0.25)


def test_pattern_hazard_compares_rates_per_frame_lived():
    games = [game(1000, pattern=3, pattern_frames=100),     # died during a pattern
             game(1000, pattern=-1, pattern_frames=100)]    # died outside one
    overall = pattern_hazard(games)["overall"]
    assert overall["active"] == pytest.approx(1 / 200 * 1e4)
    assert overall["inactive"] == pytest.approx(1 / 1800 * 1e4)
    assert overall["active_frame_share"] == pytest.approx(0.1)


def test_pattern_lethality_counts_deaths_per_encounter():
    games = [game(800, pattern=28, encounters={28: 1}) for _ in range(4)]
    games += [game(5000, died=False, encounters={28: 2, 3: 12}) for _ in range(6)]
    rows = {row["pattern"]: row for row in pattern_lethality(games, min_encounters=10)}
    assert rows[28]["encounters"] == 16
    assert rows[28]["lethality"] == pytest.approx(4 / 16)
    assert rows[28]["low"] < 0.25 < rows[28]["high"]
    assert rows[3]["lethality"] == 0.0
    low, high = wilson(0, 0)
    assert (low, high) == (0.0, 0.0)


def test_evaluation_bundles_every_figure():
    result = evaluation([game(1200, fyou=1), game(300)])
    assert set(result) == {"summary", "survival", "causes", "hazard", "patterns",
                           "anti_camping"}
    assert result["anti_camping"]["games_triggered"] == 0.5
