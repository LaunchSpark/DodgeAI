import json

import pytest

from dodge.game_settings import BOUNDS, DEFAULTS, load, save


def test_defaults_when_the_file_is_absent(tmp_path):
    assert load(tmp_path / "missing.json") == DEFAULTS


def test_round_trips_through_the_file(tmp_path):
    path = tmp_path / "game-config.json"
    save({**DEFAULTS, "difficulty": 3, "lives": 2, "patterns": True}, path)
    restored = load(path)
    assert restored["difficulty"] == 3
    assert restored["lives"] == 2
    assert restored["patterns"] is True


def test_saved_file_is_readable_and_editable_as_text(tmp_path):
    path = tmp_path / "game-config.json"
    save(DEFAULTS, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) == set(DEFAULTS)


def test_settings_the_panel_does_not_own_are_not_written(tmp_path):
    """save() is handed the session's env kwargs, which carry more than this."""
    path = tmp_path / "game-config.json"
    save({**DEFAULTS, "max_episode_frames": 900}, path)
    assert "max_episode_frames" not in json.loads(path.read_text(encoding="utf-8"))


def test_a_hand_edited_file_only_needs_the_keys_it_changes(tmp_path):
    path = tmp_path / "game-config.json"
    path.write_text('{"difficulty": 2}', encoding="utf-8")
    config = load(path)
    assert config["difficulty"] == 2
    assert config["action_repeat"] == DEFAULTS["action_repeat"]


def test_malformed_file_falls_back_instead_of_killing_a_launch(tmp_path):
    """A stray comma in a hand edit must not stop the dashboard from starting."""
    path = tmp_path / "game-config.json"
    path.write_text('{"difficulty": 2,}', encoding="utf-8")
    assert load(path) == DEFAULTS


def test_unknown_and_ill_typed_keys_are_ignored(tmp_path):
    path = tmp_path / "game-config.json"
    path.write_text('{"difficulty": 3, "nonsense": 1, "action_repeat": "x"}',
                    encoding="utf-8")
    config = load(path)
    assert config["difficulty"] == 3
    assert config["action_repeat"] == DEFAULTS["action_repeat"]


def test_values_are_clamped_to_the_supported_ranges(tmp_path):
    path = tmp_path / "game-config.json"
    path.write_text('{"difficulty": 9, "lives": -3, "spawn_rate_percent": 900}',
                    encoding="utf-8")
    config = load(path)
    assert config["difficulty"] == 3
    assert config["lives"] == 0
    assert config["spawn_rate_percent"] == 150


def test_toggles_accept_the_numbers_a_hand_edit_might_use(tmp_path):
    path = tmp_path / "game-config.json"
    path.write_text('{"patterns": 1, "powerups": 0}', encoding="utf-8")
    config = load(path)
    assert config["patterns"] is True
    assert config["powerups"] is False


@pytest.mark.parametrize("name", sorted(BOUNDS), ids=sorted(BOUNDS))
def test_every_bounded_setting_has_a_default_inside_its_range(name):
    minimum, maximum = BOUNDS[name]
    assert minimum < maximum
    assert minimum <= DEFAULTS[name] <= maximum
