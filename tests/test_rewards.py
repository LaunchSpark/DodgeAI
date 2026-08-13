import json

import pytest

from dodge.rewards import SLIDERS, RewardConfig, load, save


def test_defaults_when_the_file_is_absent(tmp_path):
    assert load(tmp_path / "missing.json") == RewardConfig()


def test_round_trips_through_the_file(tmp_path):
    path = tmp_path / "rewards.json"
    save(RewardConfig(death_penalty=2.5, score_weight=0.25), path)
    restored = load(path)
    assert restored.death_penalty == 2.5
    assert restored.score_weight == 0.25


def test_saved_file_is_readable_and_editable_as_text(tmp_path):
    path = tmp_path / "rewards.json"
    save(RewardConfig(), path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["death_penalty"] == 1.0
    assert set(raw) == {field.name for field in RewardConfig.__dataclass_fields__.values()}


def test_a_hand_edited_file_only_needs_the_keys_it_changes(tmp_path):
    path = tmp_path / "rewards.json"
    path.write_text('{"death_penalty": 3.0}', encoding="utf-8")
    config = load(path)
    assert config.death_penalty == 3.0
    assert config.survival_per_frame == RewardConfig().survival_per_frame


def test_malformed_file_falls_back_instead_of_killing_training(tmp_path):
    """A stray comma in a hand edit must not take down a training run."""
    path = tmp_path / "rewards.json"
    path.write_text('{"death_penalty": 3.0,}', encoding="utf-8")
    assert load(path) == RewardConfig()


def test_unknown_and_non_numeric_keys_are_ignored(tmp_path):
    path = tmp_path / "rewards.json"
    path.write_text('{"death_penalty": 2.0, "nonsense": 1, "score_weight": "x"}',
                    encoding="utf-8")
    config = load(path)
    assert config.death_penalty == 2.0
    assert config.score_weight == RewardConfig().score_weight


def test_values_are_clamped_to_the_slider_ranges(tmp_path):
    path = tmp_path / "rewards.json"
    path.write_text('{"death_penalty": 999, "survival_per_frame": -5}',
                    encoding="utf-8")
    config = load(path)
    assert config.death_penalty == 5.0
    assert config.survival_per_frame == 0.0


@pytest.mark.parametrize("spec", SLIDERS, ids=lambda spec: spec.name)
def test_every_slider_matches_a_config_field_and_a_sane_range(spec):
    assert hasattr(RewardConfig(), spec.name)
    assert spec.minimum < spec.maximum
    assert 0 < spec.step <= (spec.maximum - spec.minimum)
    default = getattr(RewardConfig(), spec.name)
    assert spec.minimum <= default <= spec.maximum


def test_the_old_centre_bonus_key_still_loads():
    """rewards.json is hand-edited and lives outside git, so a rename must not
    silently reset a weight the user chose."""
    import json
    from pathlib import Path
    import tempfile
    from dodge.rewards import load

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rewards.json"
        path.write_text(json.dumps({"center_bonus": 0.35, "death_penalty": 2.0}),
                        encoding="utf-8")
        config = load(path)

    assert config.edge_penalty == 0.35
    assert config.death_penalty == 2.0
