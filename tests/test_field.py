"""The field policy: a CNN that only draws danger, and a controller that reads it."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from dodge.observe import GRID_SIZE, OBSERVATION_SIZE
from dodge.actions import ACTION_NAMES
from dodge.policies import policy_class, policy_kwargs, architecture_name


def _model(seed=0):
    from dodge.env import DodgeEnv
    return PPO(policy_class("field-cnn"), DodgeEnv(max_episode_frames=20),
               n_steps=16, batch_size=8, n_epochs=1, seed=seed, device="cpu",
               policy_kwargs=policy_kwargs("field-cnn"))


def test_action_offsets_are_distinct_and_symmetric():
    """A clamp at the arena edge once made 'left' identical to 'idle'."""
    from dodge.field import action_offsets

    offsets = action_offsets()
    assert np.allclose(offsets[ACTION_NAMES.index("idle")], 0.0)

    def final(name):
        return offsets[ACTION_NAMES.index(name)][-1]

    assert final("left")[0] < 0 and final("right")[0] > 0
    assert final("up")[1] < 0 and final("down")[1] > 0
    np.testing.assert_allclose(final("left"), -final("right"))
    np.testing.assert_allclose(final("up"), -final("down"))
    # Every action must reach somewhere its neighbours do not.
    assert len({tuple(np.round(o[-1], 3)) for o in offsets}) == len(offsets)


def test_the_network_only_produces_a_field():
    from dodge.field import DodgeFieldExtractor

    model = _model()
    extractor = model.policy.features_extractor
    assert isinstance(extractor, DodgeFieldExtractor)
    field = extractor.field(torch.rand(4, OBSERVATION_SIZE))
    assert field.shape == (4, 1, GRID_SIZE, GRID_SIZE)


def test_it_is_far_smaller_than_the_hybrid_network():
    """The point of dropping the flatten: 87% of hybrid-cnn was one dense layer."""
    from dodge.env import DodgeEnv

    field = sum(p.numel() for p in _model().policy.parameters())
    hybrid = PPO("MlpPolicy", DodgeEnv(max_episode_frames=20), n_steps=16,
                 batch_size=8, seed=0, device="cpu",
                 policy_kwargs=policy_kwargs("hybrid-cnn"))
    hybrid_count = sum(p.numel() for p in hybrid.policy.parameters())
    assert field < hybrid_count / 5, f"{field:,} vs {hybrid_count:,}"


def test_logits_ignore_the_fields_magnitude():
    """Only the shape of the field should reach the distribution.

    Untouched, a fresh field spreads its readings ~30x too little to move a
    softmax, so the policy would sit at uniform until the magnitude grew.
    """
    model = _model()
    observations = torch.rand(16, OBSERVATION_SIZE)
    with torch.no_grad():
        before = model.policy.get_distribution(observations).distribution.probs
        # Scaling the field head scales every reading; the policy must not care.
        model.policy.features_extractor.field_head.weight.mul_(7.0)
        model.policy.features_extractor.field_head.bias.mul_(7.0)
        after = model.policy.get_distribution(observations).distribution.probs
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-4)


def test_the_policy_starts_exploratory_but_state_dependent():
    model = _model()
    observations = torch.rand(64, OBSERVATION_SIZE)
    with torch.no_grad():
        distribution = model.policy.get_distribution(observations).distribution
    entropy = float(distribution.entropy().mean())
    assert entropy > 0.9 * np.log(9), "should start near uniform, not committed"
    assert float(distribution.probs.std(dim=0).max()) > 0.0, \
        "but must already vary with the observation"


def test_it_trains_and_survives_a_save_load_cycle(tmp_path):
    model = _model(seed=3)
    assert architecture_name(model) == "field-cnn"
    model.learn(total_timesteps=32)

    path = tmp_path / "field-cnn"
    model.save(path)
    restored = PPO.load(path, device="cpu")
    assert architecture_name(restored) == "field-cnn"
    action, _ = restored.predict(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
    assert 0 <= int(action) < 9
