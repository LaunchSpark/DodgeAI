"""velocity-flow-v3: v2's actor, and a critic that also reads hidden state."""

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces

from dodge.critic import CRITIC_FEATURE_COUNT, WIDE_OBSERVATION_SIZE
from dodge.observe import OBSERVATION_SIZE
from dodge.policies import (ARCHITECTURES, EXTRACTOR_ARCHITECTURE,
                            architecture_from_kwargs, policy_class, policy_kwargs)
from dodge.training import GPU_ARCHITECTURES, hyperparameters
from dodge.velocity import VelocityFlowV2Extractor, VelocityFlowV3Policy

pytestmark = pytest.mark.skipif(CRITIC_FEATURE_COUNT == 0,
                                reason="the native engine has no critic features")


def _policy():
    space = spaces.Box(-1, 1, (WIDE_OBSERVATION_SIZE,), np.float32)
    return VelocityFlowV3Policy(space, spaces.Discrete(9), lambda _: 3e-4,
                                **policy_kwargs("velocity-flow-v3"))


def test_actor_ignores_critic_columns():
    policy = _policy()
    with torch.no_grad():  # make the zero-initialised critic columns live
        policy.features_extractor.value_net[0].weight[:, -CRITIC_FEATURE_COUNT:].normal_()
    obs = torch.rand(16, WIDE_OBSERVATION_SIZE) * 2 - 1
    other = obs.clone()
    other[:, OBSERVATION_SIZE:] = torch.rand(16, CRITIC_FEATURE_COUNT)
    a, va = policy._distribution_inputs(obs)
    b, vb = policy._distribution_inputs(other)
    assert torch.equal(a, b) and not torch.equal(va, vb)


def test_new_critic_columns_start_at_zero():
    w = _policy().features_extractor.value_net[0].weight
    assert torch.count_nonzero(w[:, -CRITIC_FEATURE_COUNT:]) == 0


def test_field_accepts_wide_and_narrow_observations():
    extractor = _policy().features_extractor
    narrow = torch.rand(2, OBSERVATION_SIZE) * 2 - 1
    wide = torch.cat((narrow, torch.rand(2, CRITIC_FEATURE_COUNT)), dim=1)
    assert torch.equal(extractor.field(narrow), extractor.field(wide))


def test_predict_accepts_actor_only_observations():
    policy = _policy()
    narrow = np.random.default_rng(0).uniform(-1, 1, (4, OBSERVATION_SIZE)).astype(np.float32)
    wide = np.concatenate([narrow, np.zeros((4, CRITIC_FEATURE_COUNT), np.float32)], 1)
    assert np.array_equal(policy.predict(narrow, deterministic=True)[0],
                          policy.predict(wide, deterministic=True)[0])
    one = policy.predict(narrow[0], deterministic=True)[0]
    assert np.array_equal(one, policy.predict(wide[0], deterministic=True)[0])


def test_v2_forward_equals_its_helper():
    space = spaces.Box(-1, 1, (OBSERVATION_SIZE,), np.float32)
    extractor = VelocityFlowV2Extractor(space)
    obs = torch.rand(4, OBSERVATION_SIZE) * 2 - 1
    with torch.no_grad():
        danger, inputs = extractor._danger_and_value_inputs(obs)
        expected = torch.cat((-danger, extractor.value_net(inputs)), dim=1)
        assert torch.equal(extractor(obs), expected)


def test_v3_is_registered():
    assert ARCHITECTURES[0] == "velocity-flow" and "velocity-flow-v3" in ARCHITECTURES
    assert "velocity-flow-v3" in GPU_ARCHITECTURES
    assert hyperparameters("velocity-flow-v3") == hyperparameters("velocity-flow-v2")
    assert policy_class("velocity-flow-v3") is VelocityFlowV3Policy
    assert architecture_from_kwargs(policy_kwargs("velocity-flow-v3")) == "velocity-flow-v3"
    assert EXTRACTOR_ARCHITECTURE["VelocityFlowV3Extractor"] == "velocity-flow-v3"
