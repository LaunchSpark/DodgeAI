"""Training velocity-flow-v3 from the dashboard, and starting it from v2."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from dodge.critic import CRITIC_FEATURE_COUNT, WIDE_OBSERVATION_SIZE
from dodge.env import DodgeVecEnv, native_available
from dodge.observe import OBSERVATION_SIZE
from dodge.policies import ARCHITECTURES, architecture_name, policy_class, policy_kwargs
from dodge.training import (DashboardPPO, PPOTrainingSession,
                            checkpoint_architecture)
from tools.continue_with_critic import continue_with_critic
from tools.new_policy import main as new_policy_main

pytestmark = [
    pytest.mark.skipif(not native_available(), reason="dodge_native not installed"),
    pytest.mark.skipif(CRITIC_FEATURE_COUNT == 0,
                       reason="the native engine has no critic features"),
]


def _save_fresh(path, architecture, *, critic_features):
    env = DodgeVecEnv(n_envs=1, max_episode_frames=20,
                      critic_features=critic_features)
    model = DashboardPPO(policy_class(architecture), env, n_steps=32,
                         batch_size=32, seed=0, device="cpu",
                         policy_kwargs=policy_kwargs(architecture))
    env.close()
    model.save(path)
    return model


def test_continuation_reproduces_v2(tmp_path):
    v2 = _save_fresh(tmp_path / "v2.zip", "velocity-flow-v2", critic_features=False)
    v2.num_timesteps = 4096
    v2._n_updates = 40
    v2.completed_updates = 5
    v2.save(tmp_path / "v2.zip")
    continue_with_critic(tmp_path / "v2.zip", tmp_path / "v3.zip")

    old = DashboardPPO.load(tmp_path / "v2.zip", device="cpu")
    new = DashboardPPO.load(tmp_path / "v3.zip", device="cpu")
    assert checkpoint_architecture(tmp_path / "v3.zip") == "velocity-flow-v3"
    assert (new.num_timesteps, new._n_updates, new.completed_updates) == (4096, 40, 5)
    assert new.gamma == old.gamma

    obs = torch.as_tensor(np.random.default_rng(0).uniform(
        -1, 1, (32, WIDE_OBSERVATION_SIZE)).astype(np.float32))
    with torch.no_grad():
        old_pmf, old_values = old.policy._distribution_inputs(obs[:, :OBSERVATION_SIZE])
        new_pmf, new_values = new.policy._distribution_inputs(obs)
    assert (new_pmf - old_pmf).abs().max() < 1e-6
    assert (new_values - old_values).abs().max() < 1e-6


def test_continuation_refuses_a_non_v2_source(tmp_path):
    _save_fresh(tmp_path / "v1.zip", "velocity-flow", critic_features=False)
    with pytest.raises(ValueError):
        continue_with_critic(tmp_path / "v1.zip", tmp_path / "v3.zip")


def _session(tmp_path):
    return PPOTrainingSession(model_dir=tmp_path, architecture=None, autostart=False,
                              n_envs=2, rollout_steps=32, device="cpu")


def test_session_builds_a_wide_env_for_a_v3_checkpoint(tmp_path):
    _save_fresh(tmp_path / "v2.zip", "velocity-flow-v2", critic_features=False)
    continue_with_critic(tmp_path / "v2.zip", tmp_path / "v3.zip")
    session = _session(tmp_path)
    env = session._env_for(tmp_path / "v3.zip")
    try:
        assert env.observation_space.shape == (WIDE_OBSERVATION_SIZE,)
        model = session._load_model(tmp_path / "v3.zip", env)
        assert architecture_name(model) == "velocity-flow-v3"
    finally:
        env.close()


def test_v2_checkpoint_still_gets_a_plain_env(tmp_path):
    _save_fresh(tmp_path / "v2.zip", "velocity-flow-v2", critic_features=False)
    session = _session(tmp_path)
    env = session._env_for(tmp_path / "v2.zip")
    try:
        assert env.observation_space.shape == (OBSERVATION_SIZE,)
    finally:
        env.close()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_new_policy_writes_a_loadable_checkpoint(tmp_path, architecture):
    assert new_policy_main([architecture, "--models-dir", str(tmp_path)]) == 0
    path = tmp_path / f"{architecture}.zip"
    assert checkpoint_architecture(path) == architecture
    model = DashboardPPO.load(path, device="cpu")
    assert architecture_name(model) == architecture
