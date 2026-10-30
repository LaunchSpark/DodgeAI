"""Many envs on the GPU: device selection, the minibatch cap, the callback."""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from dodge.env import native_available
from dodge.telemetry import EVENT_KEYS

pytestmark = pytest.mark.skipif(not native_available(), reason="dodge_native not installed")


# --- the update schedule ----------------------------------------------------

@pytest.mark.parametrize("architecture,n_envs,rollout_steps,expected", [
    ("velocity-flow-v2", 8, 256, 256),     # the default buffer: unchanged
    ("velocity-flow-v2", 16, 256, 512),    # 4096 / 8 lands exactly on the cap
    ("velocity-flow-v2", 64, 256, 512),    # was 2048
    ("velocity-flow-v2", 256, 64, 512),    # was 2048
    ("velocity-flow-v2", 256, 256, 512),   # was 8192: more than an 8GiB card holds
    ("velocity-flow", 8, 256, 128),        # v1's default buffer: unchanged
    ("velocity-flow", 64, 256, 512),       # was 1024
    ("mlp", 2, 8, 16),                     # a tiny buffer still clamps to itself
])
def test_the_minibatch_stops_growing_with_the_buffer(architecture, n_envs,
                                                     rollout_steps, expected):
    from dodge.training import minibatch_size

    assert minibatch_size(n_envs * rollout_steps, architecture) == expected


def test_the_gpu_preset_is_turners_schedule_for_v2():
    """64 x 256, minibatch 512, 4 epochs, target_kl 0.02: 128 optimizer steps."""
    from dodge.training import SCALE_PRESETS, update_schedule

    preset = SCALE_PRESETS["gpu"]
    schedule = update_schedule(preset["n_envs"], preset["rollout_steps"],
                               "velocity-flow-v2")
    assert schedule == {"buffer": 16384, "minibatch": 512,
                        "minibatches_per_epoch": 32, "epochs": 4,
                        "optimizer_steps": 128, "target_kl": 0.02}
    assert SCALE_PRESETS["default"] == {"n_envs": 8, "rollout_steps": 256}


@pytest.mark.parametrize("n_envs,rollout_steps", [(8, 256), (64, 256), (256, 64),
                                                  (256, 256), (1024, 256)])
def test_an_update_never_balloons_or_shrinks_its_minibatches(n_envs, rollout_steps):
    """Whatever the env count, a v2 minibatch stays between 256 and 512, so
    optimizer steps grow with the samples and never faster."""
    from dodge.training import update_schedule

    schedule = update_schedule(n_envs, rollout_steps, "velocity-flow-v2")
    assert 256 <= schedule["minibatch"] <= 512
    assert schedule["optimizer_steps"] * 256 <= 4 * schedule["buffer"]


# --- device selection -------------------------------------------------------

def test_auto_sends_both_field_policies_to_a_gpu(monkeypatch):
    from dodge import training

    monkeypatch.setattr(training, "get_device", lambda name: f"resolved:{name}")
    assert training.resolve_device("auto", "velocity-flow") == "resolved:auto"
    assert training.resolve_device("auto", "velocity-flow-v2") == "resolved:auto"
    assert training.resolve_device("auto", None) == "resolved:cpu"


def _save(tmp_path, architecture):
    from stable_baselines3 import PPO

    from dodge.env import DodgeVecEnv
    from dodge.policies import policy_class, policy_kwargs

    env = DodgeVecEnv(n_envs=1, max_episode_frames=20)
    try:
        PPO(policy_class(architecture), env, n_steps=16, batch_size=8, seed=0,
            device="cpu", policy_kwargs=policy_kwargs(architecture),
            ).save(tmp_path / architecture)
    finally:
        env.close()
    return tmp_path / f"{architecture}.zip"


@pytest.mark.parametrize("architecture,expected", [("velocity-flow-v2", "auto"),
                                                   ("mlp", "cpu")])
def test_a_loaded_checkpoint_resolves_auto_by_its_own_architecture(
        tmp_path, monkeypatch, architecture, expected):
    """The dashboard starts with no architecture and waits for the Models
    panel, so resolving "auto" once at construction put every loaded run on
    the CPU -- v2 included, at about a tenth of the GPU's update speed."""
    import torch

    from dodge import training
    from dodge.env import DodgeVecEnv

    path = _save(tmp_path, architecture)
    asked = []

    def fake_get_device(name):
        asked.append(name)
        return torch.device("cpu")

    monkeypatch.setattr(training, "get_device", fake_get_device)
    session = training.PPOTrainingSession(model_dir=tmp_path, logs_dir=tmp_path,
                                          architecture=None, autostart=False,
                                          device="auto", n_envs=1, rollout_steps=16)
    assert asked == ["cpu"], "no architecture yet, so nothing to want a GPU for"
    env = DodgeVecEnv(n_envs=1, max_episode_frames=20)
    try:
        session._load_model(path, env)
    finally:
        env.close()
    assert asked[-1] == expected


def test_the_cli_resolves_auto_by_the_checkpoint_it_resumes(tmp_path):
    import train_ppo

    assert train_ppo.checkpoint_architecture(_save(tmp_path, "velocity-flow-v2")) \
        == "velocity-flow-v2"
    assert train_ppo.checkpoint_architecture(_save(tmp_path, "mlp")) == "mlp"


# --- the dashboard callback at many envs ------------------------------------

def _callback(tmp_path, monkeypatch):
    from dodge.training import DashboardCallback, PPOTrainingSession

    session = PPOTrainingSession(model_dir=tmp_path, autostart=False, device="cpu")
    monkeypatch.setattr(session, "_should_stop", lambda model: False)
    callback = DashboardCallback(session)
    callback.model = None
    callback.num_timesteps = 0
    callback.n_calls = 0
    return session, callback


def _step(callback, infos, dones, rewards):
    callback.n_calls += 1
    callback.num_timesteps += len(infos)
    callback.locals = {"infos": infos, "dones": np.asarray(dones),
                       "rewards": np.asarray(rewards, dtype=np.float32)}
    callback._on_step()


def _events(**counts):
    return {key: counts.get(key, 0) for key in EVENT_KEYS}


def test_the_callback_banks_each_lane_separately_across_many_envs(tmp_path,
                                                                 monkeypatch):
    """Reward and kills are per-lane running totals, released in lane order
    when that lane's episode ends -- the bookkeeping the vectorised step has
    to keep exactly."""
    session, callback = _callback(tmp_path, monkeypatch)
    lanes = 64

    def infos(kills=(), done=()):
        return [{"frames": 60, "run_frames": 60, "run_over": True, "score": 0.0,
                 "training_events": _events(enemies_destroyed=int(lane in kills),
                                            survival_frames=1),
                 **({"episode": {"r": 0.0}} if lane in done else {})}
                for lane in range(lanes)]

    def mask(*lanes_done):
        return np.isin(np.arange(lanes), lanes_done)

    rewards = np.arange(lanes, dtype=np.float32)
    _step(callback, infos(kills={3, 40}), mask(), rewards)
    _step(callback, infos(kills={40}, done={3, 40}), mask(3, 40), rewards)

    data = session.data
    # Lane 3 first: finished episodes are reported in lane order.
    assert data.reward.as_list() == [pytest.approx(6.0), pytest.approx(80.0)]
    assert data.enemies_killed.as_list() == [1.0, 2.0]
    assert data.events["enemies_destroyed"].total == 3
    assert data.events["survival_frames"].total == 2 * lanes
    assert callback.rollout_events["survival_frames"] == 2 * lanes
    assert len(data.episode_length) == 2
    # A finished lane starts its next episode from zero; the rest carry on.
    _step(callback, infos(done={3, 5}), mask(3, 5), rewards)
    assert data.reward.as_list()[-2:] == [pytest.approx(3.0), pytest.approx(15.0)]


def test_the_callback_tolerates_partial_events_and_short_rewards(tmp_path,
                                                                monkeypatch):
    """Test doubles and other wrappers send partial event dicts, none at all,
    or fewer rewards than infos; none of it may raise."""
    session, callback = _callback(tmp_path, monkeypatch)
    infos = [{"frames": 30, "run_over": True, "training_events": {"deaths": 1}},
             {"frames": 30, "run_over": True},
             {"frames": 30, "run_over": True,
              "training_events": {"enemies_destroyed": 2, "not_an_event": 9}}]
    _step(callback, infos, [True, True, True], [1.0])

    data = session.data
    assert data.reward.as_list() == [1.0, 0.0, 0.0]
    assert data.enemies_killed.as_list() == [0.0, 0.0, 2.0]
    assert data.events["deaths"].total == 1
    assert data.events["enemies_destroyed"].total == 2


def test_the_callback_follows_a_change_in_lane_count(tmp_path, monkeypatch):
    session, callback = _callback(tmp_path, monkeypatch)
    live = {"frames": 30, "run_over": True, "training_events": {}}
    _step(callback, [dict(live), dict(live)], [False, False], [1.0, 2.0])
    _step(callback, [dict(live) for _ in range(4)], [False, True, False, True],
          [1.0] * 4)
    assert session.data.reward.as_list() == [3.0, 1.0]


def test_the_callback_reads_batched_infos_exactly_as_full_dicts(tmp_path,
                                                                monkeypatch):
    """DodgeVecEnv's StepInfos give a running lane an empty dict and carry its
    fields on `batch`; the callback must total them exactly as it would the
    full dicts `lane(i)` rebuilds."""
    from dodge.env import DodgeVecEnv

    batched_session, batched = _callback(tmp_path / "batched", monkeypatch)
    full_session, full = _callback(tmp_path / "full", monkeypatch)
    env = DodgeVecEnv(n_envs=16, seed=0, difficulty=3, patterns=True, lives=2,
                      max_episode_frames=400)
    env.reset()
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(600):
        _, rewards, dones, infos = env.step(rng.integers(0, 9, 16))
        assert hasattr(infos, "batch")
        finished += int(np.sum(dones))
        _step(batched, infos, dones, rewards)
        _step(full, [infos.lane(i) for i in range(len(infos))], dones, rewards)
    env.close()

    assert finished > 0
    a, b = batched_session.data, full_session.data
    assert a.reward.as_list() == b.reward.as_list()
    assert a.enemies_killed.as_list() == b.enemies_killed.as_list()
    assert a.episode_length.as_list() == b.episode_length.as_list()
    assert [x.as_list() for x in a.life_lengths] == [x.as_list() for x in b.life_lengths]
    assert {k: e.total for k, e in a.events.items()} == {k: e.total for k, e in b.events.items()}
    assert batched.rollout_events == full.rollout_events
