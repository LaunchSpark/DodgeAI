"""DodgeVecEnv must report exactly what the original implementation reported.

The step was rewritten for throughput. These tests run it side by side with
the frozen original (tests/reference_env.py) on fixed seeds and fixed action
sequences, across lane counts, action repeats, training lives and frame caps,
so that deaths, spent lives, truncations and game restarts all happen, and
demand bit-for-bit agreement: the same observations, the same float32 reward
bits, the same dones, and per-lane infos equal in type and value.

The optimised env builds a full info dict only for a lane whose episode ended
and carries every lane's per-step fields as arrays on `infos.batch`;
`infos.lane(i)` rebuilds lane i's full dict from them. Both views are checked
against the original's dicts, and so are the aggregates the dashboard
callback and SB3 compute from them.
"""

from collections import Counter
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dodge import rewards
from dodge.env import DodgeVecEnv, native_available
from dodge.rewards import RewardConfig
from dodge.telemetry import DashboardSnapshot

from .reference_env import ReferenceDodgeVecEnv

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")


@pytest.fixture(autouse=True)
def _weights_and_clock(tmp_path, monkeypatch):
    """Non-default weights, so every reward term moves; and a shared clock.

    Episode "t" is wall-clock time. Both envs read the same fake clock, which
    only moves between steps, so it is reproducible too.
    """
    path = tmp_path / "parity-rewards.json"
    rewards.save(RewardConfig(survival_per_frame=0.013, edge_penalty=0.7,
                              death_penalty=1.7, score_weight=1.3,
                              uncontrolled_score_weight=0.3), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    return clock


GAME = dict(difficulty=3, patterns=True, powerups=True)

CONFIGS = {
    # One lane, no lives, no cap: every episode is a whole game ending in death.
    "1-lane": dict(n_envs=1, seed=101, action_repeat=1, lives=0,
                   max_episode_frames=None, steps=3000),
    # Held actions with lives and a cap: spent lives, deaths and truncations.
    "7-lanes-repeat-4": dict(n_envs=7, seed=202, action_repeat=4, lives=2,
                             max_episode_frames=360, steps=2000),
    # Many lanes at one frame a decision: several restarts in one step.
    "64-lanes": dict(n_envs=64, seed=303, action_repeat=1, lives=1,
                     max_episode_frames=400, steps=1500),
    "16-lanes-repeat-4-cap": dict(n_envs=16, seed=404, action_repeat=4, lives=0,
                                  max_episode_frames=240, steps=1500),
}


def action_sequence(n_envs, steps, seed):
    """Random play mixed with idling (which trips the anti-camping wave) and
    long held directions (which reach the walls, where the edge taper bites)."""
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 9, size=(steps, n_envs))
    for lane in range(n_envs):
        start = 0
        while start < steps:
            length = int(rng.integers(20, 200))
            mode = rng.integers(0, 3)
            if mode == 1:
                actions[start:start + length, lane] = 0
            elif mode == 2:
                actions[start:start + length, lane] = int(rng.integers(1, 9))
            start += length
    return actions


def assert_identical(expected, actual, where="info"):
    """Equal in type and value, floats to the bit, dict keys in order."""
    assert type(actual) is type(expected), f"{where}: {type(actual)} != {type(expected)}"
    if isinstance(expected, dict):
        assert list(actual) == list(expected), f"{where}: keys {list(actual)} != {list(expected)}"
        for key in expected:
            assert_identical(expected[key], actual[key], f"{where}[{key!r}]")
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, where
        assert actual.tobytes() == expected.tobytes(), f"{where}: arrays differ"
    elif isinstance(expected, float):
        assert actual.hex() == expected.hex(), f"{where}: {actual!r} != {expected!r}"
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def full_infos(infos):
    """Every lane's full info dict, however the env chose to deliver them."""
    if hasattr(infos, "lane"):
        return [infos.lane(index) for index in range(len(infos))]
    return list(infos)


def make_pair(config, restart_observe):
    settings = {key: value for key, value in config.items() if key != "steps"}
    reference = ReferenceDodgeVecEnv(**settings, **GAME)
    candidate = DodgeVecEnv(**settings, **GAME)
    candidate._restart_observe = restart_observe
    return reference, candidate


def assert_step_identical(expected, actual, step):
    obs_e, reward_e, done_e, infos_e = expected
    obs_a, reward_a, done_a, infos_a = actual
    assert obs_a.dtype == obs_e.dtype and obs_a.shape == obs_e.shape
    assert obs_a.tobytes() == obs_e.tobytes(), f"step {step}: observations differ"
    assert reward_a.dtype == np.float32 and reward_e.dtype == np.float32
    assert reward_a.tobytes() == reward_e.tobytes(), f"step {step}: rewards differ"
    assert done_a.dtype == done_e.dtype and np.array_equal(done_a, done_e), f"step {step}"
    assert len(infos_a) == len(infos_e)
    full = full_infos(infos_a)
    for lane, (info_e, info_a) in enumerate(zip(infos_e, full)):
        assert_identical(info_e, info_a, f"step {step} lane {lane}")
        # What the list itself holds: the full dict at an episode's end,
        # where SB3 and the dashboard read it, and nothing otherwise.
        if done_e[lane]:
            assert_identical(info_e, infos_a[lane], f"step {step} lane {lane} (list)")
        elif hasattr(infos_a, "lane"):
            assert infos_a[lane] == {}


@pytest.mark.parametrize("restart_observe", ["auto", "nursery", "batch"])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_step_is_bit_identical_to_the_original(name, restart_observe, _weights_and_clock):
    clock = _weights_and_clock
    config = CONFIGS[name]
    reference, candidate = make_pair(config, restart_observe)
    steps = config["steps"]
    actions = action_sequence(config["n_envs"], steps, config["seed"])

    observation_e, observation_a = reference.reset(), candidate.reset()
    assert observation_a.tobytes() == observation_e.tobytes()
    assert_identical(reference.reset_infos, candidate.reset_infos, "reset_infos")

    seen = Counter()
    for step in range(steps):
        clock.now += 0.25 + 0.001 * step
        if step == steps // 2:
            # A segment boundary, as SegmentCallback takes it.
            boundary_e, abandoned_e = reference.abandon_episodes()
            boundary_a, abandoned_a = candidate.abandon_episodes()
            assert boundary_a.tobytes() == boundary_e.tobytes()
            assert_identical(abandoned_e, abandoned_a, "abandoned")
            assert_identical(reference.seed_state(), candidate.seed_state(), "seed_state")
        expected = reference.step(actions[step])
        actual = candidate.step(actions[step])
        assert_step_identical(expected, actual, step)
        for info in expected[3]:
            seen["death"] += "death" in info
            seen["run"] += "run" in info
            seen["truncated"] += bool(info.get("TimeLimit.truncated"))
            seen["spent_life"] += (info["training_events"]["lives_spent"] > 0
                                   and not info["training_events"]["deaths"])
            seen["pickup"] += sum(info["training_events"][key] for key in
                                  ("explosion_pickups", "freeze_pickups", "shrink_pickups"))
            seen["pattern_survived"] += info["training_events"]["patterns_survived"]
            seen["edge"] += (info["survival_reward"]
                             < 0.013 * info["survived_frames"] * (1 - 1e-9))

    assert_identical(reference.seed_state(), candidate.seed_state(), "final seed_state")
    assert reference.total_frames == candidate.total_frames

    # The run has to have exercised what it claims to.
    assert seen["death"] > 0 and seen["run"] > 0, seen
    assert seen["edge"] > 0, "no lane ever earned tapered credit near a wall"
    if config["lives"]:
        assert seen["spent_life"] > 0, seen
    if config["max_episode_frames"]:
        assert seen["truncated"] > 0, seen


def make_callback():
    from dodge.training import DashboardCallback

    session = SimpleNamespace(lock=threading.Lock(), data=DashboardSnapshot(),
                              rollout_steps=64, _should_stop=lambda model: False,
                              _completed_updates=lambda model: 0)
    callback = DashboardCallback(session)
    callback.model = None
    return callback


def snapshot_aggregates(data):
    return {
        "best_score": data.best_score,
        "reward": data.reward.as_list(),
        "episode_length": data.episode_length.as_list(),
        "enemies_killed": data.enemies_killed.as_list(),
        "life_lengths": [band.as_list() for band in data.life_lengths],
        "events": {key: (count.rollout, count.total, list(count.history))
                   for key, count in data.events.items()},
    }


@pytest.mark.parametrize("name", ["7-lanes-repeat-4", "64-lanes"])
def test_dashboard_and_sb3_aggregates_are_unchanged(name, _weights_and_clock):
    """The dashboard reads the batched arrays; SB3 reads the dicts. Fed the
    original's per-lane dicts and the optimised env's output respectively,
    both must arrive at the same numbers."""
    clock = _weights_and_clock
    config = CONFIGS[name]
    reference, candidate = make_pair(config, "auto")
    steps = config["steps"]
    actions = action_sequence(config["n_envs"], steps, config["seed"] + 1)
    callbacks = {"reference": make_callback(), "candidate": make_callback()}
    episodes = {"reference": [], "candidate": []}
    bootstraps = {"reference": [], "candidate": []}
    reference.reset()
    candidate.reset()
    for step in range(steps):
        clock.now += 0.5
        for key, env in (("reference", reference), ("candidate", candidate)):
            _, reward, done, infos = env.step(actions[step])
            callback = callbacks[key]
            callback.n_calls = step + 1
            callback.num_timesteps = (step + 1) * env.num_envs
            callback.locals = {"infos": infos, "dones": done, "rewards": reward}
            assert callback._on_step()
            if (step + 1) % callback.session.rollout_steps == 0:
                callback._on_rollout_end()
            # What SB3's collect_rollouts and _update_info_buffer read.
            for index, info in enumerate(infos):
                if info.get("episode") is not None:
                    episodes[key].append(info["episode"])
                if (done[index] and info.get("terminal_observation") is not None
                        and info.get("TimeLimit.truncated", False)):
                    bootstraps[key].append((step, index,
                                            info["terminal_observation"].tobytes()))

    expected = snapshot_aggregates(callbacks["reference"].session.data)
    actual = snapshot_aggregates(callbacks["candidate"].session.data)
    assert actual == expected
    assert len(expected["reward"]) > 0 and expected["events"]["deaths"][1] > 0
    assert_identical(episodes["reference"], episodes["candidate"], "episodes")
    assert bootstraps["candidate"] == bootstraps["reference"]
