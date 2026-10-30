from types import SimpleNamespace

import numpy as np
import pytest

from dodge.env import (DodgeVecEnv, center_survival_scalar, make_engine,
                       native_available, play_games)
from dodge.observe import OBSERVATION_SIZE, observe
from dodge.rewards import RewardConfig

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")


@pytest.fixture(autouse=True)
def _default_reward_weights(tmp_path, monkeypatch):
    """Keep these tests off whatever weights the developer is currently trying.

    Reward weights are tuned live from the dashboard, so reading the repository's
    own rewards.json would make the suite fail whenever someone moved a slider.
    """
    from dodge import rewards

    path = tmp_path / "default-rewards.json"
    rewards.save(RewardConfig(), path)
    monkeypatch.setattr(rewards, "CONFIG_PATH", path)


def make(**kwargs):
    settings = dict(n_envs=4, seed=3, difficulty=3, patterns=True, powerups=True,
                    action_repeat=1)
    settings.update(kwargs)
    return DodgeVecEnv(**settings)


def idle(env):
    return np.zeros(env.num_envs, dtype=np.int64)


def test_reset_returns_one_observation_per_lane():
    env = make()
    observation = env.reset()
    assert observation.shape == (4, OBSERVATION_SIZE)
    assert observation.dtype == np.float32
    assert np.all(np.abs(observation) <= 1.0)
    # The player starts in the middle of the arena.
    assert np.allclose(observation[:, 0], 64 / 127, atol=0.02)


def test_alive_frame_pays_survival_credit():
    env = make()
    env.reset()
    _, reward, done, info = env.step(idle(env))
    assert not done.any()
    # Centre of the arena, no score yet: exactly one frame of survival credit.
    assert np.allclose(reward, RewardConfig().survival_per_frame)
    # Per-step fields of lanes still mid-episode travel as arrays.
    assert all(lane == {} for lane in info)
    assert np.all(info.batch.survived_frames == 1)
    assert all(info.lane(lane)["survived_frames"] == 1 for lane in range(env.num_envs))


def test_same_seed_and_actions_are_deterministic():
    rng = np.random.default_rng(5)
    actions = [rng.integers(0, 9, size=4) for _ in range(300)]

    def run():
        env = make(seed=11)
        observations = [env.reset()]
        for action in actions:
            observations.append(env.step(action)[0])
        return np.stack(observations)

    np.testing.assert_array_equal(run(), run())


def test_frame_cap_truncates_and_restarts_the_lane():
    env = make(max_episode_frames=5)
    env.reset()
    for _ in range(4):
        _, _, done, _ = env.step(idle(env))
        assert not done.any()
    observation, _, done, info = env.step(idle(env))
    assert done.all()
    assert all(lane["TimeLimit.truncated"] and lane["run_over"] for lane in info)
    assert all(lane["episode"]["l"] == 5 for lane in info)
    assert info[0]["terminal_observation"].shape == (OBSERVATION_SIZE,)
    # The returned observation is the restarted game, not the terminal one.
    assert np.allclose(observation[:, 0], 64 / 127, atol=0.02)


def test_non_positive_frame_cap_is_rejected():
    with pytest.raises(ValueError):
        make(max_episode_frames=0)


def test_death_terminates_with_the_death_penalty_and_names_the_cause():
    env = make()
    env.reset()
    # Idling in the centre trips the anti-camping wave, which kills quickly.
    for _ in range(2_000):
        _, reward, done, info = env.step(idle(env))
        if done.any():
            lane = int(np.flatnonzero(done)[0])
            assert info[lane]["run_over"]
            assert not info[lane]["TimeLimit.truncated"]
            assert reward[lane] == pytest.approx(-RewardConfig().death_penalty)
            assert info[lane]["death"]["cause"] in {"normal", "kamikaze", "kamikaze_blast",
                                                    "static_rect", "moving_rect"}
            assert info[lane]["death"]["fyou_triggers"] >= 1
            return
    pytest.fail("no lane died while idling")


def test_an_absorbed_hit_ends_the_episode_but_leaves_the_game_running():
    env = make(lives=1, n_envs=8)
    env.reset()
    rng = np.random.default_rng(2)
    for _ in range(5_000):
        _, reward, done, info = env.step(rng.integers(0, 9, size=env.num_envs))
        # Two enemies can land on the same frame, spending the life and then
        # killing; look for a hit the game survived.
        events = info.batch.events
        spent = np.flatnonzero((events["lives_spent"] > 0)
                               & (events["deaths"] == 0)).tolist()
        if spent:
            lane = spent[0]
            assert done[lane]
            assert not info[lane]["run_over"]
            assert info[lane]["training_events"]["deaths"] == 0
            assert reward[lane] == pytest.approx(-RewardConfig().death_penalty)
            run_frames = info[lane]["run_frames"]
            _, _, _, info = env.step(idle(env))
            # A new episode on the same game: run frames keep counting.
            assert info.lane(lane)["frames"] == 1
            assert info.lane(lane)["run_frames"] == run_frames + 1
            return
    pytest.fail("no lane survived spending a life")


def test_disabled_enemy_types_never_appear():
    env = make(normal_enemies=False, kamikaze_enemies=False, n_envs=2)
    env.reset()
    for _ in range(600):
        env.step(idle(env))
        for lane in range(env.num_envs):
            assert all(enemy["p"] >= 2 for enemy in env.lane_state(lane)["enemies"])


def test_spawn_rate_outside_the_supported_range_is_rejected():
    with pytest.raises(ValueError, match="spawn rate"):
        make(spawn_rate_percent=40)


def test_survival_credit_is_flat_across_the_interior():
    """The centre must not out-pay its surroundings.

    A peak at the centre paid the agent to stand still, because that is where
    it spawns and any movement reduced its income. Flat interior, no attractor.
    """
    centre = center_survival_scalar(5, 5, screen_width=10)
    assert centre == pytest.approx(1.0)
    for x, y in ((5, 6), (6, 5), (6.5, 6.5), (7, 5), (3, 5)):
        assert center_survival_scalar(x, y, screen_width=10) == pytest.approx(centre)


def test_survival_credit_tapers_only_near_the_wall():
    scalar = center_survival_scalar
    # Plateau ends at 0.6 of the half-arena: 5 +/- 3 on a 10-wide screen.
    assert scalar(8, 5, screen_width=10) == pytest.approx(1.0)
    assert scalar(9, 5, screen_width=10) == pytest.approx(0.75)
    assert scalar(10, 5, screen_width=10) == pytest.approx(0.5)
    # Corners lie beyond a cardinal edge; the taper clamps instead of inverting.
    assert scalar(10, 10, screen_width=10) == pytest.approx(0.5)
    assert scalar(0, 0, screen_width=10) == pytest.approx(0.5)


@pytest.mark.parametrize("edge_penalty", [0.0, 0.35, 0.5, 0.7, 1.0])
def test_batched_survival_credit_matches_the_scalar_to_the_bit(edge_penalty):
    """The env credits every lane at once; each must equal the scalar exactly,
    or the float32 reward moves."""
    from dodge.env import center_survival_scalars
    from dodge.observe import ARENA_SIZE

    rng = np.random.default_rng(7)
    centre, radius = ARENA_SIZE / 2.0, ARENA_SIZE / 2.0
    on_boundaries = [centre + radius * r for r in (0.6, 1.0)]
    x = np.concatenate([rng.uniform(-4, ARENA_SIZE + 4, 20_000),
                        rng.integers(0, ARENA_SIZE * 256, 5_000) / 256.0,
                        on_boundaries, [0.0, ARENA_SIZE, centre]])
    y = np.concatenate([rng.uniform(-4, ARENA_SIZE + 4, 20_000),
                        rng.integers(0, ARENA_SIZE * 256, 5_000) / 256.0,
                        [centre, centre], [0.0, ARENA_SIZE, centre]])
    batched = center_survival_scalars(x, y, edge_penalty=edge_penalty)
    expected = np.array([center_survival_scalar(a, b, edge_penalty=edge_penalty)
                         for a, b in zip(x, y)])
    assert batched.dtype == np.float64
    assert batched.tobytes() == expected.tobytes()
    # Small batches take a per-lane path of their own.
    small = np.concatenate([center_survival_scalars(x[i:i + 8], y[i:i + 8],
                                                    edge_penalty=edge_penalty)
                            for i in range(0, len(x), 8)])
    assert small.dtype == np.float64
    assert small.tobytes() == expected.tobytes()


def test_edge_penalty_of_zero_pays_every_position_equally():
    for x, y in ((5, 5), (10, 5), (10, 10)):
        assert center_survival_scalar(x, y, screen_width=10,
                                      edge_penalty=0.0) == pytest.approx(1.0)


class _LuaList:
    """1-based indexing, the shape dodge/observe.py reads."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index - 1]


def _reference_globals(state):
    cp = None
    if state["cp"] is not None:
        rects = []
        for rect in state["cp"]["rects"]:
            rect = dict(rect)
            rect["warnings"] = (_LuaList([_LuaList(list(w)) for w in rect["warnings"]])
                                if rect["warnings"] else None)
            rects.append(SimpleNamespace(**rect))
        cp = SimpleNamespace(tpe=state["cp"]["tpe"], timer=state["cp"]["timer"],
                             rects=_LuaList(rects))
    return SimpleNamespace(
        px=state["px"], py=state["py"], dpx=state["dpx"], dpy=state["dpy"],
        spd=state["spd"], frz=state["frz"], cp=cp,
        enemies=_LuaList([SimpleNamespace(**enemy) for enemy in state["enemies"]]))


def test_the_engine_observation_matches_the_reference_encoder():
    """dodge/observe.py is the specification; the engine must match it exactly."""
    engine = make_engine(difficulty=3, patterns=True, powerups=True)
    lanes = 8
    engine.reset_ml_batch(np.arange(lanes, dtype=np.uint32) + 500)
    rng = np.random.default_rng(1)
    checked = 0
    for step in range(900):
        if step % 8 == 0:
            actions = rng.integers(0, 9, lanes, dtype=np.uint8)
        done = engine.step_ml_batch(actions)["done"]
        if step % 13 == 0:
            native = engine.observe_velocity()["observation"]
            for lane in range(lanes):
                expected = observe(SimpleNamespace(
                    globals=_reference_globals(engine.lane_state(lane))))
                np.testing.assert_array_equal(native[lane], expected)
                checked += 1
        finished = np.flatnonzero(done).astype(np.uint32)
        if len(finished):
            engine.reset_ml_lanes(finished, finished + 9_000 + step)
    assert checked > 400


def test_play_games_reports_one_result_per_seed():
    results = play_games(lambda observations, _: np.zeros(len(observations)),
                         [1, 2, 3], difficulty=3, patterns=True, powerups=True)
    assert [result["seed"] for result in results] == [1, 2, 3]
    assert all(result["died"] and result["frames"] > 0 for result in results)
