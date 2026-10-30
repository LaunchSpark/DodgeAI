"""No-replacement evaluation: one game per seed, finished lanes masked out."""

import numpy as np
import pytest
import torch

from dodge.env import native_available, play_games
from dodge.evaluate import (EvalResult, engine_settings, evaluate_model,
                            evaluate_policy, quarter_means)

pytestmark = pytest.mark.skipif(not native_available(),
                                reason="dodge_native with observe_velocity not installed")

GAME = dict(step_frames=4, difficulty=3, patterns=True, powerups=True)


def scripted(observations):
    """Deterministic in the observation, so lanes agree across evaluators."""
    return (np.abs(observations[:, :64]).sum(axis=1) * 997).astype(np.int64) % 9


def test_quarter_means_follow_the_paper():
    frames = np.arange(1, 9, dtype=float)  # 1..8
    q1, q4 = quarter_means(frames)
    assert q1 == 1.5 and q4 == 7.5
    result = EvalResult(seeds=list(range(8)), frames=frames, scores=np.zeros(8),
                        died=[True] * 8)
    assert result.mean == 4.5
    assert result.q1 == 1.5 and result.q4 == 7.5
    summary = result.summary()
    assert summary["n"] == 8 and summary["mean"] == 4.5
    # Fewer than four games: the worst quarter is the single worst game.
    assert quarter_means(np.array([3.0, 9.0])) == (3.0, 9.0)


def test_each_seed_plays_exactly_one_game_matching_the_refilling_reference():
    seeds = list(range(500, 512))
    result = evaluate_policy(scripted, seeds, frame_cap=600, **GAME)
    reference = play_games(lambda obs, _: scripted(obs), seeds, frame_cap=600, **GAME)
    assert result.seeds == seeds
    assert list(result.frames) == [game["frames"] for game in reference]
    assert list(result.scores) == [game["score"] for game in reference]
    assert result.died == [game["died"] for game in reference]


def test_finished_lanes_are_masked_out_and_never_refilled():
    batches = []

    def policy(observations):
        batches.append(len(observations))
        return scripted(observations)

    result = evaluate_policy(policy, list(range(40, 56)), frame_cap=2_000, **GAME)
    assert batches[0] == 16
    assert all(later <= earlier for earlier, later in zip(batches, batches[1:]))
    # The batch shrinks as games end: some lanes finish before others.
    assert batches[-1] < 16
    assert len(result.frames) == 16
    assert max(result.frames) <= 2_000 + GAME["step_frames"]


def test_a_frame_cap_ends_a_game_without_a_death():
    result = evaluate_policy(lambda obs: np.zeros(len(obs), dtype=np.int64),
                             [1, 2], frame_cap=8, **GAME)
    assert list(result.frames) == [8, 8]
    assert result.died == [False, False]


def test_engine_settings_translate_the_training_env():
    settings = engine_settings({
        "difficulty": 3, "patterns": True, "powerups": False,
        "normal_enemies": True, "kamikaze_enemies": False,
        "spawn_rate_percent": 120, "action_repeat": 2, "lives": 3,
        "max_episode_frames": 100, "execution": "serial",
    })
    # Evaluation plays whole games: no training lives, no training cap.
    assert settings == dict(step_frames=2, difficulty=3, patterns=True,
                            powerups=False, normal_enemies=True,
                            kamikaze_enemies=False, spawn_rate_percent=120,
                            execution="serial")


def test_model_evaluation_is_argmax_and_leaves_the_global_rngs_alone():
    from stable_baselines3 import PPO
    from dodge.env import DodgeVecEnv
    from dodge.policies import policy_class, policy_kwargs

    model = PPO(policy_class("mlp"), DodgeVecEnv(n_envs=1, max_episode_frames=20),
                n_steps=16, batch_size=16, policy_kwargs=policy_kwargs("mlp"),
                device="cpu", seed=0)
    torch.manual_seed(5)
    np.random.seed(5)
    torch_before = torch.get_rng_state()
    numpy_before = np.random.get_state()[1].copy()
    first = evaluate_model(model, [7, 8, 9], frame_cap=400, **GAME)
    second = evaluate_model(model, [7, 8, 9], frame_cap=400, **GAME)
    assert list(first.frames) == list(second.frames)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert np.array_equal(np.random.get_state()[1], numpy_before)
