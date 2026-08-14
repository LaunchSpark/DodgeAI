#!/usr/bin/env python3
"""Play Dodge from an analytic danger field, or measure how well it does.

    python gradient_bot.py                  # 30 episodes, report survival
    python gradient_bot.py --watch          # play one episode in a window
    python gradient_bot.py --compare MODEL  # against random and a checkpoint

The point of this bot is calibration. Nothing in this project knows what a good
survival time is, so an agent's score cannot be judged. A hand-written
controller establishes that, and doubles as a check on whether following a
danger field is a viable strategy here at all before one is learned.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from dodge.env import DodgeEnv
from dodge.gradient import ControllerConfig, GradientBot

DEFAULT_EPISODES = 30
FIRST_SEED = 90_000


def play(policy, episodes, seed=FIRST_SEED, render=False, **env_kwargs):
    """Run whole episodes and return each one's length in seconds.

    ``policy`` is called with the machine and returns an action. Episodes are
    never truncated: the question is how long the player survives, and a cap
    would answer a different one.
    """
    env = DodgeEnv(max_episode_frames=None,
                   render_mode="human" if render else None,
                   caption="Gradient Bot", **env_kwargs)
    lengths, scores = [], []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            frames = 0
            while True:
                _, _, terminated, truncated, info = env.step(policy(env.machine))
                frames = info["frames"]
                if terminated or truncated:
                    break
            lengths.append(frames / 60.0)
            scores.append(info["score"])
    finally:
        env.close()
    return lengths, scores


def summarise(label, lengths, scores, baseline=None):
    mean = statistics.mean(lengths)
    line = (f"{label:26s} {mean:6.2f}s mean   {statistics.median(lengths):6.2f}s median   "
            f"{min(lengths):5.2f}-{max(lengths):5.2f}s   score {statistics.mean(scores):6.1f}")
    if baseline:
        line += f"   {mean / baseline:5.2f}x random"
    print(line)
    return mean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=FIRST_SEED)
    parser.add_argument("--watch", action="store_true",
                        help="render one episode instead of measuring many")
    parser.add_argument("--compare", type=Path, default=None,
                        help="a PPO checkpoint to measure alongside")
    parser.add_argument("--difficulty", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--patterns", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--horizon", type=int, default=ControllerConfig.horizon)
    args = parser.parse_args(argv)

    env_kwargs = dict(difficulty=args.difficulty, powerups=args.powerups,
                      patterns=args.patterns, action_repeat=args.action_repeat)
    config = ControllerConfig(horizon=args.horizon)
    bot = GradientBot(config, action_repeat=args.action_repeat)

    if args.watch:
        lengths, scores = play(bot.act, 1, args.seed, render=True, **env_kwargs)
        print(f"survived {lengths[0]:.2f}s, score {scores[0]:.1f}")
        return 0

    started = time.perf_counter()
    rng = np.random.default_rng(0)
    random_lengths, random_scores = play(lambda machine: int(rng.integers(9)),
                                         args.episodes, args.seed, **env_kwargs)
    baseline = summarise("random", random_lengths, random_scores)

    bot_lengths, bot_scores = play(bot.act, args.episodes, args.seed, **env_kwargs)
    summarise("gradient bot", bot_lengths, bot_scores, baseline)

    if args.compare:
        from stable_baselines3 import PPO
        from dodge.observe import observe
        model = PPO.load(args.compare, device="cpu")

        def act(machine):
            action, _ = model.predict(observe(machine), deterministic=True)
            return int(action)

        lengths, scores = play(act, args.episodes, args.seed, **env_kwargs)
        summarise(args.compare.stem[:26], lengths, scores, baseline)

    print(f"\n{args.episodes} episodes each, seeds {args.seed}..{args.seed + args.episodes - 1}, "
          f"difficulty {args.difficulty}, {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
