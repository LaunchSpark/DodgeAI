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

from dodge.env import make_engine, play_games
from dodge.gradient import ControllerConfig, GradientBot

DEFAULT_EPISODES = 30
FIRST_SEED = 90_000


def play(policy, episodes, seed=FIRST_SEED, action_repeat=4, **game_settings):
    """Play whole games, one per seed, and return each one's length in seconds.

    ``policy(observations, lane_state)`` returns one action per game. Games are
    never truncated: the question is how long the player survives, and a cap
    would answer a different one.
    """
    results = play_games(policy, range(seed, seed + episodes),
                         step_frames=action_repeat, **game_settings)
    return ([result["frames"] / 60.0 for result in results],
            [result["score"] for result in results])


def watch(bot, seed, action_repeat, **game_settings):
    """Play one game in a window and return its length in seconds."""
    import pygame
    from dodge.display import GameWindow

    window = GameWindow(caption="Gradient Bot")
    engine = make_engine(step_frames=1, pixels=True, execution="serial", **game_settings)
    frame = engine.reset_batch(np.array([seed], dtype=np.uint32))
    clock = pygame.time.Clock()
    frames = held = action = 0
    while not frame["done"][0]:
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            break
        if held == 0:
            action, held = bot.act(engine.lane_state(0)), action_repeat
        frame = engine.step_pixels(np.array([action], dtype=np.uint8))
        held -= 1
        frames += 1
        window.show(frame["pixels"][0])
        clock.tick(60)
    score = float(engine.observe_velocity()["score"][0])
    pygame.quit()
    return frames / 60.0, score


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

    game_settings = dict(difficulty=args.difficulty, powerups=args.powerups,
                         patterns=args.patterns)
    config = ControllerConfig(horizon=args.horizon)
    bot = GradientBot(config, action_repeat=args.action_repeat)

    if args.watch:
        length, score = watch(bot, args.seed, args.action_repeat, **game_settings)
        print(f"survived {length:.2f}s, score {score:.1f}")
        return 0

    started = time.perf_counter()
    rng = np.random.default_rng(0)
    random_lengths, random_scores = play(
        lambda observations, _: rng.integers(9, size=len(observations)),
        args.episodes, args.seed, args.action_repeat, **game_settings)
    baseline = summarise("random", random_lengths, random_scores)

    bot_lengths, bot_scores = play(bot.policy, args.episodes, args.seed,
                                   args.action_repeat, **game_settings)
    summarise("gradient bot", bot_lengths, bot_scores, baseline)

    if args.compare:
        from stable_baselines3 import PPO
        model = PPO.load(args.compare, device="cpu")

        def act(observations, _):
            actions, _ = model.predict(observations, deterministic=True)
            return actions

        lengths, scores = play(act, args.episodes, args.seed, args.action_repeat,
                               **game_settings)
        summarise(args.compare.stem[:26], lengths, scores, baseline)

    print(f"\n{args.episodes} episodes each, seeds {args.seed}..{args.seed + args.episodes - 1}, "
          f"difficulty {args.difficulty}, {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
