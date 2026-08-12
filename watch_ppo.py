#!/usr/bin/env python3
"""Watch a trained keyboard-only PPO policy play Dodge."""

import argparse
from pathlib import Path
import secrets

import pygame
from stable_baselines3 import PPO

from dodge.env import DodgeEnv


def window_title(model_name, update, seed, episode=1):
    return f"{model_name} | Update {update} | Episode {episode} | Seed {seed}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update", type=int, default=0,
                        help="policy update number shown in the window title")
    parser.add_argument("--model-name", default=None,
                        help="active model name shown in the window title")
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="easy")
    parser.add_argument("--patterns", action="store_true")
    parser.add_argument("--powerups", action="store_true")
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--sample", action="store_true",
                        help="sample actions instead of choosing the most likely action")
    parser.add_argument("--frames", type=int,
                        help="exit after this many rendered game frames")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    model_name = args.model_name or args.model.stem
    seed = args.seed
    env = DodgeEnv(
        render_mode="human",
        scale=args.scale,
        caption=window_title(model_name, args.update, seed),
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns,
        powerups=args.powerups,
        action_repeat=args.action_repeat,
        # Watching is not training: an episode runs until the agent dies rather
        # than being cut short to keep episodes turning over.
        max_episode_frames=None,
    )
    observation, _ = env.reset(seed=seed)
    running = True
    episode = 0
    frames = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        action, _ = model.predict(observation, deterministic=not args.sample)
        observation, _, terminated, truncated, info = env.step(action)
        frames += info["survived_frames"]
        if args.frames is not None and frames >= args.frames:
            running = False
        if terminated or truncated:
            episode += 1
            seed = secrets.randbelow(2**31 - 1)
            observation, _ = env.reset(seed=seed)
            pygame.display.set_caption(
                window_title(model_name, args.update, seed, episode + 1)
            )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
