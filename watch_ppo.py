#!/usr/bin/env python3
"""Watch a trained keyboard-only PPO policy play Dodge."""

import argparse
from pathlib import Path
import secrets

import pygame
import torch
from stable_baselines3 import PPO

from dodge.env import DodgeEnv
from dodge.training import newest_snapshot


def window_title(model_name, update, seed, episode=1):
    return f"{model_name} | Update {update} | Episode {episode} | Seed {seed}"


def reload_if_newer(model, live_dir, current_update):
    """Adopt the newest published snapshot above `current_update`.

    Called only between episodes. A snapshot that will not load leaves the
    running policy untouched and the update number unchanged, so the next
    death tries again rather than the window dying with the file.
    """
    if live_dir is None:
        return model, current_update, None
    found = newest_snapshot(live_dir, after=current_update)
    if not found:
        return model, current_update, None
    path, update = found
    try:
        return PPO.load(path, device="cpu"), update, None
    except Exception as error:
        return model, current_update, str(error)


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
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--normal-enemies", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--kamikaze-enemies", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--spawn-rate-percent", type=int, default=100)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--sample", action="store_true",
                        help="sample actions instead of choosing the most likely action")
    parser.add_argument("--field", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="draw a velocity-flow policy's danger field as the "
                             "background (ignored by other architectures)")
    parser.add_argument("--frames", type=int,
                        help="exit after this many rendered game frames")
    parser.add_argument("--live-dir", type=Path, default=None,
                        help="directory of published update-XXXXXXXX.zip "
                             "snapshots to reload from between episodes")
    parser.add_argument("--current-update", type=int, default=-1,
                        help="the update already loaded, so only newer "
                             "snapshots are adopted")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    model_name = args.model_name or args.model.stem
    seed = args.seed
    current_update = args.current_update if args.current_update >= 0 else args.update
    env = DodgeEnv(
        render_mode="human",
        scale=args.scale,
        caption=window_title(model_name, current_update, seed),
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns,
        powerups=args.powerups,
        normal_enemies=args.normal_enemies,
        kamikaze_enemies=args.kamikaze_enemies,
        spawn_rate_percent=args.spawn_rate_percent,
        action_repeat=args.action_repeat,
        # Watching is not training: an episode runs until the agent dies rather
        # than being cut short to keep episodes turning over.
        max_episode_frames=None,
    )
    observation, _ = env.reset(seed=seed)
    # Only velocity-flow has a field to show. Computed once per decision, left
    # in place for the four frames the environment renders before deciding
    # again, so watching costs one extra forward pass per decision rather than
    # one per frame.
    # Detected by asking whether the extractor can produce a field, rather
    # than by architecture name: that keeps this working for anything that
    # exposes one, and does not assume the model has a policy at all.
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    field_source = extractor if args.field and hasattr(extractor, "field") else None
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

        if field_source is not None:
            with torch.no_grad():
                tensor, _ = model.policy.obs_to_tensor(observation)
                env.backend.set_field(field_source.field(tensor)[0, 0].cpu().numpy())
        action, _ = model.predict(observation, deterministic=not args.sample)
        observation, _, terminated, truncated, info = env.step(action)
        frames += info["survived_frames"]
        if args.frames is not None and frames >= args.frames:
            running = False
        if terminated or truncated:
            episode += 1
            model, new_update, error = reload_if_newer(
                model, args.live_dir, current_update)
            if error:
                print(f"Keeping update {current_update}: {error}")
            elif new_update != current_update:
                current_update = new_update
                # The extractor belongs to the old policy object.
                extractor = getattr(getattr(model, "policy", None),
                                    "features_extractor", None)
                field_source = (extractor if args.field
                                and hasattr(extractor, "field") else None)
            seed = secrets.randbelow(2**31 - 1)
            observation, _ = env.reset(seed=seed)
            pygame.display.set_caption(
                window_title(model_name, current_update, seed, episode + 1)
            )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
