#!/usr/bin/env python3
"""Watch a trained keyboard-only PPO policy play Dodge."""

import argparse
from pathlib import Path
import secrets

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from dodge.display import GameWindow
from dodge.env import make_engine
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


def field_source_of(model, enabled):
    """The extractor that can draw a danger field, or None.

    Detected by asking whether the extractor can produce a field, rather than
    by architecture name: that keeps this working for anything that exposes
    one, and does not assume the model has a policy at all.
    """
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    return extractor if enabled and hasattr(extractor, "field") else None


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
    window = GameWindow(scale=args.scale,
                        caption=window_title(model_name, current_update, seed))
    # One frame per step, holding each decision for --action-repeat frames
    # here, so every frame is drawn even when decisions are coarser.
    engine = make_engine(
        step_frames=1, pixels=True, execution="serial",
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns, powerups=args.powerups,
        normal_enemies=args.normal_enemies, kamikaze_enemies=args.kamikaze_enemies,
        spawn_rate_percent=args.spawn_rate_percent,
    )
    frame = engine.reset_batch(np.array([seed], dtype=np.uint32))
    # Only velocity-flow has a field to show. Computed once per decision and
    # left in place for the frames it is held, so watching costs one extra
    # forward pass per decision rather than one per frame.
    field_source = field_source_of(model, args.field)
    clock = pygame.time.Clock()
    episode = frames = held = 0
    action = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                running = False
        if not running:
            break

        if held == 0:
            observation = engine.observe_velocity()["observation"][0]
            if field_source is not None:
                with torch.no_grad():
                    tensor, _ = model.policy.obs_to_tensor(observation)
                    window.set_field(field_source.field(tensor)[0, 0].cpu().numpy())
            action, _ = model.predict(observation, deterministic=not args.sample)
            held = args.action_repeat
        frame = engine.step_pixels(np.array([int(action)], dtype=np.uint8))
        held -= 1
        window.show(frame["pixels"][0])
        clock.tick(60)
        frames += 1
        if args.frames is not None and frames >= args.frames:
            running = False
        if frame["done"][0]:
            episode += 1
            held = 0
            model, new_update, error = reload_if_newer(
                model, args.live_dir, current_update)
            if error:
                print(f"Keeping update {current_update}: {error}")
            elif new_update != current_update:
                current_update = new_update
                # The extractor belongs to the old policy object.
                field_source = field_source_of(model, args.field)
            seed = secrets.randbelow(2**31 - 1)
            frame = engine.reset_lanes(np.array([0], dtype=np.uint32),
                                       np.array([seed], dtype=np.uint32))
            window.set_caption(window_title(model_name, current_update, seed, episode + 1))
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
