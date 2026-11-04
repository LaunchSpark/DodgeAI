#!/usr/bin/env python3
"""Watch a hand-written heuristic (dodge.heuristics) play Dodge.

Press O to show or hide the bot's own overlay (its predictions, its plan, any
map it keeps), if it draws one; Esc closes the window."""

import argparse
import secrets

import numpy as np
import pygame

from dodge.display import GameWindow
from dodge.env import make_engine
from dodge.heuristics import BOTS


def window_title(label, seed, episode, frames, overlay):
    shown = "overlay on" if overlay else "O for overlay"
    return f"{label} | Episode {episode} | Seed {seed} | Frame {frames:,} | {shown}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bot", choices=sorted(BOTS))
    parser.add_argument("--seed", type=int, default=None,
                        help="first game's seed (random if omitted)")
    parser.add_argument("--next-seed", choices=("random", "increment"), default="random",
                        help="seed for later games: random, or one more than the last")
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="hard")
    parser.add_argument("--patterns", action="store_true")
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normal-enemies", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--kamikaze-enemies", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--spawn-rate-percent", type=int, default=100)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False,
                        help="start with the bot's overlay shown (O toggles it)")
    args = parser.parse_args()

    label, make_bot = BOTS[args.bot]
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
    window = GameWindow(scale=args.scale)
    # The heuristics decide every frame, from the engine's exact state.
    engine = make_engine(
        step_frames=1, pixels=True, execution="serial",
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns, powerups=args.powerups,
        normal_enemies=args.normal_enemies, kamikaze_enemies=args.kamikaze_enemies,
        spawn_rate_percent=args.spawn_rate_percent,
    )
    pixels = engine.reset_batch(np.array([seed], dtype=np.uint32))["pixels"][0]
    bot = make_bot()
    clock = pygame.time.Clock()
    episode, frames = 1, 0
    show_overlay = args.overlay
    window.set_caption(window_title(label, seed, episode, frames, show_overlay))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                show_overlay = not show_overlay
                window.set_caption(window_title(label, seed, episode, frames, show_overlay))
        if not running:
            break

        state = engine.lane_state(0)
        bot.observe(state)
        action = bot.act(state)
        # Drawn over the frame the decision was made from, so predictions start
        # where things are now.
        overlay = bot.overlay() if show_overlay and hasattr(bot, "overlay") else None
        window.set_field(None if overlay is None else overlay.field)
        window.show(pixels, overlay)
        frame = engine.step_pixels(np.array([action], dtype=np.uint8))
        pixels = frame["pixels"][0]
        clock.tick(args.fps)
        frames += 1
        if frames % 30 == 0:
            window.set_caption(window_title(label, seed, episode, frames, show_overlay))
        if frame["done"][0]:
            print(f"{label}: seed {seed} lasted {frames:,} frames")
            episode += 1
            frames = 0
            seed = seed + 1 if args.next_seed == "increment" else secrets.randbelow(2**31 - 1)
            pixels = engine.reset_lanes(np.array([0], dtype=np.uint32),
                                        np.array([seed], dtype=np.uint32))["pixels"][0]
            bot = make_bot()
            window.set_caption(window_title(label, seed, episode, frames, show_overlay))
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
