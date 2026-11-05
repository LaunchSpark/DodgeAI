#!/usr/bin/env python3
"""Watch a hand-written heuristic (dodge.heuristics) play Dodge.

Press O to show or hide the bot's own overlay (its predictions, its plan, any
map it keeps), if it draws one; Esc closes the window.

--save-run game.run.gz saves the game (seed, actions, the overlay of every
frame) so render_run.py can turn it into a video later at any speed, size, or
with the overlay on or off, without the bot deciding again. --record game.mp4
saves what the window shows as a video directly. Either one stops after one game
by default; with --headless no window opens and the bot plays as fast as it
can decide."""

import argparse
import os
import secrets

import numpy as np
import pygame

from dodge.display import GameWindow
from dodge.env import make_engine
from dodge.heuristics import BOTS
from dodge.runfile import RunWriter, open_video


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
    parser.add_argument("--record", metavar="PATH", default=None,
                        help="save the window as a video (.mp4), at --fps")
    parser.add_argument("--save-run", metavar="PATH", default=None,
                        help="save the game for render_run.py (.run.gz)")
    parser.add_argument("--games", type=int, default=None,
                        help="stop after this many games (default: 1 when saving, else never)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many frames in all")
    parser.add_argument("--headless", action="store_true",
                        help="no window, no frame limit: for recording")
    args = parser.parse_args()
    saving = bool(args.record or args.save_run)
    games = args.games if args.games is not None else (1 if saving else None)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    label, make_bot = BOTS[args.bot]
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
    window = GameWindow(scale=args.scale)
    writer = None
    if args.record:
        writer = open_video(args.record, args.fps)
    # The heuristics decide every frame, from the engine's exact state.
    settings = dict(
        step_frames=1, pixels=True, execution="serial",
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns, powerups=args.powerups,
        normal_enemies=args.normal_enemies, kamikaze_enemies=args.kamikaze_enemies,
        spawn_rate_percent=args.spawn_rate_percent,
    )
    engine = make_engine(**settings)
    pixels = engine.reset_batch(np.array([seed], dtype=np.uint32))["pixels"][0]
    run = None
    if args.save_run:
        run = RunWriter(args.save_run, dict(settings, bot=args.bot, label=label))
        run.game(seed)
    # headless and saving only the run: nothing needs drawing
    draw = not (args.headless and not args.record)
    bot = make_bot()
    clock = pygame.time.Clock()
    episode, frames, total_frames = 1, 0, 0
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
        # a saved run keeps every frame's overlay, so the render can show it or not
        want = show_overlay or run is not None
        overlay = bot.overlay() if want and hasattr(bot, "overlay") else None
        if run is not None:
            run.frame(action, pixels, overlay)
        if draw:
            shown = overlay if show_overlay else None
            window.set_field(None if shown is None else shown.field)
            window.show(pixels, shown)
        if writer is not None:
            writer.append_data(pygame.surfarray.array3d(window.window).transpose(1, 0, 2))
        frame = engine.step_pixels(np.array([action], dtype=np.uint8))
        pixels = frame["pixels"][0]
        if not args.headless:
            clock.tick(args.fps)
        frames += 1
        total_frames += 1
        if args.max_frames is not None and total_frames >= args.max_frames:
            print(f"{label}: stopped at {args.max_frames:,} frames")
            break
        if frames % 30 == 0:
            window.set_caption(window_title(label, seed, episode, frames, show_overlay))
        if frame["done"][0]:
            print(f"{label}: seed {seed} lasted {frames:,} frames")
            if games is not None and episode >= games:
                break
            episode += 1
            frames = 0
            seed = seed + 1 if args.next_seed == "increment" else secrets.randbelow(2**31 - 1)
            pixels = engine.reset_lanes(np.array([0], dtype=np.uint32),
                                        np.array([seed], dtype=np.uint32))["pixels"][0]
            if run is not None:
                run.game(seed)
            bot = make_bot()
            window.set_caption(window_title(label, seed, episode, frames, show_overlay))
    if writer is not None:
        writer.close()
        print(f"saved {args.record}")
    if run is not None:
        run.close()
        print(f"saved {args.save_run} ({total_frames:,} frames)")
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
