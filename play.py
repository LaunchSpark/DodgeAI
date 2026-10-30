#!/usr/bin/env python3
"""Play Dodge yourself on the native engine.

    python play.py                     # normal difficulty, patterns and powerups
    python play.py --difficulty hard   # the setting training uses
    python play.py --scale 4           # smaller window
    python play.py --seed 3            # fixed game seed

Controls: arrows or WASD to move, X/V or Z/C to start the next game after a
death, Escape to quit.
"""

import argparse
import secrets
import sys

import numpy as np
import pygame

from dodge.actions import ACTION_MASKS, DOWN, LEFT, RIGHT, UP
from dodge.display import GameWindow
from dodge.env import make_engine


# PICO-8 button index -> keys that press it
KEYMAP = {
    0: (pygame.K_LEFT, pygame.K_a),
    1: (pygame.K_RIGHT, pygame.K_d),
    2: (pygame.K_UP, pygame.K_w),
    3: (pygame.K_DOWN, pygame.K_s),
    4: (pygame.K_z, pygame.K_c),
    5: (pygame.K_x, pygame.K_v),
}

RESTART = (1 << 4) | (1 << 5)


def button_mask(pressed):
    mask = 0
    for index, keys in KEYMAP.items():
        if any(pressed[k] for k in keys):
            mask |= 1 << index
    return mask


def action_for(mask):
    """The movement action for a button mask; opposite directions cancel."""
    move = mask & (LEFT | RIGHT | UP | DOWN)
    if move & LEFT and move & RIGHT:
        move &= ~(LEFT | RIGHT)
    if move & UP and move & DOWN:
        move &= ~(UP | DOWN)
    return ACTION_MASKS.index(move)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=5, help="window scale (default 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the first game (default: random)")
    parser.add_argument("--difficulty", choices=("easy", "normal", "hard"), default="normal")
    parser.add_argument("--patterns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--powerups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frames", type=int, default=None,
                        help="run this many frames then exit "
                             "(default: unlimited, i.e. run until closed)")
    args = parser.parse_args()

    window = GameWindow(scale=args.scale)
    engine = make_engine(
        step_frames=1, pixels=True, execution="serial",
        difficulty={"easy": 1, "normal": 2, "hard": 3}[args.difficulty],
        patterns=args.patterns, powerups=args.powerups,
    )
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
    frame = engine.reset_batch(np.array([seed], dtype=np.uint32))
    clock = pygame.time.Clock()

    running = True
    frames_run = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        mask = button_mask(pygame.key.get_pressed())
        if frame["done"][0]:
            # The last frame stays up until the player asks for another game.
            if mask & RESTART:
                seed = secrets.randbelow(2**31 - 1)
                frame = engine.reset_lanes(np.array([0], dtype=np.uint32),
                                           np.array([seed], dtype=np.uint32))
        else:
            frame = engine.step_pixels(np.array([action_for(mask)], dtype=np.uint8))
        window.show(frame["pixels"][0])
        clock.tick(60)

        frames_run += 1
        if args.frames is not None and frames_run >= args.frames:
            running = False

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
