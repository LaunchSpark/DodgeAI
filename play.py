#!/usr/bin/env python3
"""Play the extracted PICO-8 cart locally.

    python play.py                 # normal
    python play.py --scale 4       # smaller window
    python play.py --seed 3        # fixed RNG seed

Controls: arrows or WASD to move, Z/C for O, X/V for X, Escape to quit.
The mouse also works -- the cart switches control mode on its own, exactly
as the browser build does.
"""

import argparse
import sys

import pygame

from pico8.backends import PygameBackend
from pico8.machine import boot

# PICO-8 button index -> keys that press it
KEYMAP = {
    0: (pygame.K_LEFT, pygame.K_a),
    1: (pygame.K_RIGHT, pygame.K_d),
    2: (pygame.K_UP, pygame.K_w),
    3: (pygame.K_DOWN, pygame.K_s),
    4: (pygame.K_z, pygame.K_c),
    5: (pygame.K_x, pygame.K_v),
}


def button_mask(pressed):
    mask = 0
    for index, keys in KEYMAP.items():
        if any(pressed[k] for k in keys):
            mask |= 1 << index
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=5, help="window scale (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--js", default="logic.js", help="path to the HTML5 export")
    parser.add_argument("--frames", type=int, default=None,
                         help="run this many frames of the main loop then exit "
                              "(default: unlimited, i.e. run until closed)")
    args = parser.parse_args()

    backend = PygameBackend(scale=args.scale)
    machine = boot(js_path=args.js, backend=backend, seed=args.seed)
    clock = pygame.time.Clock()

    running = True
    frames_run = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        pressed = pygame.key.get_pressed()
        machine.set_buttons(button_mask(pressed))

        mx, my = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed()
        machine.set_mouse(mx // args.scale, my // args.scale,
                          (1 if buttons[0] else 0) | (2 if buttons[2] else 0))

        machine.step()
        clock.tick(60)

        frames_run += 1
        if args.frames is not None and frames_run >= args.frames:
            running = False

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
