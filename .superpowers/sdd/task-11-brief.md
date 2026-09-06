### Task 11: The playable entry point

**Files:**
- Create: `play.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `pico8.machine.boot`, `pico8.backends.PygameBackend`
- Produces: `python play.py` — a playable window

- [ ] **Step 1: Write the entry point**

The window must exist *before* the cart is booted: the cart's top-level code runs a 66-frame splash animation calling `flip()`, so the backend has to be ready to present.

```python
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
    args = parser.parse_args()

    backend = PygameBackend(scale=args.scale)
    machine = boot(js_path=args.js, backend=backend, seed=args.seed)
    clock = pygame.time.Clock()

    running = True
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

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Handle events during the boot splash**

The splash calls `flip()` 66 times before `main()` reaches its loop, so the window would be unresponsive and Windows may mark it "not responding". Add an event pump to `PygameBackend.flip`, just before `pygame.display.flip()`:

```python
        pygame.event.pump()
```

- [ ] **Step 3: Play it**

Run: `python play.py`

Expected, in order:
1. A CoolMathGames splash animation (an expanding oval and circle) for about a second
2. The title screen with "press X" and the three credited names
3. `X` starts a round; the player is a white circle, enemies grow from the corners
4. Arrow keys move with visible momentum — the player drifts after keys are released
5. `O` from the title screen opens Settings; left/right change theme, and the colours change immediately

Troubleshooting:

- **Everything renders blue.** `pal(c, c1, 1)` is not reaching the surface palette. Check that `_apply_display_palette` runs in `flip`.
- **The player does not respond to keys.** The cart picks its control mode from `input = anybtn() and 1 or anymouse() and 0 or input`, so a moving mouse can hold it in mouse mode. Move the mouse off the window and press a direction key.
- **Sprites are missing but shapes draw.** `load_sheet` was not called — confirm `PygameBackend` is passed to `boot` before the machine is constructed.

- [ ] **Step 4: Update the README**

```markdown
# DodgeAI

A local, editable copy of the PICO-8 game `dodge`, extracted from its HTML5
export, plus (eventually) a training environment for it.

## Play it

    pip install -r requirements.txt
    python play.py

Arrows or WASD to move, Z/C and X/V for the O and X buttons, Escape to quit.

## Re-extract the cart

`dodge.lua` and `dodge.p8` are generated from `logic.js`:

    python tools/extract_cart.py

## Layout

| Path | What |
| --- | --- |
| `logic.js` | the original HTML5 export (PICO-8 runtime + 32KB cart ROM) |
| `dodge.lua` | the game's Lua source, extracted |
| `dodge.p8` | the full cartridge in PICO-8's text format |
| `pico8/` | a partial PICO-8 runtime in Python, hosting the real Lua under LuaJIT |
| `tools/` | the cart extractor |
| `docs/superpowers/` | design spec and implementation plan |
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add play.py pico8/backends.py README.md
git commit -m "feat: playable local build"
```

---

## Done when

- `python play.py` opens a window, plays the splash, shows the menu, and a round is playable with the keyboard
- The settings screen changes theme, difficulty, patterns and powerups, and the changes take effect
- `python -m pytest tests/ -v` is green
- `python tools/extract_cart.py` still reproduces `dodge.lua` and `dodge.p8` byte-for-byte
- `dodge.lua` has not been modified
