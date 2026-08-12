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
