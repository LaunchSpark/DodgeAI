"""Boots the cart and steps it one frame at a time."""

from pathlib import Path

from lupa.luajit21 import LuaRuntime

from . import api, cart
from .backends import NullBackend
from .preprocess import preprocess

PRELUDE = Path(__file__).with_name("prelude.lua")


class Machine:
    def __init__(self, rom, code, backend, seed=0):
        self.rom = rom
        self.backend = backend
        self.sheet = cart.sprite_sheet(rom)
        self.mem = bytearray(0x10000)
        self.frame = 0

        self.buttons = 0
        self.btnp_mask = 0
        self._held = [0] * 6
        self.mouse_x, self.mouse_y, self.mouse_buttons = 64, 64, 0

        if hasattr(backend, "load_sheet"):
            backend.load_sheet(self.sheet)

        self.lua = LuaRuntime(unpack_returned_tuples=True)
        api.install(self, self.lua)
        self.lua.execute(PRELUDE.read_text(encoding="utf-8"))
        self.lua.execute(f"srand({seed})")
        self.lua.execute(preprocess(code))
        self.globals = self.lua.globals()

    # -- input ------------------------------------------------------------
    def set_buttons(self, mask):
        self.buttons = mask & 0x3F

    def set_mouse(self, x, y, buttons):
        self.mouse_x, self.mouse_y, self.mouse_buttons = x, y, buttons

    def _update_btnp(self):
        """PICO-8 btnp: true on the press frame, then repeating while held."""
        mask = 0
        for i in range(6):
            if self.buttons & (1 << i):
                self._held[i] += 1
                held = self._held[i]
                if held == 1 or (held > api.BTNP_DELAY and
                                 (held - 1 - api.BTNP_DELAY) % api.BTNP_INTERVAL == 0):
                    mask |= 1 << i
            else:
                self._held[i] = 0
        self.btnp_mask = mask

    # -- stepping ---------------------------------------------------------
    def flip(self):
        """Called by the cart's own flip(); also ends a frame during boot."""
        self.backend.flip()

    def step(self):
        self.frame += 1
        self._update_btnp()
        if self.globals._update60 is not None:
            self.globals._update60()
        self.globals._draw()
        self.backend.flip()

    def start_game(self, difficulty=2, patterns=True, powerups=True):
        """Skip the menus and begin a round immediately.

        ``difficulty`` uses the cart's native values: 1 easy, 2 normal, 3 hard.
        A freshly constructed Machine gives training a fully independent round;
        the cart's own reset intentionally carries some state across restarts.
        """
        if difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1 (easy), 2 (normal), or 3 (hard)")
        g = self.globals
        g.smallsettings[2] = difficulty
        g.smallsettings[3] = bool(patterns)
        g.smallsettings[4] = bool(powerups)
        g._upd, g._drw = g.updategame, g.drawgame
        g.reset()
        # Make the native spawn check succeed on the first gameplay frame.
        # spawnenemies() resets ed to zero, preserving its usual cadence after that.
        g.ed = g.est * 60 * (2 - g.frz)
        # Mouse mode teleports the player to stat(32/33). Training uses the
        # real keyboard movement model with velocity and friction.
        g.input = 1
        g.prevxpos, g.prevypos = self.mouse_x, self.mouse_y

    def install_training_hooks(self):
        """Count game events without modifying the generated cart source."""
        self.lua.execute("""
            training_events = {
                enemies_destroyed = 0,
                explosion_pickups = 0,
                freeze_pickups = 0,
                shrink_pickups = 0
            }

            local original_collide = collide
            function collide(enemy)
                if enemy.p == 2 then
                    training_events.explosion_pickups = training_events.explosion_pickups + 1
                elseif enemy.p == 3 then
                    training_events.freeze_pickups = training_events.freeze_pickups + 1
                elseif enemy.p == 4 then
                    training_events.shrink_pickups = training_events.shrink_pickups + 1
                end
                return original_collide(enemy)
            end

            local original_difficultycurve = difficultycurve
            function difficultycurve(half)
                if half then
                    training_events.enemies_destroyed = training_events.enemies_destroyed + 1
                end
                return original_difficultycurve(half)
            end
        """)


def boot(js_path=Path("logic.js"), backend=None, seed=0):
    rom, _ = cart.read_rom(Path(js_path))
    code = cart.decompress_code(rom)
    return Machine(rom, code, backend or NullBackend(), seed=seed)
