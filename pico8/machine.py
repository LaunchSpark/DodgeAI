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

    def start_game(self):
        """Skip the menus and begin a round immediately."""
        g = self.globals
        g._upd, g._drw = g.updategame, g.drawgame
        g.reset()


def boot(js_path=Path("logic.js"), backend=None, seed=0):
    rom, _ = cart.read_rom(Path(js_path))
    code = cart.decompress_code(rom)
    return Machine(rom, code, backend or NullBackend(), seed=seed)
