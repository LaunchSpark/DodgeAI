"""Drawing targets for the PICO-8 machine.

The machine always calls every drawing entry point. NullBackend discards the
output rather than skipping the call, because the cart's draw path consumes
randomness from the same stream as gameplay -- see spawntrail() inside
drawgame(). Skipping draws would change which enemies spawn.
"""


class Backend:
    """Interface every backend implements. Coordinates are PICO-8 pixels."""

    def cls(self, col): raise NotImplementedError
    def pset(self, x, y, c): raise NotImplementedError
    def rect(self, x0, y0, x1, y1, c): raise NotImplementedError
    def rectfill(self, x0, y0, x1, y1, c): raise NotImplementedError
    def circfill(self, x, y, r, c): raise NotImplementedError
    def ovalfill(self, x0, y0, x1, y1, c): raise NotImplementedError
    def line(self, x0, y0, x1, y1, c): raise NotImplementedError
    def spr(self, n, x, y, w, h): raise NotImplementedError
    def print_(self, text, x, y, c): raise NotImplementedError
    def pal(self, c0, c1, mode): raise NotImplementedError
    def fillp(self, pattern): raise NotImplementedError
    def camera(self, x, y): raise NotImplementedError
    def flip(self): raise NotImplementedError


class NullBackend(Backend):
    """Accepts and discards every draw call."""

    def cls(self, col): pass
    def pset(self, x, y, c): pass
    def rect(self, x0, y0, x1, y1, c): pass
    def rectfill(self, x0, y0, x1, y1, c): pass
    def circfill(self, x, y, r, c): pass
    def ovalfill(self, x0, y0, x1, y1, c): pass
    def line(self, x0, y0, x1, y1, c): pass
    def spr(self, n, x, y, w, h): pass
    def print_(self, text, x, y, c): pass
    def pal(self, c0, c1, mode): pass
    def fillp(self, pattern): pass
    def camera(self, x, y): pass
    def flip(self): pass
