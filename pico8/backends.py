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


import pygame

from .palette import PICO8


class PygameBackend(Backend):
    """Draws to a 128x128 indexed surface, scaled to the window on flip.

    Colour values passed in are PICO-8 palette indices, and the surface stores
    them directly. That makes pal(c, c2, 1) a palette swap rather than a
    per-pixel rewrite -- which matters because the cart draws the entire game
    in colours 12 and 1 and remaps them at the end of every frame.
    """

    SIZE = 128

    def __init__(self, scale=5, caption="dodge"):
        pygame.init()
        self.scale = scale
        self.window = pygame.display.set_mode((self.SIZE * scale, self.SIZE * scale))
        pygame.display.set_caption(caption)

        self.surface = pygame.Surface((self.SIZE, self.SIZE), depth=8)
        self.surface.set_palette(PICO8)
        self.sheet = None

        self.draw_pal = list(range(16))     # pal(c, c1, 0)
        self.display_pal = list(range(16))  # pal(c, c1, 1)
        self.cam_x = self.cam_y = 0
        self.pen = 6
        self.text_queue = []

    # -- helpers ----------------------------------------------------------
    def load_sheet(self, rows):
        """Build a transparent surface from 128x128 palette indices."""
        sheet = pygame.Surface((self.SIZE, self.SIZE), depth=8)
        sheet.set_palette(PICO8)
        for y, row in enumerate(rows):
            for x, index in enumerate(row):
                sheet.set_at((x, y), index)
        sheet.set_colorkey(0)               # PICO-8 treats colour 0 as transparent
        self.sheet = sheet

    def _colour(self, c):
        if c is None:
            c = self.pen
        else:
            self.pen = int(c) & 15
        return self.draw_pal[int(c) & 15]

    def _pt(self, x, y):
        return int(x) - self.cam_x, int(y) - self.cam_y

    # -- Backend ----------------------------------------------------------
    def cls(self, col=0):
        self.surface.fill(self.draw_pal[int(col) & 15])
        # Note: real PICO-8's cls() does not touch the camera offset. See
        # dodge.lua's updatemenu(): camera(shakex, shakey) is called, then
        # cls(12), then _drw() -- if cls() reset the camera here, the
        # screen-shake effect on the pause/settings screen would be
        # silently discarded on every frame it's active.

    def pset(self, x, y, c=None):
        px, py = self._pt(x, y)
        if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
            self.surface.set_at((px, py), self._colour(c))

    def rect(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        pygame.draw.rect(self.surface, colour,
                         pygame.Rect(min(ax, bx), min(ay, by),
                                     abs(bx - ax) + 1, abs(by - ay) + 1), width=1)

    def rectfill(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        pygame.draw.rect(self.surface, colour,
                         pygame.Rect(min(ax, bx), min(ay, by),
                                     abs(bx - ax) + 1, abs(by - ay) + 1))

    def circfill(self, x, y, r=4, c=None):
        colour = self._colour(c)
        px, py = self._pt(x, y)
        radius = int(r)
        if radius <= 0:
            if 0 <= px < self.SIZE and 0 <= py < self.SIZE:
                self.surface.set_at((px, py), colour)
            return
        pygame.draw.circle(self.surface, colour, (px, py), radius)

    def ovalfill(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        ax, ay = self._pt(x0, y0)
        bx, by = self._pt(x1, y1)
        rect = pygame.Rect(min(ax, bx), min(ay, by), abs(bx - ax) + 1, abs(by - ay) + 1)
        pygame.draw.ellipse(self.surface, colour, rect)

    def line(self, x0, y0, x1, y1, c=None):
        colour = self._colour(c)
        pygame.draw.line(self.surface, colour, self._pt(x0, y0), self._pt(x1, y1))

    def spr(self, n, x, y, w=1, h=1):
        if self.sheet is None:
            return
        n = int(n)
        sx, sy = (n % 16) * 8, (n // 16) * 8
        area = pygame.Rect(sx, sy, int(w) * 8, int(h) * 8)
        self.surface.blit(self.sheet, self._pt(x, y), area)

    def print_(self, text, x=0, y=0, c=None):
        self.text_queue.append((str(text), int(x) - self.cam_x,
                                int(y) - self.cam_y, self._colour(c)))

    def pal(self, c0=None, c1=None, mode=0):
        if c0 is None:
            self.draw_pal = list(range(16))
            self.display_pal = list(range(16))
            self._apply_display_palette()
            return
        if int(mode) == 1:
            self.display_pal[int(c0) & 15] = int(c1) & 15
            self._apply_display_palette()
        else:
            self.draw_pal[int(c0) & 15] = int(c1) & 15

    def _apply_display_palette(self):
        self.surface.set_palette([PICO8[self.display_pal[i]] for i in range(16)])

    def fillp(self, pattern=None):
        pass                                 # approximated: solid fills

    def camera(self, x=0, y=0):
        self.cam_x, self.cam_y = int(x), int(y)

    def flip(self):
        self._apply_display_palette()
        # pygame.transform.scale() into a destination surface requires the
        # destination to match the source's format exactly; our indexed
        # 8-bit surface can't be scaled directly into the 32-bit display
        # surface (raises ValueError: "Source and destination surfaces need
        # the same format."). Scale into a fresh surface instead and blit
        # that -- blit converts formats, transform.scale-into-dest does not.
        scaled = pygame.transform.scale(self.surface, self.window.get_size())
        self.window.blit(scaled, (0, 0))
        self._draw_text()
        pygame.display.flip()
        self.text_queue.clear()

    def _draw_text(self):
        pass                                 # implemented in Task 10
