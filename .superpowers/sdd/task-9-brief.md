### Task 9: Pygame rendering

An 8-bit indexed surface is what makes `pal(c, c2, 1)` cheap: the cart draws with palette *indices*, and the display palette is applied by swapping the surface's palette before scaling. Without that the game renders flat blue, because every draw call in `drawgame` uses colour 12 or 1.

**Files:**
- Create: `pico8/palette.py`, `tests/test_pygame_backend.py`
- Modify: `pico8/backends.py`

**Interfaces:**
- Consumes: `pico8.backends.Backend`
- Produces:
  - `pico8.palette.PICO8: list[tuple[int, int, int]]` — 16 RGB triples
  - `pico8.backends.PygameBackend(scale=5)` with `.load_sheet(rows)`, `.surface`, `.window`, and the `Backend` methods

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pygame_backend.py
import os
import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pico8.backends import PygameBackend
from pico8.palette import PICO8


@pytest.fixture
def backend():
    b = PygameBackend(scale=1)
    yield b
    pygame.quit()


def test_palette_has_sixteen_colours():
    assert len(PICO8) == 16
    assert PICO8[0] == (0, 0, 0)
    assert PICO8[7] == (255, 241, 232)


def test_cls_fills_with_the_index(backend):
    backend.cls(12)
    assert backend.surface.get_at((0, 0))[:3] == PICO8[12]
    assert backend.surface.get_at((127, 127))[:3] == PICO8[12]


def test_rectfill_uses_the_requested_index(backend):
    backend.cls(0)
    backend.rectfill(2, 2, 5, 5, 8)
    assert backend.surface.get_at((3, 3))[:3] == PICO8[8]
    assert backend.surface.get_at((10, 10))[:3] == PICO8[0]


def test_camera_offsets_subsequent_draws(backend):
    backend.cls(0)
    backend.camera(10, 0)
    backend.rectfill(10, 0, 12, 2, 8)
    assert backend.surface.get_at((0, 0))[:3] == PICO8[8]


def test_draw_palette_remaps_colours(backend):
    backend.cls(0)
    backend.pal(8, 12, 0)
    backend.rectfill(0, 0, 4, 4, 8)
    assert backend.surface.get_at((1, 1))[:3] == PICO8[12]


def test_display_palette_does_not_change_the_drawn_index(backend):
    backend.cls(0)
    backend.rectfill(0, 0, 4, 4, 12)
    backend.pal(12, 3, 1)
    # the surface still holds index 12; the swap happens at flip time
    assert backend.surface.get_at((1, 1))[:3] == PICO8[3]


def test_spr_blits_from_the_sheet(backend):
    sheet = [[0] * 128 for _ in range(128)]
    sheet[0][0] = 9              # sprite 0, top-left pixel
    backend.load_sheet(sheet)
    backend.cls(0)
    backend.spr(0, 20, 30, 1, 1)
    assert backend.surface.get_at((20, 30))[:3] == PICO8[9]


def test_colour_zero_is_transparent_in_sprites(backend):
    sheet = [[0] * 128 for _ in range(128)]
    backend.load_sheet(sheet)
    backend.cls(7)
    backend.spr(0, 0, 0, 1, 1)
    assert backend.surface.get_at((2, 2))[:3] == PICO8[7]


def test_backends_produce_identical_trajectories():
    """The two backends must simulate identically.

    This is not a formality: drawgame() calls spawntrail(), which draws from
    the same RNG stream as gameplay. If rendering consumed randomness that
    headless did not, the trained policy and the watched policy would be
    playing different games.
    """
    import pathlib
    if not pathlib.Path("logic.js").exists():
        pytest.skip("logic.js not present")

    from pico8.machine import boot
    from pico8.backends import NullBackend

    def run(backend):
        m = boot(backend=backend, seed=21)
        m.start_game()
        trace = []
        for _ in range(900):        # long enough for powerups to appear
            m.step()
            trace.append((round(m.globals.px, 6), round(m.globals.py, 6),
                          round(m.globals.score, 6), len(m.globals.enemies)))
        return trace

    headless = run(NullBackend())
    rendered = run(PygameBackend(scale=1))
    pygame.quit()
    assert headless == rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pygame_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pico8.palette'`

- [ ] **Step 3: Implement the palette**

```python
# pico8/palette.py
"""The 16 PICO-8 colours, in index order."""

PICO8 = [
    (0, 0, 0),         # 0  black
    (29, 43, 83),      # 1  dark blue
    (126, 37, 83),     # 2  dark purple
    (0, 135, 81),      # 3  dark green
    (171, 82, 54),     # 4  brown
    (95, 87, 79),      # 5  dark grey
    (194, 195, 199),   # 6  light grey
    (255, 241, 232),   # 7  white
    (255, 0, 77),      # 8  red
    (255, 163, 0),     # 9  orange
    (255, 236, 39),    # 10 yellow
    (0, 228, 54),      # 11 green
    (41, 173, 255),    # 12 blue
    (131, 118, 156),   # 13 lavender
    (255, 119, 168),   # 14 pink
    (255, 204, 170),   # 15 peach
]
```

- [ ] **Step 4: Implement the backend**

Append to `pico8/backends.py`:

```python
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
        self.cam_x = self.cam_y = 0

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
        pygame.transform.scale(self.surface, self.window.get_size(), self.window)
        self._draw_text()
        pygame.display.flip()
        self.text_queue.clear()

    def _draw_text(self):
        pass                                 # implemented in Task 10
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pygame_backend.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add pico8/palette.py pico8/backends.py tests/test_pygame_backend.py
git commit -m "feat: pygame backend with indexed surface and palette swaps"
```

---

