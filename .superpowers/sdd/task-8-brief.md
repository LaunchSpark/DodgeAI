### Task 8: Backend protocol, NullBackend, and the machine

The machine owns input state, memory and the Lua runtime, and steps exactly one frame. The headless smoke test at the end is the moment the real game first runs.

**Files:**
- Create: `pico8/backends.py`, `pico8/api.py`, `pico8/machine.py`, `tests/test_machine.py`

**Interfaces:**
- Consumes: `pico8.cart.read_rom`, `pico8.cart.decompress_code`, `pico8.cart.sprite_sheet`, `pico8.preprocess.preprocess`, `pico8/prelude.lua`
- Produces:
  - `pico8.backends.Backend` — methods `cls(col)`, `pset(x,y,c)`, `rect(x0,y0,x1,y1,c)`, `rectfill(x0,y0,x1,y1,c)`, `circfill(x,y,r,c)`, `ovalfill(x0,y0,x1,y1,c)`, `line(x0,y0,x1,y1,c)`, `spr(n,x,y,w,h)`, `print_(text,x,y,c)`, `pal(c0,c1,mode)`, `fillp(pattern)`, `camera(x,y)`, `flip()`
  - `pico8.backends.NullBackend()`
  - `pico8.machine.Machine(rom, code, backend, seed=0)` with `.step()`, `.set_buttons(mask: int)`, `.set_mouse(x, y, buttons)`, `.globals`, `.frame`
  - `pico8.machine.boot(js_path=Path("logic.js"), backend=None, seed=0) -> Machine`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_machine.py
from pathlib import Path
import pytest

pytest.importorskip("lupa")
from pico8 import machine
from pico8.backends import NullBackend

pytestmark = pytest.mark.skipif(not Path("logic.js").exists(), reason="logic.js not present")


@pytest.fixture(scope="module")
def booted():
    return machine.boot(backend=NullBackend(), seed=1)


def test_cart_boots_and_defines_its_entry_points(booted):
    g = booted.globals
    assert g._draw is not None
    assert g.updategame is not None
    assert g.drawgame is not None


def test_stepping_runs_frames_without_error(booted):
    start = booted.frame
    for _ in range(120):
        booted.step()
    assert booted.frame == start + 120


def test_gameplay_state_advances_and_kills_the_idle_player():
    m = machine.boot(backend=NullBackend(), seed=3)
    m.start_game()
    assert m.globals.isdead is False
    for _ in range(3600):          # a minute of standing still
        m.step()
        if m.globals.isdead:
            break
    assert m.globals.isdead is True, "an idle player should eventually be hit"


def test_enemies_actually_spawn():
    m = machine.boot(backend=NullBackend(), seed=5)
    m.start_game()
    for _ in range(600):
        m.step()
        if len(m.globals.enemies) > 0:
            break
    assert len(m.globals.enemies) > 0


def test_same_seed_gives_the_same_trajectory():
    def run(seed):
        m = machine.boot(backend=NullBackend(), seed=seed)
        m.start_game()
        trace = []
        for _ in range(300):
            m.step()
            trace.append((round(m.globals.px, 6), round(m.globals.py, 6),
                          len(m.globals.enemies)))
        return trace

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_buttons_move_the_player():
    m = machine.boot(backend=NullBackend(), seed=7)
    m.start_game()
    m.set_buttons(1 << 1)          # hold right
    x0 = m.globals.px
    for _ in range(20):
        m.step()
    assert m.globals.px > x0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_machine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pico8.backends'`

- [ ] **Step 3: Implement the backends**

```python
# pico8/backends.py
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
```

- [ ] **Step 4: Implement the Python-backed API**

```python
# pico8/api.py
"""PICO-8 builtins that need Python state: input, memory and drawing.

Everything here is installed as a Lua global. Pure-Lua builtins live in
prelude.lua instead.
"""

# PICO-8 button repeat: first repeat after 15 frames, then every 4.
BTNP_DELAY = 15
BTNP_INTERVAL = 4


def _num(value, default=0):
    """Lua passes nil for omitted arguments."""
    return default if value is None else value


def install(machine, lua):
    g = lua.globals()

    def btn(i=None):
        if i is None:
            return machine.buttons
        return bool(machine.buttons & (1 << int(i)))

    def btnp(i=None):
        if i is None:
            return machine.btnp_mask
        return bool(machine.btnp_mask & (1 << int(i)))

    def stat(n):
        n = int(n)
        if n == 32:
            return machine.mouse_x
        if n == 33:
            return machine.mouse_y
        if n == 34:
            return machine.mouse_buttons
        return 0

    def peek(addr):
        return machine.mem[int(addr) & 0xFFFF]

    def poke(addr, value):
        machine.mem[int(addr) & 0xFFFF] = int(value) & 0xFF

    def time():
        return machine.frame / 60.0

    def noop(*args):
        return None

    b = machine.backend
    g.btn = btn
    g.btnp = btnp
    g.stat = stat
    g.peek = peek
    g.poke = poke
    g.time = time
    g.t = time
    g.sfx = noop
    g.music = noop
    g.printh = noop

    g.cls = lambda col=None: b.cls(int(_num(col)))
    g.pset = lambda x, y, c=None: b.pset(x, y, c)
    g.rect = lambda x0, y0, x1, y1, c=None: b.rect(x0, y0, x1, y1, c)
    g.rectfill = lambda x0, y0, x1, y1, c=None: b.rectfill(x0, y0, x1, y1, c)
    g.circfill = lambda x, y, r=None, c=None: b.circfill(x, y, _num(r, 4), c)
    g.ovalfill = lambda x0, y0, x1, y1, c=None: b.ovalfill(x0, y0, x1, y1, c)
    g.line = lambda x0, y0, x1, y1, c=None: b.line(x0, y0, x1, y1, c)
    g.spr = lambda n, x, y, w=None, h=None, *rest: b.spr(int(n), x, y, _num(w, 1), _num(h, 1))
    g.print = lambda text, x=None, y=None, c=None: b.print_(text, _num(x), _num(y), c)
    g.pal = lambda c0=None, c1=None, mode=None: b.pal(c0, c1, int(_num(mode)))
    g.palt = noop
    g.fillp = lambda pattern=None: b.fillp(pattern)
    g.camera = lambda x=None, y=None: b.camera(_num(x), _num(y))
    g.flip = lambda: machine.flip()
```

- [ ] **Step 5: Implement the machine**

```python
# pico8/machine.py
"""Boots the cart and steps it one frame at a time."""

from pathlib import Path

from lupa import LuaRuntime

from . import api, cart
from .backends import NullBackend
from .preprocess import preprocess

PRELUDE = Path(__file__).with_name("prelude.lua")

# smallsettings indices used by the cart: [theme, difficulty, patterns, powerups]
DIFFICULTY = {"normal": 1, "hard": 2, "easy": 3}


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
                                 (held - api.BTNP_DELAY) % api.BTNP_INTERVAL == 0):
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_machine.py -v`
Expected: 6 passed

If `test_gameplay_state_advances_and_kills_the_idle_player` fails, check that `smallsettings` is populated — `updatesettings()` runs once at cart load and fills it. If enemies never spawn, print `m.globals.spd` and `m.globals.est` after `start_game()`; both must be non-nil.

- [ ] **Step 7: Commit**

```bash
git add pico8/backends.py pico8/api.py pico8/machine.py tests/test_machine.py
git commit -m "feat: PICO-8 machine running the cart headless"
```

---

