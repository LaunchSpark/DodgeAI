"""PICO-8 builtins that need Python state: input, memory and drawing.

Everything here is installed as a Lua global. Pure-Lua builtins live in
prelude.lua instead.
"""

# PICO-8 button repeat: first repeat after 15 frames, then every 4.
BTNP_DELAY = 15
BTNP_INTERVAL = 4

# Everything a backend that declares draws = False replaces with a Lua no-op.
# None of these return a value the cart reads, so removing them changes only
# what reaches the screen.
DRAWING_BUILTINS = ("cls", "pset", "rect", "rectfill", "circfill", "ovalfill",
                    "line", "spr", "print", "pal", "palt", "fillp", "camera")


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

    def dget(index):
        return machine.cart_data[int(index) & 0x3F]

    def dset(index, value):
        machine.cart_data[int(index) & 0x3F] = float(value)

    def noop(*args):
        return None

    g.btn = btn
    g.btnp = btnp
    g.stat = stat
    g.peek = peek
    g.poke = poke
    g.time = time
    g.t = time
    g.cartdata = noop
    g.dget = dget
    g.dset = dset
    g.menuitem = noop
    g.cursor = noop
    g.sfx = noop
    g.music = noop
    g.printh = noop

    # A headless backend would discard all of this anyway, so bind Lua no-ops
    # and skip the crossing. flip() is excluded: the machine ends frames
    # through it, including during the cart's boot splash.
    #
    # This is the one binding decided at construction rather than per call, so
    # a machine built headless stays headless -- reassigning machine.backend
    # afterwards will not restore drawing. Build a new Machine instead, which
    # is what DodgeEnv.reset() already does.
    if not getattr(machine.backend, "draws", True):
        lua.execute("local function noop() end\n"
                    + "\n".join(f"{name}=noop" for name in DRAWING_BUILTINS))
        g.flip = lambda: machine.flip()
        return

    g.cls = lambda col=None: machine.backend.cls(int(_num(col)))
    g.pset = lambda x, y, c=None: machine.backend.pset(x, y, c)
    g.rect = lambda x0, y0, x1, y1, c=None: machine.backend.rect(x0, y0, x1, y1, c)
    g.rectfill = lambda x0, y0, x1, y1, c=None: machine.backend.rectfill(x0, y0, x1, y1, c)
    g.circfill = lambda x, y, r=None, c=None: machine.backend.circfill(x, y, _num(r, 4), c)
    g.ovalfill = lambda x0, y0, x1, y1, c=None: machine.backend.ovalfill(x0, y0, x1, y1, c)
    g.line = lambda x0, y0, x1, y1, c=None: machine.backend.line(x0, y0, x1, y1, c)
    g.spr = lambda n, x, y, w=None, h=None, *rest: machine.backend.spr(int(n), x, y, _num(w, 1), _num(h, 1))
    g.print = lambda text, x=None, y=None, c=None: machine.backend.print_(text, _num(x), _num(y), c)
    g.pal = lambda c0=None, c1=None, mode=None: machine.backend.pal(c0, c1, int(_num(mode)))
    g.palt = noop
    g.fillp = lambda pattern=None: machine.backend.fillp(pattern)
    g.camera = lambda x=None, y=None: machine.backend.camera(_num(x), _num(y))
    g.flip = lambda: machine.flip()
