import faulthandler
from pathlib import Path
import pytest

luajit = pytest.importorskip("lupa.luajit21")
from pico8 import machine
from pico8.backends import NullBackend

pytestmark = pytest.mark.skipif(not Path("logic.js").exists(), reason="logic.js not present")


@pytest.fixture(scope="module", autouse=True)
def _quiet_luajit_seh():
    """LuaJIT uses structured exception handling internally on Windows (see
    the faulthandler_disabled() note in test_preprocess_cart.py); running the
    real 2,263-line cart trips it on effectively every trace compile, which
    would otherwise flood the run with harmless "Windows fatal exception"
    dumps despite every test passing. Disable pytest's faulthandler dump for
    this whole module -- scoped to module, not session, so it doesn't mask a
    genuine crash dump in unrelated test files.

    Module-scoped so it wraps the module-scoped `booted` fixture too: same-
    scope fixtures set up in dependency order, and `booted` depends on this
    one explicitly (see its signature) to guarantee that ordering.
    """
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


@pytest.fixture(scope="module")
def booted(_quiet_luajit_seh):
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
    # px/py cannot distinguish seeds here: an idle mouse-mode player is pinned
    # to the (fixed) cursor position, and player movement never consumes rnd().
    # len(enemies) can't either: spawn *timing* is a deterministic frame-count
    # threshold (ed>est*60*...), not random. What the seed actually drives is
    # which enemy spawns -- its size (maxs) and personality (p), drawn via
    # rnd() in spawnenemies(). Track those, or two different seeds produce
    # byte-identical traces and the "!=" assertion below can never pass.
    def run(seed):
        m = machine.boot(backend=NullBackend(), seed=seed)
        m.start_game()
        trace = []
        for _ in range(300):
            m.step()
            enemies = m.globals.enemies
            n = len(enemies)
            fingerprint = tuple(
                (round(enemies[i].x, 3), round(enemies[i].y, 3), enemies[i].p, enemies[i].maxs)
                for i in range(1, n + 1)
            )
            trace.append((round(m.globals.px, 6), round(m.globals.py, 6), fingerprint))
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
