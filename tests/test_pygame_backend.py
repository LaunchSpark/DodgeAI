import faulthandler
import os
import pathlib

import pytest

pygame = pytest.importorskip("pygame")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pico8.backends import NullBackend, PygameBackend
from pico8.palette import PICO8


@pytest.fixture(scope="module", autouse=True)
def _quiet_luajit_seh():
    """LuaJIT uses structured exception handling internally on Windows (see
    tests/test_machine.py); the parity test below boots the real 2,263-line
    cart, which trips it on effectively every trace compile and would
    otherwise flood the run with harmless "Windows fatal exception" dumps
    despite every test passing. Disable pytest's faulthandler dump for this
    whole module -- scoped to module, not session, so it doesn't mask a
    genuine crash dump in unrelated test files.
    """
    was_enabled = faulthandler.is_enabled()
    faulthandler.disable()
    yield
    if was_enabled:
        faulthandler.enable()


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


def test_camera_persists_across_cls():
    """cls() must not reset the camera offset.

    dodge.lua's global _draw() (dodge.lua:2214-2262) calls camera(shakex,
    shakey) to apply a screen shake and only then cls(12) followed by
    _drw(). The shake itself comes from real gameplay hits (shake += 0.07
    in dodge.lua:565, 746, 875), not from a pause/settings effect. If cls()
    reset the camera, that shake would be silently discarded every frame
    it's active. Real PICO-8's cls() does not touch the camera offset, so
    this backend must not either.
    """
    b = PygameBackend(scale=1)
    try:
        b.camera(10, 0)
        b.cls(0)
        b.rectfill(10, 0, 12, 2, 8)
        assert b.surface.get_at((0, 0))[:3] == PICO8[8]
    finally:
        pygame.quit()


def test_backends_produce_identical_trajectories():
    """The two backends must simulate identically.

    This is not a formality: drawgame() calls spawntrail(), which draws from
    the same RNG stream as gameplay. If rendering consumed randomness that
    headless did not, the trained policy and the watched policy would be
    playing different games.
    """
    if not pathlib.Path("dodge.p8.png").exists():
        pytest.skip("dodge.p8.png not present")

    pytest.importorskip("lupa.luajit21")
    from pico8.machine import boot

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
