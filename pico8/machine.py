"""Boots the cart and steps it one frame at a time."""

from pathlib import Path

from lupa.luajit21 import LuaRuntime

from . import api, cart
from .backends import NullBackend
from .preprocess import preprocess

PRELUDE = Path(__file__).with_name("prelude.lua")


def validate_spawn_rate_percent(value):
    """Return an integer spawn percentage in the supported curriculum range."""
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("spawn rate must be an integer percentage") from None
    if numeric != integer:
        raise ValueError("spawn rate must be an integer percentage")
    if not 50 <= integer <= 150:
        raise ValueError("spawn rate must be between 50% and 150%")
    return integer


class Machine:
    def __init__(self, rom, code, backend, seed=0):
        self.rom = rom
        self.backend = backend
        self.sheet = cart.sprite_sheet(rom)
        self.mem = bytearray(0x10000)
        self.cart_data = [0.0] * 64
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
        if self.globals._init is not None:
            self.globals._init()

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

    def start_game(self, difficulty=2, patterns=True, powerups=True,
                   normal_enemies=True, kamikaze_enemies=True,
                   spawn_rate_percent=100):
        """Skip the menus and begin a round immediately.

        ``difficulty`` uses the cart's native values: 1 easy, 2 normal, 3 hard.
        ``spawn_rate_percent`` scales both the ordinary enemy timer and the
        between-pattern timer. Enemy switches apply to every creation path,
        including the cart's forced anti-camping wave.
        A freshly constructed Machine gives training a fully independent round;
        the cart's own reset intentionally carries some state across restarts.
        """
        if difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1 (easy), 2 (normal), or 3 (hard)")
        spawn_rate_percent = validate_spawn_rate_percent(spawn_rate_percent)
        g = self.globals
        g.smallsettings[2] = difficulty
        g.smallsettings[3] = bool(patterns)
        g.smallsettings[4] = bool(powerups)
        g._upd, g._drw = g.updategame, g.drawgame
        g.reset()
        self._install_spawn_controls(
            normal_enemies=bool(normal_enemies),
            kamikaze_enemies=bool(kamikaze_enemies),
            spawn_rate=spawn_rate_percent / 100.0,
        )
        # Make the native spawn check succeed on the first gameplay frame.
        # spawnenemies() resets ed to zero, preserving its usual cadence after that.
        g.ed = g.est * 60 * (2 - g.frz)
        # Mouse mode teleports the player to stat(32/33). Training uses the
        # real keyboard movement model with velocity and friction.
        g.input = 1
        g.prevxpos, g.prevypos = self.mouse_x, self.mouse_y

    def _install_spawn_controls(self, *, normal_enemies, kamikaze_enemies,
                                spawn_rate):
        """Layer training controls over the unmodified cartridge functions."""
        self.lua.execute("""
            training_normal_enemies = true
            training_kamikaze_enemies = true
            training_spawn_rate = 1

            local cartridge_addenemy = addenemy
            function addenemy(x, y, isdying, size, max_size, personality)
                if personality == nil then
                    return
                end
                if personality == 0 and not training_normal_enemies then
                    return
                end
                if (personality == 1 or personality == -1)
                        and not training_kamikaze_enemies then
                    return
                end
                return cartridge_addenemy(
                    x, y, isdying, size, max_size, personality)
            end

            local cartridge_spawnenemies = spawnenemies
            function spawnenemies()
                -- The cartridge adds one inside spawnenemies. Supplying the
                -- remainder here changes cadence without touching its dynamic
                -- `est`, freeze, or special-pattern multipliers.
                ed = ed + training_spawn_rate - 1
                if not training_normal_enemies then
                    enemiestats[1] = 0
                end
                if not training_kamikaze_enemies then
                    enemiestats[2] = 0
                end

                local total = 0
                for i = 1, #enemiestats do
                    total = total + enemiestats[i]
                end
                if total > 0 then
                    return cartridge_spawnenemies()
                end

                -- Preserve the timer's effective increment when there is
                -- nothing eligible to spawn, without asking rnd(0) for an
                -- absent personality.
                ed = ed + 1
            end

            local cartridge_updatepatterns = updatepatterns
            function updatepatterns()
                -- updatepatterns adds one only while this same countdown is
                -- active. Applying the remainder under the same conditions
                -- keeps freeze and active-pattern behavior byte-for-byte.
                if dopatterns and cp == nil and not freeze then
                    cpt = cpt + training_spawn_rate - 1
                end
                return cartridge_updatepatterns()
            end
        """)
        g = self.lua.globals()
        g.training_normal_enemies = normal_enemies
        g.training_kamikaze_enemies = kamikaze_enemies
        g.training_spawn_rate = float(spawn_rate)

    def install_training_hooks(self, lives=0):
        """Count game events without modifying the generated cart source.

        `lives` is a training-only device. With it set, the first `lives` enemy
        collisions delete the enemy as usual but leave the player alive, so one
        cartridge run supplies several RL episodes starting from mid-game
        states. The environment still ends the episode at each one; this only
        keeps the cart running underneath it.
        """
        if lives < 0:
            raise ValueError("lives cannot be negative")
        self.lua.execute("""
            training_events = {
                enemies_destroyed = 0,
                explosion_pickups = 0,
                freeze_pickups = 0,
                shrink_pickups = 0,
                lives_spent = 0
            }

            -- Set from Python below; a life is spent rather than a death taken
            -- while this is positive.
            training_lives = 0
            local in_enemy_collide = false

            local original_collide = collide
            function collide(enemy)
                if enemy.p == 2 then
                    training_events.explosion_pickups = training_events.explosion_pickups + 1
                elseif enemy.p == 3 then
                    training_events.freeze_pickups = training_events.freeze_pickups + 1
                elseif enemy.p == 4 then
                    training_events.shrink_pickups = training_events.shrink_pickups + 1
                end
                in_enemy_collide = true
                local result = original_collide(enemy)
                in_enemy_collide = false
                return result
            end

            local original_die = die
            function die()
                -- Only an enemy collision spends a life. A pattern crush calls
                -- die() straight from collisioncheck() with the player still
                -- inside the rectangle, so absorbing one would burn every life
                -- in consecutive frames without the player ever getting away.
                --
                -- Returning early also skips checkhs(), which would otherwise
                -- write a highscore into cartdata for a death that did not
                -- happen, and the death music.
                if in_enemy_collide and training_lives > 0 then
                    training_lives = training_lives - 1
                    training_events.lives_spent = training_events.lives_spent + 1
                    return
                end
                return original_die()
            end

            local original_difficultycurve = difficultycurve
            function difficultycurve(half)
                if half then
                    training_events.enemies_destroyed = training_events.enemies_destroyed + 1
                end
                return original_difficultycurve(half)
            end
        """)
        self.lua.globals().training_lives = int(lives)


def boot(js_path=Path("dodge.p8.png"), backend=None, seed=0):
    rom, _ = cart.read_rom(Path(js_path))
    code = cart.decompress_code(rom)
    return Machine(rom, code, backend or NullBackend(), seed=seed)
