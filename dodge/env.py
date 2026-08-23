"""Gymnasium environment for the first state-based Dodge PPO bot."""

from pathlib import Path
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pico8 import cart
from pico8.backends import NullBackend, PygameBackend
from pico8.machine import Machine

from . import rewards
from .actions import ACTION_MASKS, action_mask
from .observe import ARENA_SIZE, OBSERVATION_SIZE, observe


# Reward weights live in dodge/rewards.py so the dashboard and a text editor
# can both change them; see RewardConfig for the defaults and their ranges.


# How much of the half-arena pays full survival credit.
CENTRE_PLATEAU = 0.6


def center_survival_scalar(x, y, screen_width=ARENA_SIZE, edge_penalty=0.5):
    """Full survival credit across the interior, tapering only near the walls.

    This used to peak at the exact centre and fall away in every direction,
    which made standing perfectly still the best-paying action available: the
    agent spawns at the centre, so any movement at all reduced its income. That
    is a reward for doing nothing, in a game about moving.

    A plateau keeps the part that was wanted -- do not cower against a wall,
    where there is nowhere left to dodge -- without paying the agent to freeze
    on one pixel. Inside the plateau every position pays the same, so position
    only matters once the agent nears an edge.
    """
    center = screen_width / 2.0
    distance = math.hypot(float(x) - center, float(y) - center) / center
    if distance <= CENTRE_PLATEAU:
        return 1.0
    # Corners sit past 1.0, so the taper is clamped rather than going negative.
    taper = min((distance - CENTRE_PLATEAU) / (1.0 - CENTRE_PLATEAU), 1.0)
    return max(0.0, 1.0 - edge_penalty * taper)


class DodgeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        js_path=Path("dodge.p8.png"),
        difficulty=1,
        patterns=False,
        powerups=False,
        action_repeat=4,
        lives=0,
        max_episode_frames=3600,
        render_mode=None,
        scale=5,
        caption="Dodge PPO Agent",
        reward_config_path=None,
    ):
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")
        self.js_path = Path(js_path)
        self.difficulty = difficulty
        self.patterns = patterns
        self.powerups = powerups
        self.action_repeat = action_repeat
        # Training-only. Each life turns one enemy collision into the end of an
        # RL episode that the cartridge run survives, so the next episode starts
        # from a mid-game state instead of the spawn point. The episode still
        # ends -- that is what stops the agent valuing a life as a resource.
        if lives < 0:
            raise ValueError("lives cannot be negative")
        self.lives = int(lives)
        # None means an episode only ends when the player dies. Training caps
        # this so episodes keep turning over; watching a policy does not.
        if max_episode_frames is not None and max_episode_frames < 1:
            raise ValueError("max_episode_frames must be positive or None")
        self.max_episode_frames = max_episode_frames
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.scale = scale
        self.caption = caption

        self.action_space = spaces.Discrete(len(ACTION_MASKS))
        # Velocities are signed, so the box is not the unit interval.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.rom, _ = cart.read_rom(self.js_path)
        self.code = cart.decompress_code(self.rom)
        # Re-read when an episode starts so dashboard sliders and hand edits
        # both reach every environment process without any message passing.
        # Resolved here rather than as a default argument so the location can be
        # redirected at runtime, which keeps tests off the developer's own file.
        self.reward_config_path = Path(reward_config_path
                                       if reward_config_path is not None
                                       else rewards.CONFIG_PATH)
        self.reward_config = rewards.load(self.reward_config_path)
        self.machine = None
        self.episode_frames = 0
        self.run_frames = 0
        self.episode_seed = 0
        # True once the cartridge run itself is finished, as opposed to the
        # agent's episode. Only then does reset() build a new Machine.
        self._run_over = True
        self.previous_score = 0.0
        self.previous_events = {}
        self.previous_pattern_active = False
        self.render_clock = None
        self.backend = None
        if self.render_mode == "human":
            self.backend = PygameBackend(scale=self.scale, caption=self.caption)
            import pygame
            self.render_clock = pygame.time.Clock()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Picked up here rather than at construction so a weight changed mid-run
        # reaches every environment process from the next episode onward.
        self.reward_config = rewards.load(self.reward_config_path)
        # With lives in play an episode ends while the cartridge run carries on.
        # Rebuilding here would return the agent to the spawn point and discard
        # the mid-game state the life was spent reaching, so the run is kept and
        # only the per-episode bookkeeping is cleared. An explicit seed always
        # starts a fresh run: SB3's auto-reset never passes one, while play.py,
        # watch_ppo.py and the tests always do.
        if self.machine is not None and not self._run_over and seed is None:
            return self._resume()
        episode_seed = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else int(seed)
        if self.render_mode == "human":
            backend = self.backend
        else:
            backend = NullBackend()
        self.machine = Machine(self.rom, self.code, backend, seed=episode_seed)
        self.machine.start_game(
            difficulty=self.difficulty,
            patterns=self.patterns,
            powerups=self.powerups,
        )
        self.machine.install_training_hooks(lives=self.lives)
        self.episode_seed = episode_seed
        self._run_over = False
        self.run_frames = 0
        self.episode_frames = 0
        self.previous_score = float(self.machine.globals.score)
        self.previous_events = self._event_totals()
        self.previous_pattern_active = self.machine.globals.cp is not None
        return self._observation(), {"seed": episode_seed, "score": 0.0,
                                     "frames": 0, "run_frames": 0}

    def _resume(self):
        """Open a new episode on the running cartridge without advancing a frame.

        The observation returned is the one the finished episode ended on, so
        the agent picks up exactly where the collision left the arena.
        """
        score = float(self.machine.globals.score)
        self.episode_frames = 0
        self.previous_score = score
        self.previous_events = self._event_totals()
        self.previous_pattern_active = self.machine.globals.cp is not None
        return self._observation(), {"seed": self.episode_seed,
                                     "score": score,
                                     "frames": 0,
                                     "run_frames": self.run_frames}

    def _observation(self):
        return observe(self.machine)

    def _lives_spent(self):
        """Total lives spent this cartridge run. Zero-cost when lives are off."""
        if not self.lives:
            return 0
        return int(self.machine.globals.training_events.lives_spent)

    def _event_totals(self):
        events = self.machine.globals.training_events
        return {
            "enemies_destroyed": int(events.enemies_destroyed),
            "explosion_pickups": int(events.explosion_pickups),
            "freeze_pickups": int(events.freeze_pickups),
            "shrink_pickups": int(events.shrink_pickups),
            "lives_spent": int(events.lives_spent),
        }

    def step(self, action):
        if self.machine is None:
            raise RuntimeError("reset() must be called before step()")
        self.machine.set_buttons(action_mask(action))

        survived = 0
        survival_credit = 0.0
        patterns_survived = 0
        lives_at_start = self._lives_spent()
        for _ in range(self.action_repeat):
            # Keep training strictly on the cart's keyboard movement branch.
            self.machine.globals.input = 1
            self.machine.step()
            self.episode_frames += 1
            self.run_frames += 1
            pattern_active = self.machine.globals.cp is not None
            if self.previous_pattern_active and not pattern_active \
                    and not bool(self.machine.globals.isdead):
                patterns_survived += 1
            self.previous_pattern_active = pattern_active
            if self.render_clock is not None:
                self.render_clock.tick(self.metadata["render_fps"])
            if bool(self.machine.globals.isdead) or self._lives_spent() > lives_at_start:
                break
            survived += 1
            survival_credit += center_survival_scalar(
                self.machine.globals.px,
                self.machine.globals.py,
                edge_penalty=self.reward_config.edge_penalty,
            )


        score = float(self.machine.globals.score)
        score_delta = score - self.previous_score
        self.previous_score = score
        lives_spent = self._lives_spent() - lives_at_start
        real_death = bool(self.machine.globals.isdead)
        # A spent life ends the episode too. terminated, never truncated: SB3
        # bootstraps V=0 at a terminal and V(s') at a truncation, and it is the
        # zero that keeps the value of surviving the hit from being credited
        # back to the actions that led into it.
        terminated = real_death or lives_spent > 0
        # The cap bounds the cartridge run, not the episode. Counting the
        # episode instead would let a run that keeps absorbing hits carry on
        # forever, holding one seed and one difficulty ramp in the buffer.
        truncated = (self.max_episode_frames is not None
                     and self.run_frames >= self.max_episode_frames
                     and not terminated)
        # Only these end the cartridge run; a spent life ends the episode alone.
        self._run_over = real_death or truncated
        totals = self._event_totals()
        training_events = {
            key: totals[key] - self.previous_events.get(key, 0)
            for key in totals
        }
        self.previous_events = totals

        weights = self.reward_config
        survival_reward = weights.survival_per_frame * survival_credit
        # Score the agent did not cause, priced separately from the rest.
        uncontrolled_score = (rewards.UNCONTROLLED_SCORE_PER_ENEMY
                              * training_events["enemies_destroyed"])
        agent_score = score_delta - uncontrolled_score
        reward = (
            survival_reward
            + weights.score_weight * agent_score
            + weights.uncontrolled_score_weight * uncontrolled_score
            - weights.death_penalty * (lives_spent + int(real_death))
        )
        training_events.update({
            "survival_frames": survived,
            "patterns_survived": patterns_survived,
            "deaths": int(real_death),
            "lives_spent": lives_spent,
        })
        info = {
            "score": score,
            "score_delta": score_delta,
            "frames": self.episode_frames,
            "run_frames": self.run_frames,
            # The episode always ended; this says whether the cartridge run did
            # too, which is what the dashboard's survival metric follows.
            "run_over": self._run_over,
            "survived_frames": survived,
            "survival_reward": survival_reward,
            "uncontrolled_score": uncontrolled_score,
            "training_events": training_events,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.machine = None
        self.backend = None
        if self.render_mode == "human":
            import pygame
            pygame.quit()
