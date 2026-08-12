"""Gymnasium environment for the first state-based Dodge PPO bot."""

from pathlib import Path
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pico8 import cart
from pico8.backends import NullBackend, PygameBackend
from pico8.machine import Machine

from .actions import ACTION_MASKS, action_mask
from .observe import ARENA_SIZE, FRAME_STACK_SIZE, OBSERVATION_SIZE, observe_frame


SURVIVAL_REWARD_PER_FRAME = 0.01
DEATH_PENALTY = 2.0


def center_survival_scalar(x, y, screen_width=ARENA_SIZE):
    """Scale survival from 1.5x at center to 0.5x at a cardinal edge."""
    center = screen_width / 2.0
    distance = math.hypot(float(x) - center, float(y) - center)
    return max(0.0, 1.5 - distance / center)


class DodgeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        js_path=Path("logic.js"),
        difficulty=1,
        patterns=False,
        powerups=False,
        action_repeat=4,
        max_episode_frames=3600,
        render_mode=None,
        scale=5,
        caption="Dodge PPO Agent",
    ):
        super().__init__()
        if action_repeat < 1:
            raise ValueError("action_repeat must be at least 1")
        self.js_path = Path(js_path)
        self.difficulty = difficulty
        self.patterns = patterns
        self.powerups = powerups
        self.action_repeat = action_repeat
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
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.rom, _ = cart.read_rom(self.js_path)
        self.code = cart.decompress_code(self.rom)
        self.machine = None
        self.episode_frames = 0
        self.previous_score = 0.0
        self.previous_events = {}
        self.previous_pattern_active = False
        self.observation_frames = None
        self.render_clock = None
        self.backend = None
        if self.render_mode == "human":
            self.backend = PygameBackend(scale=self.scale, caption=self.caption)
            import pygame
            self.render_clock = pygame.time.Clock()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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
        self.machine.install_training_hooks()
        self.episode_frames = 0
        self.previous_score = float(self.machine.globals.score)
        self.previous_events = self._event_totals()
        self.previous_pattern_active = self.machine.globals.cp is not None
        initial_frame = observe_frame(self.machine)
        self.observation_frames = [initial_frame.copy() for _ in range(FRAME_STACK_SIZE)]
        return self._observation(), {"seed": episode_seed, "score": 0.0, "frames": 0}

    def _observation(self):
        return np.concatenate(self.observation_frames).astype(np.float32, copy=False)

    def _event_totals(self):
        events = self.machine.globals.training_events
        return {
            "enemies_destroyed": int(events.enemies_destroyed),
            "explosion_pickups": int(events.explosion_pickups),
            "freeze_pickups": int(events.freeze_pickups),
            "shrink_pickups": int(events.shrink_pickups),
        }

    @staticmethod
    def _rect_gap(left_a, top_a, right_a, bottom_a,
                  left_b, top_b, right_b, bottom_b):
        dx = max(left_b - right_a, left_a - right_b, 0)
        dy = max(top_b - bottom_a, top_a - bottom_b, 0)
        return max(dx, dy)

    def _is_near_miss(self):
        g = self.machine.globals
        size = float(g.size)
        player = (float(g.px) - size + 1, float(g.py) - size + 1,
                  float(g.px) + size - 1, float(g.py) + size - 1)
        enemies = g.enemies
        for index in range(1, len(enemies) + 1):
            enemy = enemies[index]
            if int(enemy.p) >= 2:
                continue
            hazard = (float(enemy.x), float(enemy.y),
                      float(enemy.x) + float(enemy.s),
                      float(enemy.y) + float(enemy.s))
            gap = self._rect_gap(*player, *hazard)
            if 0 < gap <= 4:
                return True

        pattern = g.cp
        if pattern is not None:
            rects = pattern.rects
            for index in range(1, len(rects) + 1):
                rect = rects[index]
                if int(pattern.tpe) != 1 and float(rect.sh) != 2:
                    continue
                hazard = (float(rect.x), float(rect.y),
                          float(rect.x) + float(rect.w),
                          float(rect.y) + float(rect.h))
                gap = self._rect_gap(*player, *hazard)
                if 0 < gap <= 4:
                    return True
        return False

    def step(self, action):
        if self.machine is None:
            raise RuntimeError("reset() must be called before step()")
        self.machine.set_buttons(action_mask(action))
        previous_decision_frame = self.observation_frames[-1]

        survived = 0
        survival_credit = 0.0
        patterns_survived = 0
        near_misses = 0
        for _ in range(self.action_repeat):
            # Keep training strictly on the cart's keyboard movement branch.
            self.machine.globals.input = 1
            self.machine.step()
            self.episode_frames += 1
            pattern_active = self.machine.globals.cp is not None
            if self.previous_pattern_active and not pattern_active \
                    and not bool(self.machine.globals.isdead):
                patterns_survived += 1
            self.previous_pattern_active = pattern_active
            if not bool(self.machine.globals.isdead) and self._is_near_miss():
                near_misses += 1
            if self.render_clock is not None:
                self.render_clock.tick(self.metadata["render_fps"])
            if bool(self.machine.globals.isdead):
                break
            survived += 1
            survival_credit += center_survival_scalar(
                self.machine.globals.px,
                self.machine.globals.py,
            )

        self.observation_frames = [
            previous_decision_frame,
            observe_frame(self.machine),
        ]

        score = float(self.machine.globals.score)
        score_delta = score - self.previous_score
        self.previous_score = score
        terminated = bool(self.machine.globals.isdead)
        truncated = (self.max_episode_frames is not None
                     and self.episode_frames >= self.max_episode_frames
                     and not terminated)
        survival_reward = SURVIVAL_REWARD_PER_FRAME * survival_credit
        reward = (
            survival_reward
            + score_delta
            - (DEATH_PENALTY if terminated else 0.0)
        )
        totals = self._event_totals()
        training_events = {
            key: totals[key] - self.previous_events.get(key, 0)
            for key in totals
        }
        self.previous_events = totals
        training_events.update({
            "survival_frames": survived,
            "patterns_survived": patterns_survived,
            "near_misses": near_misses,
            "deaths": int(terminated),
        })
        info = {
            "score": score,
            "score_delta": score_delta,
            "frames": self.episode_frames,
            "survived_frames": survived,
            "survival_reward": survival_reward,
            "training_events": training_events,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.machine = None
        self.backend = None
        if self.render_mode == "human":
            import pygame
            pygame.quit()
