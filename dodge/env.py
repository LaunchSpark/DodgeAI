"""Gymnasium environment for the first state-based Dodge PPO bot."""

from collections import deque
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
from .observe import (ARENA_SIZE, DECISION_LOOKBACK, OBSERVATION_SIZE,
                      observe_frame)


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
        max_episode_frames=3600,
        observation_lookback=DECISION_LOOKBACK,
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
        # None means an episode only ends when the player dies. Training caps
        # this so episodes keep turning over; watching a policy does not.
        if max_episode_frames is not None and max_episode_frames < 1:
            raise ValueError("max_episode_frames must be positive or None")
        self.max_episode_frames = max_episode_frames
        if observation_lookback < 1:
            raise ValueError("observation_lookback must be at least 1")
        # The observation stays two frames; this is how many decisions apart
        # they are. Intermediate frames are retained but not sent.
        self.observation_lookback = observation_lookback
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
        self.previous_score = 0.0
        self.previous_events = {}
        self.previous_pattern_active = False
        self.decision_frames = None
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
        self.decision_frames = deque(
            (initial_frame.copy() for _ in range(self.observation_lookback + 1)),
            maxlen=self.observation_lookback + 1,
        )
        return self._observation(), {"seed": episode_seed, "score": 0.0, "frames": 0}

    def _observation(self):
        # Oldest and newest: the frames the policy sees, a lookback apart.
        return np.concatenate(
            (self.decision_frames[0], self.decision_frames[-1])
        ).astype(np.float32, copy=False)

    def _event_totals(self):
        events = self.machine.globals.training_events
        return {
            "enemies_destroyed": int(events.enemies_destroyed),
            "explosion_pickups": int(events.explosion_pickups),
            "freeze_pickups": int(events.freeze_pickups),
            "shrink_pickups": int(events.shrink_pickups),
        }

    def step(self, action):
        if self.machine is None:
            raise RuntimeError("reset() must be called before step()")
        self.machine.set_buttons(action_mask(action))

        survived = 0
        survival_credit = 0.0
        patterns_survived = 0
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
            if self.render_clock is not None:
                self.render_clock.tick(self.metadata["render_fps"])
            if bool(self.machine.globals.isdead):
                break
            survived += 1
            survival_credit += center_survival_scalar(
                self.machine.globals.px,
                self.machine.globals.py,
                edge_penalty=self.reward_config.edge_penalty,
            )

        self.decision_frames.append(observe_frame(self.machine))

        score = float(self.machine.globals.score)
        score_delta = score - self.previous_score
        self.previous_score = score
        terminated = bool(self.machine.globals.isdead)
        truncated = (self.max_episode_frames is not None
                     and self.episode_frames >= self.max_episode_frames
                     and not terminated)
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
            - (weights.death_penalty if terminated else 0.0)
        )
        training_events.update({
            "survival_frames": survived,
            "patterns_survived": patterns_survived,
            "deaths": int(terminated),
        })
        info = {
            "score": score,
            "score_delta": score_delta,
            "frames": self.episode_frames,
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
