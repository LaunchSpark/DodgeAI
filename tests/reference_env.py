"""A frozen copy of DodgeVecEnv's step logic as of f5f01ee, the parity oracle.

DodgeVecEnv's step was rewritten for speed. Everything it reports has to stay
bit-for-bit what this implementation reports: the same observations, the same
float32 rewards, the same dones and the same per-lane info dicts. This class
keeps the original methods verbatim (only renamed where noted), over the same
engine and seed stream, so the parity tests can run both side by side on the
same seeds and actions. Do not optimise this file.
"""

import math
import time

import numpy as np

from dodge import rewards
from dodge.env import DEATH_CAUSES, DodgeVecEnv
from dodge.seeds import SeedStream


def center_survival_scalar(x, y, screen_width=128, edge_penalty=0.5):
    center = screen_width / 2.0
    distance = math.hypot(float(x) - center, float(y) - center) / center
    if distance <= 0.6:
        return 1.0
    taper = min((distance - 0.6) / (1.0 - 0.6), 1.0)
    return max(0.0, 1.0 - edge_penalty * taper)


class ReferenceDodgeVecEnv(DodgeVecEnv):
    """DodgeVecEnv with the original, per-lane step and reset bookkeeping."""

    def _draw_seeds(self, lanes):
        lanes = np.asarray(lanes, dtype=np.int64)
        seeds = self._stream.draw(lanes)
        self._game_seeds[lanes] = seeds
        self._game_indices[lanes] = self._stream.next_episode_indices[lanes] - 1
        return seeds

    def _counters(self, state):
        return {
            "score": state["score"].astype(np.float64),
            "kills": state["uncontrolled_kills"].astype(np.int64),
            "lives": state["lives_spent"].astype(np.int64),
            "pickups": state["pickups"].astype(np.int64),
            "pattern": state["pattern_active"].copy(),
            "pattern_id": state["pattern_id"].astype(np.int64),
        }

    def _start_runs(self, lanes, state):
        self._run_frames[lanes] = 0
        self._pattern_frames[lanes] = 0
        for lane in lanes:
            self._encounters[lane] = {}
        self._start_episodes(lanes)
        fresh = self._counters(state)
        if self._previous is None:
            self._previous = fresh
        else:
            for key, values in fresh.items():
                self._previous[key][lanes] = values[lanes]

    def _start_episodes(self, lanes):
        self._episode_started[lanes] = time.monotonic()
        self._episode_reward[lanes] = 0.0
        self._episode_frames[lanes] = 0

    def reset(self):
        if (self._seeds and self._seeds[0] is not None
                and isinstance(self._stream, SeedStream)):
            self._stream = SeedStream(self._seeds[0], self.num_envs,
                                      stream=self._stream.stream)
        self._reset_seeds()
        return self._start_all(self._draw_seeds(np.arange(self.num_envs)))

    def _start_all(self, seeds):
        self.reward_config = rewards.load(self.reward_config_path)
        self._game_seeds[:] = seeds
        self.engine.reset_ml_batch(self._game_seeds)
        state = self.engine.observe_velocity()
        self._previous = None
        self._start_runs(np.arange(self.num_envs), state)
        self.reset_infos = [{"seed": int(seed), "score": 0.0, "frames": 0, "run_frames": 0}
                            for seed in self._game_seeds]
        return state["observation"]

    def step_wait(self):
        lanes = self.num_envs
        weights = self.reward_config
        stepped = self.engine.step_ml_batch(self._actions)
        frames = stepped["frames_advanced"].astype(np.int64)
        state = self.engine.observe_velocity()
        self._run_frames += frames
        self._episode_frames += frames
        self.total_frames += int(frames.sum())

        now = self._counters(state)
        dead = state["dead"]
        lives_spent = now["lives"] - self._previous["lives"]
        hit = dead | (lives_spent > 0)
        survived = np.maximum(frames - hit, 0)
        credit = survived * np.array([
            center_survival_scalar(x, y, edge_penalty=weights.edge_penalty)
            for x, y in zip(state["player_x"], state["player_y"])
        ])
        patterns_survived = (self._previous["pattern"] & ~now["pattern"]
                             & ~dead).astype(np.int64)
        self._pattern_frames += frames * now["pattern"]
        met = (now["pattern_id"] >= 0) & (now["pattern_id"] != self._previous["pattern_id"])
        for lane in np.flatnonzero(met):
            pattern = int(now["pattern_id"][lane])
            self._encounters[lane][pattern] = self._encounters[lane].get(pattern, 0) + 1
        score_delta = now["score"] - self._previous["score"]
        kills = now["kills"] - self._previous["kills"]
        pickups = now["pickups"] - self._previous["pickups"]
        uncontrolled = rewards.UNCONTROLLED_SCORE_PER_ENEMY * kills
        survival_reward = weights.survival_per_frame * credit
        reward = (survival_reward
                  + weights.score_weight * (score_delta - uncontrolled)
                  + weights.uncontrolled_score_weight * uncontrolled
                  - weights.death_penalty * (lives_spent + dead))
        terminated = dead | (lives_spent > 0)
        truncated = (np.zeros(lanes, dtype=bool) if self.max_episode_frames is None
                     else (self._run_frames >= self.max_episode_frames) & ~terminated)
        run_over = dead | truncated
        done = terminated | truncated
        self._episode_reward += reward
        self._previous = now

        observation = state["observation"]
        infos = []
        for lane in range(lanes):
            info = {
                "score": float(now["score"][lane]),
                "score_delta": float(score_delta[lane]),
                "frames": int(self._episode_frames[lane]),
                "run_frames": int(self._run_frames[lane]),
                "run_over": bool(run_over[lane]),
                "survived_frames": int(survived[lane]),
                "survival_reward": float(survival_reward[lane]),
                "uncontrolled_score": float(uncontrolled[lane]),
                "training_events": {
                    "enemies_destroyed": int(kills[lane]),
                    "explosion_pickups": int(pickups[lane, 0]),
                    "freeze_pickups": int(pickups[lane, 1]),
                    "shrink_pickups": int(pickups[lane, 2]),
                    "lives_spent": int(lives_spent[lane]),
                    "survival_frames": int(survived[lane]),
                    "patterns_survived": int(patterns_survived[lane]),
                    "deaths": int(dead[lane]),
                },
            }
            if dead[lane]:
                info["death"] = {
                    "cause": DEATH_CAUSES.get(int(state["death_cause"][lane])),
                    "pattern": int(state["death_pattern"][lane]),
                    "fyou_triggers": int(state["fyou_triggers"][lane]),
                }
            if run_over[lane]:
                info["run"] = {
                    "seed": int(self._game_seeds[lane]),
                    "frames": int(self._run_frames[lane]),
                    "score": float(now["score"][lane]),
                    "died": bool(dead[lane]),
                    "cause": DEATH_CAUSES.get(int(state["death_cause"][lane])),
                    "pattern": int(state["death_pattern"][lane]),
                    "fyou_triggers": int(state["fyou_triggers"][lane]),
                    "pattern_frames": int(self._pattern_frames[lane]),
                    "encounters": dict(self._encounters[lane]),
                    "lives_spent": int(now["lives"][lane]),
                }
            if done[lane]:
                info["terminal_observation"] = observation[lane].copy()
                info["TimeLimit.truncated"] = bool(truncated[lane])
                info["episode"] = {
                    "r": float(self._episode_reward[lane]),
                    "l": int(math.ceil(self._episode_frames[lane] / self.action_repeat)),
                    "t": round(time.monotonic() - self._episode_started[lane], 6),
                }
            infos.append(info)

        self._start_episodes(np.flatnonzero(done & ~run_over))
        restart = np.flatnonzero(run_over).astype(np.uint32)
        if len(restart):
            seeds = self._draw_seeds(restart)
            self.engine.reset_ml_lanes(restart, seeds)
            fresh = self.engine.observe_velocity()
            self._start_runs(restart, fresh)
            observation = observation.copy()
            observation[restart] = fresh["observation"][restart]
        return observation, reward.astype(np.float32), done, infos
