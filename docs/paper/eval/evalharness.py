"""Batched evaluation of Dodge checkpoints with observation-only death telemetry.

usage: python evalharness.py MODEL_OR_POLICY K MODE N FIRST_SEED OUT.jsonl
  MODEL_OR_POLICY: path to .zip, or 'random' / 'gradient'
  MODE: argmax | sampled
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from dodge.env import DodgeEnv  # noqa: E402

CAP = 18_000

TELEMETRY = r"""
ev = {fyou=0, cause=nil, pat=nil, ptype=nil, killer_fy=false,
      enc_id=nil, encs={}, nenc=0, fy_spawned=0}
local ev_cur_e = nil
local ev_in_fyou = false

local function ev_pattern_id()
  if cp == nil then return nil end
  for i = 1, #patterns do
    if patterns[i] == cp then return i end
  end
  return -1
end

local ev_addenemy = addenemy
function addenemy(...)
  local before = #enemies
  local r = ev_addenemy(...)
  if ev_in_fyou and #enemies > before then
    for i = before + 1, #enemies do enemies[i].ev_fy = true end
    ev.fy_spawned = ev.fy_spawned + (#enemies - before)
  end
  return r
end

local ev_fyou = fyou
function fyou()
  ev.fyou = ev.fyou + 1
  ev_in_fyou = true
  local r = ev_fyou()
  ev_in_fyou = false
  return r
end

local ev_updatepatterns = updatepatterns
function updatepatterns()
  local prev = cp
  local r = ev_updatepatterns()
  if prev == nil and cp ~= nil then
    ev.enc_id = ev_pattern_id()
    ev.nenc = ev.nenc + 1
    ev.encs[ev.nenc] = ev.enc_id
  elseif prev ~= nil and cp == nil then
    ev.enc_id = nil
  end
  return r
end

local ev_collide = collide
function collide(e)
  ev_cur_e = e
  local r = ev_collide(e)
  ev_cur_e = nil
  return r
end

local ev_die = die
function die()
  if not isdead and ev.cause == nil then
    if ev_cur_e ~= nil then
      if ev_cur_e.p == 0 then ev.cause = "normal"
      elseif ev_cur_e.p == 1 then ev.cause = "kamikaze_body"
      else ev.cause = "kamikaze_blast" end
      ev.killer_fy = ev_cur_e.ev_fy == true
    elseif cp ~= nil then
      ev.cause = (cp.tpe == 1) and "moving_rect" or "static_rect"
    else
      ev.cause = "unknown"
    end
    if cp ~= nil then
      ev.pat = ev.enc_id
      ev.ptype = cp.tpe
    end
  end
  return ev_die()
end
"""


def make_env(k):
    return DodgeEnv(difficulty=3, patterns=True, powerups=True,
                    normal_enemies=True, kamikaze_enemies=True,
                    spawn_rate_percent=100, action_repeat=k, lives=0,
                    max_episode_frames=CAP)


class Game:
    def __init__(self, k, seed):
        self.env = make_env(k)
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.env.machine.lua.execute(TELEMETRY)
        self.ev = self.env.machine.globals.ev
        self.pattern_frames = 0
        self.fyou_frames = []
        self.last_fyou = 0
        self.frames = 0

    def step(self, action):
        pattern_was = self.env.machine.globals.cp is not None
        self.obs, _, term, trunc, info = self.env.step(int(action))
        self.frames = info["run_frames"]
        if pattern_was or self.env.machine.globals.cp is not None:
            self.pattern_frames += info["survived_frames"] + int(term)
        fy = int(self.ev.fyou)
        if fy > self.last_fyou:
            self.fyou_frames.extend([self.frames] * (fy - self.last_fyou))
            self.last_fyou = fy
        if term or trunc:
            ev = self.ev
            encs = [int(ev.encs[i + 1]) for i in range(int(ev.nenc))]
            died = bool(term)
            return {
                "seed": self.seed, "frames": self.frames,
                "score": info["score"], "died": died, "capped": bool(trunc),
                "cause": ev.cause if died else None,
                "pattern": (int(ev.pat) if ev.pat is not None else None),
                "ptype": (int(ev.ptype) if ev.ptype is not None else None),
                "killer_fyou": bool(ev.killer_fy),
                "fyou_frames": self.fyou_frames,
                "fyou_spawned": int(ev.fy_spawned),
                "pattern_frames": self.pattern_frames,
                "encounters": encs,
            }
        return None


def main():
    policy, k, mode, n, first, out = sys.argv[1:7]
    k, n, first = int(k), int(n), int(first)
    deterministic = mode == "argmax"
    torch.manual_seed(first)
    rng = np.random.default_rng(first)
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 48

    model = bot = None
    if policy == "random":
        pass
    elif policy == "gradient":
        from dodge.gradient import ControllerConfig, GradientBot
        bot = GradientBot(ControllerConfig(), action_repeat=k)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(policy, device="cuda")

    seeds = list(range(first, first + n))
    pending = iter(seeds)
    games = []
    for _ in range(min(batch, n)):
        games.append(Game(k, next(pending)))
    out_f = open(out, "w")
    done = 0
    started = time.perf_counter()
    frames_total = 0
    while games:
        if model is not None:
            obs = np.stack([g.obs for g in games])
            actions, _ = model.predict(obs, deterministic=deterministic)
        elif bot is not None:
            actions = [bot.act(g.env.machine) for g in games]
        else:
            actions = rng.integers(0, 9, size=len(games))
        survivors = []
        for g, a in zip(games, actions):
            result = g.step(a)
            if result is None:
                survivors.append(g)
                continue
            frames_total += result["frames"]
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            done += 1
            g.env.close()
            nxt = next(pending, None)
            if nxt is not None:
                survivors.append(Game(k, nxt))
        games = survivors
        if done and done % 32 == 0 and not games == []:
            pass
    out_f.close()
    dt = time.perf_counter() - started
    print(f"{out}: {done} games, {frames_total/done:.0f} mean frames, "
          f"{dt:.0f}s, {frames_total/dt:.0f} frames/s")


if __name__ == "__main__":
    main()
