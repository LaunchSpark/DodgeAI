"""Replay recorded games and keep the last 60 frames of player path before death.

usage: python replay_paths.py MODEL K MODE RUN.jsonl OUT.jsonl [BATCH]
Replays every game in RUN.jsonl that died while a static pattern was active,
and checks that the replay dies on the same frame with the same cause.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent))
from evalharness import Game  # noqa: E402

TAIL = 60

# Wraps the telemetry die() so it runs first: capture the layout as it was
# at the lethal contact, before the cart's death handling.
CAPTURE = r"""
ev.rects = {}
ev.kx, ev.ky, ev.ks = nil, nil, nil
local ev_cur = nil
local ev_collide2 = collide
function collide(e)
  ev_cur = e
  local r = ev_collide2(e)
  ev_cur = nil
  return r
end
local ev_die2 = die
function die()
  if not isdead and ev.cause == nil then
    if cp ~= nil then
      for i, r in ipairs(cp.rects) do
        ev.rects[i] = {x=r.x, y=r.y, w=r.w, h=r.h, sh=r.sh}
      end
    end
    if ev_cur ~= nil then
      ev.kx, ev.ky, ev.ks = ev_cur.x, ev_cur.y, ev_cur.s
    end
  end
  return ev_die2()
end
"""


class PathGame(Game):
    def __init__(self, k, seed, expected):
        super().__init__(k, seed)
        self.env.machine.lua.execute(CAPTURE)
        self.expected = expected
        self.path = deque(maxlen=TAIL + 1)
        g = self.env.machine.globals
        self.path.append((float(g.px), float(g.py)))

    def step(self, action):
        result = super().step(action)
        g = self.env.machine.globals
        self.path.append((float(g.px), float(g.py)))
        if result is None:
            return None
        ev = self.ev
        rects = []
        if ev.rects is not None:
            for i in range(1, len(ev.rects) + 1):
                r = ev.rects[i]
                rects.append({"x": float(r.x), "y": float(r.y), "w": float(r.w),
                              "h": float(r.h), "sh": float(r.sh)})
        killer = None
        if ev.kx is not None:
            killer = [float(ev.kx), float(ev.ky), float(ev.ks or 0)]
        result.update({
            "path": list(self.path), "rects": rects, "killer": killer,
            "replay_match": (result["frames"] == self.expected["frames"]
                             and result["cause"] == self.expected["cause"]),
        })
        return result


def main():
    model_path, k, mode, run, out = sys.argv[1:6]
    batch = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    k = int(k)
    games = [json.loads(line) for line in open(run)]
    targets = [g for g in games if g["died"] and g["ptype"] == 0]
    print(f"{len(targets)} of {len(games)} games died during a static pattern")
    model = PPO.load(model_path, device="cuda")
    pending = iter(targets)
    live = [PathGame(k, t["seed"], t) for _, t in zip(range(batch), pending)]
    mismatches = 0
    with open(out, "w") as f:
        while live:
            obs = np.stack([g.obs for g in live])
            actions, _ = model.predict(obs, deterministic=mode == "argmax")
            nxt = []
            for g, a in zip(live, actions):
                r = g.step(a)
                if r is None:
                    nxt.append(g)
                    continue
                mismatches += not r["replay_match"]
                f.write(json.dumps(r) + "\n")
                g.env.close()
                t = next(pending, None)
                if t is not None:
                    nxt.append(PathGame(k, t["seed"], t))
            live = nxt
    print(f"replayed {len(targets)}, mismatches {mismatches}")


if __name__ == "__main__":
    main()
