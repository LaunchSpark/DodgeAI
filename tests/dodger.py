"""A scripted dodger for engine-backed tests: it lives long enough to meet
patterns, and survives some of them, on easy.

It steers away from walls, the anti-camping box, nearby enemies and pattern
rectangles, reading each lane's raw state.
"""

import math
import numpy as np

DIRECTIONS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))


def dodge(observations, lane_state):
    actions = np.zeros(len(observations), dtype=np.uint8)
    for lane in range(len(observations)):
        s = lane_state(lane)
        px, py = s["px"], s["py"]
        # Walls push inward; the centre box (the anti-camping guard) pushes out.
        fx = 30 / max(px - 4, 1) ** 2 - 30 / max(124 - px, 1) ** 2
        fy = 30 / max(py - 4, 1) ** 2 - 30 / max(124 - py, 1) ** 2
        if abs(px - 64) < 10 and abs(py - 64) < 10:
            fx += 0.05 if px >= 64 else -0.05
        for e in s["enemies"]:
            if e["p"] >= 2:
                continue
            dx = px - (e["x"] + e["s"] / 2)
            dy = py - (e["y"] + e["s"] / 2)
            d2 = dx * dx + dy * dy + 1
            if d2 < 1600:
                fx += dx / d2 * 20
                fy += dy / d2 * 20
        cp = s["cp"]
        if cp is not None:
            for r in cp["rects"]:
                cx = min(max(px, r["x"]), r["x"] + r["w"])
                cy = min(max(py, r["y"]), r["y"] + r["h"])
                dx, dy = px - cx, py - cy
                if dx == 0 and dy == 0:
                    dx, dy = px - (r["x"] + r["w"] / 2), py - (r["y"] + r["h"] / 2)
                d2 = dx * dx + dy * dy + 1
                if d2 < 400:
                    fx += dx / d2 * 40
                    fy += dy / d2 * 40
        if abs(fx) < 0.02 and abs(fy) < 0.02:
            continue
        best = max(range(1, 9), key=lambda a: (DIRECTIONS[a][0] * fx + DIRECTIONS[a][1] * fy)
                   / math.hypot(*DIRECTIONS[a]))
        actions[lane] = best
    return actions
