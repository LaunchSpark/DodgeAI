"""Faithful Python port of Taylor Turner's m0_policy.rs (PolicyKind::AabbDodger),
from TaylorTurnerIT/dodge-native crates/dodge-rl-env/tests/support/m0_policy.rs,
played on dodge-ai-3 through NativeBatchEnv.lane_state."""
import numpy as np


HORIZON = 12
SAFE_CLEARANCE = 4.0
ONE = 65536
F08 = 52428          # PicoFixed::from_f32(0.8)
HALF = 32768         # PicoFixed::from_f32(0.5)
LO, HI = 2 * ONE, 125 * ONE
# Action::ALL order and masks: neutral, L, R, U, D, UL, UR, DL, DR
MASKS = [0, 1, 2, 4, 8, 5, 6, 9, 10]


def raw(value):
    return int(round(value * ONE))


def player_path(x, y, vx, vy):
    """(9, H+1, 2) player centre for t = 0..H; t=0 is now, as in choose_action."""
    out = np.empty((9, HORIZON + 1, 2))
    for a, mask in enumerate(MASKS):
        px, py, pvx, pvy = x, y, vx, vy
        for t in range(HORIZON + 1):
            out[a, t] = (px / ONE, py / ONE)
            pvx = (pvx * F08) >> 16
            pvy = (pvy * F08) >> 16
            if mask & 1: pvx -= HALF
            if mask & 2: pvx += HALF
            if mask & 4: pvy -= HALF
            if mask & 8: pvy += HALF
            px = min(max(px + pvx, LO), HI)
            py = min(max(py + pvy, LO), HI)
    return out


def clearance(bx0, by0, bx1, by1, ox0, oy0, ox1, oy1):
    x = np.maximum(ox0 - bx1, bx0 - ox1)
    y = np.maximum(oy0 - by1, by0 - oy1)
    inside = (x <= 0) & (y <= 0)
    return np.where(inside, np.maximum(x, y),
                    np.hypot(np.maximum(x, 0), np.maximum(y, 0)))


class Policy:
    def __init__(self):
        self.previous_pattern = None
        self.previous_rects = []
        self.previous_sh = []
        self.rect_changes = []
        self.sh_changes = []

    @staticmethod
    def _rects(s):
        cp = s["cp"]
        if cp is None:
            return [], []
        boxes = [(r["x"], r["y"], r["x"] + r["w"] - 1, r["y"] + r["h"] - 1) for r in cp["rects"]]
        return boxes, [r["sh"] for r in cp["rects"]]

    def observe(self, s):
        pattern = None if s["cp"] is None else s["cp"]["id"]
        current, current_sh = self._rects(s)
        same = self.previous_pattern == pattern
        self.rect_changes = [
            tuple(b - o for b, o in zip(box, self.previous_rects[i]))
            if same and i < len(self.previous_rects) else (0.0, 0.0, 0.0, 0.0)
            for i, box in enumerate(current)]
        self.sh_changes = [
            sh - self.previous_sh[i] if same and i < len(self.previous_sh) else None
            for i, sh in enumerate(current_sh)]
        self.previous_rects = current
        self.previous_sh = current_sh
        self.previous_pattern = pattern

    def entities(self, s):
        size = s["size"]
        rows = []  # x0, y0, x1, y1, dx0, dy0, dx1, dy1, half, lethal_after, beneficial
        for e in s["enemies"]:
            beneficial = e["p"] >= 2
            if beneficial:
                box = (e["x"] - 4, e["y"] - 4, e["x"] + 4, e["y"] + 4)
            else:
                box = (e["x"], e["y"], e["x"] + e["s"], e["y"] + e["s"])
            scale = s["spd"] * s["frz"] * e["spd"]
            dx = 0.0 if e["p"] == -1 else e["vx"] * scale
            dy = 0.0 if e["p"] == -1 else e["vy"] * scale
            rows.append((*box, dx, dy, dx, dy, size if beneficial else size - 1, 0.0, beneficial))
        cp = s["cp"]
        if cp is not None:
            for i, r in enumerate(cp["rects"]):
                if (cp["id"] == 23 and cp["tpe"] == 1 and r["x"] == -50 and r["y"] == -50
                        and r["w"] == 0 and r["h"] == 0):
                    continue
                if cp["tpe"] == 1 or r["sh"] == 2:
                    lethal_after = 0.0
                elif 0 < r["sh"] < 2:
                    change = self.sh_changes[i] if i < len(self.sh_changes) else None
                    if change is None or change <= 0:
                        continue
                    lethal_after = (2.0 - r["sh"]) / change
                else:
                    continue
                box = (r["x"], r["y"], r["x"] + r["w"] - 1, r["y"] + r["h"] - 1)
                ch = self.rect_changes[i] if i < len(self.rect_changes) else (0.0,) * 4
                rows.append((*box, *ch, size - 2, lethal_after, False))
        return np.array(rows, dtype=float).reshape(-1, 11)

    def action(self, s):
        ents = self.entities(s)
        path = player_path(raw(s["px"]), raw(s["py"]), raw(s["dpx"]), raw(s["dpy"]))
        if len(ents) == 0:
            return 0
        x0, y0, x1, y1, cx0, cy0, cx1, cy1, half, lethal_after, beneficial = ents.T
        beneficial = beneficial.astype(bool)

        # nearest lethal entity, measured from the current position, unextrapolated
        px, py = s["px"], s["py"]
        dist = clearance(px - half, py - half, px + half, py + half, x0, y0, x1, y1)
        nearest = None
        for mask in (~beneficial & (lethal_after == 0.0),
                     ~beneficial & (lethal_after <= HORIZON)):
            if mask.any():
                idx = np.flatnonzero(mask)
                nearest = idx[np.argmin(dist[idx])]   # first minimum, like min_by
                break

        t = np.arange(HORIZON + 1, dtype=float)                        # (T,)
        bx0 = x0[None] + t[:, None] * cx0[None]                         # (T, N)
        by0 = y0[None] + t[:, None] * cy0[None]
        bx1 = x1[None] + t[:, None] * cx1[None]
        by1 = y1[None] + t[:, None] * cy1[None]
        cx = path[..., 0][..., None]                                   # (9, T, 1)
        cy = path[..., 1][..., None]
        b0x, b0y, b1x, b1y = cx - half, cy - half, cx + half, cy + half  # (9, T, N)
        clr = clearance(b0x, b0y, b1x, b1y, bx0[None], by0[None], bx1[None], by1[None])
        overlap = (b1x > bx0[None]) & (b1y > by0[None]) & (b0x < bx1[None]) & (b0y < by1[None])
        live = (~beneficial)[None, None] & (t[None, :, None] >= lethal_after[None, None])

        hit_t = np.where(overlap & live, t[None, :, None], np.inf).min(axis=(1, 2))
        free = np.where(np.isinf(hit_t), HORIZON + 1, hit_t)
        all_clr = np.where(live, clr, 1e9).min(axis=(1, 2))
        nearest_clr = (clr[:, HORIZON, nearest] if nearest is not None
                       and HORIZON >= lethal_after[nearest] else np.full(9, 1e9))
        if beneficial.any():
            power = clr[:, HORIZON][:, beneficial].min(axis=1)
            safe = (free > HORIZON) & (all_clr >= SAFE_CLEARANCE)
            if safe.any():
                return int(np.argmin(np.where(safe, power, np.inf)))
        best = 0
        for a in range(1, 9):
            if (free[a], nearest_clr[a], all_clr[a]) > (free[best], nearest_clr[best], all_clr[best]):
                best = a
        return best
