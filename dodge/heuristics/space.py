"""Space-control heuristic: stay away from enemies, stay where there is room,
and leave early when the room is closing.

Each frame, for each of the 9 actions held for H frames:
  1. simulate the player exactly and every enemy with its real steering rule
     (homing toward the player's predicted position, friction, wall bounces,
     kamikaze slow-down), blasts growing, rectangles extrapolated;
  2. reject actions that collide;
  3. score the end position by "room": over a grid, how much sooner the player
     can reach each cell than any threat can, summed.
No pattern-specific knowledge; no RNG.
"""
import numpy as np

from ..overlay import HAZARD, MUTED, PLAYER, THREAT, Overlay


ONE = 65536
F08, HALF = 52428, 32768
LO, HI = 2 * ONE, 125 * ONE
MASKS = [0, 1, 2, 4, 8, 5, 6, 9, 10]
DIRS = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                 [-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=float)
E_FRICTION = 64880 / ONE
E_ACCEL = 655 / ONE

CFG = dict(
    horizon=24,          # frames each action is held and checked for collisions
    room_tmax=60,        # how far ahead threats are projected for the room map
    room_cap=30.0,       # breathing room per cell is capped at this many frames
    grid=12,             # room map is grid x grid cells
    overlay_grid=48,     # the room map drawn this finely when the overlay is shown,
    overlay_tmax=120,    # and this far ahead
    w_clear=0.3,         # weight of nearest-threat clearance next to room
    clear_cap=16.0,
    centre_guard=35,     # leave the anti-camping box once its timer passes this
    w_sweep=1.0,         # penalty for ending in a sweeping rect's path with little time to exit
    sweep_margin=30.0,   # frames of slack wanted between exiting the band and the rect arriving
)


def player_paths(x, y, vx, vy, horizon):
    """(9, H+1, 2) exact player centres for t = 0..H."""
    out = np.empty((9, horizon + 1, 2))
    for a, mask in enumerate(MASKS):
        px, py, pvx, pvy = x, y, vx, vy
        out[a, 0] = (px / ONE, py / ONE)
        for t in range(1, horizon + 1):
            pvx = (pvx * F08) >> 16
            pvy = (pvy * F08) >> 16
            if mask & 1: pvx -= HALF
            if mask & 2: pvx += HALF
            if mask & 4: pvy -= HALF
            if mask & 8: pvy += HALF
            px = min(max(px + pvx, LO), HI)
            py = min(max(py + pvy, LO), HI)
            out[a, t] = (px / ONE, py / ONE)
    return out


def raw(v):
    return int(round(v * ONE))


def box_at(B, t):
    """Boxes (R, 12) at times t (T,) -> x0, y0, x1, y1 each (T, R).
    Columns: x0 y0 x1 y1 vx vy lethal_after half move_from grow emin emax.
    Moves at (vx, vy) from move_from on; each side grows by clip(grow*t, emin, emax)."""
    t = np.asarray(t, dtype=float)[:, None]
    m = np.maximum(t - B[:, 8][None], 0.0)
    e = np.clip(t * B[:, 9][None], B[:, 10][None], B[:, 11][None])
    return (B[:, 0][None] + m * B[:, 4][None] - e, B[:, 1][None] + m * B[:, 5][None] - e,
            B[:, 2][None] + m * B[:, 4][None] + e, B[:, 3][None] + m * B[:, 5][None] + e)


def gap(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    x = np.maximum(bx0 - ax1, ax0 - bx1)
    y = np.maximum(by0 - ay1, ay0 - by1)
    inside = (x <= 0) & (y <= 0)
    return np.where(inside, np.maximum(x, y), np.hypot(np.maximum(x, 0), np.maximum(y, 0)))


class SpaceBot:
    def __init__(self, cfg=None):
        self.cfg = dict(CFG, **(cfg or {}))
        self.prev_pattern = None
        self.prev_rects = []
        self.prev_sh = []
        self.rect_change = []
        self.sh_change = []
        self.prev_blasts = []       # (cx, cy, size)
        g = self.cfg["grid"]
        centres = (np.arange(g) + 0.5) * (128.0 / g)
        self.cx, self.cy = np.meshgrid(centres, centres, indexing="ij")
        self.cx, self.cy = self.cx.ravel(), self.cy.ravel()
        self.T = np.arange(0, self.cfg["room_tmax"] + 1, 4, dtype=float)
        self.T_view = np.arange(0, self.cfg["overlay_tmax"] + 1, 4, dtype=float)

    # ---- frame-to-frame tracking (rects like Taylor's; blasts by centre) ----
    def observe(self, s):
        cp = s["cp"]
        pid = None if cp is None else cp["id"]
        rects = [] if cp is None else [(r["x"], r["y"], r["x"] + r["w"] - 1, r["y"] + r["h"] - 1)
                                       for r in cp["rects"]]
        shs = [] if cp is None else [r["sh"] for r in cp["rects"]]
        same = pid == self.prev_pattern
        self.rect_change = [tuple(b - o for b, o in zip(box, self.prev_rects[i]))
                            if same and i < len(self.prev_rects) else (0.0,) * 4
                            for i, box in enumerate(rects)]
        self.sh_change = [sh - self.prev_sh[i] if same and i < len(self.prev_sh) else None
                          for i, sh in enumerate(shs)]
        self.prev_pattern, self.prev_rects, self.prev_sh = pid, rects, shs

    def _blast_growing(self, e):
        cx, cy = e["x"] + e["s"] / 2, e["y"] + e["s"] / 2
        for px, py, ps in self.prev_blasts:
            if abs(px - cx) < 0.75 and abs(py - cy) < 0.75:
                return e["s"] > ps
        return e["s"] < 30

    def act(self, s):
        cfg = self.cfg
        H = cfg["horizon"]
        size = s["size"]
        step = s["spd"] * s["frz"]
        paths = player_paths(raw(s["px"]), raw(s["py"]), raw(s["dpx"]), raw(s["dpy"]), H)

        lethal = [e for e in s["enemies"] if e["p"] < 2 and e["p"] != -1]
        blasts = [e for e in s["enemies"] if e["p"] == -1]
        growing = [self._blast_growing(e) for e in blasts]
        self.prev_blasts = [(e["x"] + e["s"] / 2, e["y"] + e["s"] / 2, e["s"]) for e in blasts]

        hit_t = np.full(9, H + 1, dtype=float)
        hit_p = np.full(9, H + 1, dtype=float)
        room_extra = []     # blasts born during a plan: room map only
        min_clear = np.full(9, 1e9)
        end_enemies = None
        traj, sizes, alive = [], None, None     # for overlay(): enemy centres per frame

        # -- lit rectangles reflect enemies: (x, y, w, h, dx, dy, mdx, mdy, lit_from)
        bounce = []
        cp = s["cp"]
        if cp is not None:
            for i, r in enumerate(cp["rects"]):
                if r["sh"] == 2:
                    lit_from = 0.0
                else:
                    ch = self.sh_change[i] if i < len(self.sh_change) else None
                    if not (0 < r["sh"] < 2) or ch is None or ch <= 0:
                        continue
                    lit_from = (2.0 - r["sh"]) / ch
                d = self.rect_change[i] if i < len(self.rect_change) else (0.0,) * 4
                bounce.append((r["x"], r["y"], r["w"], r["h"], r["dx"], r["dy"], d[0], d[1], lit_from))
        cap = self.cfg.get("bounce_cap", 0.6)

        # -- homing enemies, simulated jointly with each candidate path --------
        if lethal:
            n = len(lethal)
            ex = np.tile([e["x"] for e in lethal], (9, 1)).astype(float)
            ey = np.tile([e["y"] for e in lethal], (9, 1)).astype(float)
            vx = np.tile([e["vx"] for e in lethal], (9, 1)).astype(float)
            vy = np.tile([e["vy"] for e in lethal], (9, 1)).astype(float)
            sp = np.tile([e["spd"] for e in lethal], (9, 1)).astype(float)
            es = np.array([e["s"] for e in lethal], dtype=float)[None]
            kam = np.array([e["p"] == 1 for e in lethal])[None]
            inside = (((ex >= 1) & (ex <= 128)) | ((ex < 1) & (vx < 0)) | ((ex > 128) & (vx > 0)))
            half = size - 1
            alive = np.ones((9, n), dtype=bool)
            born = np.full((9, n), np.inf)          # frame a kamikaze turns into a blast
            bcx = np.zeros((9, n))
            bcy = np.zeros((9, n))
            # explode powerups (personality 2): picking one up detonates every kamikaze
            bombs = [e for e in s["enemies"] if e["p"] == 2]
            for t in range(0, H + 1):
                px = paths[:, t, 0][:, None]
                py = paths[:, t, 1][:, None]
                if t > 0:
                    vx = vx * E_FRICTION + E_ACCEL * np.sign(px - ex)
                    vy = vy * E_FRICTION + E_ACCEL * np.sign(py - ey)
                    for (rx, ry, rw, rh, rdx, rdy, mdx, mdy, lit_from) in bounce:
                        if t - 1 < lit_from:
                            continue
                        rx_t, ry_t = rx + mdx * (t - 1), ry + mdy * (t - 1)
                        hitr = ((ex + vx < rx_t + rw) & (ex + es > rx_t + rdx)
                                & (ey + es > ry_t + rdy) & (ey + vy < ry_t + rh))
                        if not hitr.any():
                            continue
                        nvx = np.where(ex + vx < rx_t, -np.maximum(np.abs(vx), cap),
                                       np.where(ex + vx + es > rx_t + rw, np.maximum(np.abs(vx), cap), vx))
                        nvy = np.where(ey + vy < ry_t, -np.maximum(np.abs(vy), cap),
                                       np.where(ey + vy + es > ry_t + rh, np.maximum(np.abs(vy), cap), vy))
                        nvx = np.sign(nvx) * np.minimum(np.abs(nvx), cap)
                        nvy = np.sign(nvy) * np.minimum(np.abs(nvy), cap)
                        vx = np.where(hitr, nvx, vx)
                        vy = np.where(hitr, nvy, vy)
                    d2 = (ex - px) ** 2 + (ey - py) ** 2
                    sp = np.where(kam, np.where(d2 <= 625, sp - 0.01,
                                                np.minimum(sp + 0.01, 1.0)), sp)
                    vy = np.where(inside & (ey >= 129 - es), -np.abs(vy),
                                  np.where(inside & (ey <= -2), np.abs(vy), vy))
                    vx = np.where(inside & (ex >= 129 - es), -np.abs(vx),
                                  np.where(inside & (ex <= -2), np.abs(vx), vx))
                    ex = ex + vx * step * sp
                    ey = ey + vy * step * sp
                    inside |= (ex >= 1) & (ex <= 128)
                    # enemy-enemy contact: both die; a kamikaze leaves a blast
                    both = (alive & inside)
                    touch = ((ex[:, :, None] + es[:, :, None] > ex[:, None, :])
                             & (ey[:, :, None] + es[:, :, None] > ey[:, None, :])
                             & (ex[:, :, None] < ex[:, None, :] + es[:, None, :])
                             & (ey[:, :, None] < ey[:, None, :] + es[:, None, :])
                             & both[:, :, None] & both[:, None, :])
                    touch[:, np.arange(n), np.arange(n)] = False
                    dying = touch.any(axis=2)
                    # picking up an explode powerup this frame detonates every kamikaze
                    for b in bombs:
                        bx = b["x"] + b["vx"] * b["spd"] * step * t
                        by = b["y"] + b["vy"] * b["spd"] * step * t
                        got = ((px[:, 0] + size > bx - 4) & (py[:, 0] + size > by - 4)
                               & (px[:, 0] - size < bx + 4) & (py[:, 0] - size < by + 4))
                        dying |= got[:, None] & kam & alive
                    newly = dying & kam & np.isinf(born)
                    bcx = np.where(newly, ex + es / 2, bcx)
                    bcy = np.where(newly, ey + es / 2, bcy)
                    born = np.where(newly, t, born)
                    alive &= ~dying
                touching = ((px + half > ex) & (py + half > ey)
                            & (px - half < ex + es) & (py - half < ey + es))
                # pessimistic: predicted enemy-enemy kills are not relied on
                hit_p = np.where(touching.any(axis=1) & (hit_p > H), t, hit_p)
                over = (touching & alive).any(axis=1)
                hb = np.clip((t - born) * s["frz"], 0, 30) / 2      # blast half-size
                live_b = np.isfinite(born) & (born < t)
                over |= ((px + half > bcx - hb) & (py + half > bcy - hb)
                         & (px - half < bcx + hb) & (py - half < bcy + hb) & live_b).any(axis=1)
                hit_t = np.where(over & (hit_t > H), t, hit_t)
                hit_p = np.minimum(hit_p, hit_t)
                clr = gap(px - half, py - half, px + half, py + half, ex, ey, ex + es, ey + es)
                clr = np.where(alive, clr, 1e9)
                bclr = gap(px - half, py - half, px + half, py + half,
                           bcx - 15, bcy - 15, bcx + 15, bcy + 15)
                clr = np.minimum(clr, np.where(np.isfinite(born), bclr, 1e9))
                min_clear = np.minimum(min_clear, clr.min(axis=1))
                traj.append((ex + es / 2, ey + es / 2))
            sizes = es[0]
            # a kamikaze can become a 30-wide blast: give it 15 extra reach in the room map
            es_reach = np.where(kam, es + 30, es) * np.ones_like(ex)
            ex_r = np.where(alive, np.where(kam, ex - 15, ex), 1e6)   # dead: far away
            ey_r = np.where(alive, np.where(kam, ey - 15, ey), 1e6)
            end_enemies = (ex_r, ey_r, vx, vy, sp, es_reach)
            # blasts born during the plan sit where they were born, full size
            for k in np.flatnonzero(np.isfinite(born).any(axis=0)):
                a = int(np.argmax(np.isfinite(born[:, k])))
                c = (bcx[a, k], bcy[a, k])
                room_extra.append((c[0] - 15, c[1] - 15, c[0] + 15, c[1] + 15,
                                   0.0, 0.0, 0.0, size - 1, 0.0, 0.0, 0.0, 0.0))

        # -- blasts: each side grows frz/2 per frame up to 30 wide, then shrinks
        frz = s["frz"]
        static_boxes = []   # see box_at for the 12 columns
        sweepers = []       # moving rects
        for e, grow in zip(blasts, growing):
            x0, y0, w = e["x"], e["y"], e["s"]
            if grow:
                g, lo, hi = frz / 2, 0.0, (30.0 - w) / 2
            else:
                g, lo, hi = -frz / 2, -w / 2, 0.0
            static_boxes.append((x0, y0, x0 + w, y0 + w, 0.0, 0.0, 0.0, size - 1, 0.0, g, lo, hi))
        # -- pattern rectangles: Taylor's rules, generic ----------------------
        cp = s["cp"]
        if cp is not None:
            for i, r in enumerate(cp["rects"]):
                if r["w"] == 0 and r["h"] == 0 and (r["x"] < -20 or r["y"] < -20):
                    continue
                if cp["tpe"] == 1 or r["sh"] == 2:
                    after = 0.0
                elif 0 < r["sh"] < 2:
                    ch = self.sh_change[i] if i < len(self.sh_change) else None
                    if ch is None or ch <= 0:
                        continue
                    after = (2.0 - r["sh"]) / ch
                else:
                    continue
                d = self.rect_change[i] if i < len(self.rect_change) else (0.0,) * 4
                vx_r, vy_r, move_from = d[0], d[1], 0.0
                # a moving rect shows its direction (the warning arrow) before it moves;
                # until it is seen moving, assume it sweeps that way at its speed
                if cp["tpe"] == 1 and not any(d) and (r["dx"] or r["dy"]):
                    v = r["spd"] * frz
                    vx_r, vy_r = np.sign(r["dx"]) * v, np.sign(r["dy"]) * v
                    if r["sh"] < 2:
                        ch = self.sh_change[i] if i < len(self.sh_change) else None
                        move_from = (2.0 - r["sh"]) / (ch if ch and ch > 0 else 0.02 * frz)
                static_boxes.append((r["x"], r["y"], r["x"] + r["w"] - 1, r["y"] + r["h"] - 1,
                                     vx_r, vy_r, after, size - 2, move_from, 0.0, 0.0, 0.0))
                if vx_r or vy_r:
                    sweepers.append(static_boxes[-1])
        if static_boxes:
            B = np.array(static_boxes)
            t = np.arange(H + 1, dtype=float)
            bx0, by0, bx1, by1 = box_at(B, t)
            live = t[:, None] >= B[:, 6][None]
            hh = B[:, 7][None, None]
            px = paths[..., 0][..., None]
            py = paths[..., 1][..., None]
            over = ((px + hh > bx0[None]) & (py + hh > by0[None])
                    & (px - hh < bx1[None]) & (py - hh < by1[None])) & live[None]
            first = np.where(over.any(axis=2), np.arange(H + 1)[None], H + 1).min(axis=1)
            hit_t = np.minimum(hit_t, first)
            hit_p = np.minimum(hit_p, first)
            clr = gap(px - hh, py - hh, px + hh, py + hh, bx0[None], by0[None], bx1[None], by1[None])
            min_clear = np.minimum(min_clear, np.where(live[None], clr, 1e9).min(axis=(1, 2)))

        # -- room at the end of each path --------------------------------------
        room = self._room(paths[:, -1], size, step, end_enemies, static_boxes + room_extra, H)
        score = room + cfg["w_clear"] * np.minimum(min_clear, cfg["clear_cap"]) / cfg["clear_cap"]
        if sweepers:
            score = score - cfg["w_sweep"] * self._sweep_risk(paths[:, -1], sweepers, H)

        # anti-camping box: don't end inside it once its timer is high
        if s["fy"] and s["ftimer"] + H > 50 + (50 - cfg["centre_guard"]):
            end = paths[:, -1]
            in_box = ((end[:, 0] + size - 2 > 60) & (end[:, 1] + size - 2 > 60)
                      & (end[:, 0] - size + 2 < 68) & (end[:, 1] - size + 2 < 68))
            score = score - 2.0 * in_box

        best = 0
        for a in range(1, 9):
            if (hit_p[a], hit_t[a], score[a]) > (hit_p[best], hit_t[best], score[best]):
                best = a
        self.last = (float(hit_t[best]), float(room[best]), len(lethal), float(hit_p[best]),
                     hit_p.copy(), hit_t.copy())
        self._decision = dict(best=best, path=paths[best], traj=traj, sizes=sizes,
                              alive=None if alive is None else alive[best],
                              boxes=np.array(static_boxes) if static_boxes else None, H=H,
                              # the room map's inputs for the chosen plan, redrawn finer on demand
                              room=(paths[best:best + 1, -1], size, step,
                                    None if end_enemies is None
                                    else tuple(a[best:best + 1] for a in end_enemies),
                                    static_boxes + room_extra, H))
        return best

    def overlay(self):
        """The last decision as an Overlay: breathing room as the field, the chosen
        path, each enemy's simulated path under it, and hazards where they will be."""
        d = self._decision
        # Decisions use a coarse grid, and clip the margin to [0, room_cap] frames;
        # the picture samples the same map finely for the chosen plan, unclipped and
        # further ahead, so it shows how much room there is, not only whether
        # there is enough.
        g = self.cfg["overlay_grid"]
        centres = (np.arange(g) + 0.5) * (128.0 / g)
        cy, cx = np.meshgrid(centres, centres, indexing="ij")     # rows are y
        room = self._room_map(*d["room"], cx.ravel(), cy.ravel(), T=self.T_view, margin=True)
        out = Overlay(field=room[0].reshape(g, g))
        out.paths.append(([tuple(p) for p in d["path"]], PLAYER))
        x, y = d["path"][-1]        # the room map is measured from here
        out.boxes.append((x - 2, y - 2, x + 2, y + 2, PLAYER))
        if d["traj"]:
            best = d["best"]
            for k, s in enumerate(d["sizes"]):
                points = [(cx[best, k], cy[best, k]) for cx, cy in d["traj"]]
                colour = THREAT if d["alive"][k] else MUTED
                out.paths.append((points, colour))
                x, y = points[-1]
                out.boxes.append((x - s / 2, y - s / 2, x + s / 2, y + s / 2, colour))
        if d["boxes"] is not None:
            B = d["boxes"]
            now = box_at(B, [0])
            end = box_at(B, [d["H"]])
            for i in range(len(B)):
                x0, y0, x1, y1 = (a[0, i] for a in end)
                out.boxes.append((x0, y0, x1, y1, HAZARD))
                (a0, b0, a1, b1) = (a[0, i] for a in now)
                if B[i, 4] or B[i, 5]:
                    out.paths.append(([((a0 + a1) / 2, (b0 + b1) / 2),
                                       ((x0 + x1) / 2, (y0 + y1) / 2)], HAZARD))
        return out

    def _sweep_risk(self, ends, sweepers, H):
        """A rect sweeping across the screen can only be escaped sideways, out of its
        band. Risk in [0, 1]: how little time is left to get out before it arrives."""
        B = np.array(sweepers)
        x0, y0, x1, y1 = (a[0] for a in box_at(B, [H]))
        vx, vy, move_from, hh = B[:, 4], B[:, 5], B[:, 8], B[:, 7]
        ex, ey = ends[:, 0:1], ends[:, 1:2]                         # (9, 1)
        along_x = np.abs(vx) >= np.abs(vy)
        speed = np.maximum(np.where(along_x, np.abs(vx), np.abs(vy)), 1e-6)
        # distance from the rect's leading edge to the player, along the motion
        ahead = np.where(along_x,
                         np.where(vx > 0, (ex - hh) - x1, x0 - (ex + hh)),
                         np.where(vy > 0, (ey - hh) - y1, y0 - (ey + hh)))
        behind = np.where(along_x,
                          np.where(vx > 0, ex + hh <= x0, ex - hh >= x1),
                          np.where(vy > 0, ey + hh <= y0, ey - hh >= y1))
        # sideways exits: to just past either long side, if that is on screen
        lo_edge = np.where(along_x, y0, x0) - hh - 1
        hi_edge = np.where(along_x, y1, x1) + hh + 1
        pos = np.where(along_x, ey, ex)
        in_band = (pos > lo_edge) & (pos < hi_edge)
        d_lo = np.where(lo_edge >= 2, pos - lo_edge, np.inf)
        d_hi = np.where(hi_edge <= 125, hi_edge - pos, np.inf)
        t_exit = np.minimum(d_lo, d_hi) / 2.5
        t_arrive = np.maximum(ahead, 0) / speed + np.maximum(move_from - H, 0)
        margin = self.cfg["sweep_margin"]
        risk = np.clip((t_exit + margin - t_arrive) / margin, 0, 1)
        risk = np.where(in_band & ~behind, risk, 0.0)
        return risk.max(axis=1)

    def _room(self, ends, size, step, enemies, static_boxes, H):
        """Mean capped breathing room (threat arrival - player arrival) over the grid."""
        cx, cy = self.cx, self.cy
        valid = (cx >= 2) & (cx <= 125) & (cy >= 2) & (cy <= 125)
        roomy = self._room_map(ends, size, step, enemies, static_boxes, H, cx, cy)
        return (roomy * valid[None]).sum(axis=1) / valid.sum()

    def _room_map(self, ends, size, step, enemies, static_boxes, H, cx, cy, T=None, margin=False):
        """Breathing room in [0, 1] at each point (cx, cy), for each plan end in `ends`.

        With `margin`, the unclipped margin in frames instead: threat arrival minus
        player arrival, negative where a threat gets there first, and T's last
        value standing in for threats that never arrive within T."""
        cfg = self.cfg
        T = self.T if T is None else T
        drift = 99.0 * (1 - 0.99 ** T)          # sum of friction-decayed velocity
        spread = T - drift                      # extra reach from steering, per unit speed
        threat = np.full((len(ends), cx.size), np.inf)
        if enemies is not None:
            ex, ey, vx, vy, sp, es = enemies        # (9, N)
            s_ = (step * sp)[..., None]             # (9, N, 1)
            cen_x = (ex + es / 2)[..., None] + s_ * vx[..., None] * drift[None, None]
            cen_y = (ey + es / 2)[..., None] + s_ * vy[..., None] * drift[None, None]
            reach = s_ * spread[None, None] + (es / 2)[..., None] + size  # (9, N, T)
            ok = ((np.abs(cx[None, None, None] - cen_x[..., None]) <= reach[..., None])
                  & (np.abs(cy[None, None, None] - cen_y[..., None]) <= reach[..., None]))
            arrival = np.where(ok, T[None, None, :, None], np.inf).min(axis=2)  # (9, N, C)
            threat = np.minimum(threat, arrival.min(axis=1))
        if static_boxes:
            B = np.array(static_boxes)
            # positions continue from the end of the plan: time offset H
            tt = T + H
            x0, y0, x1, y1 = (a.T for a in box_at(B, tt))
            x0, y0, x1, y1 = x0 - size, y0 - size, x1 + size, y1 + size
            live = tt[None] >= B[:, 6][:, None]
            inbox = ((cx[None, None] > x0[..., None]) & (cx[None, None] < x1[..., None])
                     & (cy[None, None] > y0[..., None]) & (cy[None, None] < y1[..., None])
                     & live[..., None])                                     # (R, T, C)
            arr = np.where(inbox, T[None, :, None], np.inf).min(axis=(0, 1))
            threat = np.minimum(threat, arr[None])
        dist = np.maximum(np.abs(cx[None] - ends[:, 0:1]), np.abs(cy[None] - ends[:, 1:2]))
        t_player = dist / 2.5
        if margin:
            return np.minimum(threat, T[-1]) - t_player
        return np.clip(threat - t_player, 0.0, cfg["room_cap"]) / cfg["room_cap"]
