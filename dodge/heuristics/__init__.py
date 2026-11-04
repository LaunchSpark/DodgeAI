"""Hand-written, non-learned players, each deciding from `engine.lane_state(0)`.

Every bot has `observe(state)` (called once per frame, before deciding) and
`act(state) -> action index` (0-8 in the game's action order). A bot may also
have `overlay() -> dodge.overlay.Overlay`, a picture of its last decision for
the watch window; it is only called when the viewer turns overlays on.
"""
from ..overlay import HAZARD, MUTED, PICKUP, PLAYER, THREAT, Overlay
from .m0 import HORIZON, Policy as _M0, player_path, raw
from .space import SpaceBot


class TaylorM0(_M0):
    """Taylor Turner's 12-frame AABB dodger, under the shared `act` name.

    m0.py stays a line-for-line port; the overlay rebuilds his prediction
    from the same state rather than reaching into it.
    """

    def act(self, state):
        self._state = state
        self._chosen = self.action(state)
        return self._chosen

    def overlay(self):
        """Every entity's straight-line extrapolation to frame 12 and the chosen path."""
        s = self._state
        out = Overlay()
        path = player_path(raw(s["px"]), raw(s["py"]), raw(s["dpx"]), raw(s["dpy"]))[self._chosen]
        out.paths.append(([tuple(p) for p in path], PLAYER))
        enemies = s["enemies"]
        # entities() lists every enemy first, in order, then the pattern rects
        for i, (x0, y0, x1, y1, cx0, cy0, cx1, cy1, _, after, _) in enumerate(self.entities(s)):
            ex0, ey0 = x0 + HORIZON * cx0, y0 + HORIZON * cy0
            ex1, ey1 = x1 + HORIZON * cx1, y1 + HORIZON * cy1
            if i < len(enemies):
                p = enemies[i]["p"]
                colour = PICKUP if p >= 2 else HAZARD if p == -1 else THREAT
            else:
                colour = MUTED if after > HORIZON else HAZARD
            out.paths.append(([((x0 + x1) / 2, (y0 + y1) / 2), ((ex0 + ex1) / 2, (ey0 + ey1) / 2)],
                              colour))
            out.boxes.append((ex0, ey0, ex1, ey1, colour))
        return out


BOTS = {
    "space": ("Space bot", SpaceBot),
    "taylor-m0": ("Taylor's m0 heuristic", TaylorM0),
}
