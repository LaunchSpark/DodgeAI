"""The paper's figures, as the dashboard draws them, exported to vector PDF.

Survival, death causes, pattern hazard and pattern lethality are dodge/charts.py's
own Evaluation-page charts on the eval1024 runs; the decision-frequency grid and
the death paths are new charts in the same style. All render with the light theme
of web/src/lib/vega-theme.ts, sized for IEEEtran (column 252 pt, text 516 pt).

    python paper_figs.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import altair as alt
import vl_convert as vlc

sys.path.insert(0, "D:/projects/DodgeAI/.claude/worktrees/train-plan")
from dodge import charts  # noqa: E402

HERE = Path(__file__).parent / "runs"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "paper1024" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
STATS = json.loads((HERE / "paper1024" / "stats.json").read_text())
TEL, SUM = STATS["telemetry"], STATS["summary"]

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CATEGORY = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FONT = "Helvetica, Arial, sans-serif"
SIZE = 9
CONFIG = {
    "background": "white", "font": FONT, "view": {"stroke": None},
    "padding": {"left": 2, "right": 4, "top": 2, "bottom": 2},
    "range": {"category": CATEGORY}, "mark": {"color": CATEGORY[0]},
    "line": {"strokeWidth": 1.6, "strokeCap": "round"},
    "point": {"stroke": SURFACE, "strokeWidth": 1},
    "bar": {"stroke": SURFACE, "strokeWidth": 1, "cornerRadiusEnd": 2},
    "rule": {"color": MUTED}, "text": {"color": INK, "fontSize": SIZE - 1},
    "axis": {"labelColor": MUTED, "titleColor": SECONDARY, "titleFontWeight": 500,
             "labelFontSize": SIZE - 1, "titleFontSize": SIZE, "gridColor": GRID,
             "domainColor": AXIS, "tickColor": AXIS, "labelFont": FONT, "titleFont": FONT},
    "axisX": {"grid": False},
    "legend": {"labelColor": SECONDARY, "titleColor": SECONDARY, "labelFontSize": SIZE - 1,
               "symbolType": "circle", "symbolSize": 40},
    "header": {"labelFontSize": SIZE, "labelColor": SECONDARY, "titleFontSize": SIZE, "labelFont": FONT},
}
COLUMN, TEXT, HEIGHT = 250, 516, 150


def save(name, spec, width, height=HEIGHT):
    spec = {k: v for k, v in spec.items() if k != "usermeta"}
    if "hconcat" not in spec and "facet" not in spec:
        spec["width"], spec["height"] = width, height
    (OUT / f"{name}.pdf").write_bytes(vlc.vegalite_to_pdf(spec, config=CONFIG))
    (OUT / f"{name}.svg").write_text(vlc.vegalite_to_svg(spec, config=CONFIG), encoding="utf-8")


LABELS = {"k1_argmax": "Best, 1f argmax", "k1_sampled": "Best, 1f sampled", "k4_argmax": "Best, 4f argmax",
          "k4_sampled": "Best, 4f sampled", "c100": "can-reach-100, 1f", "random": "Random"}

# Fig. survival: the Evaluation page's survival curves, one line per condition.
curves = {LABELS[k]: TEL[k]["survival"] for k in LABELS if k in TEL}
spec = charts.survival_curves(curves)
# Six sources: the page's one-row top legend, wrapped to three columns for a paper column.
for layer in spec["layer"]:
    color = layer.get("encoding", {}).get("color", {})
    if isinstance(color, dict) and "legend" in color and color["legend"]:
        color["legend"].update(columns=3, orient="top")
save("fig_survival", spec, COLUMN, 170)

# Fig. K grid: mean and Q1 by decision interval and action mode, 95% bootstrap intervals.
rows = []
for k in (1, 4):
    for mode in ("argmax", "sampled"):
        s = SUM.get(f"k{k}_{mode}")
        if s:
            for stat, ci in (("mean", "mean_ci"), ("q1", "q1_ci")):
                rows.append({"K": f"{k} frame{'s' if k > 1 else ''}", "mode": mode,
                             "stat": "Mean" if stat == "mean" else "Worst quarter (Q1)",
                             "v": s[stat], "lo": s[ci][0], "hi": s[ci][1]})
for r in rows:
    r["cond"] = f"{r['K'].split()[0]}f {r['mode']}"
conds = [f"{k}f {m}" for k in (1, 4) for m in ("argmax", "sampled")]
base = alt.Chart(alt.Data(values=rows)).encode(
    x=alt.X("cond:N", title=None, sort=conds,
            axis=alt.Axis(labelAngle=0, labelFontSize=7, labelExpr="split(datum.label, ' ')")))
bars = base.mark_bar(width={"band": 0.6}).encode(
    y=alt.Y("v:Q", title="Alive frames", axis=alt.Axis(format=",d")),
    color=alt.Color("mode:N", title=None, scale=alt.Scale(domain=["argmax", "sampled"]), legend=None))
err = base.mark_rule(color=INK, strokeWidth=1).encode(y="lo:Q", y2="hi:Q")
grid = alt.layer(bars, err).properties(width=128, height=130).facet(
    column=alt.Column("stat:N", title=None, sort=["Mean", "Worst quarter (Q1)"])).resolve_scale(y="independent")
save("fig_k_grid", grid.to_dict(), COLUMN)

# Fig. causes: the page's death causes by window, 1f and 4f argmax side by side.
panels = []
for key, title in (("k1_argmax", "One frame per decision"), ("k4_argmax", "Four frames per decision")):
    if key in TEL:
        spec = charts.causes_by_window(TEL[key]["by_window"])
        spec = {k: v for k, v in spec.items() if k not in ("usermeta", "$schema", "config")}
        spec.update(width=230, height=150, title={"text": title, "fontSize": SIZE, "color": SECONDARY, "font": FONT})
        panels.append(spec)
if panels:
    save("fig_causes", {"$schema": "https://vega.github.io/schema/vega-lite/v6.json", "hconcat": panels,
                        "resolve": {"legend": {"color": "shared"}}}, TEXT)

# Fig. hazard: the page's deaths per 10,000 frames with and without a pattern, by window.
if "k1_argmax" in TEL:
    hazard = {"windows": TEL["k1_argmax"]["hazard_windows"]}
    save("fig_hazard", charts.pattern_hazard(hazard), COLUMN, 150)
    # Fig. patterns: the page's per-pattern lethality with 95% intervals.
    save("fig_patterns", charts.pattern_lethality(TEL["k1_argmax"]["patterns"]), TEXT, 170)

# Fig. paths: last 60 frames before death inside the four deadliest static patterns.
paths_file = HERE / "paper1024" / "paths.json"
if paths_file.exists():
    P = json.loads(paths_file.read_text())
    types = {int(k): v for k, v in P["types"].items()}
    static = [d for d in P["deaths"] if d["pid"] != -1 and types.get(d["pid"]) == "static" and d["replay_match"]]
    counts = {}
    for d in static:
        counts[d["pid"]] = counts.get(d["pid"], 0) + 1
    top = sorted(counts, key=lambda p: -counts[p])[:4]
    line_rows, end_rows, rect_rows = [], [], []
    for pid in top:
        mine = sorted((d for d in static if d["pid"] == pid), key=lambda d: d["seed"])[:45]
        for d in mine:
            cause = "Died to a rectangle" if d["cause"] == "rect" else "Died to an enemy"
            for i, (x, y) in enumerate(zip(d["x"], d["y"])):
                line_rows.append({"pid": f"Pattern {pid} ({counts[pid]} deaths)", "seed": d["seed"], "i": i,
                                  "x": x, "y": y, "cause": cause})
            end_rows.append({"pid": f"Pattern {pid} ({counts[pid]} deaths)", "x": d["x"][-1], "y": d["y"][-1], "cause": cause})
            for x, y, w, h, lethal in d["rects"]:
                if lethal:
                    rect_rows.append({"pid": f"Pattern {pid} ({counts[pid]} deaths)", "x0": x, "y0": y, "x1": x + w, "y1": y + h})
    order = [f"Pattern {p} ({counts[p]} deaths)" for p in top]
    sx = alt.Scale(domain=[0, 128], nice=False)
    sy = alt.Scale(domain=[128, 0], nice=False)
    causes = ["Died to a rectangle", "Died to an enemy"]
    rects = alt.Chart().mark_rect(color=MUTED, opacity=0.04).encode(
        x=alt.X("x0:Q", scale=sx, title=None, axis=None), x2="x1:Q",
        y=alt.Y("y0:Q", scale=sy, title=None, axis=None), y2="y1:Q")
    lines = alt.Chart().mark_line(strokeWidth=0.8, opacity=0.6).encode(
        x=alt.X("x:Q", scale=sx), y=alt.Y("y:Q", scale=sy), detail="seed:N", order="i:Q",
        color=alt.Color("cause:N", title=None, scale=alt.Scale(domain=causes), legend=alt.Legend(orient="top")))
    ends = alt.Chart().mark_point(filled=True, size=14, opacity=0.9).encode(
        x=alt.X("x:Q", scale=sx), y=alt.Y("y:Q", scale=sy), color=alt.Color("cause:N", scale=alt.Scale(domain=causes)))
    datasets = {"lines": line_rows, "ends": end_rows, "rects": rect_rows}
    # One panel per pattern; each layer reads its own named dataset, filtered to the panel.
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "datasets": datasets,
            "hconcat": []}
    for name in order:
        sub = []
        for layer_chart, ds in ((rects, "rects"), (lines, "lines"), (ends, "ends")):
            d = layer_chart.to_dict()
            d.pop("$schema", None)
            d.pop("config", None)
            d.pop("datasets", None)
            d["data"] = {"name": ds}
            d["transform"] = [{"filter": {"field": "pid", "equal": name}}]
            sub.append(d)
        spec["hconcat"].append({"layer": sub, "width": 112, "height": 112,
                                "title": {"text": name, "fontSize": SIZE - 1, "color": SECONDARY, "font": FONT},
                                "view": {"stroke": AXIS}})
    save("fig_paths", spec, TEXT)
print("figures:", sorted(p.name for p in OUT.glob("*.pdf")))
