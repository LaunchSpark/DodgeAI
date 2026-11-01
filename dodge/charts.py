"""The dashboard's charts, as Vega-Lite specs built with Altair.

Specs carry data and encoding only. Colours, fonts and ink come from the
frontend's theme config, so one spec renders correctly in light and dark mode;
every categorical scale has a fixed domain so a series keeps its colour (taken
from the theme's category range in that order) whatever else is on screen.
"""

import altair as alt

from .metrics import CAUSES

alt.data_transformers.disable_max_rows()

CAUSE_LABELS = {
    "normal": "Normal enemy",
    "kamikaze": "Kamikaze (body + blast)",
    "static_rect": "Static pattern rectangle",
    "moving_rect": "Moving pattern rectangle",
}
CAUSE_ORDER = [CAUSE_LABELS[cause] for cause in CAUSES]
QUARTERS = ["Worst quarter (Q1)", "Mean", "Best quarter (Q4)"]
HAZARD_SERIES = ["Pattern active", "No pattern"]

WIDTH = "container"
HEIGHT = 220


def _spec(chart, table=None):
    """The chart's spec; `table` replaces the plotted rows in the table view."""
    spec = chart.properties(width=WIDTH, height=HEIGHT).to_dict()
    if table is not None:
        spec["usermeta"] = {"table": table}
    return spec


def _percent(share):
    return None if share is None else round(share * 100, 1)


def _hover_line(base, x, y_fields, tooltip):
    """Lines with a nearest-x crosshair, a dot on each line and a tooltip."""
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x],
                                  empty=False, clear="pointerout")
    rule = base.mark_rule(strokeWidth=1).encode(
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        tooltip=tooltip,
    ).add_params(nearest)
    points = base.mark_point(filled=True, size=64).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    return rule, points


def training_series(series, title, unit="", per="Episodes"):
    """One run metric over its samples: the raw points faint, the trend solid."""
    rows = [{"episode": (index + 1) * series.interval, "value": value, "trend": trend}
            for index, (value, trend) in enumerate(zip(series.values, series.trend))]
    suffix = f" ({unit})" if unit else ""
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("episode:Q", title=per, axis=alt.Axis(format=",d")),
    )
    raw = base.mark_line(strokeWidth=1, opacity=0.3).encode(
        y=alt.Y("value:Q", title=f"{title}{suffix}", scale=alt.Scale(zero=False)))
    trend = base.mark_line(strokeWidth=2).encode(y="trend:Q")
    tooltip = [alt.Tooltip("episode:Q", title=per.rstrip("s"), format=","),
               alt.Tooltip("trend:Q", title=f"{title} (trend)", format=",.2f"),
               alt.Tooltip("value:Q", title=title, format=",.2f")]
    rule, points = _hover_line(base.encode(y="trend:Q"), "episode", ["trend"], tooltip)
    table = [{per.rstrip("s"): row["episode"], title: row["value"],
              f"{title} (trend)": row["trend"]} for row in rows]
    return _spec(alt.layer(raw, trend, rule, points), table)


def quarters_over_training(rows):
    """Q1, mean and Q4 of recent games at each update: is the lower tail moving?"""
    long = [{"update": row["update"], "measure": label, "frames": row[key]}
            for row in rows
            for key, label in zip(("q1", "mean", "q4"), QUARTERS)]
    color = alt.Color("measure:N", title=None, scale=alt.Scale(domain=QUARTERS),
                      legend=alt.Legend(orient="top"))
    base = alt.Chart(alt.Data(values=long)).encode(
        x=alt.X("update:Q", title="PPO update", axis=alt.Axis(format=",d", tickMinStep=1)),
        y=alt.Y("frames:Q", title="Alive frames"),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["update"],
                                  empty=False, clear="pointerout")
    wide = alt.Chart(alt.Data(values=rows)).mark_rule(strokeWidth=1).encode(
        x="update:Q",
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        tooltip=[alt.Tooltip("update:Q", title="Update", format=","),
                 alt.Tooltip("q1:Q", title=QUARTERS[0], format=",.0f"),
                 alt.Tooltip("mean:Q", title=QUARTERS[1], format=",.0f"),
                 alt.Tooltip("q4:Q", title=QUARTERS[2], format=",.0f"),
                 alt.Tooltip("games:Q", title="Games", format=",")],
    ).add_params(nearest)
    points = base.mark_point(filled=True, size=64).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    table = [{"Update": row["update"], QUARTERS[0]: row["q1"], QUARTERS[1]: row["mean"],
              QUARTERS[2]: row["q4"], "Games": row["games"]} for row in rows]
    return _spec(alt.layer(lines, wide, points), table)


def survival_curves(curves):
    """Share of games still alive at each 1,000 frames, one line per source."""
    names = list(curves)
    rows = [{"source": name, "frames": point["frames"], "alive": point["alive"]}
            for name, points in curves.items() for point in points]
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("frames:Q", title="Gameplay frames survived", axis=alt.Axis(format=",d")),
        y=alt.Y("alive:Q", title="Games still alive", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("source:N", title=None, scale=alt.Scale(domain=names),
                        legend=alt.Legend(orient="top") if len(names) > 1 else None),
    )
    lines = base.mark_line(strokeWidth=2, interpolate="step-after")
    tooltip = [alt.Tooltip("source:N", title="Games"),
               alt.Tooltip("frames:Q", title="Frames", format=","),
               alt.Tooltip("alive:Q", title="Still alive", format=".1%")]
    rule, points = _hover_line(base, "frames", ["alive"], tooltip)
    # Sources can step at different frame counts, so rows join on frames.
    table = {}
    for row in rows:
        table.setdefault(row["frames"], {"Frames": row["frames"]})[
            f"{row['source']} (% alive)"] = _percent(row["alive"])
    table = [table[frames] for frames in sorted(table)]
    return _spec(alt.layer(lines, rule.encode(color=alt.value("#898781")), points), table)


def causes_by_window(rows):
    """What ended each game, by the window it ended in (100% stacked bars)."""
    long = []
    for row in rows:
        for cause in CAUSES:
            long.append({"window": row["window"], "bucket": row.get("bucket"),
                         "cause": CAUSE_LABELS[cause],
                         "share": row["causes"][cause], "games": row["games"],
                         "window_share": row["share_of_games"],
                         "order": CAUSES.index(cause)})
    windows = [row["window"] for row in rows]
    base = alt.Chart(alt.Data(values=long)).encode(
        x=alt.X("window:N", title="Frames at death (shortest 10% … longest 10%)",
                sort=windows, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share:Q", title="Deaths in window", stack="zero",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        order=alt.Order("order:Q"),
    )
    bars = base.mark_bar(width={"band": 0.6}).encode(
        color=alt.Color("cause:N", title=None, scale=alt.Scale(domain=CAUSE_ORDER),
                        legend=alt.Legend(orient="top", columns=2)),
        tooltip=[alt.Tooltip("window:N", title="Frames"),
                 alt.Tooltip("bucket:N", title="Deaths"),
                 alt.Tooltip("cause:N", title="Cause"),
                 alt.Tooltip("share:Q", title="Share of deaths", format=".0%"),
                 alt.Tooltip("games:Q", title="Deaths in window", format=","),
                 alt.Tooltip("window_share:Q", title="Share of all deaths", format=".0%")],
    )
    # Visible labels are the relief for the two light-mode fills under 3:1.
    # Inside a fill the ink is white, except on yellow (the fourth slot).
    labels = alt.Chart(alt.Data(values=long)).transform_stack(
        stack="share", as_=["y0", "y1"], groupby=["window"],
        sort=[alt.SortField("order")],
    ).transform_calculate(
        middle="(datum.y0 + datum.y1) / 2",
    ).transform_filter("datum.share >= 0.08").mark_text(fontSize=11).encode(
        x=alt.X("window:N", sort=windows),
        y="middle:Q",
        text=alt.Text("share:Q", format=".0%"),
        color=alt.condition(f"datum.cause === '{CAUSE_ORDER[3]}'",
                            alt.value("#0b0b0b"), alt.value("#ffffff")),
    )
    table = [{"Window": row.get("bucket"), "Frames": row["window"], "Deaths": row["games"],
              "% of all deaths": _percent(row["share_of_games"]),
              **{f"{CAUSE_LABELS[cause]} %": _percent(row["causes"][cause])
                 for cause in CAUSES}} for row in rows]
    return _spec(alt.layer(bars, labels), table)


def pattern_hazard(hazard):
    """Deaths per 10,000 frames lived with and without a pattern, by window."""
    long = []
    for row in hazard["windows"]:
        for key, label in (("active", HAZARD_SERIES[0]), ("inactive", HAZARD_SERIES[1])):
            if row[key] is not None and row[key] > 0:
                long.append({"window": row["window"], "bucket": row.get("bucket"),
                             "series": label, "rate": row[key]})
    windows = [row["window"] for row in hazard["windows"]]
    # Lines with markers rather than bars: on a log scale a bar has no zero to
    # stand on, and the gap between the two lines is the reading.
    chart = alt.Chart(alt.Data(values=long)).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=64)).encode(
        x=alt.X("window:N", title="Frames (shortest 10% … longest 10% of deaths)",
                sort=windows, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Deaths per 10,000 frames", scale=alt.Scale(type="log")),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=HAZARD_SERIES),
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=HAZARD_SERIES,
                                                               range=[[1, 0], [6, 4]]),
                                  legend=None),
        tooltip=[alt.Tooltip("window:N", title="Frames"),
                 alt.Tooltip("bucket:N", title="Deaths"),
                 alt.Tooltip("series:N", title="Pattern"),
                 alt.Tooltip("rate:Q", title="Deaths per 10k frames", format=".2f")],
    )
    table = [{"Window": row.get("bucket"), "Frames": row["window"],
              f"{HAZARD_SERIES[0]} (deaths per 10k frames)": row["active"],
              f"{HAZARD_SERIES[1]} (deaths per 10k frames)": row["inactive"]}
             for row in hazard["windows"]]
    return _spec(chart, table)


def pattern_lethality(rows):
    """Share of encounters with each pattern that ended in death, with 95% CIs."""
    order = [str(row["pattern"]) for row in rows]
    data = [{**row, "pattern": str(row["pattern"])} for row in rows]
    base = alt.Chart(alt.Data(values=data)).encode(
        x=alt.X("pattern:N", title="Pattern id, deadliest first", sort=order,
                axis=alt.Axis(labelAngle=0)),
    )
    interval = base.mark_rule(strokeWidth=2, opacity=0.5).encode(
        y=alt.Y("low:Q", title="Encounters ending in death",
                axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y2="high:Q",
    )
    dots = base.mark_point(filled=True, size=80).encode(
        y="lethality:Q",
        tooltip=[alt.Tooltip("pattern:N", title="Pattern"),
                 alt.Tooltip("lethality:Q", title="Lethality", format=".0%"),
                 alt.Tooltip("deaths:Q", title="Deaths", format=","),
                 alt.Tooltip("encounters:Q", title="Encounters", format=","),
                 alt.Tooltip("low:Q", title="95% CI low", format=".0%"),
                 alt.Tooltip("high:Q", title="95% CI high", format=".0%")],
    )
    table = [{"Pattern": row["pattern"], "Encounters": row["encounters"],
              "Deaths": row["deaths"], "Lethality %": _percent(row["lethality"]),
              "95% CI low %": _percent(row["low"]), "95% CI high %": _percent(row["high"])}
             for row in rows]
    return _spec(alt.layer(interval, dots), table)
