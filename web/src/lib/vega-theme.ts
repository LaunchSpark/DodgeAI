// Chart chrome and categorical hues for each mode, from the validated
// reference palette. Hues are assigned in this fixed order by each chart's
// fixed scale domain, so a series keeps its colour whatever else is shown.

const light = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  category: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
}

const dark = {
  surface: "#1a1a19",
  ink: "#f0efec",
  secondary: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
  category: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}

export function vegaConfig(mode: "light" | "dark") {
  const t = mode === "dark" ? dark : light
  const font = "system-ui, -apple-system, 'Segoe UI', sans-serif"
  return {
    background: "transparent",
    font,
    view: { stroke: null },
    padding: { left: 4, right: 8, top: 4, bottom: 4 },
    range: { category: t.category },
    mark: { color: t.category[0] },
    line: { strokeWidth: 2, strokeCap: "round" },
    point: { stroke: t.surface, strokeWidth: 2 },
    // A 2px surface gap between adjacent fills.
    bar: { stroke: t.surface, strokeWidth: 2, cornerRadiusEnd: 4 },
    rule: { color: t.muted },
    text: { color: t.ink },
    axis: {
      labelColor: t.muted,
      titleColor: t.secondary,
      titleFontWeight: 500,
      labelFontSize: 11,
      titleFontSize: 11,
      gridColor: t.grid,
      domainColor: t.axis,
      tickColor: t.axis,
      labelFont: font,
      titleFont: font,
    },
    axisX: { grid: false },
    legend: {
      labelColor: t.secondary,
      titleColor: t.secondary,
      labelFontSize: 11,
      symbolType: "circle",
      symbolSize: 80,
    },
  }
}
