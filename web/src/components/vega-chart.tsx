import { useEffect, useRef, useState } from "react"
import { useTheme } from "next-themes"
import embed, { type Result } from "vega-embed"
import { TableIcon, ChartLineIcon } from "lucide-react"

import type { VegaSpec } from "@/lib/api"
import { vegaConfig } from "@/lib/vega-theme"
import { Button } from "@/components/ui/button"

type Row = Record<string, unknown>

/** The data rows of a layered or single spec, for the table view. */
function rowsOf(spec: VegaSpec): Row[] {
  const data = spec.data as { values?: Row[] } | undefined
  if (data?.values) return data.values
  const layers = (spec.layer as VegaSpec[] | undefined) ?? []
  for (const layer of layers) {
    const rows = rowsOf(layer)
    if (rows.length) return rows
  }
  return []
}

function format(value: unknown) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString()
    return Math.abs(value) < 1 ? value.toFixed(3) : value.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  if (value && typeof value === "object") return JSON.stringify(value)
  return String(value ?? "")
}

function DataTable({ rows }: { rows: Row[] }) {
  const columns = Object.keys(rows[0] ?? {})
  return (
    <div className="max-h-[260px] overflow-auto rounded-md border text-xs">
      <table className="w-full tabular-nums">
        <thead className="sticky top-0 bg-muted">
          <tr>{columns.map((c) => <th key={c} className="px-2 py-1 text-left font-medium">{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t">
              {columns.map((c) => <td key={c} className="px-2 py-1">{format(row[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** A Vega-Lite spec rendered with the dashboard theme, with a table view. */
export function VegaChart({ spec, height = 220, empty = "Waiting for games to finish", onSelect }: {
  spec?: VegaSpec; height?: number; empty?: string
  /** Called with the data row of a clicked mark; marks with a row show a pointer. */
  onSelect?: (row: Row) => void
}) {
  const container = useRef<HTMLDivElement>(null)
  const view = useRef<Result | null>(null)
  const select = useRef(onSelect)
  select.current = onSelect
  const [table, setTable] = useState(false)
  const { resolvedTheme } = useTheme()
  const mode = resolvedTheme === "dark" ? "dark" : "light"

  useEffect(() => {
    if (!spec || table || !container.current) return
    let cancelled = false
    embed(container.current, { ...spec, height } as never, {
      actions: false,
      renderer: "svg",
      config: vegaConfig(mode) as never,
      tooltip: { theme: mode },
    }).then((result) => {
      if (cancelled) result.finalize()
      else {
        view.current?.finalize()
        view.current = result
        const row = (item: unknown) =>
          (item as { datum?: Row } | null)?.datum
        result.view.addEventListener("click", (_event, item) => {
          const datum = row(item)
          if (datum && select.current) select.current(datum)
        })
        result.view.addEventListener("mousemove", (_event, item) => {
          if (container.current) container.current.style.cursor = select.current && row(item) ? "pointer" : ""
        })
      }
    })
    return () => {
      cancelled = true
    }
  }, [spec, mode, table, height])

  useEffect(() => () => view.current?.finalize(), [])

  if (!spec) {
    return (
      <div className="flex items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground" style={{ height }}>
        {empty}
      </div>
    )
  }
  return (
    <div>
      <div className="-mt-2 mb-1 flex justify-end">
        <Button
          variant="ghost"
          size="icon"
          className="size-7 text-muted-foreground"
          aria-label={table ? "Show chart" : "Show table"}
          title={table ? "Show chart" : "Show table"}
          onClick={() => setTable((value) => !value)}
        >
          {table ? <ChartLineIcon className="size-4" /> : <TableIcon className="size-4" />}
        </Button>
      </div>
      {table ? <DataTable rows={rowsOf(spec)} /> : <div ref={container} className="w-full" />}
    </div>
  )
}
