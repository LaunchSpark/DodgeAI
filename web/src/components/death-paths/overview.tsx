import { useCallback, useEffect, useRef } from "react"
import { useTheme } from "next-themes"

import type { Appearance, PatternView, Rect } from "@/lib/api"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ENEMY, RECT, SURVIVED, pathColor, patternFrames } from "@/lib/death-paths"
import { OutcomeBadge } from "./outcome-badge"

const SIZE = 640
const X = (value: number) => (value * SIZE) / 128

const surfaces = {
  light: { surface: "#fcfcfb", border: "#e1e0d9", muted: "#898781" },
  dark: { surface: "#1a1a19", border: "#2c2c2a", muted: "#898781" },
}

function alpha(hex: string, value: number) {
  const n = parseInt(hex.slice(1), 16)
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${value})`
}

/** Every appearance of one pattern at once, over its rectangles, and a table of them. */
export function Overview({ view, appearances, hover, onHover, onOpen }: {
  view: PatternView
  appearances: Appearance[]
  hover: number | null
  onHover: (id: number | null) => void
  onOpen: (appearance: Appearance) => void
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const { resolvedTheme } = useTheme()
  const colors = surfaces[resolvedTheme === "dark" ? "dark" : "light"]

  const draw = useCallback(() => {
    const g = canvas.current?.getContext("2d")
    if (!g) return
    g.fillStyle = colors.surface
    g.fillRect(0, 0, SIZE, SIZE)
    const rects = (list: Rect[], fill: number) => {
      for (const [x, y, w, h, lethal] of list) {
        g.setLineDash(lethal ? [] : [6, 4])
        g.fillStyle = alpha(RECT, lethal ? fill : fill / 3)
        g.strokeStyle = alpha(RECT, 0.7)
        g.lineWidth = 1.2
        g.fillRect(X(x), X(y), X(w), X(h))
        g.strokeRect(X(x), X(y), X(w), X(h))
      }
      g.setLineDash([])
    }
    // One shared layout when every game agrees; otherwise each game's own, faint.
    if (view.layout) rects(view.layout, 0.12)
    else for (const a of appearances) rects(a.rects, 0.03)

    const path = (a: Appearance, width: number, opacity: number) => {
      g.strokeStyle = pathColor(a)
      g.globalAlpha = opacity
      g.lineWidth = width
      g.lineJoin = "round"
      g.beginPath()
      a.x.forEach((x, i) => (i ? g.lineTo(X(x), X(a.y[i])) : g.moveTo(X(x), X(a.y[i]))))
      g.stroke()
      g.globalAlpha = 1
    }
    const survived = appearances.filter((a) => a.outcome !== "died")
    const died = appearances.filter((a) => a.outcome === "died")
    for (const a of survived) path(a, a.id === hover ? 2.4 : 1, a.id === hover ? 1 : 0.5)
    for (const a of died) path(a, a.id === hover ? 2.4 : 1.2, a.id === hover ? 1 : 0.75)
    for (const a of died) {
      const last = a.x.length - 1
      g.fillStyle = pathColor(a)
      g.beginPath()
      g.arc(X(a.x[last]), X(a.y[last]), a.id === hover ? 6 : 4, 0, 7)
      g.fill()
      g.strokeStyle = colors.surface
      g.lineWidth = 1
      g.stroke()
    }
    const highlighted = appearances.find((a) => a.id === hover)
    if (highlighted && highlighted.outcome !== "died") path(highlighted, 2.4, 1)
    g.strokeStyle = colors.border
    g.strokeRect(0.5, 0.5, SIZE - 1, SIZE - 1)
  }, [view, appearances, hover, colors])

  useEffect(draw, [draw])

  const nearest = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    const mx = ((event.clientX - box.left) * SIZE) / box.width
    const my = ((event.clientY - box.top) * SIZE) / box.height
    let best: Appearance | null = null
    let distance = 14
    for (const a of appearances) {
      for (let i = 0; i < a.x.length; i += 2) {
        const d = Math.hypot(X(a.x[i]) - mx, X(a.y[i]) - my)
        if (d < distance) {
          distance = d
          best = a
        }
      }
      const last = a.x.length - 1
      // A death's dot wins over a line passing near it.
      if (a.outcome === "died" && Math.hypot(X(a.x[last]) - mx, X(a.y[last]) - my) < 8) return a
    }
    return best
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,560px)_1fr]">
      <div className="grid content-start gap-2">
        <canvas ref={canvas} width={SIZE} height={SIZE}
          className="aspect-square w-full max-w-[560px] rounded-md"
          style={{ cursor: hover !== null ? "pointer" : "default" }}
          onMouseMove={(event) => onHover(nearest(event)?.id ?? null)}
          onMouseLeave={() => onHover(null)}
          onClick={(event) => {
            const hit = nearest(event)
            if (hit) onOpen(hit)
          }} />
        <div className="flex flex-wrap justify-center gap-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-1.5"><i className="inline-block size-2.5 rounded-full" style={{ background: RECT }} />died on a rectangle</span>
          <span className="flex items-center gap-1.5"><i className="inline-block size-2.5 rounded-full" style={{ background: ENEMY }} />died on an enemy</span>
          <span className="flex items-center gap-1.5"><i className="inline-block h-0.5 w-3" style={{ background: SURVIVED }} />survived</span>
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-3">
        <ScrollArea className="h-[420px] rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Seed</TableHead>
                <TableHead>Outcome</TableHead>
                <TableHead className="text-right">Frames in pattern</TableHead>
                <TableHead className="text-right">Game frame</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {appearances.map((a) => (
                <TableRow key={a.id} className="cursor-pointer tabular-nums"
                  data-state={a.id === hover ? "selected" : undefined}
                  onMouseEnter={() => onHover(a.id)} onMouseLeave={() => onHover(null)}
                  onClick={() => onOpen(a)}>
                  <TableCell>{a.seed}</TableCell>
                  <TableCell><OutcomeBadge appearance={a} /></TableCell>
                  <TableCell className="text-right">{patternFrames(a).toLocaleString()}</TableCell>
                  <TableCell className="text-right">{a.end.toLocaleString()}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
        <p className="text-xs text-muted-foreground">
          {view.pid < 0
            ? "Each line is a game's last 120 frames before it died with no pattern on screen."
            : "Each line is one appearance of the pattern, from its start to its end or the death."}
          {" "}Click a line, a dot or a row to watch that appearance.
        </p>
      </div>
    </div>
  )
}
