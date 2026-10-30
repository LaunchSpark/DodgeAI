import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTheme } from "next-themes"
import { PauseIcon, PlayIcon, RouteIcon } from "lucide-react"

import { get, post, type State } from "@/lib/api"
import { run } from "@/components/controls"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

type Rect = [number, number, number, number, number]

interface DeathRun {
  pid: number
  seed: number
  cause: "rect" | "enemy"
  kind: string
  frames: number
  since: number
  score: number
  psize: number
  x: number[]
  y: number[]
  st: number[]
  rects: (Rect[] | null)[]
  n: number[]
  gap: (number | null)[]
  kp: ([number, number] | null)[]
  killer: { p: number; size: number; age: number } | null
  alive: number[]
  armed: boolean
}

interface DeathData {
  pids: number[]
  stats: Record<string, { type: string | null; encounters: number; deaths: number; rate: number }>
  runs: DeathRun[]
  games: number
}

interface Response {
  status: string
  label: string
  data: DeathData | null
}

const SIZE = 640
const SCALE = SIZE / 128
const X = (value: number) => value * SCALE

const palette = {
  light: { rect: "#2a78d6", enemy: "#eb6834", ink: "#0b0b0b", muted: "#898781", faint: "#c3c2b7", surface: "#fcfcfb", border: "#e1e0d9" },
  dark: { rect: "#3987e5", enemy: "#d95926", ink: "#f0efec", muted: "#898781", faint: "#52514e", surface: "#1a1a19", border: "#2c2c2a" },
}

function hexAlpha(hex: string, alpha: number) {
  const value = parseInt(hex.slice(1), 16)
  return `rgba(${value >> 16}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`
}

function kindLabel(kind: string) {
  return kind.split("_").join(" ")
}

export function DeathPaths({ state }: { state: State }) {
  const job = state.death_paths
  const [response, setResponse] = useState<Response | null>(null)
  const [pid, setPid] = useState<number | null>(null)
  const [selected, setSelected] = useState<DeathRun | null>(null)
  const [hover, setHover] = useState<DeathRun | null>(null)
  const [frame, setFrame] = useState(119)
  const [playing, setPlaying] = useState(false)
  const [games, setGames] = useState("256")
  const canvas = useRef<HTMLCanvasElement>(null)
  const { resolvedTheme } = useTheme()
  const colors = palette[resolvedTheme === "dark" ? "dark" : "light"]

  // Fetch the (large) recording once each time a job finishes.
  const label = job?.status === "done" ? job.label : null
  useEffect(() => {
    if (!label) return
    get<Response>("death-paths").then((next) => {
      setResponse(next)
      setPid(next.data?.pids[0] ?? null)
      setSelected(null)
    })
  }, [label, job?.total])

  const data = response?.data ?? null
  const runs = useMemo(() => (data && pid !== null ? data.runs.filter((r) => r.pid === pid) : []), [data, pid])

  const draw = useCallback(() => {
    const element = canvas.current
    if (!element) return
    const g = element.getContext("2d")!
    g.clearRect(0, 0, SIZE, SIZE)
    g.fillStyle = colors.surface
    g.fillRect(0, 0, SIZE, SIZE)
    const color = (r: DeathRun) => (r.cause === "rect" ? colors.rect : colors.enemy)
    const rects = (list: Rect[] | null, alpha: number, charging = false) => {
      for (const [x, y, w, h, lethal] of list ?? []) {
        const dashed = charging || !lethal
        g.setLineDash(dashed ? [6, 4] : [])
        g.fillStyle = hexAlpha(colors.rect, dashed ? 0.06 : alpha)
        g.strokeStyle = hexAlpha(colors.rect, 0.8)
        g.lineWidth = 1.5
        g.fillRect(X(x), X(y), X(w), X(h))
        g.strokeRect(X(x), X(y), X(w), X(h))
      }
      g.setLineDash([])
    }
    const path = (r: DeathRun, from: number, to: number, stroke: string, width: number, alpha: number) => {
      g.strokeStyle = stroke
      g.globalAlpha = alpha
      g.lineWidth = width
      g.lineJoin = "round"
      g.beginPath()
      for (let i = from; i <= to; i++) {
        if (i === from) g.moveTo(X(r.x[i]), X(r.y[i]))
        else g.lineTo(X(r.x[i]), X(r.y[i]))
      }
      g.stroke()
      g.globalAlpha = 1
    }
    g.font = "12px system-ui"
    if (!selected) {
      // Each run's rectangles at its death frame, faint, so the layout reads
      // as how often each spot was lethal.
      for (const r of runs) rects(r.rects[r.rects.length - 1], 0.05)
      for (const r of runs) {
        const last = r.x.length - 1
        path(r, Math.max(0, last - 60), last, color(r), r === hover ? 2.4 : 1, r === hover ? 1 : 0.45)
      }
      for (const r of runs) {
        const last = r.x.length - 1
        g.fillStyle = color(r)
        g.globalAlpha = r === hover ? 1 : 0.8
        g.beginPath()
        g.arc(X(r.x[last]), X(r.y[last]), r === hover ? 5 : 3.4, 0, 7)
        g.fill()
        g.strokeStyle = colors.surface
        g.lineWidth = 1
        g.stroke()
        g.globalAlpha = 1
      }
      g.fillStyle = colors.muted
      g.fillText("last 60 frames; dots mark where each game ended", 10, 18)
    } else {
      const r = selected
      const last = r.x.length - 1
      const i = Math.min(frame, last)
      if (r.st[i]) rects(r.rects[i], 0.3, r.st[i] === 1)
      path(r, 0, last, colors.faint, 1.2, 0.9)
      path(r, Math.max(0, i - 30), i, color(r), 2.4, 1)
      const k = r.kp[i]
      if (k && r.killer) {
        const s = r.killer.size
        g.strokeStyle = colors.enemy
        g.lineWidth = 2
        g.fillStyle = hexAlpha(colors.enemy, r.killer.p === -1 ? 0.25 : 0.85)
        g.fillRect(X(k[0]), X(k[1]), X(s), X(s))
        g.strokeRect(X(k[0]), X(k[1]), X(s), X(s))
        g.globalAlpha = 0.35
        g.setLineDash([4, 3])
        g.lineWidth = 1.2
        g.beginPath()
        r.kp.forEach((point, j) => {
          if (point && j <= i) g.lineTo(X(point[0] + s / 2), X(point[1] + s / 2))
        })
        g.stroke()
        g.setLineDash([])
        g.globalAlpha = 1
      }
      // The player at its collision half-width, translucent so the path it
      // is on stays visible.
      const radius = X(r.psize)
      g.fillStyle = hexAlpha(colors.ink, 0.25)
      g.strokeStyle = colors.ink
      g.lineWidth = 1.5
      g.beginPath()
      g.arc(X(r.x[i]), X(r.y[i]), radius, 0, 7)
      g.fill()
      g.stroke()
      g.fillStyle = colors.muted
      g.fillText(i === last ? "death frame" : `${last - i} frames before death`, 10, 18)
    }
    g.strokeStyle = colors.border
    g.strokeRect(0.5, 0.5, SIZE - 1, SIZE - 1)
  }, [runs, selected, hover, frame, colors])

  useEffect(draw, [draw])

  useEffect(() => {
    if (!playing || !selected) return
    const timer = window.setInterval(() => {
      setFrame((value) => {
        if (value >= selected.x.length - 1) {
          setPlaying(false)
          return value
        }
        return value + 1
      })
    }, 50)
    return () => window.clearInterval(timer)
  }, [playing, selected])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelected(null)
        setPlaying(false)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const nearest = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect()
    const mx = ((event.clientX - box.left) * SIZE) / box.width
    const my = ((event.clientY - box.top) * SIZE) / box.height
    let best: DeathRun | null = null
    let distance = 1e9
    for (const r of runs) {
      for (let i = Math.max(0, r.x.length - 60); i < r.x.length; i++) {
        const d = Math.hypot(X(r.x[i]) - mx, X(r.y[i]) - my)
        if (d < distance) {
          distance = d
          best = r
        }
      }
    }
    return distance < 14 ? best : null
  }

  const choose = (r: DeathRun | null) => {
    setPlaying(false)
    setSelected(r)
    setFrame(r ? r.x.length - 1 : 119)
  }

  const stats = data && pid !== null ? data.stats[pid] : null
  const running = job?.status === "running"

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Where games end inside a pattern</CardTitle>
        <CardDescription>
          {data
            ? `${response?.label}: ${data.games.toLocaleString()} fixed seeds. Each line is a game's last frames before dying while the chosen pattern was active.`
            : "Plays the live policy on fixed seeds and records the last 120 frames of every game that ends inside a pattern, for every pattern at once."}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup type="single" variant="outline" size="sm" value={games}
            onValueChange={(value) => value && setGames(value)}>
            {["128", "256", "512"].map((value) => <ToggleGroupItem key={value} value={value}>{value} games</ToggleGroupItem>)}
          </ToggleGroup>
          <Button size="sm" disabled={running}
            onClick={() => run(() => post("death-paths", { games: Number(games), mode: "argmax" }))}>
            <RouteIcon />{running ? "Recording…" : data ? "Record again" : "Record death paths"}
          </Button>
          {running && job && (
            <div className="flex min-w-48 flex-1 items-center gap-2 text-sm text-muted-foreground">
              <span className="tabular-nums">{job.finished}/{job.total}</span>
              <Progress value={(job.finished / Math.max(job.total, 1)) * 100} className="h-1.5" />
            </div>
          )}
          {job?.status === "error" && <span className="text-sm text-destructive">{job.message}</span>}
        </div>

        {data && (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <Select value={pid !== null ? String(pid) : undefined}
                onValueChange={(value) => { setPid(Number(value)); choose(null) }}>
                <SelectTrigger className="w-[420px] max-w-full"><SelectValue placeholder="Pattern" /></SelectTrigger>
                <SelectContent>
                  {data.pids.map((p) => {
                    const s = data.stats[p]
                    const count = data.runs.filter((r) => r.pid === p).length
                    return (
                      <SelectItem key={p} value={String(p)}>
                        Pattern {p} ({s.type ?? "?"}) · {s.rate}% of {s.encounters} encounters · {count} paths
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
              <div className="ml-auto flex gap-4 text-sm text-muted-foreground">
                <span className="flex items-center gap-1.5"><i className="inline-block size-2.5 rounded-full" style={{ background: colors.rect }} />died on a rectangle</span>
                <span className="flex items-center gap-1.5"><i className="inline-block size-2.5 rounded-full" style={{ background: colors.enemy }} />died on an enemy</span>
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-[minmax(0,560px)_1fr]">
              <div className="grid gap-2">
                <canvas ref={canvas} width={SIZE} height={SIZE}
                  className="aspect-square w-full max-w-[560px] rounded-md"
                  style={{ cursor: hover ? "pointer" : "default" }}
                  onMouseMove={(event) => { if (!selected) setHover(nearest(event)) }}
                  onMouseLeave={() => setHover(null)}
                  onClick={(event) => {
                    const hit = selected ? null : nearest(event)
                    choose(hit)
                  }} />
                <div className="flex items-center gap-3">
                  <Button size="sm" variant="outline" disabled={!selected}
                    onClick={() => {
                      if (!selected) return
                      if (!playing && frame >= selected.x.length - 1) setFrame(0)
                      setPlaying(!playing)
                    }}>
                    {playing ? <PauseIcon /> : <PlayIcon />}{playing ? "Pause" : "Play"}
                  </Button>
                  <Slider disabled={!selected} min={0} max={(selected?.x.length ?? 120) - 1} step={1}
                    value={[frame]} onValueChange={([value]) => { setPlaying(false); setFrame(value) }} />
                  <span className="w-24 text-right text-sm tabular-nums text-muted-foreground">
                    {selected ? `frame −${selected.x.length - 1 - Math.min(frame, selected.x.length - 1)}` : ""}
                  </span>
                </div>
              </div>

              <div className="flex min-w-0 flex-col gap-3">
                <div className="h-[300px] overflow-auto rounded-md border text-sm" role="listbox" aria-label="Games">
                  {runs.map((r) => (
                    <div key={r.seed} role="option" aria-selected={r === selected}
                      onClick={() => choose(r)}
                      className={`flex cursor-pointer gap-3 border-b px-3 py-1.5 tabular-nums last:border-b-0 hover:bg-muted ${r === selected ? "bg-muted font-medium" : ""}`}>
                      <i className="mt-1.5 inline-block size-2.5 flex-none rounded-full" style={{ background: r.cause === "rect" ? colors.rect : colors.enemy }} />
                      <span className="w-20">{r.seed}</span>
                      <span className="flex-1">{kindLabel(r.kind).replace(" pattern", "")}</span>
                      <span>{r.frames.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
                <Details run={selected} frame={frame} runs={runs} pid={pid} stats={stats} />
                <p className="text-xs text-muted-foreground">
                  Click a line, a dot or a row to isolate one game, then scrub or play it; click empty space or press Esc to see them all.
                  The killer's path is traced back from the frame before death by matching enemies frame to frame.
                </p>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}

function Details({ run: r, frame, runs, pid, stats }: {
  run: DeathRun | null
  frame: number
  runs: DeathRun[]
  pid: number | null
  stats: { type: string | null; encounters: number; deaths: number; rate: number } | null
}) {
  if (!r) {
    if (!runs.length) return <div className="rounded-md border p-3 text-sm text-muted-foreground">No recorded deaths in pattern {pid}.</div>
    const enemy = runs.filter((x) => x.cause === "enemy").length
    const early = runs.filter((x) => !x.armed).length
    return (
      <div className="rounded-md border p-3 text-sm text-muted-foreground">
        <b className="text-foreground">{runs.length} deaths in pattern {pid}</b>: {runs.length - enemy} on a rectangle, {enemy} on an enemy
        {early ? `; ${early} before the rectangles became lethal` : ""}.<br />
        {stats && <>Share of encounters ending in death: {stats.rate}% of {stats.encounters}. Small counts are noisy.<br /></>}
        Hover a line to highlight it; click to isolate.
      </div>
    )
  }
  const i = Math.min(frame, r.x.length - 1)
  const killer = r.killer
  const killerName = killer ? (killer.p === 0 ? "normal enemy" : killer.p === 1 ? "kamikaze body" : "kamikaze blast") : ""
  const phase = ["not active", "active, rectangles charging (not lethal yet)", "active, rectangles lethal"][r.st[i]]
  return (
    <div className="rounded-md border p-3 text-sm text-muted-foreground">
      <b className="text-foreground">seed {r.seed}</b> · died at frame {r.frames.toLocaleString()}, {r.since} frames into pattern {r.pid}, score {r.score}<br />
      cause: <b className="text-foreground">{kindLabel(r.kind)}</b>
      {killer ? ` · killer: ${killerName}, traced ${killer.age} frames, size ${killer.size}px` : ""}<br />
      at this frame: pattern {phase} · {r.n[i]} enemies on screen · nearest enemy gap {r.gap[i] === null ? "–" : `${r.gap[i]} px`}<br />
      enemies alive at death: {r.alive[0]} normal, {r.alive[1]} kamikaze, {r.alive[2]} explosion
    </div>
  )
}
