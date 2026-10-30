import { useEffect, useRef, useState } from "react"
import { ArrowLeftIcon, RotateCcwIcon } from "lucide-react"

import { get, type Appearance, type ReplayResponse, type ReplaySample } from "@/lib/api"
import { PICO8 } from "@/lib/pico8"
import { Button } from "@/components/ui/button"
import { KINDS, patternFrames, patternName } from "@/lib/death-paths"
import { OutcomeBadge } from "./outcome-badge"

const PIXELS = 128 * 128

async function inflate(base64: string) {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

function gap(value: number | null) {
  return value === null ? "–" : `${value.toFixed(1)} px`
}

function frameLine(sample: ReplaySample, index: number, count: number, loose: boolean) {
  const where = loose ? `Frame ${index + 1} of the last ${count}` : `Frame ${index + 1} of the pattern`
  const state = ["no pattern", "rectangles charging", "rectangles lethal"][sample.st]
  return `${where} · ${state} · ${sample.enemies} enemies · nearest enemy ${gap(sample.enemy_gap)} · lethal rectangle ${gap(sample.rect_gap)}`
}

function summary(a: Appearance) {
  const into = a.pid < 0 ? `died at frame ${a.end.toLocaleString()}, outside any pattern`
    : `${(patternFrames(a) - 1).toLocaleString()} frames into ${patternName(a.pid, null).toLowerCase()}`
  const parts = [into]
  if (a.death) parts.push(`score ${a.death.score.toLocaleString()}`)
  const killer = a.death?.killer
  if (killer) {
    const kind = KINDS[{ 0: "normal", 1: "kamikaze", [-1]: "kamikaze_blast" }[killer.personality] ?? "unknown"]
    parts.push(`killer: ${kind}${killer.age === null ? "" : `, ${killer.age} frames old`}`)
    parts.push(killer.fyou_spawned === null ? "guard origin unknown"
      : killer.fyou_spawned ? "spawned by the anti-camping guard" : "not from the anti-camping guard")
  }
  if (a.outcome === "unfinished") parts.push("the game hit the frame cap first")
  return parts.join(" · ")
}

/** The real game picture of one appearance, played straight through. */
export function Replay({ runId, appearance, onBack }: {
  runId: string
  appearance: Appearance
  onBack: () => void
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [replay, setReplay] = useState<{ meta: ReplayResponse; frames: Uint8Array } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [index, setIndex] = useState(0)
  const [take, setTake] = useState(0)

  useEffect(() => {
    let live = true
    setReplay(null)
    setError(null)
    get<ReplayResponse>(`death-paths/replay/${appearance.id}?run=${runId}`)
      .then(async (meta) => {
        const frames = await inflate(meta.frames)
        if (live) setReplay({ meta, frames })
      })
      .catch((reason: Error) => live && setError(reason.message))
    return () => { live = false }
  }, [runId, appearance.id])

  // Play at the game's own rate from the first frame; `take` restarts it.
  useEffect(() => {
    if (!replay || !replay.meta.matched || replay.meta.count === 0) return
    const g = canvas.current?.getContext("2d")
    if (!g) return
    const image = g.createImageData(128, 128)
    const { fps, count } = replay.meta
    let started: number | null = null
    let shown = -1
    let handle = 0
    const tick = (now: number) => {
      started ??= now
      const next = Math.min(count - 1, Math.floor(((now - started) * fps) / 1000))
      if (next !== shown) {
        shown = next
        const frame = replay.frames.subarray(next * PIXELS, (next + 1) * PIXELS)
        for (let i = 0; i < PIXELS; i++) {
          const [r, gr, b] = PICO8[frame[i] & 15]
          image.data[i * 4] = r
          image.data[i * 4 + 1] = gr
          image.data[i * 4 + 2] = b
          image.data[i * 4 + 3] = 255
        }
        g.putImageData(image, 0, 0)
        setIndex(next)
      }
      if (next < count - 1) handle = requestAnimationFrame(tick)
    }
    handle = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(handle)
  }, [replay, take])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onBack()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onBack])

  const meta = replay?.meta
  const finished = !!meta && meta.matched && index >= meta.count - 1
  const sample = meta?.samples[index]

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Button size="sm" variant="outline" onClick={onBack}><ArrowLeftIcon />All appearances</Button>
        <span className="text-sm text-muted-foreground tabular-nums">
          seed {appearance.seed} · {patternName(appearance.pid, appearance.type).toLowerCase()}
        </span>
      </div>
      <div className="relative mx-auto aspect-square w-full max-w-[560px] overflow-hidden rounded-md bg-[#1d2b53]">
        <canvas ref={canvas} width={128} height={128}
          className={`size-full transition-opacity ${finished ? "opacity-40" : ""}`}
          style={{ imageRendering: "pixelated" }} />
        {!meta && !error && (
          <div className="absolute inset-0 grid place-items-center text-sm text-white/80">Re-simulating…</div>
        )}
        {(error || (meta && !meta.matched)) && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-white">
            <div className="grid gap-1">
              <b>{error ? "Could not load this replay." : "This replay does not match the recording."}</b>
              <span className="text-white/80">{error ?? meta?.reason}</span>
            </div>
          </div>
        )}
        {finished && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <OutcomeBadge appearance={appearance} long />
            <Button size="sm" onClick={() => { setIndex(0); setTake((n) => n + 1) }}>
              <RotateCcwIcon />Replay
            </Button>
          </div>
        )}
      </div>
      <div className="rounded-md border p-3 text-sm text-muted-foreground tabular-nums">
        {finished ? summary(appearance)
          : sample && meta ? frameLine(sample, index, meta.count, appearance.pid < 0)
          : summary(appearance)}
      </div>
    </div>
  )
}
