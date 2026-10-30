import { useEffect, useRef, useState } from "react"

export type VegaSpec = Record<string, unknown>

export interface SeriesSummary {
  key: string
  label: string
  unit: string
  rising: boolean
  latest?: number
  baseline?: number
  trend?: number
  delta?: number
}

export interface Summary {
  games: number
  worst: number
  q1: number
  median: number
  mean: number
  q4: number
  best: number
  past_1000: number
}

export interface Figures {
  summary: Summary | null
  anti_camping: { games_triggered: number; triggers_per_game: number } | null
  hazard: {
    overall: { active: number | null; inactive: number | null; active_frame_share: number }
  }
  patterns: { pattern: number; encounters: number; deaths: number; lethality: number }[]
}

export interface Slider {
  name: string
  label: string
  min: number
  max: number
  step: number
  hint?: string
}

export interface Toast {
  id: number
  ok: boolean
  message: string
}

export interface EvaluationState {
  status: "idle" | "running" | "done" | "error"
  message: string
  finished: number
  total: number
  label: string
  figures: Figures | null
}

export interface DeathPathsState {
  status: "idle" | "running" | "done" | "error"
  message: string
  finished: number
  total: number
  label: string
}

export interface State {
  status: string
  model_name: string
  update: number
  seed: number
  steps_per_second: number
  best_score: number
  elapsed_seconds: number
  progress_phase: string
  rollout_progress: number
  learning_progress: number
  dead_units: number
  progress: SeriesSummary[]
  events: Record<string, { rollout: number; total: number; history: number[] }>
  game_config: Record<string, number | boolean>
  game_controls: { sliders: Slider[]; toggles: { name: string; label: string }[] }
  rewards: Record<string, number>
  reward_controls: Slider[]
  training_figures: Figures | null
  evaluation: EvaluationState
  death_paths?: DeathPathsState
  charts: Record<string, VegaSpec>
  system: { cpu: number; memory: number; gpu: { name: string; memory: number } | null }
  toasts: Toast[]
}

export interface ModelEntry {
  name: string
  path: string
}

export async function post<T = unknown>(path: string, body: unknown = {}): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(payload.detail ?? response.statusText)
  return payload as T
}

export async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api/${path}`)
  if (!response.ok) throw new Error(response.statusText)
  return (await response.json()) as T
}

/** The trainer's state, pushed once a second; reconnects if the server restarts. */
export function useLiveState(): { state: State | null; connected: boolean } {
  const [state, setState] = useState<State | null>(null)
  const [connected, setConnected] = useState(false)
  const retry = useRef<number | undefined>(undefined)

  useEffect(() => {
    let socket: WebSocket | null = null
    let closed = false
    const connect = () => {
      const scheme = location.protocol === "https:" ? "wss" : "ws"
      socket = new WebSocket(`${scheme}://${location.host}/api/live`)
      socket.onopen = () => setConnected(true)
      socket.onmessage = (event) => setState(JSON.parse(event.data) as State)
      socket.onclose = () => {
        setConnected(false)
        if (!closed) retry.current = window.setTimeout(connect, 1500)
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry.current)
      socket?.close()
    }
  }, [])

  return { state, connected }
}
