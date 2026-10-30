import { useEffect, useRef, type ReactNode } from "react"
import { ThemeProvider, useTheme } from "next-themes"
import { toast } from "sonner"
import {
  BrainIcon, ClapperboardIcon, FlaskConicalIcon, MoonIcon, PauseIcon, PlayIcon,
  SaveIcon, SlidersHorizontalIcon, SunIcon, Gamepad2Icon, CircleIcon,
} from "lucide-react"

import { post, useLiveState, type Figures, type SeriesSummary, type State } from "@/lib/api"
import { VegaChart } from "@/components/vega-chart"
import { DeathPaths } from "@/components/death-paths"
import { EvaluateDialog, GameConfigDialog, ModelsDialog, RewardsDialog, run } from "@/components/controls"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"

const FPS = 60

function duration(seconds: number) {
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`
}

function frames(value: number | undefined) {
  if (value === undefined || value === null) return "–"
  return Math.round(value).toLocaleString()
}

function ChartCard({ title, description, children, className }: {
  title: string; description?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-lg border bg-card px-4 py-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
      {detail && <div className="text-xs text-muted-foreground">{detail}</div>}
    </div>
  )
}

function seriesDescription(series?: SeriesSummary) {
  if (!series || series.trend === undefined) return "No episodes yet"
  const unit = series.unit ? ` ${series.unit}` : ""
  const delta = series.delta ?? 0
  const better = series.rising ? delta >= 0 : delta <= 0
  return (
    <span>
      Trend {series.trend.toFixed(2)}{unit} ·{" "}
      <span className={better ? "text-emerald-700 dark:text-emerald-500" : "text-muted-foreground"}>
        {delta >= 0 ? "+" : ""}{delta.toFixed(2)}{unit}
      </span>{" "}since the start
    </span>
  )
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const dark = resolvedTheme === "dark"
  return (
    <Button variant="ghost" size="icon" aria-label="Toggle dark mode" onClick={() => setTheme(dark ? "light" : "dark")}>
      {dark ? <SunIcon /> : <MoonIcon />}
    </Button>
  )
}

function Header({ state, connected }: { state: State; connected: boolean }) {
  const paused = state.status === "paused"
  const phase = state.progress_phase === "learning" ? state.learning_progress : state.rollout_progress
  return (
    <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-3">
        <div className="mr-2 flex items-center gap-2">
          <BrainIcon className="size-5" />
          <span className="font-semibold">DodgeAI</span>
        </div>
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <span className="truncate font-medium">{state.model_name || "No model"}</span>
          <Badge variant={state.status === "error" ? "destructive" : "secondary"}>{state.status}</Badge>
          <span className="text-muted-foreground tabular-nums">
            update {state.update.toLocaleString()} · {Math.round(state.steps_per_second).toLocaleString()} steps/s · {duration(state.elapsed_seconds)}
          </span>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-1">
          <Button size="sm" variant="outline" onClick={() => run(() => post("pause"))}>
            {paused ? <PlayIcon /> : <PauseIcon />}{paused ? "Resume" : "Pause"}
          </Button>
          <Button size="sm" variant="outline" onClick={() => run(() => post("watch"), "Opening the watch window")}>
            <ClapperboardIcon />Watch
          </Button>
          <EvaluateDialog state={state} trigger={<Button size="sm" variant="outline"><FlaskConicalIcon />Evaluate</Button>} />
          <GameConfigDialog state={state} trigger={<Button size="sm" variant="outline"><Gamepad2Icon />Game</Button>} />
          <RewardsDialog state={state} trigger={<Button size="sm" variant="outline"><SlidersHorizontalIcon />Rewards</Button>} />
          <ModelsDialog state={state} trigger={<Button size="sm" variant="outline"><SaveIcon />Models</Button>} />
          <ThemeToggle />
          <span title={connected ? "Connected" : "Reconnecting"} className="flex items-center">
            <CircleIcon className={`size-3 ${connected ? "fill-emerald-600 text-emerald-600" : "fill-amber-500 text-amber-500"}`} />
          </span>
        </div>
      </div>
      <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 pb-2 text-xs text-muted-foreground">
        <span className="w-16">{state.progress_phase}</span>
        <Progress value={phase * 100} className="h-1.5" />
      </div>
    </header>
  )
}

function TrainingTab({ state }: { state: State }) {
  const series = Object.fromEntries(state.progress.map((row) => [row.key, row]))
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ChartCard className="lg:col-span-2" title="Worst quarter, mean and best quarter of recent games"
        description="Over the last 256 finished training games at each update. The worst quarter is the lower tail the paper tracks.">
        <VegaChart spec={state.charts.quarters} />
      </ChartCard>
      {state.progress.map((row) => (
        <ChartCard key={row.key} title={row.label} description={seriesDescription(series[row.key])}>
          <VegaChart spec={state.charts[row.key]} height={180} />
        </ChartCard>
      ))}
      <ChartCard className="lg:col-span-2" title="Events" description="This rollout, and the whole run.">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Object.entries(state.events).map(([key, event]) => (
            <Stat key={key} label={key.split("_").join(" ")} value={event.rollout.toLocaleString()}
              detail={`${event.total.toLocaleString()} in total`} />
          ))}
        </div>
      </ChartCard>
    </div>
  )
}

function SummaryTiles({ figures }: { figures: Figures | null }) {
  const s = figures?.summary
  if (!s) return null
  const hazard = figures?.hazard.overall
  const ratio = hazard?.active && hazard?.inactive ? hazard.active / hazard.inactive : null
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
      <Stat label="Worst" value={frames(s.worst)} detail={`${(s.worst / FPS).toFixed(1)} s`} />
      <Stat label="Worst quarter (Q1)" value={frames(s.q1)} detail={`${(s.q1 / FPS).toFixed(1)} s`} />
      <Stat label="Median" value={frames(s.median)} detail={`${(s.median / FPS).toFixed(1)} s`} />
      <Stat label="Mean" value={frames(s.mean)} detail={`${(s.mean / FPS).toFixed(1)} s`} />
      <Stat label="Best quarter (Q4)" value={frames(s.q4)} detail={`${(s.q4 / FPS).toFixed(1)} s`} />
      <Stat label="Best" value={frames(s.best)} detail={`${(s.best / FPS).toFixed(1)} s`} />
      <Stat label="Past 1,000 frames" value={`${(s.past_1000 * 100).toFixed(1)}%`} detail={`${s.games.toLocaleString()} games`} />
      <Stat label="Pattern hazard" value={ratio ? `${ratio.toFixed(1)}×` : "–"}
        detail={`deaths/frame with a pattern vs without`} />
    </div>
  )
}

function EvaluationTab({ state }: { state: State }) {
  const evaluation = state.evaluation
  const fromEvaluation = evaluation.status === "done" && evaluation.figures
  const figures = fromEvaluation ? evaluation.figures : state.training_figures
  const source = fromEvaluation
    ? `Evaluation of ${evaluation.label}: ${evaluation.total.toLocaleString()} fixed seeds.`
    : "Recent training games (sampled actions, current game settings). Run an evaluation for the paper's protocol."
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
        <span>{source}</span>
        {evaluation.status === "running" && (
          <div className="flex min-w-64 flex-1 items-center gap-2">
            <span>Evaluating {evaluation.finished}/{evaluation.total}</span>
            <Progress value={(evaluation.finished / Math.max(evaluation.total, 1)) * 100} className="h-1.5" />
          </div>
        )}
        {evaluation.status === "error" && <span className="text-destructive">Evaluation failed: {evaluation.message}</span>}
      </div>
      <SummaryTiles figures={figures} />
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Games still alive" description="Share of games still alive at each milestone.">
          <VegaChart spec={state.charts.survival} />
        </ChartCard>
        <ChartCard title="What ended each game" description="First lethal contact, by the window the game ended in.">
          <VegaChart spec={state.charts.causes} />
        </ChartCard>
        <ChartCard title="Deaths with and without a pattern" description="Deaths per 10,000 frames lived, log scale.">
          <VegaChart spec={state.charts.hazard} />
        </ChartCard>
        <ChartCard title="Deadliest patterns" description="Share of encounters ending in death, with 95% Wilson intervals (patterns met 10+ times).">
          <VegaChart spec={state.charts.patterns}
            empty={figures?.summary ? "No pattern met 10 times yet" : "Waiting for games to finish"} />
        </ChartCard>
      </div>
    </div>
  )
}

function SystemTab({ state }: { state: State }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Stat label="CPU" value={`${state.system.cpu.toFixed(0)}%`} />
      <Stat label="Memory" value={`${state.system.memory.toFixed(0)}%`} />
      <Stat label="GPU memory" value={state.system.gpu ? `${state.system.gpu.memory.toFixed(0)}%` : "–"}
        detail={state.system.gpu?.name ?? "No CUDA device"} />
      <Stat label="Dead units" value={`${(state.dead_units * 100).toFixed(1)}%`}
        detail="Trunk units silent for a whole rollout" />
      <Stat label="Seed" value={String(state.seed)} />
      <Stat label="Best score" value={state.best_score.toFixed(1)} />
    </div>
  )
}

function Toasts({ state }: { state: State | null }) {
  const seen = useRef(0)
  useEffect(() => {
    for (const item of state?.toasts ?? []) {
      if (item.id <= seen.current) continue
      seen.current = item.id
      if (item.ok) toast.success(item.message)
      else toast.error(item.message)
    }
  }, [state?.toasts])
  return null
}

function Dashboard() {
  const { state, connected } = useLiveState()
  if (!state) {
    return (
      <div className="flex min-h-svh items-center justify-center text-muted-foreground">
        {connected ? "Loading…" : "Connecting to the trainer…"}
      </div>
    )
  }
  return (
    <div className="min-h-svh bg-muted/30">
      <Header state={state} connected={connected} />
      <Toasts state={state} />
      <main className="mx-auto max-w-7xl px-4 py-4">
        <Tabs defaultValue="training">
          <TabsList>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
            <TabsTrigger value="death-paths">Death paths</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
          </TabsList>
          <TabsContent value="training"><TrainingTab state={state} /></TabsContent>
          <TabsContent value="evaluation"><EvaluationTab state={state} /></TabsContent>
          <TabsContent value="death-paths"><DeathPaths state={state} /></TabsContent>
          <TabsContent value="system"><SystemTab state={state} /></TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <Dashboard />
      <Toaster position="bottom-right" />
    </ThemeProvider>
  )
}
