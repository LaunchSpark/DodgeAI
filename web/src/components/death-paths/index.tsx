import { useCallback, useEffect, useMemo, useState } from "react"
import { RouteIcon } from "lucide-react"

import { get, post, type Appearance, type DeathPathsResponse, type PatternView, type State } from "@/lib/api"
import { run } from "@/components/controls"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Overview } from "./overview"
import { Replay } from "./replay"

type Filter = "all" | "died" | "survived"

function shown(appearance: Appearance, filter: Filter) {
  if (filter === "all") return true
  return (appearance.outcome === "died") === (filter === "died")
}

export function DeathPaths({ state, focus }: { state: State; focus?: { pid: number; at: number } | null }) {
  const job = state.death_paths
  const [response, setResponse] = useState<DeathPathsResponse | null>(null)
  const [pid, setPid] = useState<number | null>(null)
  const [view, setView] = useState<PatternView | null>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [filter, setFilter] = useState<Filter>("all")
  const [hover, setHover] = useState<number | null>(null)
  const [open, setOpen] = useState<Appearance | null>(null)
  const [games, setGames] = useState("256")
  const [minutes, setMinutes] = useState("5")
  // The pattern asked for from another tab, until a recording can show it.
  const [wanted, setWanted] = useState<number | null>(null)

  // Fetch the summary each time a sampling run finishes.
  const label = job?.status === "done" ? job.label : null
  useEffect(() => {
    if (!label) return
    get<DeathPathsResponse>("death-paths")
      .then((next) => {
        setResponse(next)
        setPid(next.data?.patterns[0]?.pid ?? null)
        setOpen(null)
      })
      .catch((error: Error) => setProblem(error.message))
  }, [label, job?.total])

  const summary = response?.data ?? null

  useEffect(() => {
    if (focus) {
      setWanted(focus.pid)
      setOpen(null)
    }
  }, [focus])
  useEffect(() => {
    if (wanted !== null && summary?.patterns.some((p) => p.pid === wanted)) {
      setPid(wanted)
      setWanted(null)
    }
  }, [wanted, summary])
  const runId = summary?.run_id ?? null
  useEffect(() => {
    if (runId === null || pid === null) return
    let live = true
    setView(null)
    setProblem(null)
    get<PatternView>(`death-paths/pattern/${pid}?run=${runId}`)
      .then((next) => live && setView(next))
      .catch((error: Error) => live && setProblem(error.message))
    return () => { live = false }
  }, [runId, pid])

  const appearances = useMemo(
    () => (view ? view.appearances.filter((a) => shown(a, filter)) : []), [view, filter])
  const back = useCallback(() => setOpen(null), [])
  const running = job?.status === "running"
  const chosen = summary?.patterns.find((p) => p.pid === pid)

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">How the policy plays each pattern</CardTitle>
        <CardDescription>
          {summary
            ? `${response?.label}: ${summary.games.toLocaleString()} fixed seeds. Every appearance of the chosen pattern, the ones it survived and the ones that killed it.`
            : "Pauses training, plays the live policy on fixed seeds until every game ends or the time limit runs out, and records every pattern appearance, survived or died. Training resumes afterwards."}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup type="single" variant="outline" size="sm" value={games}
            onValueChange={(value) => value && setGames(value)}>
            {["128", "256", "512"].map((value) => <ToggleGroupItem key={value} value={value}>{value} games</ToggleGroupItem>)}
          </ToggleGroup>
          <ToggleGroup type="single" variant="outline" size="sm" value={minutes} aria-label="Time limit"
            onValueChange={(value) => value && setMinutes(value)}>
            {["2", "5", "10"].map((value) => <ToggleGroupItem key={value} value={value}>{value} min</ToggleGroupItem>)}
          </ToggleGroup>
          <Button size="sm" disabled={running}
            onClick={() => run(() => post("death-paths", { games: Number(games), mode: "argmax", minutes: Number(minutes) }))}>
            <RouteIcon />{running ? "Recording…" : summary ? "Record again" : "Record pattern appearances"}
          </Button>
          {running && job && (
            <div className="flex min-w-48 flex-1 items-center gap-2 text-sm text-muted-foreground">
              <span className="tabular-nums">{job.finished}/{job.total}</span>
              <Progress value={(job.finished / Math.max(job.total, 1)) * 100} className="h-1.5" />
            </div>
          )}
          {job?.status === "error" && <span className="text-sm text-destructive">{job.message}</span>}
        </div>

        {summary && !open && (
          <div className="flex flex-wrap items-center gap-3">
            <Select value={pid !== null ? String(pid) : undefined}
              onValueChange={(value) => { setPid(Number(value)); setHover(null) }}>
              <SelectTrigger className="w-[360px] max-w-full"><SelectValue placeholder="Pattern" /></SelectTrigger>
              <SelectContent>
                {summary.patterns.map((p) => (
                  <SelectItem key={p.pid} value={String(p.pid)}>
                    {p.pid < 0
                      ? `No pattern · ${p.deaths} deaths`
                      : `Pattern ${p.pid} (${p.type}) · ${p.rate}% of ${p.encounters}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <ToggleGroup type="single" variant="outline" size="sm" value={filter}
              onValueChange={(value) => value && setFilter(value as Filter)}>
              <ToggleGroupItem value="died">Died</ToggleGroupItem>
              <ToggleGroupItem value="survived">Survived</ToggleGroupItem>
              <ToggleGroupItem value="all">All</ToggleGroupItem>
            </ToggleGroup>
            {chosen && chosen.pid >= 0 && (
              <span className="text-sm text-muted-foreground tabular-nums">
                {chosen.deaths} of {chosen.encounters} appearances died
              </span>
            )}
          </div>
        )}

        {wanted !== null && (
          <p className="text-sm text-muted-foreground">
            {summary
              ? `Pattern ${wanted} did not appear in the last sampling run; record again to look for it.`
              : `Record pattern appearances to see pattern ${wanted}.`}
          </p>
        )}
        {summary && summary.cut_short > 0 && !open && (
          <p className="text-sm text-muted-foreground">
            The time limit stopped {summary.cut_short} of {summary.games} games before they ended; their patterns still on screen are listed as unfinished.
          </p>
        )}
        {problem && <p className="text-sm text-destructive">{problem}</p>}
        {summary && runId && open && <Replay runId={runId} appearance={open} onBack={back} />}
        {summary && view && !open && (
          <Overview view={view} appearances={appearances} hover={hover} onHover={setHover}
            onOpen={(appearance) => { setHover(null); setOpen(appearance) }} />
        )}
      </CardContent>
    </Card>
  )
}
