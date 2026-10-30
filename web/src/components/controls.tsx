import { useEffect, useState, type ReactNode } from "react"
import { toast } from "sonner"

import { get, post, type ModelEntry, type Slider as SliderSpec, type State } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

async function run(action: () => Promise<unknown>, success?: string) {
  try {
    await action()
    if (success) toast.success(success)
    return true
  } catch (error) {
    toast.error(error instanceof Error ? error.message : String(error))
    return false
  }
}

function SliderRow({ spec, value, onChange, format }: {
  spec: SliderSpec
  value: number
  onChange: (value: number) => void
  format?: (value: number) => string
}) {
  return (
    <div className="grid gap-2">
      <div className="flex items-baseline justify-between gap-4">
        <Label htmlFor={spec.name}>{spec.label}</Label>
        <span className="text-sm tabular-nums text-muted-foreground">{format ? format(value) : value}</span>
      </div>
      <Slider id={spec.name} min={spec.min} max={spec.max} step={spec.step}
        value={[value]} onValueChange={([next]) => onChange(next)} />
      {spec.hint && <p className="text-xs text-muted-foreground">{spec.hint}</p>}
    </div>
  )
}

const DIFFICULTY = ["", "Easy", "Normal", "Hard"]

function gameValue(name: string, value: number) {
  if (name === "difficulty") return DIFFICULTY[value]
  if (name === "spawn_rate_percent") return `${value}%`
  if (name === "action_repeat") return `${value} frame${value === 1 ? "" : "s"}`
  return String(value)
}

export function GameConfigDialog({ state, trigger }: { state: State; trigger: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState(state.game_config)
  useEffect(() => {
    if (open) setDraft(state.game_config)
    // Only reset when the dialog opens, not on every live update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const apply = async () => {
    if (await run(() => post("game-config", draft), "Game settings applied; training restarts from the live policy"))
      setOpen(false)
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Game settings</DialogTitle>
          <DialogDescription>Applied at the next safe point; the current policy keeps training.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-5 py-2">
          {state.game_controls.sliders.map((spec) => (
            <SliderRow key={spec.name} spec={spec} value={Number(draft[spec.name])}
              format={(value) => gameValue(spec.name, value)}
              onChange={(value) => setDraft({ ...draft, [spec.name]: value })} />
          ))}
          <div className="grid grid-cols-2 gap-3">
            {state.game_controls.toggles.map((toggle) => (
              <div key={toggle.name} className="flex items-center justify-between gap-2 rounded-md border px-3 py-2">
                <Label htmlFor={toggle.name}>{toggle.label}</Label>
                <Switch id={toggle.name} checked={Boolean(draft[toggle.name])}
                  onCheckedChange={(checked) => setDraft({ ...draft, [toggle.name]: checked })} />
              </div>
            ))}
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={apply}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function RewardsDialog({ state, trigger }: { state: State; trigger: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState(state.rewards)
  useEffect(() => {
    if (open) setDraft(state.rewards)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const apply = async () => {
    if (await run(() => post("rewards", draft), "Reward weights saved; they apply from the next episodes"))
      setOpen(false)
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Reward weights</DialogTitle>
          <DialogDescription>Read by every game when its next episode starts.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-5 py-2">
          {state.reward_controls.map((spec) => (
            <SliderRow key={spec.name} spec={spec} value={Number(draft[spec.name])}
              format={(value) => value.toFixed(spec.step < 0.01 ? 3 : 2)}
              onChange={(value) => setDraft({ ...draft, [spec.name]: value })} />
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={apply}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function trainingValue(name: string, value: number) {
  if (name === "learning_rate") return value.toExponential(1)
  if (name === "gamma") return value.toFixed(3)
  if (name === "retain_every") return value ? `${value} updates` : "off"
  if (name === "patience") return value ? String(value) : "never"
  return String(value)
}

export function TrainingDialog({ state, trigger }: { state: State; trigger: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [draft, setDraft] = useState(state.training)
  useEffect(() => {
    if (open) setDraft(state.training)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const status = state.training_status
  const apply = async () => {
    if (await run(() => post("training", draft), "Training schedule saved; it applies from the next update"))
      setOpen(false)
  }
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Training schedule</DialogTitle>
          <DialogDescription>
            {state.training_applied
              ? "Read at every update boundary."
              : "Not in use: the architecture's defaults apply until you save."}
          </DialogDescription>
        </DialogHeader>
        {state.training_applied && status.last && (
          <p className="text-sm text-muted-foreground tabular-nums">
            Dev cohort at update {status.last.update}: mean {status.last.mean.toFixed(0)},
            Q1 {status.last.q1.toFixed(0)}. Best {status.best_mean?.toFixed(0) ?? "–"}
            {status.since_best ? `, ${status.since_best} scoring${status.since_best === 1 ? "" : "s"} since` : ""}.
          </p>
        )}
        <div className="grid gap-5 py-2">
          {state.training_controls.map((spec) => (
            <SliderRow key={spec.name} spec={spec} value={Number(draft[spec.name])}
              format={(value) => trainingValue(spec.name, value)}
              onChange={(value) => setDraft({ ...draft, [spec.name]: value })} />
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
          <Button onClick={apply}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function ModelsDialog({ state, trigger }: { state: State; trigger: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [models, setModels] = useState<ModelEntry[]>([])
  const [name, setName] = useState(state.model_name)
  useEffect(() => {
    if (!open) return
    setName(state.model_name)
    get<ModelEntry[]>("models").then(setModels).catch(() => setModels([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Models</DialogTitle>
          <DialogDescription>Save the live policy, or load a checkpoint to continue training it.</DialogDescription>
        </DialogHeader>
        <form className="flex gap-2" onSubmit={async (event) => {
          event.preventDefault()
          await run(() => post("save", { name }), `Saving ${name}`)
        }}>
          <Input value={name} onChange={(event) => setName(event.target.value)} aria-label="Model name" />
          <Button type="submit">Save</Button>
        </form>
        <ScrollArea className="h-72 rounded-md border">
          {models.length === 0 && <p className="p-4 text-sm text-muted-foreground">No checkpoints yet.</p>}
          {models.map((model) => (
            <div key={model.path} className="flex items-center justify-between gap-3 border-b px-3 py-2 last:border-b-0">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">{model.name}</div>
                <div className="truncate text-xs text-muted-foreground">{model.path}</div>
              </div>
              <Button size="sm" variant="outline" onClick={async () => {
                if (await run(() => post("load", { path: model.path }), `Loading ${model.name}`)) setOpen(false)
              }}>Load</Button>
            </div>
          ))}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  )
}

export function EvaluateDialog({ state, trigger }: { state: State; trigger: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [games, setGames] = useState("256")
  const [mode, setMode] = useState("argmax")
  const running = state.evaluation.status === "running"

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>{trigger}</DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Evaluate the live policy</DialogTitle>
          <DialogDescription>
            The paper's protocol: a frozen copy plays one game on each fixed seed from 1,000,000,
            capped at 18,000 frames, with the current game settings. Training carries on.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-2">
          <div className="grid gap-2">
            <Label>Games</Label>
            <ToggleGroup type="single" variant="outline" value={games} onValueChange={(value) => value && setGames(value)}>
              {["64", "256", "512", "1024"].map((value) => <ToggleGroupItem key={value} value={value}>{value}</ToggleGroupItem>)}
            </ToggleGroup>
          </div>
          <div className="grid gap-2">
            <Label>Actions</Label>
            <ToggleGroup type="single" variant="outline" value={mode} onValueChange={(value) => value && setMode(value)}>
              <ToggleGroupItem value="argmax">Most likely (argmax)</ToggleGroupItem>
              <ToggleGroupItem value="sampled">Sampled</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>Cancel</Button>
          <Button disabled={running} onClick={async () => {
            if (await run(() => post("evaluate", { games: Number(games), mode }))) setOpen(false)
          }}>{running ? "Evaluation running" : "Start"}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export { run }
