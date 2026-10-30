import type { Appearance } from "@/lib/api"
import { ENEMY, KINDS, RECT } from "@/lib/death-paths"
import { Badge } from "@/components/ui/badge"

export function OutcomeBadge({ appearance, long = false }: { appearance: Appearance; long?: boolean }) {
  if (appearance.outcome === "died") {
    const color = appearance.cause === "rect" ? RECT : ENEMY
    const what = appearance.cause === "rect" ? "rectangle" : "enemy"
    return (
      <Badge variant="outline" style={{ color, borderColor: color }}>
        {long ? `Died on ${KINDS[appearance.kind ?? "unknown"]}` : what}
      </Badge>
    )
  }
  if (appearance.outcome === "unfinished") {
    return <Badge variant="outline">{long ? "Reached the frame cap" : "unfinished"}</Badge>
  }
  return <Badge variant="secondary">{long ? "Survived" : "survived"}</Badge>
}
