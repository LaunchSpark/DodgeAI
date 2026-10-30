import type { Appearance } from "@/lib/api"

/** Taylor Turner's colours for the two ways to die; survived paths are grey. */
export const RECT = "#0072B2"
export const ENEMY = "#D55E00"
export const SURVIVED = "#a1a1aa"

export const KINDS: Record<string, string> = {
  normal: "normal enemy",
  kamikaze: "kamikaze body",
  kamikaze_blast: "kamikaze blast",
  static_rect: "static rectangle",
  moving_rect: "moving rectangle",
  unknown: "unknown",
}

export function pathColor(appearance: Appearance) {
  if (appearance.outcome !== "died") return SURVIVED
  return appearance.cause === "rect" ? RECT : ENEMY
}

export function patternName(pid: number, type: string | null) {
  return pid < 0 ? "No pattern" : `Pattern ${pid}${type ? ` (${type})` : ""}`
}

/** Frames from the pattern's start to its end, or to death. */
export function patternFrames(appearance: Appearance) {
  return appearance.end - appearance.start + 1
}
