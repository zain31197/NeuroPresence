/** Number and time formatting shared by the screens. */

export const dash = '–' // shown where there is no value yet

export function fixed(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || Number.isNaN(value) ? dash : value.toFixed(digits)
}

export function percent(share: number | null | undefined, digits = 0): string {
  return share === null || share === undefined ? dash : `${(share * 100).toFixed(digits)}%`
}

/** 75 -> "1:15", 3725 -> "1:02:05" */
export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return dash
  const total = Math.floor(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`
}

/** Time of day for an event, from seconds since 1970. */
export function timeOfDay(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export function longDate(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })
}

/** "7 October 2026, 14:32" from a local ISO time. */
export function dateTime(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  const time = date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false })
  return `${longDate(iso)}, ${time}`
}

/**
 * Split a check's hint into what is wrong and what to do about it.
 * Hints are written as two sentences: "The head is turned to one side. Face the camera."
 */
export function splitHint(hint: string): [problem: string, fix: string] {
  const end = hint.indexOf('. ')
  return end === -1 ? ['', hint] : [hint.slice(0, end + 1), hint.slice(end + 2)]
}

export type Verdict = 'good' | 'off' | 'unknown'

/** Does a value meet its target? `atLeast` targets want bigger numbers, the others smaller. */
export function verdict(value: number | null | undefined, target: number, atLeast: boolean): Verdict {
  if (value === null || value === undefined) return 'unknown'
  return (atLeast ? value >= target : value <= target) ? 'good' : 'off'
}
