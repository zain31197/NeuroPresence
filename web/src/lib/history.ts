/** Words for the session history and the security log (see neuropresence/database.py). */

import type { Tone } from '../components/ui/Pill'
import type { SecurityItem, SessionRecord } from './api'

/** "45 s", "12 min 4 s", "1 h 3 min". */
export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s} s`
  if (s < 3600) {
    const minutes = Math.floor(s / 60)
    const rest = s % 60
    return rest ? `${minutes} min ${rest} s` : `${minutes} min`
  }
  const hours = Math.floor(s / 3600)
  const minutes = Math.round((s % 3600) / 60)
  return minutes ? `${hours} h ${minutes} min` : `${hours} h`
}

/** "Today", "Yesterday", or "21 November" (with the year when it is not this one), for a heading over a day's items. */
export function dayLabel(epochSeconds: number, now = Date.now()): string {
  const day = new Date(epochSeconds * 1000)
  const today = new Date(now)
  const start = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
  const days = Math.round((start(today) - start(day)) / 86_400_000)
  if (days === 0) return 'Today'
  if (days === 1) return 'Yesterday'
  return day.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    ...(day.getFullYear() === today.getFullYear() ? {} : { year: 'numeric' }),
  })
}

/** "14:02", in the 24-hour clock. */
export function clockTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false })
}

/** Items grouped under their day, newest day first, keeping their order inside each day. */
export function byDay<T extends { at: number }>(items: T[], now = Date.now()): { day: string; items: T[] }[] {
  const groups: { day: string; items: T[] }[] = []
  for (const item of items) {
    const day = dayLabel(item.at, now)
    const last = groups[groups.length - 1]
    if (last && last.day === day) last.items.push(item)
    else groups.push({ day, items: [item] })
  }
  return groups
}

/** At most `most` of the items, evenly spread and always keeping the last, so a long session's chart stays light. */
export function thin<T>(items: T[], most = 240): T[] {
  if (items.length <= most) return items
  const step = Math.ceil(items.length / most)
  return items.filter((_, index) => index % step === 0 || index === items.length - 1)
}

/** The middle value of those known, or null when none is. */
export function median(values: (number | null | undefined)[]): number | null {
  const known = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v)).sort((a, b) => a - b)
  if (!known.length) return null
  const middle = Math.floor(known.length / 2)
  return known.length % 2 ? known[middle] : (known[middle - 1] + known[middle]) / 2
}

export const OUTCOME: Record<SessionRecord['outcome'], { label: string; tone: Tone }> = {
  running: { label: 'Live now', tone: 'accent' },
  stopped: { label: 'Completed', tone: 'good' },
  refused: { label: 'Check not passed', tone: 'warning' },
  error: { label: 'Stopped on a problem', tone: 'critical' },
  interrupted: { label: 'Interrupted', tone: 'serious' },
}

export const KIND: Record<SessionRecord['kind'], string> = {
  camera: 'Camera',
  voice: 'Voice, camera off',
  sample: 'Recorded clip',
}

const ENROLMENT: Record<Extract<SecurityItem, { kind: 'enrolment' }>['action'], string> = {
  face_verified: 'Face verified with the camera',
  pose_registered: 'Pose registered',
  picture_enrolled: 'Meeting picture enrolled',
  picture_removed: 'Meeting picture removed',
  face_forgotten: 'Face and picture forgotten',
}

/** The line a security item is shown as, and whether it is good news. */
export function securityLine(item: SecurityItem): { title: string; detail: string | null; tone: Tone } {
  if (item.kind === 'enrolment') {
    const detail = item.action === 'pose_registered' && item.detail ? `${item.detail[0].toUpperCase()}${item.detail.slice(1)} pose` : item.detail
    return { title: ENROLMENT[item.action], detail, tone: item.action === 'face_forgotten' || item.action === 'picture_removed' ? 'neutral' : 'accent' }
  }
  const what = item.purpose === 'session' ? 'before a session' : 'to verify a face'
  if (item.passed) {
    const match = item.lowest_match !== null ? `, face matched at ${item.lowest_match.toFixed(2)} or better` : ''
    return { title: `Check passed ${what}`, detail: `A living person, the verified face${match}.`, tone: 'good' }
  }
  return { title: `Check not passed ${what}`, detail: item.reason, tone: 'critical' }
}
