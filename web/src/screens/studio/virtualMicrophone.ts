/** What the Live Studio says about the virtual microphone, worked out from what the engine reports. */

import type { VirtualMicrophone } from '../../lib/api'

export interface VoiceLine {
  tone: 'neutral' | 'good' | 'accent' | 'critical'
  /** A word or two for the pill. */
  state: string
  /** One sentence on what a meeting application hears right now. */
  text: string
}

export function voiceLine(mic: VirtualMicrophone | null | undefined): VoiceLine | null {
  if (!mic) return null // an engine that has no virtual microphone: nothing to say
  const name = mic.meeting_device ?? 'the virtual cable'
  if (!mic.on) {
    if (mic.reason) return { tone: 'critical', state: 'Not available', text: mic.reason }
    return {
      tone: 'neutral',
      state: 'Off',
      text: `Meeting apps hear nothing from NeuroPresence. Switch this on, then choose "${name}" as the microphone in the meeting app.`,
    }
  }
  if (mic.picture_ms === null) {
    return {
      tone: 'accent',
      state: 'Ready',
      text: `Choose "${name}" as the microphone in your meeting app. Your voice goes through as it is until a camera session is running.`,
    }
  }
  const held = Math.round(mic.delay_ms ?? mic.picture_ms + mic.offset_ms)
  return { tone: 'good', state: 'In step', text: `Your voice is held back ${held} ms, as long as the picture takes, so lips and words arrive together.` }
}

/** The adjustment as it is shown: a sign always, so that "0 ms" is plainly no change. */
export function offsetLabel(offset: number): string {
  if (offset === 0) return '0 ms'
  return `${offset > 0 ? '+' : '−'}${Math.abs(offset)} ms`
}

/** The meter's width for a level from 0 to 1: speech sits low on a straight scale, so a square root lifts it. */
export function meterShare(level: number): number {
  return Math.min(1, Math.sqrt(Math.max(0, level)))
}
