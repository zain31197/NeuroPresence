/** Why the still picture is shown although a face is in view, and what the screen says about it. */

import type { Tone } from '../../components/ui/Pill'
import type { IdentityGuard } from '../../lib/api'

/** How long the prompt offers a choice. After that the still picture simply stays. */
export const PROMPT_SECONDS = 5

export interface GuardNote {
  text: string
  tone: Tone
}

/** What the identity figure says beside its verdict, or null when there is nothing to add. */
export function guardNote(guard: IdentityGuard | null | undefined): GuardNote | null {
  if (!guard) return null
  if (!guard.enabled) return { text: 'Fallback off', tone: 'neutral' }
  switch (guard.state) {
    case 'dipping':
      return { text: 'Low for a moment', tone: 'warning' }
    case 'anchoring':
      return { text: 'Fresh neutral pose', tone: 'warning' }
    case 'fallback':
      return { text: 'Still picture shown', tone: 'critical' }
    default:
      return null
  }
}

/**
 * Whole seconds left in which the prompt still asks what to do, counted from how long the still
 * picture has been shown. Zero once the time is up or the person chose to stay.
 */
export function promptSecondsLeft(shownFor: number | null | undefined, chosenToStay: boolean): number {
  if (chosenToStay || shownFor === null || shownFor === undefined) return 0
  return Math.max(0, Math.ceil(PROMPT_SECONDS - shownFor))
}
