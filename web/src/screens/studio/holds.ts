/** Why the still picture is shown although a face is in view, and what the screen says about it. */

import type { Tone } from '../../components/ui/Pill'
import type { IdentityGuard } from '../../lib/api'

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
 * What the notice says while the identity monitor shows the still picture. The engine resumes by itself
 * once the person sits at rest, unless they chose to stay (see runtime.py, IdentityMonitor).
 */
export function identityHoldText(guard: IdentityGuard | null | undefined): string {
  const start = 'The output stopped matching your enrolled picture, so your still picture is shown.'
  if (guard?.staying) return `${start} It stays on the still picture until you resume.`
  const left = guard?.resumes_in
  if (left === null || left === undefined) return `${start} Resume with a fresh neutral pose when you are ready.`
  const when = left > 0.5 ? `, in ${Math.ceil(left)} s at the earliest` : ''
  return `${start} It resumes by itself with a fresh neutral pose once you sit facing the camera, mouth closed${when}.`
}
