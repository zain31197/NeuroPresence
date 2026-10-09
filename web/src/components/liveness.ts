/** What the liveness prompt is asking for, worked out from what the engine sends (consent/liveness.py). */

import type { ConsentPrompt, LivenessAction } from '../lib/api'

/** Is something being asked of the person right now? Not once the check has passed or been refused. */
export function isAsking(prompt: ConsentPrompt | null | undefined): prompt is ConsentPrompt {
  return !!prompt && (prompt.stage === 'settle' || prompt.stage === 'act' || prompt.stage === 'confirming')
}

/** How much of the time for the action asked for is left, from 1 down to 0. Null when nothing is being timed. */
export function timeLeftShare(prompt: ConsentPrompt): number | null {
  if (prompt.stage !== 'act' || prompt.seconds_left === null || prompt.seconds <= 0) return null
  return Math.min(1, Math.max(0, prompt.seconds_left / prompt.seconds))
}

/**
 * Which side of the picture the head should turn towards, or null for an action that is not a turn.
 * The camera is shown mirrored while a prompt is up, so the person's own left is the left of the picture.
 */
export function turnSide(action: LivenessAction | null): 'left' | 'right' | null {
  if (action === 'turn_left') return 'left'
  if (action === 'turn_right') return 'right'
  return null
}

/** "Step 1 of 2": which of the actions is being asked for, or comes next. Nothing once they are all done. */
export function stepLine(prompt: ConsentPrompt): string {
  if (prompt.steps === 0 || prompt.stage === 'confirming') return ''
  return `Step ${prompt.step} of ${prompt.steps}`
}
