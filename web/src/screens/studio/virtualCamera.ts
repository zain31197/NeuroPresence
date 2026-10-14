/** What the Live Studio says about the virtual camera, worked out from what the engine reports. */

import type { VirtualCamera } from '../../lib/api'

export interface CameraLine {
  tone: 'neutral' | 'good' | 'accent' | 'critical'
  /** A word or two for the pill. */
  state: string
  /** One sentence on what a meeting application sees right now. */
  text: string
}

export function cameraLine(camera: VirtualCamera | null | undefined): CameraLine | null {
  if (!camera) return null // an engine that has no virtual camera: nothing to say
  if (!camera.on) {
    if (camera.reason) return { tone: 'critical', state: 'Not available', text: camera.reason }
    return { tone: 'neutral', state: 'Off', text: 'Meeting apps cannot see your picture. Switch this on, then choose the camera in the meeting app.' }
  }
  const name = camera.device ?? 'the virtual camera'
  if (camera.showing === 'session') {
    return { tone: 'good', state: 'In meetings', text: `Meeting apps that select "${name}" see your output now, with its mark.` }
  }
  return {
    tone: 'accent',
    state: 'Ready',
    text: `Choose "${name}" as the camera in your meeting app. It shows a plain card until a camera session is running.`,
  }
}
