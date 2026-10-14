import { describe, expect, it } from 'vitest'
import type { VirtualCamera } from '../../../lib/api'
import { cameraLine } from '../virtualCamera'

const camera = (patch: Partial<VirtualCamera>): VirtualCamera => ({
  on: false,
  device: null,
  reason: null,
  showing: 'nothing',
  size: [1280, 720],
  fps: 30,
  ...patch,
})

describe('cameraLine', () => {
  it('says nothing for an engine without a virtual camera', () => {
    expect(cameraLine(undefined)).toBeNull()
    expect(cameraLine(null)).toBeNull()
  })

  it('says that meeting apps see nothing while it is off', () => {
    const line = cameraLine(camera({}))
    expect(line?.state).toBe('Off')
    expect(line?.text).toContain('cannot see your picture')
  })

  it('gives the reason when it could not be switched on', () => {
    const line = cameraLine(camera({ reason: 'No virtual camera could be opened. Install OBS Studio.' }))
    expect(line).toEqual({ tone: 'critical', state: 'Not available', text: 'No virtual camera could be opened. Install OBS Studio.' })
  })

  it('names the camera to choose, and says that it shows a card until a session runs', () => {
    const line = cameraLine(camera({ on: true, device: 'OBS Virtual Camera', showing: 'card' }))
    expect(line?.state).toBe('Ready')
    expect(line?.text).toBe('Choose "OBS Virtual Camera" as the camera in your meeting app. It shows a plain card until a camera session is running.')
  })

  it('says so when the output is in the meeting', () => {
    const line = cameraLine(camera({ on: true, device: 'OBS Virtual Camera', showing: 'session' }))
    expect(line?.tone).toBe('good')
    expect(line?.text).toBe('Meeting apps that select "OBS Virtual Camera" see your output now, with its mark.')
  })
})
