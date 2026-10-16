import { describe, expect, it } from 'vitest'
import type { VirtualMicrophone } from '../../../lib/api'
import { meterShare, offsetLabel, voiceLine } from '../virtualMicrophone'

const mic = (patch: Partial<VirtualMicrophone>): VirtualMicrophone => ({
  on: false,
  reason: null,
  cable: 'CABLE Input (VB-Audio Virtual Cable)',
  meeting_device: 'CABLE Output (VB-Audio Virtual Cable)',
  microphone: 'Headset',
  microphones: ['Headset', 'Webcam'],
  offset_ms: 0,
  picture_ms: null,
  delay_ms: null,
  level: 0,
  ...patch,
})

describe('voiceLine', () => {
  it('says nothing for an engine without a virtual microphone', () => {
    expect(voiceLine(undefined)).toBeNull()
  })

  it('names the device to choose in the meeting app while it is off', () => {
    const line = voiceLine(mic({}))
    expect(line?.state).toBe('Off')
    expect(line?.text).toContain('choose "CABLE Output (VB-Audio Virtual Cable)" as the microphone')
  })

  it('gives the reason when it could not be switched on', () => {
    const line = voiceLine(mic({ reason: 'No virtual cable was found. Install VB-Audio Virtual Cable.' }))
    expect(line).toEqual({ tone: 'critical', state: 'Not available', text: 'No virtual cable was found. Install VB-Audio Virtual Cable.' })
  })

  it('says the voice goes through as it is until a camera session runs', () => {
    const line = voiceLine(mic({ on: true, delay_ms: 30 }))
    expect(line?.state).toBe('Ready')
    expect(line?.text).toContain('as it is until a camera session is running')
  })

  it('says how long the voice is held back once it follows the picture', () => {
    const line = voiceLine(mic({ on: true, picture_ms: 104, offset_ms: 10, delay_ms: 113.6 }))
    expect(line?.tone).toBe('good')
    expect(line?.text).toBe('Your voice is held back 114 ms, as long as the picture takes, so lips and words arrive together.')
  })
})

describe('offsetLabel', () => {
  it('always shows a sign, with a true minus', () => {
    expect(offsetLabel(0)).toBe('0 ms')
    expect(offsetLabel(20)).toBe('+20 ms')
    expect(offsetLabel(-30)).toBe('−30 ms')
  })
})

describe('meterShare', () => {
  it('lifts quiet speech and stays between 0 and 1', () => {
    expect(meterShare(0)).toBe(0)
    expect(meterShare(0.04)).toBeCloseTo(0.2)
    expect(meterShare(2)).toBe(1)
    expect(meterShare(-1)).toBe(0)
  })
})
