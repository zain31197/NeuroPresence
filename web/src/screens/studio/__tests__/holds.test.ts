import { describe, expect, it } from 'vitest'
import type { IdentityGuard } from '../../../lib/api'
import { guardNote, promptSecondsLeft, PROMPT_SECONDS } from '../holds'

const guard = (patch: Partial<IdentityGuard>): IdentityGuard => ({ enabled: true, state: 'steady', seconds: null, mean: 0.9, low: 0.75, ...patch })

describe('guardNote', () => {
  it('says nothing while the output matches the picture', () => {
    expect(guardNote(guard({}))).toBeNull()
  })

  it('names each thing the guard is doing', () => {
    expect(guardNote(guard({ state: 'dipping' }))?.text).toBe('Low for a moment')
    expect(guardNote(guard({ state: 'anchoring' }))?.text).toBe('Fresh neutral pose')
    expect(guardNote(guard({ state: 'fallback' }))).toEqual({ text: 'Still picture shown', tone: 'critical' })
  })

  it('says so when the fallback is switched off, whatever the state', () => {
    expect(guardNote(guard({ enabled: false, state: 'dipping' }))).toEqual({ text: 'Fallback off', tone: 'neutral' })
  })

  it('says nothing for an engine that does not send a guard', () => {
    expect(guardNote(undefined)).toBeNull()
    expect(guardNote(null)).toBeNull()
  })
})

describe('promptSecondsLeft', () => {
  it('counts down from the moment the still picture went up', () => {
    expect(promptSecondsLeft(0, false)).toBe(PROMPT_SECONDS)
    expect(promptSecondsLeft(1.2, false)).toBe(4)
    expect(promptSecondsLeft(4.9, false)).toBe(1)
  })

  it('stops asking when the time is up', () => {
    expect(promptSecondsLeft(5, false)).toBe(0)
    expect(promptSecondsLeft(60, false)).toBe(0)
  })

  it('stops asking as soon as the person chooses to stay', () => {
    expect(promptSecondsLeft(1, true)).toBe(0)
  })

  it('does not ask when nothing is on hold', () => {
    expect(promptSecondsLeft(null, false)).toBe(0)
    expect(promptSecondsLeft(undefined, false)).toBe(0)
  })
})
