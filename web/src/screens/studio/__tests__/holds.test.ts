import { describe, expect, it } from 'vitest'
import type { IdentityGuard } from '../../../lib/api'
import { guardNote, identityHoldText } from '../holds'

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

describe('identityHoldText', () => {
  it('says the still picture resumes by itself, and from when', () => {
    const text = identityHoldText(guard({ state: 'fallback', seconds: 1, resumes_in: 3.2, staying: false }))
    expect(text).toContain('resumes by itself with a fresh neutral pose once you sit facing the camera, mouth closed')
    expect(text).toContain('in 4 s at the earliest')
  })

  it('leaves the time out once the wait is over and only the rest is missing', () => {
    const text = identityHoldText(guard({ state: 'fallback', seconds: 9, resumes_in: 0, staying: false }))
    expect(text).toMatch(/mouth closed\.$/)
  })

  it('says it stays when the person chose to stay', () => {
    expect(identityHoldText(guard({ state: 'fallback', resumes_in: null, staying: true }))).toContain('until you resume')
  })

  it('still says what to do for an engine that does not resume by itself', () => {
    expect(identityHoldText(guard({ state: 'fallback' }))).toContain('Resume with a fresh neutral pose')
    expect(identityHoldText(undefined)).toContain('Resume with a fresh neutral pose')
  })
})
