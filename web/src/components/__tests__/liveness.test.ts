import { describe, expect, it } from 'vitest'
import type { ConsentPrompt } from '../../lib/api'
import { isAsking, missNote, mouthShare, stepLine, timeLeftShare, turnSide } from '../liveness'

const prompt = (patch: Partial<ConsentPrompt>): ConsentPrompt => ({
  stage: 'act',
  step: 1,
  steps: 2,
  action: 'turn_left',
  prompt: 'Turn your head to your left',
  seconds: 5,
  seconds_left: 5,
  progress: 0,
  reason: '',
  ...patch,
})

describe('isAsking', () => {
  it('is true while the person has something to do', () => {
    expect(isAsking(prompt({ stage: 'settle', action: null }))).toBe(true)
    expect(isAsking(prompt({}))).toBe(true)
    expect(isAsking(prompt({ stage: 'confirming', action: null }))).toBe(true)
  })

  it('is false once the check has ended, and when there is none', () => {
    expect(isAsking(prompt({ stage: 'passed', action: null }))).toBe(false)
    expect(isAsking(prompt({ stage: 'refused', action: null }))).toBe(false)
    expect(isAsking(null)).toBe(false)
    expect(isAsking(undefined)).toBe(false)
  })
})

describe('timeLeftShare', () => {
  it('runs from one to nothing over the time an action is given', () => {
    expect(timeLeftShare(prompt({ seconds_left: 5 }))).toBe(1)
    expect(timeLeftShare(prompt({ seconds_left: 2.5 }))).toBe(0.5)
    expect(timeLeftShare(prompt({ seconds_left: 0 }))).toBe(0)
  })

  it('stays within its range whatever the engine sends', () => {
    expect(timeLeftShare(prompt({ seconds_left: 6 }))).toBe(1)
    expect(timeLeftShare(prompt({ seconds_left: -1 }))).toBe(0)
  })

  it('times nothing while the face is only being waited for', () => {
    expect(timeLeftShare(prompt({ stage: 'settle', action: null, seconds_left: null }))).toBeNull()
    expect(timeLeftShare(prompt({ stage: 'confirming', action: null, seconds_left: null }))).toBeNull()
  })
})

describe('turnSide', () => {
  it('puts the marker on the side the head turns to in a mirrored picture', () => {
    expect(turnSide('turn_left')).toBe('left')
    expect(turnSide('turn_right')).toBe('right')
  })

  it('has no side for an action that is not a turn', () => {
    expect(turnSide('blink')).toBeNull()
    expect(turnSide('open_mouth')).toBeNull()
    expect(turnSide(null)).toBeNull()
  })
})

describe('mouthShare', () => {
  it('says how far the mouth has opened while an open mouth is asked for', () => {
    expect(mouthShare(prompt({ action: 'open_mouth', progress: 0.6 }))).toBe(0.6)
    expect(mouthShare(prompt({ action: 'open_mouth', progress: 1.4 }))).toBe(1)
  })

  it('says nothing for a turn, a blink, or while nothing is asked for', () => {
    expect(mouthShare(prompt({ action: 'turn_left', progress: 0.6 }))).toBeNull()
    expect(mouthShare(prompt({ action: 'blink', progress: null }))).toBeNull()
    expect(mouthShare(prompt({ stage: 'settle', action: null, progress: null }))).toBeNull()
  })
})

describe('missNote', () => {
  it('says what was missing while the action is asked for once more', () => {
    expect(missNote(prompt({ stage: 'settle', action: null, note: 'Not seen in time. Once more.' }))).toBe('Not seen in time. Once more.')
    expect(missNote(prompt({ action: 'open_mouth', note: 'Not seen in time: open your mouth wider.' }))).toBe(
      'Not seen in time: open your mouth wider.',
    )
  })

  it('says nothing when nothing was missed, for an engine that sends no note, and once the check has ended', () => {
    expect(missNote(prompt({ note: '' }))).toBe('')
    expect(missNote(prompt({}))).toBe('')
    expect(missNote(prompt({ stage: 'refused', action: null, note: 'Not seen in time. Once more.' }))).toBe('')
  })
})

describe('stepLine', () => {
  it('says which of the actions this is', () => {
    expect(stepLine(prompt({}))).toBe('Step 1 of 2')
    expect(stepLine(prompt({ stage: 'settle', step: 2, action: null }))).toBe('Step 2 of 2')
  })

  it('says nothing once every action has been seen', () => {
    expect(stepLine(prompt({ stage: 'confirming', step: 2, action: null }))).toBe('')
    expect(stepLine(prompt({ steps: 0, step: 0 }))).toBe('')
  })
})
