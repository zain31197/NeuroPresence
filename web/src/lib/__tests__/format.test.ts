import { describe, expect, it } from 'vitest'
import { clock, dash, dateTime, fixed, percent, splitHint, verdict } from '../format'

describe('fixed', () => {
  it('rounds to the asked number of digits', () => {
    expect(fixed(7.46, 1)).toBe('7.5')
    expect(fixed(134.04, 0)).toBe('134')
  })
  it('shows a dash where there is no value', () => {
    expect(fixed(null)).toBe(dash)
    expect(fixed(undefined)).toBe(dash)
    expect(fixed(Number.NaN)).toBe(dash)
  })
})

describe('percent', () => {
  it('turns a share into a percentage', () => {
    expect(percent(0.746)).toBe('75%')
    expect(percent(0.746, 1)).toBe('74.6%')
    expect(percent(null)).toBe(dash)
  })
})

describe('clock', () => {
  it('formats seconds as minutes and seconds, adding hours only when needed', () => {
    expect(clock(0)).toBe('0:00')
    expect(clock(75.9)).toBe('1:15')
    expect(clock(3725)).toBe('1:02:05')
    expect(clock(null)).toBe(dash)
  })
})

describe('dateTime', () => {
  it('writes a local time the way the enrolment record gives it', () => {
    expect(dateTime('2026-10-07T14:32:10')).toBe('7 October 2026, 14:32')
    expect(dateTime('2026-10-07T09:05:00')).toBe('7 October 2026, 09:05')
  })
  it('passes through what it cannot read', () => {
    expect(dateTime('not a date')).toBe('not a date')
  })
})

describe('splitHint', () => {
  it('separates what is wrong from what to do', () => {
    expect(splitHint('The head is turned to one side. Face the camera.')).toEqual(['The head is turned to one side.', 'Face the camera.'])
    expect(splitHint('The face is too far to one side. Move to your left, toward the middle.')).toEqual([
      'The face is too far to one side.',
      'Move to your left, toward the middle.',
    ])
  })
  it('treats a single sentence as the thing to do', () => {
    expect(splitHint('Hold still.')).toEqual(['', 'Hold still.'])
  })
})

describe('verdict', () => {
  it('judges "at least" targets', () => {
    expect(verdict(24, 24, true)).toBe('good')
    expect(verdict(7.5, 24, true)).toBe('off')
  })
  it('judges "at most" targets', () => {
    expect(verdict(150, 150, false)).toBe('good')
    expect(verdict(154, 150, false)).toBe('off')
  })
  it('gives no verdict without a reading', () => {
    expect(verdict(null, 24, true)).toBe('unknown')
    expect(verdict(undefined, 24, true)).toBe('unknown')
  })
})
