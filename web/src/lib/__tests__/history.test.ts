import { describe, expect, it } from 'vitest'
import { byDay, dayLabel, duration, median, securityLine, thin } from '../history'

describe('thin and median', () => {
  it('keeps a long series light, ending on its last value', () => {
    const series = Array.from({ length: 1000 }, (_, i) => i)
    const kept = thin(series, 240)
    expect(kept.length).toBeLessThanOrEqual(241)
    expect(kept[0]).toBe(0)
    expect(kept[kept.length - 1]).toBe(999)
    expect(thin([1, 2, 3], 240)).toEqual([1, 2, 3])
  })

  it('takes the middle of the known values', () => {
    expect(median([3, null, 1, 2])).toBe(2)
    expect(median([4, 1, 3, 2])).toBe(2.5)
    expect(median([null, undefined])).toBeNull()
  })
})

const NOW = new Date(2026, 10, 21, 15, 0, 0).getTime() // 21 November 2026, 15:00, local time
const at = (day: number, hour: number) => new Date(2026, 10, day, hour, 0, 0).getTime() / 1000

describe('duration', () => {
  it('reads as people say it', () => {
    expect(duration(45.4)).toBe('45 s')
    expect(duration(724)).toBe('12 min 4 s')
    expect(duration(600)).toBe('10 min')
    expect(duration(3780)).toBe('1 h 3 min')
    expect(duration(-3)).toBe('0 s')
  })
})

describe('dayLabel and byDay', () => {
  it('names the last two days and dates the others', () => {
    expect(dayLabel(at(21, 9), NOW)).toBe('Today')
    expect(dayLabel(at(20, 23), NOW)).toBe('Yesterday')
    expect(dayLabel(at(14, 12), NOW)).toBe('14 November')
    expect(dayLabel(new Date(2025, 11, 31, 12).getTime() / 1000, NOW)).toBe('31 December 2025')
  })

  it('groups items under their day in order', () => {
    const items = [{ at: at(21, 14) }, { at: at(21, 9) }, { at: at(20, 18) }]
    expect(byDay(items, NOW).map((group) => [group.day, group.items.length])).toEqual([
      ['Today', 2],
      ['Yesterday', 1],
    ])
  })
})

describe('securityLine', () => {
  it('says what a check was for and how it ended', () => {
    const passed = securityLine({ kind: 'check', at: 0, purpose: 'session', passed: true, failure: null, reason: null, lowest_match: 0.812, session_id: 3 })
    expect(passed.title).toBe('Check passed before a session')
    expect(passed.detail).toContain('0.81')
    expect(passed.tone).toBe('good')
    const refused = securityLine({ kind: 'check', at: 0, purpose: 'verify', passed: false, failure: 'late', reason: 'Your head did not turn.', lowest_match: null, session_id: null })
    expect(refused).toEqual({ title: 'Check not passed to verify a face', detail: 'Your head did not turn.', tone: 'critical' })
  })

  it('names enrolment changes', () => {
    expect(securityLine({ kind: 'enrolment', at: 0, action: 'pose_registered', detail: 'left' }).detail).toBe('Left pose')
    expect(securityLine({ kind: 'enrolment', at: 0, action: 'face_forgotten', detail: null }).title).toBe('Face and picture forgotten')
  })
})
