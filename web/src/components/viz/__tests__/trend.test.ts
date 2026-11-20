import { describe, expect, it } from 'vitest'
import { niceStep } from '../TrendChart'

describe('niceStep', () => {
  it('rounds an axis step up to 1, 2, 2.5 or 5 times a power of ten', () => {
    expect(niceStep(13)).toBe(20)
    expect(niceStep(0.47)).toBe(0.5)
    expect(niceStep(55)).toBe(100)
    expect(niceStep(2.2)).toBe(2.5)
    expect(niceStep(1)).toBe(1)
  })
})
