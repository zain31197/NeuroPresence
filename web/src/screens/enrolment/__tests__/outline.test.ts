import { describe, expect, it } from 'vitest'
import { OUTLINE_MARGIN, outlineEllipse } from '../outline'

// What the engine sends for a 1280 x 720 camera picture: a face 54% of its height, in the middle.
const box = { x: 0.3709, y: 0.2268, w: 0.2582, h: 0.54 }

describe('outlineEllipse', () => {
  it('draws the outline around the face box, a little larger than it', () => {
    const { cx, cy, rx, ry } = outlineEllipse(box, 1280, 720)
    expect(cx).toBeCloseTo(640, 0)
    expect(cy).toBeCloseTo((0.2268 + 0.27) * 720, 1)
    expect(2 * ry).toBeCloseTo(0.54 * 720 * OUTLINE_MARGIN, 1)
    expect(2 * rx).toBeCloseTo(0.2582 * 1280 * OUTLINE_MARGIN, 1)
  })

  it('is large enough for a face of a good size: taller than half the picture', () => {
    // The first outline was 47% of the height and 0.72 as wide as tall: smaller and narrower than a well-placed face.
    const { rx, ry } = outlineEllipse(box, 1280, 720)
    expect((2 * ry) / 720).toBeGreaterThan(0.55)
    expect(rx / ry).toBeGreaterThan(0.8)
  })

  it('swaps left and right, because the picture is shown mirrored', () => {
    const left = { x: 0.1, y: 0.2, w: 0.2, h: 0.5 }
    expect(outlineEllipse(left, 1000, 500).cx).toBeCloseTo(800, 5) // its middle is at 20%, drawn at 80%
  })

  it('keeps its place whatever size the picture is drawn at', () => {
    const small = outlineEllipse(box, 640, 360)
    const large = outlineEllipse(box, 1280, 720)
    expect(large.cx / small.cx).toBeCloseTo(2, 5)
    expect(large.ry / small.ry).toBeCloseTo(2, 5)
  })
})
