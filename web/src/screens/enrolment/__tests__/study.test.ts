import { describe, expect, it } from 'vitest'
import { detailAt, largestFace, signed, span } from '../study'

// Part of what scripts/study_enrolment.py measured on 7 October 2026.
const pictures = [
  { name: 's7', points: [{ face_height_px: 148, detail: 0.33 }, { face_height_px: 198, detail: 0.61 }, { face_height_px: 258, detail: 1.0 }, { face_height_px: 458, detail: 1.7 }] },
  { name: 's1', points: [{ face_height_px: 149, detail: 0.58 }, { face_height_px: 199, detail: 0.79 }, { face_height_px: 260, detail: 1.0 }, { face_height_px: 380, detail: 1.23 }] },
  { name: 'd13', points: [{ face_height_px: 151, detail: 0.37 }, { face_height_px: 201, detail: 0.61 }, { face_height_px: 260, detail: 1.0 }, { face_height_px: 460, detail: 1.71 }] },
]

describe('detailAt', () => {
  it('gives the range over the pictures tried near a face height', () => {
    expect(detailAt(pictures, 200)).toEqual([0.61, 0.79])
    expect(detailAt(pictures, 150)).toEqual([0.33, 0.58])
    expect(detailAt(pictures, 260)).toEqual([1.0, 1.0])
  })
  it('leaves out a picture that was not tried near that size', () => {
    expect(detailAt(pictures, 460)).toEqual([1.7, 1.71]) // s1 stops at 380 px
  })
  it('says so when no picture was tried there', () => {
    expect(detailAt(pictures, 320)).toBeNull()
    expect(detailAt([], 200)).toBeNull()
  })
})

describe('span', () => {
  it('writes a range, or one number when both ends read the same', () => {
    expect(span([0.61, 0.79])).toBe('0.61 to 0.79')
    expect(span([1, 1.001])).toBe('1.00')
  })
})

describe('signed', () => {
  it('always shows the sign, with a real minus', () => {
    expect(signed(-0.0677)).toBe('−0.068')
    expect(signed(0.004)).toBe('+0.004')
  })
})

describe('largestFace', () => {
  it('finds the tallest face any picture was tried at', () => {
    expect(largestFace(pictures)).toBe(460)
  })
})
