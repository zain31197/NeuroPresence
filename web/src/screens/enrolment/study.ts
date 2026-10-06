/** Reading the enrolment study's numbers for the Enrolment screen. */

import type { EnrolmentStudy } from '../../lib/api'

type Pictures = EnrolmentStudy['face_size']['pictures']

/**
 * The lowest and highest detail the sample pictures give at about this face height.
 *
 * Each picture was tried at slightly different sizes (198, 199 and 201 px for "200"),
 * so its nearest point counts if it is within a few pixels. null if no picture was tried there.
 */
export function detailAt(pictures: Pictures, px: number, within = 15): [low: number, high: number] | null {
  const found = pictures.flatMap((picture) => {
    const nearest = picture.points.reduce<Pictures[number]['points'][number] | null>(
      (best, point) => (best === null || Math.abs(point.face_height_px - px) < Math.abs(best.face_height_px - px) ? point : best),
      null,
    )
    return nearest && Math.abs(nearest.face_height_px - px) <= within ? [nearest.detail] : []
  })
  return found.length ? [Math.min(...found), Math.max(...found)] : null
}

/** "0.61 to 0.79", or one number when both ends are the same. */
export const span = ([low, high]: [number, number], digits = 2) =>
  low.toFixed(digits) === high.toFixed(digits) ? low.toFixed(digits) : `${low.toFixed(digits)} to ${high.toFixed(digits)}`

/** A change with a real minus sign, so "−0.068" lines up with "+0.004". */
export const signed = (value: number, digits = 3) => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(digits)}`

/** The tallest face any sample picture was tried at. */
export const largestFace = (pictures: Pictures) =>
  Math.max(...pictures.flatMap((picture) => picture.points.map((point) => point.face_height_px)))
