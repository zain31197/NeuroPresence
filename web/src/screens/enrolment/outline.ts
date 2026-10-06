/** The face outline on the camera picture: turning the box the engine asks for into the line that is drawn. */

import type { FaceBox } from '../../lib/api'

/** The line is drawn this much larger than the face it asks for, so a face that fits sits just inside it. */
export const OUTLINE_MARGIN = 1.05

/**
 * The ellipse to draw over the camera picture, in the picture's own pixels.
 *
 * The engine works out where the face should be (neuropresence/enrolment/checks.py, beside the
 * limits of the checks), so a face that fills the outline always passes them. The picture is
 * shown mirrored, so left and right swap here.
 */
export function outlineEllipse(box: FaceBox, width: number, height: number) {
  return {
    cx: (1 - (box.x + box.w / 2)) * width,
    cy: (box.y + box.h / 2) * height,
    rx: (box.w / 2) * width * OUTLINE_MARGIN,
    ry: (box.h / 2) * height * OUTLINE_MARGIN,
  }
}
