import { describe, expect, it } from 'vitest'
import type { EnrolmentRecord, FaceIdentity } from '../../../lib/api'
import { stepsDone } from '../Progress'

const pose = { verified_at: '2026-11-20T10:00:00', angle_deg: 0, checks: [] }
const face = (poses: string[]) => ({ id: 'f', verified_at: '2026-11-20T10:00:00', poses: Object.fromEntries(poses.map((p) => [p, pose])) }) as unknown as FaceIdentity
const record = { id: 'r' } as unknown as EnrolmentRecord

describe('stepsDone', () => {
  it('counts the steps from what is stored', () => {
    expect(stepsDone(null, null)).toBe(0)
    expect(stepsDone(face([]), null)).toBe(1)
    expect(stepsDone(face(['front', 'left', 'right']), null)).toBe(1)
    expect(stepsDone(face(['front', 'left', 'right', 'up', 'down']), null)).toBe(2)
    expect(stepsDone(face(['front', 'left', 'right', 'up', 'down']), record)).toBe(3)
  })

  it('lets a picture enrolled before poses existed go live', () => {
    expect(stepsDone(face([]), record)).toBe(3)
  })
})
