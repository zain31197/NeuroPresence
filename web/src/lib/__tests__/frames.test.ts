import { describe, expect, it } from 'vitest'
import { parseFrame } from '../frames'

/** Build a frame message the way the engine does (neuropresence/server/stream.py). */
function pack(header: object, camera: number[], output: number[]): ArrayBuffer {
  const headerBytes = new TextEncoder().encode(JSON.stringify(header))
  const message = new Uint8Array(4 + headerBytes.length + camera.length + output.length)
  new DataView(message.buffer).setUint32(0, headerBytes.length, true)
  message.set(headerBytes, 4)
  message.set(camera, 4 + headerBytes.length)
  message.set(output, 4 + headerBytes.length + camera.length)
  return message.buffer
}

const bytes = async (blob: Blob | null) => (blob ? Array.from(new Uint8Array(await blob.arrayBuffer())) : null)

describe('parseFrame', () => {
  it('splits a message into its header and the two pictures', async () => {
    const header = { id: 7, kind: 'live', live: true, status: 'ok', camera_bytes: 3, output_bytes: 5 }
    const [parsed, camera, output] = parseFrame(pack(header, [1, 2, 3], [9, 8, 7, 6, 5]))
    expect(parsed).toEqual(header)
    expect(camera.type).toBe('image/jpeg')
    expect(await bytes(camera)).toEqual([1, 2, 3])
    expect(await bytes(output)).toEqual([9, 8, 7, 6, 5])
  })

  it('reads headers that contain characters outside ASCII', async () => {
    const header = { id: 1, kind: 'live', live: false, status: 'no_face', camera_bytes: 1, output_bytes: 1, note: 'café' }
    const [parsed, camera, output] = parseFrame(pack(header, [42], [43]))
    expect(parsed).toEqual(header)
    expect(await bytes(camera)).toEqual([42])
    expect(await bytes(output)).toEqual([43])
  })

  it('gives no output for a frame of the enrolment preview', async () => {
    const header = { id: 3, kind: 'preview', live: false, status: 'ok', camera_bytes: 4, output_bytes: 0 }
    const [parsed, camera, output] = parseFrame(pack(header, [5, 6, 7, 8], []))
    expect(parsed.kind).toBe('preview')
    expect(await bytes(camera)).toEqual([5, 6, 7, 8])
    expect(output).toBeNull()
  })

  it('treats a frame from an engine that names no kind as a live one', () => {
    const [parsed] = parseFrame(pack({ id: 2, live: true, status: 'ok', camera_bytes: 1, output_bytes: 1 }, [1], [2]))
    expect(parsed.kind).toBe('live')
  })
})
