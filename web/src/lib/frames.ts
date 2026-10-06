import type { FrameHeader } from './api'

/**
 * Split one frame message from the engine into its parts.
 *
 * Layout (see neuropresence/server/stream.py): 4 bytes giving the length of the
 * header, little-endian; the header as UTF-8 JSON; the camera picture as JPEG;
 * the output as JPEG. The header says how long each picture is. A frame of the
 * enrolment preview has no output: its output length is zero, and null is
 * returned in its place.
 */
export function parseFrame(buffer: ArrayBuffer): [FrameHeader, Blob, Blob | null] {
  const headerLength = new DataView(buffer).getUint32(0, true)
  const sent = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)))
  // An engine from before the enrolment preview sends no kind: all of its frames are live ones.
  const header: FrameHeader = { kind: 'live', ...sent }
  const cameraStart = 4 + headerLength
  const outputStart = cameraStart + header.camera_bytes
  return [
    header,
    new Blob([new Uint8Array(buffer, cameraStart, header.camera_bytes)], { type: 'image/jpeg' }),
    header.output_bytes > 0
      ? new Blob([new Uint8Array(buffer, outputStart, header.output_bytes)], { type: 'image/jpeg' })
      : null,
  ]
}
