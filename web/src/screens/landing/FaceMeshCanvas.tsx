import { useEffect, useRef } from 'react'
import mesh from '../../data/faceMesh.json'

// The 478 landmarks the tracker follows, averaged over several sample faces so
// that this is nobody's face in particular (see scripts/make_face_mesh.py).
const POINTS = mesh.points as [number, number, number][]
const MESH_EDGES = mesh.tesselation as [number, number][]
const CONTOUR_EDGES = [...mesh.contours, ...mesh.irises] as [number, number][]

const DEPTH = 0.8 // the tracker's depth is a little exaggerated
const MOUTH_Y = Math.min(...[13, 14].map((index) => POINTS[index][1])) // between the lips
const CHIN_Y = Math.max(...POINTS.map((point) => point[1]))

/** Head pose and mouth opening at a moment in time: a slow, natural-looking wander. */
export function poseAt(seconds: number) {
  const talking = Math.max(0, Math.sin(seconds * 2.3) * Math.sin(seconds * 0.71 + 0.4))
  return {
    yaw: 0.3 * Math.sin(seconds / 2.3),
    pitch: 0.11 * Math.sin(seconds / 3.4 + 1.2),
    roll: 0.05 * Math.sin(seconds / 4.9 + 0.5),
    mouth: 0.16 * talking,
  }
}

interface Props {
  /** Line colours: the fine mesh, and the stronger outlines of eyes, brows, lips and jaw. */
  meshColor: string
  contourColor: string
  /** Seconds this face trails the shared clock, so one canvas can follow another. */
  delay?: number
  className?: string
  label: string
}

/** A face drawn from its landmarks, turning and talking. Purely illustrative. */
export function FaceMeshCanvas({ meshColor, contourColor, delay = 0, className, label }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const element = canvas.current
    const context = element?.getContext('2d')
    if (!element || !context) return
    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const projected = new Float32Array(POINTS.length * 2)
    let frame = 0

    const draw = (milliseconds: number) => {
      const ratio = Math.min(2, window.devicePixelRatio || 1)
      const width = element.clientWidth
      const height = element.clientHeight
      if (element.width !== Math.round(width * ratio) || element.height !== Math.round(height * ratio)) {
        element.width = Math.round(width * ratio)
        element.height = Math.round(height * ratio)
      }
      const { yaw, pitch, roll, mouth } = poseAt(still ? 1.6 : milliseconds / 1000 - delay)
      const [cy, sy, cp, sp, cr, sr] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch), Math.cos(roll), Math.sin(roll)]
      const scale = Math.min(width / 3.5, height / 4.15) * ratio
      const [cx, cyCentre] = [element.width / 2, element.height * 0.535]

      for (let i = 0; i < POINTS.length; i++) {
        let [x, y, z] = POINTS[i]
        z *= DEPTH
        // Open the mouth: the lower lip drops at once, the rest of the jaw more the lower it is.
        if (y > MOUTH_Y) {
          const depth = (y - MOUTH_Y) / (CHIN_Y - MOUTH_Y)
          const nearLips = Math.max(0, 1 - Math.abs(x) / 0.45)
          y += mouth * Math.min(1, depth + 0.35 * nearLips)
        }
        const x1 = x * cy + z * sy // turn left and right
        const z1 = -x * sy + z * cy
        const y1 = y * cp - z1 * sp // nod
        const x2 = x1 * cr - y1 * sr // tilt
        const y2 = x1 * sr + y1 * cr
        projected[i * 2] = cx + x2 * scale
        projected[i * 2 + 1] = cyCentre + y2 * scale
      }

      context.clearRect(0, 0, element.width, element.height)
      context.lineJoin = 'round'
      const stroke = (edges: [number, number][], color: string, lineWidth: number) => {
        context.beginPath()
        for (const [a, b] of edges) {
          context.moveTo(projected[a * 2], projected[a * 2 + 1])
          context.lineTo(projected[b * 2], projected[b * 2 + 1])
        }
        context.strokeStyle = color
        context.lineWidth = lineWidth * ratio
        context.stroke()
      }
      stroke(MESH_EDGES, meshColor, 0.6)
      stroke(CONTOUR_EDGES, contourColor, 1.5)
      if (!still) frame = requestAnimationFrame(draw)
    }

    frame = requestAnimationFrame(draw)
    const observer = new ResizeObserver(() => still && draw(0))
    observer.observe(element)
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
    }
  }, [meshColor, contourColor, delay])

  return <canvas ref={canvas} className={className} role="img" aria-label={label} />
}
