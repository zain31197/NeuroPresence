import { useEffect, useRef, useState, type PointerEvent } from 'react'

interface Picture {
  name: string
  points: { face_height_px: number; detail: number }[]
}

interface Props {
  /** One line each: the same picture enrolled at several face sizes. */
  pictures: Picture[]
  /** Detail is a multiple of what a face this tall yields, so every line passes through 1 here. */
  referencePx: number
  /** The smallest face the size check accepts. */
  minPx: number
  /** From this size on, the check stops suggesting to come closer. */
  goodPx: number
  /** Face height of the enrolled picture, if there is one. */
  yours?: number | null
  height?: number
}

const PAD = { top: 22, right: 14, bottom: 24, left: 38 }
const REACH = 28 // a point answers the pointer from this many pixels away

const steps = (from: number, to: number, step: number) =>
  Array.from({ length: Math.floor((to - from) / step + 1e-9) + 1 }, (_, index) => from + index * step)

/** Detail in the output face against the size of the face in the enrolled picture. */
export function FaceSizeChart({ pictures, referencePx, minPx, goodPx, yours, height = 240 }: Props) {
  const frame = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState<number | null>(null)
  const points = pictures.flatMap((picture) => picture.points.map((point) => ({ ...point, name: picture.name })))
  const empty = points.length === 0

  useEffect(() => {
    const element = frame.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    observer.observe(element)
    return () => observer.disconnect()
  }, [empty]) // the frame is only drawn once there is something to plot

  if (empty) return null

  const heights = points.map((point) => point.face_height_px)
  const xLow = Math.floor((Math.min(...heights, minPx) - 20) / 50) * 50
  const xHigh = Math.ceil((Math.max(...heights, yours ?? 0) + 10) / 50) * 50
  const yHigh = Math.ceil(Math.max(...points.map((point) => point.detail), 1) / 0.5) * 0.5
  const innerWidth = Math.max(0, width - PAD.left - PAD.right)
  const innerHeight = height - PAD.top - PAD.bottom
  const baseline = PAD.top + innerHeight
  const x = (px: number) => PAD.left + ((px - xLow) / (xHigh - xLow)) * innerWidth
  const y = (detail: number) => PAD.top + innerHeight * (1 - detail / yHigh)

  const onMove = (event: PointerEvent) => {
    const bounds = frame.current?.getBoundingClientRect()
    if (!bounds) return
    const [px, py] = [event.clientX - bounds.left, event.clientY - bounds.top]
    let nearest: number | null = null
    let reach = REACH
    points.forEach((point, index) => {
      const distance = Math.hypot(x(point.face_height_px) - px, y(point.detail) - py)
      if (distance < reach) [nearest, reach] = [index, distance]
    })
    setHover(nearest)
  }

  const shown = hover !== null ? points[hover] : null
  const label = { fontSize: 10, fontFamily: 'var(--font-mono)', fill: 'var(--color-ink-500)' }

  return (
    <figure>
      <div ref={frame} className="relative" style={{ height }} onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {width > 0 && (
          <svg width={width} height={height} className="block" role="img" aria-label={`Detail in the output face against face height in the enrolled picture. ${pictures.length} sample pictures; the table after this chart gives the values.`}>
            {/* Left of the minimum, a picture is refused. */}
            <rect x={x(xLow)} y={PAD.top} width={Math.max(0, x(minPx) - x(xLow))} height={innerHeight} fill="var(--color-ink-50)" />
            {x(minPx) - x(xLow) > 64 && (
              <text x={x(xLow) + 8} y={PAD.top + 15} {...label} fill="var(--color-ink-400)">
                TOO SMALL
              </text>
            )}

            {steps(0, yHigh, 0.5).map((tick) => (
              <g key={tick}>
                <line x1={PAD.left} x2={PAD.left + innerWidth} y1={y(tick)} y2={y(tick)} stroke={tick === 0 ? 'var(--color-ink-300)' : 'var(--color-ink-100)'} />
                <text x={PAD.left - 7} y={y(tick) + 3.5} textAnchor="end" {...label}>
                  {tick.toFixed(1)}×
                </text>
              </g>
            ))}
            {steps(Math.ceil(xLow / 100) * 100, xHigh, 100).map((tick) => (
              <text key={tick} x={x(tick)} y={baseline + 15} textAnchor="middle" {...label}>
                {tick}
              </text>
            ))}

            {/* The two limits of the size check: dashed, as thresholds are everywhere in the app. */}
            {[
              { px: minPx, name: 'MIN' },
              { px: goodPx, name: 'GOOD' },
            ].map(({ px, name }) => (
              <g key={name}>
                <line x1={x(px)} x2={x(px)} y1={PAD.top} y2={baseline} stroke="var(--color-ink-400)" strokeDasharray="3 3" />
                <text x={x(px)} y={PAD.top - 8} textAnchor="middle" {...label}>
                  {name} {px}
                </text>
              </g>
            ))}

            {typeof yours === 'number' && (
              <g>
                <line x1={x(yours)} x2={x(yours)} y1={PAD.top} y2={baseline} stroke="var(--color-ink-950)" strokeWidth="1.5" />
                <circle cx={x(yours)} cy={baseline} r="4" fill="var(--color-ink-950)" stroke="var(--color-surface)" strokeWidth="2" />
              </g>
            )}

            {pictures.map((picture) => (
              <path
                key={picture.name}
                d={picture.points.map((point, index) => `${index ? 'L' : 'M'}${x(point.face_height_px).toFixed(1)} ${y(point.detail).toFixed(1)}`).join(' ')}
                fill="none"
                stroke="var(--color-series-1)"
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}
            {points.map((point, index) => (
              <circle
                key={index}
                cx={x(point.face_height_px)}
                cy={y(point.detail)}
                r={index === hover ? 5 : 3.5}
                fill="var(--color-series-1)"
                stroke="var(--color-surface)"
                strokeWidth="2"
              />
            ))}
          </svg>
        )}
        {shown && (
          <div
            className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-[6px] bg-ink-950 px-2 py-1 text-[11.5px] whitespace-nowrap text-white shadow-raised"
            style={{ left: Math.min(Math.max(x(shown.face_height_px), 70), Math.max(70, width - 70)), top: y(shown.detail) - 10 }}
          >
            <span className="font-semibold tabular-nums">{shown.detail.toFixed(2)}×</span>
            <span className="ml-1.5 text-ink-300 tabular-nums">at {shown.face_height_px} px</span>
          </div>
        )}
      </div>

      <figcaption className="mt-1 flex flex-wrap items-center gap-x-5 gap-y-1.5 pl-[38px] text-[12px] text-ink-600">
        <span className="w-full text-ink-500">Face height in the enrolled picture, in pixels. 1.0× is a {referencePx} px face.</span>
        <span className="inline-flex items-center gap-2">
          <span className="h-0.5 w-4 rounded-full bg-series-1" />
          One sample picture per line
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="w-4 border-t border-dashed border-ink-400" />
          Limits of the size check
        </span>
        {typeof yours === 'number' && (
          <span className="inline-flex items-center gap-2">
            <span className="h-3 w-0.5 rounded-full bg-ink-950" />
            Your picture, <span className="tabular-nums">{Math.round(yours)} px</span>
          </span>
        )}
      </figcaption>

      <table className="sr-only">
        <caption>Detail in the output face by face height in the enrolled picture</caption>
        <thead>
          <tr>
            <th scope="col">Sample picture</th>
            <th scope="col">Face height in pixels</th>
            <th scope="col">Detail, as a multiple of a {referencePx} pixel face</th>
          </tr>
        </thead>
        <tbody>
          {points.map((point, index) => (
            <tr key={index}>
              <td>{point.name}</td>
              <td>{point.face_height_px}</td>
              <td>{point.detail.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}
