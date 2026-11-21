import { motion } from 'motion/react'
import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { EASE_OUT } from '../motion'

export interface TrendPoint {
  /** What the run was, as the tooltip and the screen-reader table name it. */
  label: string
  date: string
  value: number | null
}

interface Props {
  points: TrendPoint[]
  /** The project's target, drawn as a dashed threshold line. */
  target?: number
  /** 'up' when more is better, 'down' when less is: which side of the target is good. */
  better: 'up' | 'down'
  format: (value: number) => string
  /** What the chart shows, for screen readers and the table. */
  title: string
  height?: number
  /** The words under the two ends of the axis, and in the table. Default: the run's date. */
  xLabel?: (point: TrendPoint) => string
  /** What one point is, in the plural, for screen readers: "runs" by default, "readings" over a session. */
  noun?: string
}

// Past this many points a marker on each would be clutter: only the one pointed at is drawn.
const MARKERS_UP_TO = 36

const PAD = { top: 18, right: 14, bottom: 26, left: 40 }

/**
 * One figure across the saved benchmark runs, oldest to newest: one line, a marker per run, the target as a
 * dashed threshold. The line draws itself in the first time it is seen. Hover or arrow keys show each run.
 */
export function TrendChart({ points, target, better, format, title, height = 190, xLabel, noun = 'runs' }: Props) {
  const frame = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [active, setActive] = useState<number | null>(null)
  const gradient = useId()

  useEffect(() => {
    const element = frame.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const known = points.map((p) => p.value).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  const innerWidth = Math.max(0, width - PAD.left - PAD.right)
  const innerHeight = height - PAD.top - PAD.bottom
  // The scale always holds the target, starts at zero so differences are not exaggerated, and ends on a round number.
  const step = niceStep((Math.max(...known, target ?? 0) * 1.08 || 1) / 5)
  const top = step * Math.ceil((Math.max(...known, target ?? 0) * 1.08 || 1) / step)
  const x = (i: number) => PAD.left + (points.length <= 1 ? innerWidth / 2 : (i / (points.length - 1)) * innerWidth)
  const y = (v: number) => PAD.top + innerHeight - (v / top) * innerHeight
  const ticks = Array.from({ length: Math.round(top / step) + 1 }, (_, i) => Number((i * step).toPrecision(3)))

  const drawn = points.map((p, i) => (typeof p.value === 'number' ? [x(i), y(p.value)] : null)).filter(Boolean) as [number, number][]
  const line = drawn.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`).join(' ')
  const area = drawn.length ? `${line} L${drawn[drawn.length - 1][0].toFixed(1)},${PAD.top + innerHeight} L${drawn[0][0].toFixed(1)},${PAD.top + innerHeight} Z` : ''
  const good = (v: number) => (target === undefined ? true : better === 'up' ? v >= target : v <= target)

  const keys = (event: KeyboardEvent) => {
    if (event.key === 'ArrowRight') setActive((a) => Math.min(points.length - 1, (a ?? -1) + 1))
    else if (event.key === 'ArrowLeft') setActive((a) => Math.max(0, (a ?? points.length) - 1))
    else if (event.key === 'Escape') setActive(null)
  }
  const shown = active === null ? null : points[active]
  const endLabel = xLabel ?? ((point: TrendPoint) => point.date.slice(5, 10))
  const markers = points.length <= MARKERS_UP_TO

  return (
    <div ref={frame} className="relative">
      {width > 0 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${title}, ${points.length} ${noun}`}
          tabIndex={0}
          onKeyDown={keys}
          onBlur={() => setActive(null)}
          onPointerLeave={() => setActive(null)}
          onPointerMove={(event) => {
            const box = event.currentTarget.getBoundingClientRect()
            const share = (event.clientX - box.left - PAD.left) / Math.max(1, innerWidth)
            setActive(Math.max(0, Math.min(points.length - 1, Math.round(share * (points.length - 1)))))
          }}
          className="block overflow-visible rounded-panel outline-none focus-visible:ring-2 focus-visible:ring-accent-500"
        >
          <defs>
            <linearGradient id={gradient} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="var(--color-accent-500)" stopOpacity="0.18" />
              <stop offset="100%" stopColor="var(--color-accent-500)" stopOpacity="0" />
            </linearGradient>
          </defs>
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={PAD.left} x2={PAD.left + innerWidth} y1={y(tick)} y2={y(tick)} stroke="var(--color-ink-100)" />
              <text x={PAD.left - 8} y={y(tick)} dy="0.32em" textAnchor="end" className="fill-ink-400 font-mono text-[10px]">
                {tick}
              </text>
            </g>
          ))}
          {target !== undefined && (
            <g>
              <line x1={PAD.left} x2={PAD.left + innerWidth} y1={y(target)} y2={y(target)} stroke="var(--color-ink-700)" strokeDasharray="4 4" />
              <text x={PAD.left + innerWidth} y={known.length && known[known.length - 1] >= target ? y(target) + 14 : y(target) - 6} textAnchor="end" className="fill-ink-600 font-mono text-[10px]">
                target {format(target)}
              </text>
            </g>
          )}
          <motion.path
            d={area}
            fill={`url(#${gradient})`}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.9, delay: 0.4 }}
          />
          <motion.path
            d={line}
            fill="none"
            stroke="var(--color-accent-500)"
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            whileInView={{ pathLength: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1.2, ease: EASE_OUT }}
          />
          {active !== null && <line x1={x(active)} x2={x(active)} y1={PAD.top} y2={PAD.top + innerHeight} stroke="var(--color-ink-200)" />}
          {points.map((point, i) =>
            typeof point.value === 'number' && (markers || active === i) ? (
              <motion.circle
                key={`${point.label}-${point.date}-${i}`}
                cx={x(i)}
                cy={y(point.value)}
                r={active === i ? 5.5 : 4}
                fill={good(point.value) ? 'var(--color-surface)' : 'var(--color-serious)'}
                stroke={good(point.value) ? 'var(--color-accent-500)' : 'var(--color-surface)'}
                strokeWidth={2}
                initial={{ opacity: 0, scale: 0.4 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={markers ? { duration: 0.35, delay: 0.25 + (i / Math.max(1, points.length)) * 0.9 } : { duration: 0.12 }}
              />
            ) : null,
          )}
          {points.length > 0 && (
            <>
              <text x={x(0)} y={height - 6} textAnchor="start" className="fill-ink-400 font-mono text-[10px]">
                {endLabel(points[0])}
              </text>
              <text x={x(points.length - 1)} y={height - 6} textAnchor="end" className="fill-ink-400 font-mono text-[10px]">
                {endLabel(points[points.length - 1])}
              </text>
            </>
          )}
        </svg>
      )}
      {shown && typeof shown.value === 'number' && active !== null && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-[10px] border border-line bg-surface px-3 py-2 text-[12px] shadow-raised"
          style={{ left: Math.min(Math.max(x(active), 80), width - 80), top: y(shown.value) - 10 }}
        >
          <p className="font-semibold whitespace-nowrap text-ink-950">{shown.label}</p>
          <p className="mt-0.5 font-mono whitespace-nowrap text-ink-600 tabular-nums">
            {format(shown.value)}
            {xLabel ? '' : ` · ${shown.date.slice(0, 10)}`}
          </p>
        </div>
      )}
      <div className="sr-only">
        <table>
          <caption>{title}</caption>
          <thead>
            <tr>
              <th scope="col">Run</th>
              <th scope="col">Date</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point) => (
              <tr key={`${point.label}-${point.date}`}>
                <td>{point.label}</td>
                <td>{xLabel ? xLabel(point) : point.date.slice(0, 10)}</td>
                <td>{typeof point.value === 'number' ? format(point.value) : 'not measured'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** A round step for an axis: 1, 2, 2.5 or 5 times a power of ten, at least `rough`. */
export function niceStep(rough: number): number {
  const power = 10 ** Math.floor(Math.log10(rough))
  return ([1, 2, 2.5, 5, 10].find((m) => m * power >= rough) ?? 10) * power
}
