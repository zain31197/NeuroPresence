import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'

interface Props {
  /** Readings, oldest first. null where there was no reading. */
  values: (number | null)[]
  /** How many readings fill the full width; the newest sits at the right edge. */
  capacity: number
  /** Seconds between two readings, for the "n s ago" in the readout. */
  interval: number
  /** Drawn as a dashed threshold line and labelled with its value. */
  target?: number
  format: (value: number) => string
  /** What the line shows, for screen readers. */
  label: string
  height?: number
}

const PAD = { top: 7, right: 30, bottom: 7, left: 2 }

/** A strip chart of the last minute: a quiet line, its newest point marked, the target as a threshold. */
export function Sparkline({ values: readings, capacity, interval, target, format, label, height = 46 }: Props) {
  // Anything that is not a real number is treated as "no reading".
  const values = readings.map((value) => (typeof value === 'number' && Number.isFinite(value) ? value : null))
  const frame = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const element = frame.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const known = values.filter((value): value is number => value !== null)
  const innerWidth = Math.max(0, width - PAD.left - PAD.right)
  const innerHeight = height - PAD.top - PAD.bottom

  // The scale always includes the target, so the gap to it is visible.
  const reach = target === undefined ? known : [...known, target]
  let low = Math.min(...reach)
  let high = Math.max(...reach)
  if (!Number.isFinite(low)) [low, high] = [0, 1]
  const margin = (high - low || Math.abs(high) || 1) * 0.12
  low -= margin
  high += margin

  const step = capacity > 1 ? innerWidth / (capacity - 1) : 0
  const x = (index: number) => PAD.left + innerWidth - (values.length - 1 - index) * step
  const y = (value: number) => PAD.top + innerHeight * (1 - (value - low) / (high - low))

  // One path per unbroken run of readings, so a gap is drawn as a gap.
  const runs: string[] = []
  let run: string[] = []
  values.forEach((value, index) => {
    if (value === null) {
      if (run.length) runs.push(run.join(' '))
      run = []
    } else {
      run.push(`${run.length ? 'L' : 'M'}${x(index).toFixed(1)} ${y(value).toFixed(1)}`)
    }
  })
  if (run.length) runs.push(run.join(' '))

  const lastIndex = values.length - 1
  const last = lastIndex >= 0 ? values[lastIndex] : null
  const shown = hover !== null && values[hover] !== null && values[hover] !== undefined ? hover : null
  const baseline = PAD.top + innerHeight

  const nearest = (clientX: number) => {
    const bounds = frame.current?.getBoundingClientRect()
    if (!bounds || values.length === 0 || step === 0) return null
    const index = Math.round(values.length - 1 - (PAD.left + innerWidth - (clientX - bounds.left)) / step)
    return Math.min(values.length - 1, Math.max(0, index))
  }
  const onMove = (event: PointerEvent) => setHover(nearest(event.clientX))
  const onKey = (event: KeyboardEvent) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
    event.preventDefault()
    const from = hover ?? lastIndex
    setHover(Math.min(lastIndex, Math.max(0, from + (event.key === 'ArrowLeft' ? -1 : 1))))
  }

  return (
    <div
      ref={frame}
      className="relative rounded-[6px] outline-offset-2"
      style={{ height }}
      role="img"
      tabIndex={known.length ? 0 : -1}
      aria-label={`${label}, last minute. ${last !== null ? `Now ${format(last)}.` : 'No readings yet.'}`}
      onPointerMove={onMove}
      onPointerLeave={() => setHover(null)}
      onFocus={() => setHover(lastIndex >= 0 ? lastIndex : null)}
      onBlur={() => setHover(null)}
      onKeyDown={onKey}
    >
      {width > 0 && (
        <svg width={width} height={height} className="block overflow-visible">
          <line x1={PAD.left} x2={PAD.left + innerWidth} y1={baseline} y2={baseline} stroke="var(--color-ink-100)" />
          {target !== undefined && (
            <>
              <line
                x1={PAD.left}
                x2={PAD.left + innerWidth}
                y1={y(target)}
                y2={y(target)}
                stroke="var(--color-ink-300)"
                strokeDasharray="3 3"
              />
              <text
                x={PAD.left + innerWidth + 5}
                y={y(target) + 3.5}
                fontSize="10"
                fontFamily="var(--font-mono)"
                fill="var(--color-ink-500)"
              >
                {format(target)}
              </text>
            </>
          )}
          {runs.map((path, index) => (
            <path
              key={index}
              d={path}
              fill="none"
              stroke="var(--color-ink-400)"
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {shown !== null && (
            <line x1={x(shown)} x2={x(shown)} y1={PAD.top - 3} y2={baseline} stroke="var(--color-ink-300)" />
          )}
          {last !== null && (
            <circle cx={x(lastIndex)} cy={y(last)} r="4" fill="var(--color-accent-600)" stroke="var(--color-surface)" strokeWidth="2" />
          )}
          {shown !== null && shown !== lastIndex && (
            <circle cx={x(shown)} cy={y(values[shown] as number)} r="3.5" fill="var(--color-ink-700)" stroke="var(--color-surface)" strokeWidth="2" />
          )}
        </svg>
      )}
      {shown !== null && (
        <div
          className="pointer-events-none absolute -top-7 z-10 -translate-x-1/2 rounded-[6px] bg-ink-950 px-2 py-1 text-[11.5px] whitespace-nowrap text-white shadow-raised"
          style={{ left: Math.min(Math.max(x(shown), 44), Math.max(44, width - 44)) }}
        >
          <span className="font-semibold tabular-nums">{format(values[shown] as number)}</span>
          <span className="ml-1.5 text-ink-300">
            {shown === lastIndex ? 'now' : `${Math.round((lastIndex - shown) * interval)} s ago`}
          </span>
        </div>
      )}
    </div>
  )
}
