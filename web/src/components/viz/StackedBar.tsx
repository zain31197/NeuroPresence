import { Tooltip } from '../ui/Tooltip'

export interface Segment {
  key: string
  label: string
  value: number
  /** A CSS colour. Chart series colours carry identity; grey is for what is not a series. */
  color: string
}

interface Props {
  segments: Segment[]
  /** Value at the right edge of the track. */
  scale: number
  /** A threshold drawn across the bar, with its label above. */
  marker?: { value: number; label: string }
  format: (value: number) => string
  label: string
}

const GAP = 2 // surface-coloured gap between touching segments, in px

/** Parts of a whole on one horizontal track. Segments are separated by a gap, never by an outline. */
export function StackedBar({ segments, scale, marker, format, label }: Props) {
  const visible = segments.filter((segment) => segment.value > 0)
  const total = visible.reduce((sum, segment) => sum + segment.value, 0)
  const summary = visible.map((segment) => `${segment.label} ${format(segment.value)}`).join(', ')

  return (
    <div className="relative pt-6" role="img" aria-label={`${label}: ${summary || 'no readings yet'}.`}>
      {marker && (
        <div className="pointer-events-none absolute inset-y-0 z-10" style={{ left: `${(marker.value / scale) * 100}%` }}>
          <span className="label-caps absolute top-0 right-1.5 whitespace-nowrap text-ink-500">{marker.label}</span>
          <span className="absolute inset-y-0 left-0 border-l border-dashed border-ink-500" />
        </div>
      )}
      <div className="relative flex h-4 w-full overflow-hidden rounded-r-[4px] bg-ink-100">
        {visible.map((segment, index) => (
          <Tooltip
            key={segment.key}
            content={
              <span>
                <span className="font-semibold tabular-nums">{format(segment.value)}</span>
                <span className="ml-1.5 text-ink-300">
                  {segment.label} · {total ? Math.round((segment.value / total) * 100) : 0}%
                </span>
              </span>
            }
          >
            <div
              tabIndex={0}
              className="h-full transition-[width,filter] duration-500 ease-out hover:brightness-110 focus-visible:brightness-110"
              style={{
                width: `${(segment.value / scale) * 100}%`,
                background: segment.color,
                // The gap is drawn in the track's own colour so it reads as empty space.
                boxShadow: index < visible.length - 1 ? `${GAP}px 0 0 var(--color-surface)` : undefined,
                marginRight: index < visible.length - 1 ? GAP : 0,
                borderRadius: index === visible.length - 1 ? '0 4px 4px 0' : 0,
              }}
            />
          </Tooltip>
        ))}
      </div>
    </div>
  )
}
