import { Card, CardHeader } from '../../components/ui/Card'
import { StackedBar, type Segment } from '../../components/viz/StackedBar'
import type { StageKey } from '../../lib/api'
import { percent } from '../../lib/format'
import { useStudio } from '../../lib/studio'

// The four timed stages, in pipeline order. Each keeps its colour for good.
const STAGES: { key: StageKey; label: string; what: string; color: string }[] = [
  { key: 'tracker', label: 'Tracking', what: 'Find the face and its landmarks', color: 'var(--color-series-1)' },
  { key: 'motion', label: 'Motion', what: 'Read pose and expression from the face', color: 'var(--color-series-2)' },
  { key: 'render', label: 'Render', what: 'Warp and decode the enrolled picture', color: 'var(--color-series-3)' },
  { key: 'compose', label: 'Compose', what: 'Blend the face back into the frame', color: 'var(--color-series-4)' },
]
const WAITING_COLOR = 'var(--color-ink-300)' // not a stage, so it is grey, not a series colour

const ms = (value: number) => `${value.toFixed(value < 10 ? 1 : 0)} ms`

/** Where one frame's time goes, against the end-to-end budget. */
export function LatencyBreakdown() {
  const { status } = useStudio()
  const metrics = status?.session.metrics
  const budget = status?.targets.end_to_end_ms ?? 150
  const stages = metrics?.stages_ms
  const total = metrics?.end_to_end_ms ?? null

  const segments: Segment[] = []
  if (stages && total !== null) {
    const staged = STAGES.reduce((sum, stage) => sum + stages[stage.key], 0)
    segments.push({ key: 'waiting', label: 'Waiting', value: Math.max(0, total - staged), color: WAITING_COLOR })
    STAGES.forEach((stage) => segments.push({ key: stage.key, label: stage.label, value: stages[stage.key], color: stage.color }))
  }
  const sum = segments.reduce((all, segment) => all + segment.value, 0)
  const rows = [
    { key: 'waiting', label: 'Waiting', what: 'Until the pipeline is free, plus cropping', color: WAITING_COLOR },
    ...STAGES,
  ]

  return (
    <Card>
      <CardHeader title="Where the time goes" hint="One frame, from the camera to the finished output." />
      <div className="px-5 pb-5">
        <StackedBar
          segments={segments}
          scale={Math.max(budget, sum) * 1.06}
          marker={{ value: budget, label: `${budget} ms budget` }}
          format={ms}
          label="Time per frame by stage"
        />
        <ul className="mt-4 divide-y divide-line text-[13px]" aria-label="Time per frame by stage">
          {rows.map((row) => {
            const segment = segments.find((item) => item.key === row.key)
            return (
              <li key={row.key} className="grid grid-cols-[minmax(0,1fr)_auto_2.5rem] items-center gap-3 py-2">
                <span className="flex min-w-0 items-center gap-2.5">
                  <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: row.color }} />
                  <span className="shrink-0 font-medium text-ink-900">{row.label}</span>
                  <span className="truncate text-ink-500">{row.what}</span>
                </span>
                <span className="text-right font-medium whitespace-nowrap text-ink-900 tabular-nums">
                  {segment ? ms(segment.value) : <span className="font-normal text-ink-300">&mdash;</span>}
                </span>
                <span className="text-right text-ink-500 tabular-nums">{segment && sum > 0 ? percent(segment.value / sum) : ''}</span>
              </li>
            )
          })}
        </ul>
        <p className="mt-3 border-t border-line pt-3 text-[12.5px] leading-snug text-ink-500">
          {metrics?.dropped_share !== null && metrics?.dropped_share !== undefined ? (
            <>
              <span className="font-medium text-ink-800">{percent(metrics.dropped_share)}</span> of camera frames were skipped,
              so the output shows the newest frame and never falls behind.
            </>
          ) : (
            'Start a session to see where each frame spends its time.'
          )}
        </p>
      </div>
    </Card>
  )
}
