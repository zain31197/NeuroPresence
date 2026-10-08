import { Info } from 'lucide-react'
import type { ReactNode } from 'react'
import { Card } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { Tooltip } from '../../components/ui/Tooltip'
import { Meter } from '../../components/viz/Meter'
import { Sparkline } from '../../components/viz/Sparkline'
import type { Benchmark } from '../../lib/api'
import { verdict, type Verdict } from '../../lib/format'
import { useStudio } from '../../lib/studio'

const HISTORY_CAPACITY = 240
const STATUS_INTERVAL_S = 0.25

interface FigureProps {
  label: string
  /** What the figure means and how it is measured. */
  about: string
  value: number | null | undefined
  digits: number
  unit: string
  result: Verdict
  /** The target as a bound, for example "≥ 24". */
  target: string
  /** The same figure in the latest saved benchmark, if there is one. */
  benchmark?: number | null
  /** Shown in place of the target when the figure cannot be measured. */
  unavailable?: string | null
  children: ReactNode
}

function Figure({ label, about, value, digits, unit, result, target, benchmark, unavailable, children }: FigureProps) {
  const known = value !== null && value !== undefined
  const hasBenchmark = benchmark !== null && benchmark !== undefined
  return (
    <div className="flex min-w-0 flex-col bg-surface px-5 py-4 last:@lg:col-span-2 last:@4xl:col-span-1">
      <div className="flex items-center gap-1.5 text-[12.5px] font-medium text-ink-600">
        <span className="truncate">{label}</span>
        <Tooltip content={about}>
          <button type="button" aria-label={`About ${label.toLowerCase()}`} className="shrink-0 rounded-full text-ink-300 hover:text-ink-600">
            <Info className="size-3.5" />
          </button>
        </Tooltip>
      </div>
      <p className="mt-2 flex items-baseline gap-1.5">
        {known ? (
          <span className="display text-[30px] leading-none font-semibold text-ink-950 tabular-nums">{value.toFixed(digits)}</span>
        ) : (
          <span className="text-[30px] leading-none font-light text-ink-300" aria-label="No reading yet">
            &mdash;
          </span>
        )}
        <span className="text-[12.5px] font-medium text-ink-500">{unit}</span>
      </p>
      <div className="mt-2.5 flex h-[22px] items-center">
        {result === 'good' && <Pill tone="good">On target</Pill>}
        {result === 'off' && <Pill tone="serious">Off target</Pill>}
      </div>
      {/* Two fixed lines, so every figure's trend line starts at the same height. */}
      <p className="mt-2 h-[36px] text-[12px] leading-[18px] text-ink-500">
        {unavailable ? (
          <span className="line-clamp-2" title={unavailable}>
            {unavailable}
          </span>
        ) : (
          <>
            <span className="block truncate">Target {target}</span>
            {hasBenchmark && <span className="block truncate">Benchmark {benchmark.toFixed(digits)}</span>}
          </>
        )}
      </p>
      <div className="mt-2">{children}</div>
    </div>
  )
}

/** The five figures the project is judged on, live, each against its target. */
export function MetricTiles({ benchmark }: { benchmark: Benchmark | null }) {
  const { status, history } = useStudio()
  if (!status) return null

  const { targets, identity, gpu } = status
  const metrics = status.session.metrics
  const render = metrics?.render_ms
  const saved = benchmark?.summary
  const series = (pick: (sample: (typeof history)[number]) => number | null) => history.map(pick)
  const trend = { capacity: HISTORY_CAPACITY, interval: STATUS_INTERVAL_S }

  return (
    <Card className="@container overflow-hidden">
      {/* Hairlines between the figures are the gaps of this grid showing the line colour. */}
      <div className="grid grid-cols-1 gap-px bg-line @lg:grid-cols-2 @4xl:grid-cols-5">
      <Figure
        label="Frame rate"
        about="Output frames finished per second, averaged over the last two seconds."
        value={metrics?.fps}
        digits={1}
        unit="fps"
        result={verdict(metrics?.fps, targets.fps, true)}
        target={`≥ ${targets.fps}`}
        benchmark={saved?.fps}
      >
        <Sparkline values={series((s) => s.fps)} {...trend} target={targets.fps} format={(v) => v.toFixed(0)} label="Frame rate" />
      </Figure>

      <Figure
        label="End-to-end latency"
        about="Time from a frame arriving from the camera to its output being ready. It includes the wait for the pipeline to be free."
        value={metrics?.end_to_end_ms}
        digits={0}
        unit="ms"
        result={verdict(metrics?.end_to_end_ms, targets.end_to_end_ms, false)}
        target={`≤ ${targets.end_to_end_ms}`}
      >
        <Sparkline values={series((s) => s.endToEnd)} {...trend} target={targets.end_to_end_ms} format={(v) => v.toFixed(0)} label="End-to-end latency" />
      </Figure>

      <Figure
        label="Render time"
        about="GPU time the reenactment stage takes for one frame. It sets the ceiling on the frame rate, so it is the main thing to optimize."
        value={render}
        digits={0}
        unit="ms"
        result={verdict(render, targets.render_ms, false)}
        target={`≤ ${targets.render_ms}`}
        benchmark={saved?.render_ms}
      >
        <Sparkline values={series((s) => s.render)} {...trend} target={targets.render_ms} format={(v) => v.toFixed(0)} label="Render time" />
      </Figure>

      <Figure
        label="Identity match"
        about="How closely the output face matches the enrolled picture (cosine similarity of ArcFace embeddings), checked once a second. A face that never moved would score 1.0."
        value={identity.csim}
        digits={2}
        unit="CSIM"
        result={verdict(identity.csim, targets.csim, true)}
        target={`≥ ${targets.csim.toFixed(2)}`}
        benchmark={saved?.csim_self_reenactment}
        unavailable={identity.available ? null : identity.reason}
      >
        <Sparkline values={series((s) => s.csim)} {...trend} target={targets.csim} format={(v) => v.toFixed(2)} label="Identity match" />
      </Figure>

      <Figure
        label="GPU memory"
        about="The most GPU memory the engine has held: the drop in the card's free memory since the models began loading, as the driver reports it. Other programs on the GPU can shift it."
        value={gpu?.peak_gb}
        digits={2}
        unit="GB"
        result={verdict(gpu?.peak_gb, targets.vram_gb, false)}
        target={`≤ ${targets.vram_gb}`}
        benchmark={saved?.peak_vram_gb}
        unavailable={gpu ? null : 'Measured once the models are loaded.'}
      >
        <div className="flex h-[46px] flex-col justify-center gap-2">
          <Meter
            value={gpu?.peak_gb ?? null}
            limit={targets.vram_gb}
            label="GPU memory used of the limit"
            valueText={gpu ? `${gpu.peak_gb} of ${targets.vram_gb} gigabytes` : 'not measured yet'}
          />
          <p className="truncate text-[11.5px] text-ink-500">{gpu ? `${gpu.name.replace('NVIDIA GeForce ', '')} · ${gpu.total_gb} GB card` : 'No GPU reading yet'}</p>
        </div>
      </Figure>
      </div>
    </Card>
  )
}
