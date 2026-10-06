import { ArrowRight } from 'lucide-react'
import type { Benchmark } from '../../lib/api'
import { FaceMeshCanvas } from './FaceMeshCanvas'

/** The idea in one picture: the face in front of the camera, and the enrolled picture moving with it. */
export function HeroVisual({ benchmark }: { benchmark: Benchmark | null }) {
  const summary = benchmark?.summary
  const readouts = summary
    ? [
        { value: summary.fps?.toFixed(1), unit: 'fps', label: 'frame rate' },
        { value: summary.pipeline_ms?.toFixed(0), unit: 'ms', label: 'per frame' },
        { value: summary.csim_self_reenactment?.toFixed(2), unit: 'CSIM', label: 'identity match' },
      ].filter((readout) => readout.value !== undefined)
    : []

  return (
    <figure className="relative rounded-[26px] border border-line bg-surface p-2.5 shadow-hero">
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2">
        <Panel label="Camera" tone="plain">
          <FaceMeshCanvas
            className="size-full"
            meshColor="rgb(104 112 131 / 0.42)"
            contourColor="rgb(20 23 30 / 0.9)"
            label="A face drawn from tracked landmarks, turning and talking."
          />
        </Panel>

        <div className="flex flex-col items-center gap-2 px-0.5" aria-hidden="true">
          <span className="grid size-8 place-items-center rounded-full border border-line bg-surface text-ink-700 shadow-control">
            <ArrowRight className="size-4" />
          </span>
        </div>

        <Panel label="Output" tone="accent" live>
          <FaceMeshCanvas
            className="size-full"
            meshColor="rgb(80 98 240 / 0.45)"
            contourColor="rgb(47 59 189 / 0.95)"
            delay={0.13}
            label="The enrolled picture's face following the same movement a moment later."
          />
        </Panel>
      </div>

      <figcaption className="px-3 pt-3.5 pb-1.5">
        <p className="label-caps text-ink-500">Illustration: pose, lips, expression</p>
        {readouts.length > 0 && <p className="label-caps mt-3 border-t border-line pt-3 text-ink-400">Latest benchmark on this computer</p>}
        {readouts.length > 0 && (
          <dl className="mt-2.5 grid grid-cols-3 divide-x divide-line">
            {readouts.map((readout) => (
              <div key={readout.label} className="px-3 first:pl-0 last:pr-0">
                <dd className="flex items-baseline gap-1">
                  <span className="display text-[20px] leading-none font-semibold text-ink-950 tabular-nums">{readout.value}</span>
                  <span className="text-[12px] font-medium text-ink-500">{readout.unit}</span>
                </dd>
                <dt className="mt-1 text-[12px] text-ink-500">{readout.label}</dt>
              </div>
            ))}
          </dl>
        )}
      </figcaption>
    </figure>
  )
}

function Panel({ label, tone, live, children }: { label: string; tone: 'plain' | 'accent'; live?: boolean; children: React.ReactNode }) {
  return (
    <div
      className={`relative aspect-[4/5] overflow-hidden rounded-[18px] ${
        tone === 'accent' ? 'bg-accent-50' : 'bg-ink-50'
      }`}
    >
      {children}
      <span className="label-caps absolute top-3 left-3 rounded-chip bg-surface px-2 py-1 text-ink-700 shadow-control">{label}</span>
      {live && (
        <span className="absolute top-3 right-3 inline-flex h-[22px] items-center gap-1.5 rounded-full bg-surface px-2 text-[11px] font-semibold text-ink-950 shadow-control">
          <span className="size-1.5 animate-lamp rounded-full bg-live" />
          Live
        </span>
      )}
    </div>
  )
}
