import { ArrowDownRight, ArrowUpRight, Check, Minus, RotateCcw, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { CountUp, Reveal, Stagger, StaggerItem } from '../../components/motion'
import { PageHeader } from '../../components/PageHeader'
import { Button } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { TrendChart } from '../../components/viz/TrendChart'
import { api, ApiError, type Benchmarks as Data, type BenchmarkRun, type CallFigureKey, type CallPhase, type LiveFigures } from '../../lib/api'
import { cn } from '../../lib/cn'
import { longDate } from '../../lib/format'

/** The figures followed across runs, each with its target and which way is better. */
const FIGURES = [
  { key: 'fps', name: 'Frame rate', unit: 'fps', target: 24, better: 'up', format: (v: number) => v.toFixed(1) },
  { key: 'render_ms', name: 'Render time', unit: 'ms', target: 42, better: 'down', format: (v: number) => v.toFixed(0) },
  { key: 'csim_self_reenactment', name: 'Identity match', unit: 'CSIM', target: 0.8, better: 'up', format: (v: number) => v.toFixed(3) },
  // Three places: the latest run is 1.005, just over the target, and two places would show it as 1.00.
  { key: 'jitter_ratio', name: 'Head jitter', unit: '× real video', target: 1, better: 'down', format: (v: number) => v.toFixed(3) },
] as const

const value = (run: BenchmarkRun, key: (typeof FIGURES)[number]['key']) => {
  const v = run.summary[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export function Benchmarks() {
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [live, setLive] = useState<LiveFigures | null>(null)
  const load = () => {
    setError(null)
    api.benchmarks().then(setData, (e) => setError(e instanceof ApiError ? e.message : 'The engine did not answer.'))
  }
  useEffect(() => {
    document.title = 'Benchmarks · NeuroPresence'
    load()
    api.capabilities().then((reply) => setLive(reply.live ?? null), () => setLive(null))
  }, [])

  // The trend uses the runs on the same eight sample clips, so they can be compared; other runs are in the table.
  const runs = data?.runs ?? []
  const comparable = runs.filter((run) => run.clips === 8)
  const first = comparable[0]
  const latest = comparable[comparable.length - 1]

  return (
    <main className="mx-auto max-w-[1520px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <PageHeader
        title="Benchmarks"
        badge={data && <Pill tone="neutral" icon={null}>{runs.length} runs</Pill>}
        actions={
          <Button icon={<RotateCcw className="size-3.5" />} onClick={load}>
            Reload
          </Button>
        }
      >
        Every benchmark saved on this computer, and the engine measured during real calls. Nothing here is typed in by hand: it is
        read from the results folder, as scripts/benchmark.py and scripts/measure_call.py wrote it. A benchmark tracks and draws one
        after the other on recorded clips, so it runs slower than a live session, which does both at once.
      </PageHeader>

      {error && (
        <Card className="mt-6 p-6">
          <p className="text-[14px] font-medium text-ink-950">The results could not be read.</p>
          <p className="mt-1 text-[13.5px] text-ink-600">{error}</p>
        </Card>
      )}

      {!data && !error && <Skeleton />}

      {data && runs.length === 0 && (
        <Card className="mt-6 p-8">
          <p className="text-[15px] font-medium text-ink-950">No benchmark has been run on this computer yet.</p>
          <p className="mt-1.5 text-[13.5px] text-ink-600">
            Run <code className="rounded-[5px] bg-ink-100 px-1.5 py-0.5 font-mono text-[12.5px]">python scripts/benchmark.py</code> and reload.
          </p>
        </Card>
      )}

      {latest && first && (
        <>
          <Stagger className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {FIGURES.map((figure) => (
              <StaggerItem key={figure.key}>
                <Headline
                  figure={figure}
                  now={value(latest, figure.key)}
                  then={value(first, figure.key)}
                  since={first.date}
                  live={figure.key === 'fps' && live ? `${live.fps[0]} to ${live.fps[1]} fps in a live session, which tracks while it draws` : null}
                />
              </StaggerItem>
            ))}
          </Stagger>

          <Reveal className="mt-8">
            <h2 className="text-[15px] font-semibold text-ink-950">How the figures moved</h2>
            <p className="mt-0.5 text-[13px] text-ink-500">
              {comparable.length} runs on the same eight sample clips, from {longDate(first.date)} to {longDate(latest.date)}. Each run is
              named after the change it was made to judge. Point at a run, or use the arrow keys, to see it.
            </p>
          </Reveal>
          <Stagger className="mt-4 grid gap-4 lg:grid-cols-2">
            {FIGURES.map((figure) => (
              <StaggerItem key={figure.key}>
                <Card className="h-full">
                  <CardHeader title={`${figure.name}, ${figure.unit}`} hint={figure.better === 'up' ? 'Higher is better' : 'Lower is better'} />
                  <div className="px-4 pb-4">
                    <TrendChart
                      title={`${figure.name} per benchmark run`}
                      points={comparable.map((run) => ({ label: run.label, date: run.date, value: value(run, figure.key) }))}
                      target={figure.target}
                      better={figure.better}
                      format={figure.format}
                    />
                  </div>
                </Card>
              </StaggerItem>
            ))}
          </Stagger>
        </>
      )}

      {data?.calls && <Calls phases={data.calls.phases} date={data.calls.date} gpu={data.calls.machine.gpu} seconds={data.calls.seconds_per_phase} />}

      {runs.length > 0 && <RunTable runs={runs} />}
    </main>
  )
}

function Headline({
  figure,
  now,
  then,
  since,
  live,
}: {
  figure: (typeof FIGURES)[number]
  now: number | null
  then: number | null
  since: string
  /** For the frame rate: what a live session reaches, beside the benchmark's figure. */
  live: string | null
}) {
  const met = now !== null && (figure.better === 'up' ? now >= figure.target : now <= figure.target)
  const change = now !== null && then !== null && then !== 0 ? (now - then) / Math.abs(then) : null
  const improved = change !== null && (figure.better === 'up' ? change > 0.005 : change < -0.005)
  const worse = change !== null && (figure.better === 'up' ? change < -0.005 : change > 0.005)
  return (
    <Card className="h-full p-5">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[13px] font-medium text-ink-600">{figure.name}</p>
        {now !== null && <Pill tone={met ? 'good' : 'serious'}>{met ? 'On target' : 'Off target'}</Pill>}
      </div>
      <p className="mt-3 flex items-baseline gap-1.5">
        {now !== null ? (
          <CountUp value={now} format={figure.format} className="display text-[34px] leading-none font-semibold text-ink-950" />
        ) : (
          <span className="display text-[34px] leading-none font-semibold text-ink-300">—</span>
        )}
        <span className="text-[13px] font-medium text-ink-500">{figure.unit}</span>
      </p>
      <p className="mt-3 flex flex-wrap items-center gap-x-2 gap-y-1 text-[12.5px] text-ink-500">
        {change !== null && (
          <span
            className={cn(
              'inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 font-medium tabular-nums',
              improved ? 'bg-good-wash text-good-ink' : worse ? 'bg-serious-wash text-serious-ink' : 'bg-ink-100 text-ink-600',
            )}
          >
            {improved || worse ? (change > 0 ? <ArrowUpRight className="size-3" /> : <ArrowDownRight className="size-3" />) : <Minus className="size-3" />}
            {Math.abs(change * 100).toFixed(0)}%
          </span>
        )}
        {then !== null && (
          <span>
            from {figure.format(then)} on {longDate(since)}
          </span>
        )}
      </p>
      <p className="mt-1 font-mono text-[11.5px] text-ink-400">
        target {figure.better === 'up' ? '≥' : '≤'} {figure.format(figure.target)}
      </p>
      {live && (
        <p className="mt-3 flex items-center gap-2 rounded-[9px] bg-accent-50 px-2.5 py-1.5 text-[12px] font-medium text-accent-700">
          <span className="size-1.5 shrink-0 animate-lamp rounded-full bg-live" />
          {live}
        </p>
      )}
    </Card>
  )
}

const CALL_ROWS: { key: CallFigureKey; name: string; format: (v: number) => string }[] = [
  { key: 'fps', name: 'Frame rate', format: (v) => `${v.toFixed(1)} fps` },
  { key: 'delay_ms', name: 'Camera to picture', format: (v) => `${v.toFixed(0)} ms` },
  { key: 'render_ms', name: 'Render time', format: (v) => `${v.toFixed(1)} ms` },
  { key: 'identity', name: 'Identity match', format: (v) => v.toFixed(2) },
  { key: 'gpu_util_pct', name: 'Graphics card busy', format: (v) => `${v.toFixed(0)}%` },
  { key: 'cpu_engine_pct', name: 'Engine CPU', format: (v) => `${v.toFixed(0)}%` },
]

const ANSWERS: Record<string, string> = {
  picture_seen: 'Picture seen',
  mark_readable: 'Mark readable',
  voice_heard: 'Voice heard',
  in_step: 'Lips and voice in step',
}

function Calls({ phases, date, gpu, seconds }: { phases: CallPhase[]; date: string; gpu?: string; seconds: number | null }) {
  return (
    <>
      <Reveal className="mt-10">
        <h2 className="text-[15px] font-semibold text-ink-950">During real calls</h2>
        <p className="mt-0.5 text-[13px] text-ink-500">
          A camera session measured once a second{seconds ? ` for ${seconds / 60} minutes a phase` : ''}, with no meeting and then beside
          each meeting application{gpu ? `, on ${gpu}` : ''}, {longDate(date)}. Medians, with the range of nine readings in ten under them.
        </p>
      </Reveal>
      <Stagger className="mt-4 grid gap-4 md:grid-cols-3">
        {phases.map((phase) => (
          <StaggerItem key={phase.key}>
            <Card className="h-full overflow-hidden">
              <div className="flex items-center justify-between border-b border-line bg-ink-50/60 px-5 py-3">
                <h3 className="text-[14px] font-semibold text-ink-950">{phase.title}</h3>
                {phase.figures.fps && (
                  <span className="display text-[20px] font-semibold text-ink-950">
                    <CountUp value={phase.figures.fps.median} format={(v) => v.toFixed(1)} />
                    <span className="ml-1 text-[12px] font-medium text-ink-500">fps</span>
                  </span>
                )}
              </div>
              <dl className="divide-y divide-line px-5">
                {CALL_ROWS.filter((row) => phase.figures[row.key]).map((row) => {
                  const figure = phase.figures[row.key]!
                  return (
                    <div key={row.key} className="flex items-baseline justify-between gap-3 py-2.5">
                      <dt className="text-[13px] text-ink-600">{row.name}</dt>
                      <dd className="text-right">
                        <span className="text-[14px] font-semibold text-ink-950 tabular-nums">{row.format(figure.median)}</span>
                        <span className="block font-mono text-[10.5px] text-ink-400 tabular-nums">
                          {row.format(figure.p5)} to {row.format(figure.p95)}
                        </span>
                      </dd>
                    </div>
                  )
                })}
              </dl>
              {phase.answers && (
                <ul className="flex flex-wrap gap-1.5 border-t border-line px-5 py-3">
                  {Object.entries(phase.answers).map(([key, yes]) => (
                    <li key={key}>
                      <Pill tone={yes ? 'good' : 'critical'} icon={yes ? <Check className="size-3" strokeWidth={3} /> : <X className="size-3" strokeWidth={3} />}>
                        {ANSWERS[key] ?? key}
                      </Pill>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </StaggerItem>
        ))}
      </Stagger>
    </>
  )
}

function RunTable({ runs }: { runs: BenchmarkRun[] }) {
  const ordered = [...runs].reverse()
  const cell = (v: number | null | undefined, format: (v: number) => string) => (typeof v === 'number' ? format(v) : '—')
  return (
    <Reveal className="mt-10">
      <h2 className="text-[15px] font-semibold text-ink-950">Every run</h2>
      <p className="mt-0.5 text-[13px] text-ink-500">Newest first. A run on other clips than the eight samples is marked, since its figures are not comparable.</p>
      <Card className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[760px] text-left text-[13px]">
          <thead className="border-b border-line bg-ink-50/60">
            <tr className="label-caps text-ink-500">
              <th className="px-5 py-3 font-medium">Run</th>
              <th className="px-3 py-3 font-medium">Date</th>
              <th className="px-3 py-3 text-right font-medium">fps</th>
              <th className="px-3 py-3 text-right font-medium">Render ms</th>
              <th className="px-3 py-3 text-right font-medium">Identity</th>
              <th className="px-3 py-3 text-right font-medium">Flicker</th>
              <th className="px-3 py-3 text-right font-medium">Jitter</th>
              <th className="px-5 py-3 font-medium">Code</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {ordered.map((run) => (
              <tr key={run.file} className="transition-colors hover:bg-ink-50/60">
                <td className="px-5 py-3">
                  <span className="font-medium text-ink-950">{run.label}</span>
                  {run.clips !== 8 && (
                    <Pill tone="accent" icon={null} className="ml-2">
                      {run.clips ?? '?'} clips
                    </Pill>
                  )}
                </td>
                <td className="px-3 py-3 whitespace-nowrap text-ink-600">{longDate(run.date)}</td>
                <td className="px-3 py-3 text-right text-ink-950 tabular-nums">{cell(run.summary.fps, (v) => v.toFixed(1))}</td>
                <td className="px-3 py-3 text-right text-ink-950 tabular-nums">{cell(run.summary.render_ms, (v) => v.toFixed(1))}</td>
                <td className="px-3 py-3 text-right text-ink-950 tabular-nums">{cell(run.summary.csim_self_reenactment, (v) => v.toFixed(3))}</td>
                <td className="px-3 py-3 text-right text-ink-950 tabular-nums">{cell(run.summary.warping_error_ratio, (v) => v.toFixed(2))}</td>
                <td className="px-3 py-3 text-right text-ink-950 tabular-nums">{cell(run.summary.jitter_ratio, (v) => v.toFixed(2))}</td>
                <td className="px-5 py-3 font-mono text-[11.5px] text-ink-400">{run.code?.neuropresence?.split('+')[0] ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </Reveal>
  )
}

function Skeleton() {
  return (
    <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-hidden="true">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="h-[150px] rounded-card border border-line bg-surface p-5 shadow-card">
          <div className="skeleton h-3 w-24" />
          <div className="skeleton mt-5 h-8 w-28" />
          <div className="skeleton mt-4 h-3 w-40" />
        </div>
      ))}
    </div>
  )
}
