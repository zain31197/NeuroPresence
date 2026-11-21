import { CircleAlert, Info, ShieldCheck, ShieldX, TriangleAlert, X, type LucideIcon } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { EASE_OUT } from '../../components/motion'
import { Pill } from '../../components/ui/Pill'
import { TrendChart, type TrendPoint } from '../../components/viz/TrendChart'
import { api, type SessionDetail, type Targets } from '../../lib/api'
import { cn } from '../../lib/cn'
import { clock, fixed } from '../../lib/format'
import { clockTime, dayLabel, duration, KIND, median, OUTCOME, thin } from '../../lib/history'
import { useNow } from '../../lib/useNow'
import { KIND_ICON, lasted } from './SessionRow'

const LEVEL_ICON: Record<'info' | 'warning' | 'error', { icon: LucideIcon; tone: string }> = {
  info: { icon: Info, tone: 'bg-ink-100 text-ink-600' },
  warning: { icon: TriangleAlert, tone: 'bg-warning-wash text-warning-ink' },
  error: { icon: CircleAlert, tone: 'bg-critical-wash text-critical-ink' },
}

/**
 * One past session in full, in a panel that slides in from the right: how it ended, its typical figures, how
 * they moved over the session, its consent check and what happened in it. A running one is read again every
 * few seconds. Escape, the close button or a click beside the panel closes it.
 */
export function SessionDrawer({ id, targets, onClose }: { id: number | null; targets: Targets | null; onClose: () => void }) {
  return createPortal(
    <AnimatePresence>
      {id !== null && (
        <motion.div key="drawer" className="fixed inset-0 z-40" initial="closed" animate="open" exit="closed">
          <motion.div
            aria-hidden="true"
            className="absolute inset-0 bg-ink-950/25 backdrop-blur-[2px]"
            variants={{ closed: { opacity: 0 }, open: { opacity: 1 } }}
            transition={{ duration: 0.25 }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-modal="true"
            aria-labelledby="session-title"
            className="absolute inset-y-0 right-0 flex w-full max-w-[580px] flex-col border-l border-line bg-surface shadow-[0_0_60px_-12px_rgb(11_13_18/0.35)]"
            variants={{ closed: { x: '100%' }, open: { x: 0 } }}
            transition={{ type: 'spring', stiffness: 380, damping: 40 }}
          >
            <Detail id={id} targets={targets} onClose={onClose} />
          </motion.aside>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  )
}

function Detail({ id, targets, onClose }: { id: number; targets: Targets | null; onClose: () => void }) {
  const [detail, setDetail] = useState<SessionDetail | 'missing' | null>(null)
  const close = useRef<HTMLButtonElement>(null)
  const running = detail !== null && detail !== 'missing' && detail.outcome === 'running'
  const now = useNow(running)
  // The latest close, so the effect below runs once per opening and focus is not moved again on every render.
  const closing = useRef(onClose)
  useEffect(() => {
    closing.current = onClose
  })

  // Focus goes into the panel when it opens and back to where it was when it closes.
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null
    close.current?.focus()
    const escape = (event: KeyboardEvent) => event.key === 'Escape' && closing.current()
    window.addEventListener('keydown', escape)
    return () => {
      window.removeEventListener('keydown', escape)
      before?.focus?.()
    }
  }, [])

  useEffect(() => {
    let stopped = false
    const read = () =>
      api.historySession(id).then(
        (found) => !stopped && setDetail(found),
        () => !stopped && setDetail('missing'),
      )
    setDetail(null)
    read()
    return () => {
      stopped = true
    }
  }, [id])

  // While the session is still running, its figures keep arriving.
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => api.historySession(id).then(setDetail, () => undefined), 4000)
    return () => window.clearInterval(timer)
  }, [running, id])

  return (
    <>
      <header className="flex items-start gap-3 border-b border-line px-5 py-4 sm:px-6">
        <div className="min-w-0 flex-1">
          {detail && detail !== 'missing' ? <Heading session={detail} now={now} /> : <p id="session-title" className="text-[15px] font-semibold text-ink-950">Session</p>}
        </div>
        <button
          ref={close}
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="-mr-1.5 grid size-8 shrink-0 place-items-center rounded-control text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-950 focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:outline-none"
        >
          <X className="size-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 pt-5 pb-10 sm:px-6">
        {detail === null && <Loading />}
        {detail === 'missing' && <p className="text-[13.5px] text-ink-600">This session is no longer in the history.</p>}
        {detail && detail !== 'missing' && <Body session={detail} targets={targets} />}
      </div>
    </>
  )
}

function Heading({ session, now }: { session: SessionDetail; now: number }) {
  const Icon = KIND_ICON[session.kind]
  const outcome = OUTCOME[session.outcome]
  return (
    <div className="flex items-start gap-3">
      <span className="grid size-10 shrink-0 place-items-center rounded-[12px] bg-ink-100 text-ink-800">
        <Icon className="size-[18px]" strokeWidth={1.9} />
      </span>
      <div className="min-w-0">
        <p className="label-caps text-ink-400">{KIND[session.kind]}</p>
        <h2 id="session-title" className="display mt-0.5 truncate text-[18px] leading-tight font-semibold text-ink-950">
          {session.input_label}
        </h2>
        <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[12.5px] text-ink-500">
          <span>
            {dayLabel(session.started_at)}, {clockTime(session.started_at)}
          </span>
          <span aria-hidden="true">·</span>
          <span className="tabular-nums">{duration(lasted(session, now))}</span>
          <Pill tone={outcome.tone}>{outcome.label}</Pill>
        </div>
      </div>
    </div>
  )
}

function Body({ session, targets }: { session: SessionDetail; targets: Targets | null }) {
  const readings = thin(session.figures)
  const point = (value: (f: SessionDetail['figures'][number]) => number | null) =>
    readings.map((f): TrendPoint => ({ label: clock(f.at - session.started_at), date: '', value: value(f) }))
  const hasIdentity = readings.some((f) => f.identity !== null)
  // The engine works the typical figures out when a session ends; while it runs they come from its readings so far.
  const running = session.outcome === 'running'
  const typical = (kept: number | null, value: (f: SessionDetail['figures'][number]) => number | null) =>
    running ? median(session.figures.map(value)) : kept
  const ended = session.outcome !== 'stopped' && session.outcome !== 'running' && session.message

  return (
    <div className="grid gap-6">
      {ended && (
        <p
          className={cn(
            'rounded-panel border px-3.5 py-2.5 text-[13px] leading-snug',
            session.outcome === 'interrupted' ? 'border-line bg-ink-50 text-ink-700' : 'border-critical/20 bg-critical-wash text-critical-ink',
          )}
        >
          {session.outcome === 'interrupted' ? 'The engine stopped while this session ran, so it has no proper ending.' : session.message}
        </p>
      )}

      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-panel border border-line bg-line sm:grid-cols-4">
        <Typical label="Frame rate" value={fixed(typical(session.fps, (f) => f.fps))} unit="fps" target={targets && `target ${targets.fps}`} />
        <Typical label="Delay" value={fixed(typical(session.delay_ms, (f) => f.delay_ms), 0)} unit="ms" target={targets && `target ${targets.end_to_end_ms}`} />
        <Typical label="Drawing" value={fixed(typical(session.render_ms, (f) => f.render_ms), 0)} unit="ms" target={targets && `target ${targets.render_ms}`} />
        <Typical label="Identity" value={fixed(typical(session.identity, (f) => f.identity), 2)} unit="" target={targets && `target ${targets.csim}`} />
      </dl>
      <p className="-mt-4 text-[12px] text-ink-400">The middle value of the readings taken every two seconds while the picture was live.</p>

      <div className="flex flex-wrap gap-2">
        <Flag on={session.virtual_camera} yes="Sent to the meeting as a camera" no="Not sent to a meeting" />
        <Flag on={session.voice_in_step} yes="Voice held in step with the lips" no="Voice not sent" />
        {!running && (
          <span className="inline-flex h-7 items-center rounded-chip bg-ink-50 px-2.5 font-mono text-[11.5px] text-ink-600 tabular-nums">
            {session.frames.toLocaleString('en-GB')} frames
          </span>
        )}
      </div>

      {readings.length > 1 ? (
        <div className="grid gap-5">
          <Chart title="Frame rate" hint="Pictures a second in the output.">
            <TrendChart points={point((f) => f.fps)} target={targets?.fps} better="up" format={(v) => v.toFixed(1)} title="Frame rate over the session" height={150} xLabel={(p) => p.label} noun="readings" />
          </Chart>
          <Chart title="Delay" hint="From the camera to the output, in milliseconds.">
            <TrendChart points={point((f) => f.delay_ms)} target={targets?.end_to_end_ms} better="down" format={(v) => v.toFixed(0)} title="Delay over the session" height={150} xLabel={(p) => p.label} noun="readings" />
          </Chart>
          {hasIdentity && (
            <Chart title="Identity" hint="How alike the output's face is to your verified face.">
              <TrendChart points={point((f) => f.identity)} target={targets?.csim} better="up" format={(v) => v.toFixed(2)} title="Identity over the session" height={150} xLabel={(p) => p.label} noun="readings" />
            </Chart>
          )}
        </div>
      ) : (
        <p className="rounded-panel bg-ink-50 px-4 py-6 text-center text-[13px] text-ink-500">
          {session.outcome === 'running' ? 'The first figures arrive a few seconds after the picture goes live.' : 'This session never reached the live picture, so it has no figures to draw.'}
        </p>
      )}

      {session.checks.length > 0 && (
        <section>
          <h3 className="label-caps mb-2 text-ink-400">Consent check</h3>
          <ul className="grid gap-2">
            {session.checks.map((check) => (
              <li key={check.at} className="flex items-start gap-3 rounded-panel border border-line px-3.5 py-3">
                <span className={cn('grid size-7 shrink-0 place-items-center rounded-full', check.passed ? 'bg-good-wash text-good-ink' : 'bg-critical-wash text-critical-ink')}>
                  {check.passed ? <ShieldCheck className="size-4" /> : <ShieldX className="size-4" />}
                </span>
                <div className="min-w-0 text-[13px]">
                  <p className="font-semibold text-ink-950">{check.passed ? 'Passed' : 'Not passed'}</p>
                  <p className="mt-0.5 text-ink-600">
                    {check.passed
                      ? `A living person, the verified face${check.lowest_match !== null ? `: matched at ${check.lowest_match.toFixed(2)} or better` : ''}.`
                      : check.reason}
                  </p>
                </div>
                <span className="ml-auto font-mono text-[11.5px] text-ink-400 tabular-nums">{clockTime(check.at)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {session.events.length > 0 && (
        <section>
          <h3 className="label-caps mb-3 text-ink-400">What happened</h3>
          <ol className="relative grid gap-3 before:absolute before:top-2 before:bottom-2 before:left-[11px] before:w-px before:bg-line">
            {session.events.map((event, index) => {
              const level = LEVEL_ICON[event.level]
              const Icon = level.icon
              return (
                <motion.li
                  key={`${event.at}-${index}`}
                  className="relative flex items-start gap-3"
                  initial={{ opacity: 0, x: 8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, ease: EASE_OUT, delay: Math.min(index, 12) * 0.03 }}
                >
                  <span className={cn('relative grid size-[23px] shrink-0 place-items-center rounded-full ring-4 ring-surface', level.tone)}>
                    <Icon className="size-3" strokeWidth={2.4} />
                  </span>
                  <p className="min-w-0 flex-1 pt-0.5 text-[13px] leading-snug text-ink-800">{event.message}</p>
                  <span className="shrink-0 pt-0.5 font-mono text-[11.5px] text-ink-400 tabular-nums">+{clock(Math.max(0, event.at - session.started_at))}</span>
                </motion.li>
              )
            })}
          </ol>
        </section>
      )}
    </div>
  )
}

function Typical({ label, value, unit, target }: { label: string; value: string; unit: string; target: string | null }) {
  return (
    <div className="bg-surface p-3.5">
      <dt className="text-[12px] font-medium text-ink-500">{label}</dt>
      <dd className="mt-1 flex items-baseline gap-1">
        <span className="display text-[22px] leading-none font-semibold text-ink-950 tabular-nums">{value}</span>
        {unit && <span className="text-[12px] text-ink-400">{unit}</span>}
      </dd>
      {target && <dd className="mt-1 font-mono text-[10.5px] text-ink-400">{target}</dd>}
    </div>
  )
}

function Flag({ on, yes, no }: { on: boolean; yes: string; no: string }) {
  return (
    <span className={cn('inline-flex h-7 items-center gap-1.5 rounded-chip px-2.5 text-[12px] font-medium', on ? 'bg-accent-50 text-accent-700' : 'bg-ink-50 text-ink-500')}>
      <span className={cn('size-1.5 rounded-full', on ? 'bg-accent-500' : 'bg-ink-300')} aria-hidden="true" />
      {on ? yes : no}
    </span>
  )
}

function Chart({ title, hint, children }: { title: string; hint: string; children: ReactNode }) {
  return (
    <section>
      <div className="mb-1 flex items-baseline justify-between gap-3">
        <h3 className="text-[13.5px] font-semibold text-ink-950">{title}</h3>
        <p className="truncate text-[12px] text-ink-400">{hint}</p>
      </div>
      {children}
    </section>
  )
}

function Loading() {
  return (
    <div className="grid gap-4" aria-label="Loading the session">
      {[64, 120, 150, 150].map((height, index) => (
        <div key={index} className="animate-pulse rounded-panel bg-ink-100/70" style={{ height }} />
      ))}
    </div>
  )
}
