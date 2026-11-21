import { ArrowRight, Check, ImagePlus, OctagonAlert, Play, ShieldCheck, ShieldX, TriangleAlert } from 'lucide-react'
import { motion } from 'motion/react'
import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { GlowBorder, LiveDot } from '../../components/fx'
import { CountUp, EASE_OUT } from '../../components/motion'
import { Button, buttonClass } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, enrolledPictureUrl, type History, type Security, type SystemReport } from '../../lib/api'
import { cn } from '../../lib/cn'
import { clock, dash } from '../../lib/format'
import { clockTime, dayLabel, duration, median, securityLine } from '../../lib/history'
import { usePreferredInput } from '../../lib/preferences'
import { useStudio } from '../../lib/studio'
import { useNow } from '../../lib/useNow'
import { stepsDone } from '../enrolment/Progress'
import { SessionDrawer } from '../sessions/SessionDrawer'
import { SessionRow } from '../sessions/SessionRow'
import { InputSelect } from '../studio/InputSelect'

const STEPS = ['Face', 'Poses', 'Picture', 'Live'] as const
const WEEK_SECONDS = 7 * 24 * 3600

// The readiness check takes a few seconds; one answer serves every visit for five minutes.
let readiness: { report: SystemReport; at: number } | null = null
const READINESS_FOR_MS = 5 * 60 * 1000

function greeting(hour: number): string {
  return hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
}

/** The studio's home: whether everything is ready for a call, a way to go live, and what the sessions have done. */
export function Overview() {
  const { status } = useStudio()
  const [history, setHistory] = useState<History | null>(null)
  const [security, setSecurity] = useState<Security | null>(null)
  const [open, setOpen] = useState<number | null>(null)
  const sessionState = status?.session.state

  const load = useCallback(() => {
    api.history(20).then(setHistory, () => undefined)
    api.security().then(setSecurity, () => undefined)
  }, [])

  useEffect(() => {
    document.title = 'Overview · NeuroPresence'
  }, [])
  useEffect(() => {
    load()
  }, [load, sessionState])

  const sessions = history?.sessions ?? []
  const now = useNow(sessions.some((session) => session.outcome === 'running'))
  const closeDrawer = useCallback(() => setOpen(null), [])

  return (
    <main className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <div>
        <p className="text-[13px] font-medium text-ink-500">{greeting(new Date(now).getHours())}</p>
        <h1 className="display mt-0.5 text-[26px] leading-tight font-semibold">Overview</h1>
        <p className="mt-1 max-w-[90ch] text-[13.5px] text-ink-500">
          Where everything stands before a call: your enrolment, this computer, and what your sessions have done.
        </p>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <GoLive className="lg:col-span-2" />
        <Readiness />
      </div>

      <Figures history={history} security={security} />

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="overflow-hidden lg:col-span-2">
          <CardHeader title="Recent sessions" hint="The last five. Open one for its figures over time." action={<SeeAll to="/studio/sessions" />} />
          {history && sessions.length === 0 ? (
            <p className="mx-5 mb-5 rounded-panel bg-ink-50 px-4 py-8 text-center text-[13px] text-ink-500">Your sessions will be listed here.</p>
          ) : (
            <div className="divide-y divide-line border-t border-line">
              {sessions.slice(0, 5).map((session) => (
                <SessionRow key={session.id} session={session} now={now} onOpen={() => setOpen(session.id)} />
              ))}
              {!history && <div className="h-[200px] animate-pulse bg-ink-50/60" />}
            </div>
          )}
        </Card>
        <SecuritySnippet security={security} />
      </div>

      <SessionDrawer id={open} targets={status?.targets ?? null} onClose={closeDrawer} />
    </main>
  )
}

function SeeAll({ to }: { to: string }) {
  return (
    <Link to={to} className="group inline-flex shrink-0 items-center gap-1 text-[12.5px] font-semibold text-accent-700">
      See all
      <ArrowRight className="size-3.5 transition-transform duration-200 group-hover:translate-x-0.5" />
    </Link>
  )
}

/** The next session: what is missing before one can start, or the input to start it with and one button. */
function GoLive({ className }: { className?: string }) {
  const { status, connection, refresh } = useStudio()
  const toast = useToast()
  const navigate = useNavigate()
  const inputs = status?.inputs ?? []
  const [input, setInput] = usePreferredInput(inputs)
  const [busy, setBusy] = useState(false)

  const face = status?.enrolment.face ?? null
  const record = status?.enrolment.record ?? null
  const done = stepsDone(face, record)
  const session = status?.session
  const live = session?.state === 'running' || session?.state === 'starting'
  // A recorded clip brings its own picture, so it can run before enrolment is finished.
  const needsPicture = !input.startsWith('sample') && record === null

  const start = async () => {
    setBusy(true)
    try {
      await api.startSession(input)
      await refresh().catch(() => undefined)
      navigate('/studio/live')
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The session could not start. Try again.')
    } finally {
      setBusy(false)
    }
  }

  const title = live ? 'A session is live' : done >= 3 ? 'Ready to go live' : 'Finish enrolment to go live'
  const note = live
    ? `${session?.state === 'starting' ? 'Starting' : 'Running'} on ${session?.input ?? 'the camera'}.`
    : done >= 3
      ? 'A camera session starts with two quick actions on the camera picture, to check that it is you.'
      : 'A camera session animates your enrolled picture, so the face, its poses and the picture come first.'

  const body = (
    <div className="relative flex flex-col gap-5 p-5 sm:flex-row sm:items-center sm:p-6">
      <div className="relative mx-auto size-[112px] shrink-0 sm:mx-0">
        {record ? (
          <img src={enrolledPictureUrl(record.id)} alt="Your enrolled picture" className="size-full rounded-[18px] bg-ink-100 object-cover shadow-card" />
        ) : (
          <div className="grid size-full place-items-center rounded-[18px] border border-dashed border-line-strong bg-ink-50 text-ink-400">
            <ImagePlus className="size-7" strokeWidth={1.6} />
          </div>
        )}
        {live && (
          <span className="absolute -right-1.5 -bottom-1.5 inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2 py-0.5 text-[11px] font-semibold text-ink-900 shadow-control">
            <LiveDot on />
            Live
          </span>
        )}
      </div>

      <div className="min-w-0 flex-1">
        <p className="label-caps text-ink-400">Next session</p>
        <p className="display mt-1 text-[22px] leading-tight font-semibold text-ink-950">{title}</p>
        <p className="mt-1 text-[13.5px] text-ink-600">{note}</p>

        <ol className="mt-4 flex flex-wrap items-center gap-x-1.5 gap-y-2" aria-label="Enrolment steps">
          {STEPS.map((step, index) => {
            const state = index < done || live ? 'done' : index === done ? 'current' : 'todo'
            return (
              <li key={step} className="flex items-center gap-1.5">
                <span
                  className={cn(
                    'inline-flex h-6 items-center gap-1 rounded-full px-2 text-[11.5px] font-semibold transition-colors',
                    state === 'done' && 'bg-accent-50 text-accent-700',
                    state === 'current' && 'bg-ink-950 text-white',
                    state === 'todo' && 'bg-ink-100 text-ink-400',
                  )}
                >
                  {state === 'done' && <Check className="size-3" strokeWidth={3} />}
                  {step}
                </span>
                {index < STEPS.length - 1 && <span className="h-px w-3 bg-line-strong" aria-hidden="true" />}
              </li>
            )
          })}
        </ol>

        <div className="mt-5 flex flex-wrap items-center gap-2">
          {live ? (
            <>
              <Link to="/studio/live" className={buttonClass('primary')}>
                Open Live Studio
                <ArrowRight className="size-3.5" />
              </Link>
              {session?.uptime_s != null && <span className="font-mono text-[12.5px] text-ink-500 tabular-nums">{clock(session.uptime_s)} live</span>}
            </>
          ) : done < 3 ? (
            <>
              <Link to="/studio/enrolment" className={buttonClass('primary')}>
                {done === 0 ? 'Start enrolment' : 'Continue enrolment'}
                <ArrowRight className="size-3.5" />
              </Link>
              <Link to="/studio/live" className={buttonClass('ghost')}>
                Try a recorded clip
              </Link>
            </>
          ) : (
            <>
              <InputSelect value={input} onValueChange={setInput} inputs={inputs} />
              <Button variant="primary" icon={<Play className="size-3.5 fill-current" />} busy={busy} disabled={connection !== 'open' || needsPicture} onClick={start}>
                Go live
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  )

  // Ready, or live: the card's edge carries a slow band of light. Otherwise it stays quiet.
  return done >= 3 || live ? (
    <GlowBorder className={cn('shadow-card', className)} radius={18}>
      <div className="h-full rounded-[18px] bg-surface">{body}</div>
    </GlowBorder>
  ) : (
    <Card className={className}>{body}</Card>
  )
}

/** Is this computer ready? The System screen's checks, without the internet ones, in one line and a ring. */
function Readiness() {
  const [report, setReport] = useState<SystemReport | null>(readiness?.report ?? null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (readiness && Date.now() - readiness.at < READINESS_FOR_MS) return
    api.system(false).then(
      (answer) => {
        readiness = { report: answer, at: Date.now() }
        setReport(answer)
      },
      () => setFailed(true),
    )
  }, [])

  const level = report ? (report.failed ? 'fail' : report.warned ? 'warn' : 'ok') : null
  const findings = report ? report.groups.flatMap((group) => group.checks).filter((check) => check.level !== 'ok') : []
  const ring = { ok: 'var(--color-good)', warn: 'var(--color-warning)', fail: 'var(--color-critical)' }
  const Icon = level === 'fail' ? OctagonAlert : level === 'warn' ? TriangleAlert : Check

  return (
    <Card className="flex flex-col p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[13.5px] font-semibold text-ink-950">This computer</p>
        <SeeAll to="/studio/system" />
      </div>
      <div className="mt-4 flex items-center gap-4">
        <div className="relative grid size-[60px] shrink-0 place-items-center">
          <svg viewBox="0 0 60 60" className="absolute inset-0 size-full -rotate-90" aria-hidden="true">
            <circle cx="30" cy="30" r="26" fill="none" stroke="var(--color-ink-100)" strokeWidth="5" />
            {level ? (
              <motion.circle cx="30" cy="30" r="26" fill="none" stroke={ring[level]} strokeWidth="5" strokeLinecap="round" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.9, ease: EASE_OUT }} />
            ) : (
              !failed && (
                <motion.circle
                  cx="30"
                  cy="30"
                  r="26"
                  fill="none"
                  stroke="var(--color-accent-500)"
                  strokeWidth="5"
                  strokeLinecap="round"
                  strokeDasharray="40 200"
                  animate={{ rotate: 360 }}
                  style={{ originX: '50%', originY: '50%' }}
                  transition={{ duration: 1.1, ease: 'linear', repeat: Infinity }}
                />
              )
            )}
          </svg>
          {level && (
            <motion.span
              className={cn(
                'grid size-8 place-items-center rounded-full',
                level === 'ok' ? 'bg-good-wash text-good-ink' : level === 'warn' ? 'bg-warning-wash text-warning-ink' : 'bg-critical-wash text-critical-ink',
              )}
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', stiffness: 380, damping: 22, delay: 0.45 }}
            >
              <Icon className="size-4" strokeWidth={2.6} />
            </motion.span>
          )}
        </div>
        <div className="min-w-0">
          <p className="text-[16px] leading-tight font-semibold text-ink-950">
            {failed ? 'Not checked' : !report ? 'Checking' : report.failed ? `${report.failed} to fix` : 'Ready for a session'}
          </p>
          <p className="mt-0.5 text-[12.5px] text-ink-500">
            {failed
              ? 'The engine did not answer.'
              : !report
                ? 'The camera, the graphics card and the models.'
                : report.warned && !report.failed
                  ? `${report.warned} worth a look`
                  : 'Every check passed.'}
          </p>
        </div>
      </div>
      {findings.length > 0 && (
        <ul className="mt-4 grid gap-1.5 border-t border-line pt-3">
          {findings.slice(0, 3).map((check) => (
            <li key={check.name} className="flex items-center gap-2 text-[12.5px] text-ink-700">
              <span className={cn('size-1.5 shrink-0 rounded-full', check.level === 'fail' ? 'bg-critical' : 'bg-warning')} aria-hidden="true" />
              <span className="truncate">{check.name}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

/** Four figures across the week: sessions, time live, the typical frame rate, and checks passed. */
function Figures({ history, security }: { history: History | null; security: Security | null }) {
  const week = history?.totals.since
  const finished = (history?.sessions ?? []).filter((session) => session.kind !== 'sample' && session.outcome !== 'running')
  const fps = median(finished.map((session) => session.fps))
  const since = Date.now() / 1000 - WEEK_SECONDS
  const checks = (security?.timeline ?? []).filter((item) => item.kind === 'check' && item.at >= since)
  const passed = checks.filter((item) => item.kind === 'check' && item.passed).length

  const items: { label: string; value: number | null; format: (v: number) => string; note: ReactNode }[] = [
    { label: 'Sessions this week', value: week?.sessions ?? null, format: (v) => Math.round(v).toString(), note: `${history?.totals.all.sessions ?? 0} in all` },
    { label: 'Time live this week', value: week ? week.seconds / 60 : null, format: (v) => `${Math.round(v)} min`, note: week ? duration(week.seconds) : dash },
    { label: 'Typical frame rate', value: fps, format: (v) => v.toFixed(1), note: fps === null ? 'after your first session' : 'fps, recent camera sessions' },
    { label: 'Checks passed', value: security ? passed : null, format: (v) => Math.round(v).toString(), note: security ? `of ${checks.length} this week` : dash },
  ]
  return (
    <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-card border border-line bg-line shadow-card lg:grid-cols-4">
      {items.map((item, index) => (
        <motion.div
          key={item.label}
          className="bg-surface px-5 py-4"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, ease: EASE_OUT, delay: 0.1 + index * 0.06 }}
        >
          <p className="text-[12.5px] font-medium text-ink-500">{item.label}</p>
          <p className="display mt-1 text-[28px] leading-none font-semibold text-ink-950">
            {item.value === null ? dash : <CountUp value={item.value} format={item.format} />}
          </p>
          <p className="mt-1.5 truncate text-[12px] text-ink-400">{item.note}</p>
        </motion.div>
      ))}
    </div>
  )
}

/** The retry limit in a line, and the last few entries of the security log. */
function SecuritySnippet({ security }: { security: Security | null }) {
  const attempts = security?.attempts
  const recent = (security?.timeline ?? []).slice(0, 6)
  return (
    <Card className="flex flex-col overflow-hidden">
      <CardHeader title="Security" hint="Liveness checks and enrolment changes." action={<SeeAll to="/studio/security" />} />
      {attempts && (
        <p
          className={cn(
            'mx-5 mb-3 rounded-panel px-3 py-2 text-[12.5px] font-medium',
            attempts.wait_seconds > 0 ? 'bg-critical-wash text-critical-ink' : attempts.failures > 0 ? 'bg-warning-wash text-warning-ink' : 'bg-good-wash text-good-ink',
          )}
        >
          {attempts.wait_seconds > 0
            ? `Next check waits ${duration(attempts.wait_seconds)}`
            : attempts.failures > 0
              ? `${attempts.failures} not passed in a row; ${Math.max(0, attempts.free - attempts.failures)} free attempts left`
              : 'No failed checks. Every attempt is free.'}
        </p>
      )}
      <ul className="grid gap-3 px-5 pb-5">
        {recent.map((item, index) => {
          const line = securityLine(item)
          const passed = item.kind === 'check' && item.passed
          const refused = item.kind === 'check' && !item.passed
          return (
            <li key={`${item.at}-${index}`} className="flex items-start gap-2.5">
              <span
                className={cn(
                  'mt-0.5 grid size-6 shrink-0 place-items-center rounded-full',
                  passed ? 'bg-good-wash text-good-ink' : refused ? 'bg-critical-wash text-critical-ink' : 'bg-accent-50 text-accent-700',
                )}
              >
                {refused ? <ShieldX className="size-3.5" /> : <ShieldCheck className="size-3.5" />}
              </span>
              <span className="min-w-0 flex-1 text-[12.5px] leading-snug text-ink-800">{line.title}</span>
              <span className="shrink-0 pt-px font-mono text-[11px] text-ink-400 tabular-nums">
                {dayLabel(item.at) === 'Today' ? clockTime(item.at) : dayLabel(item.at)}
              </span>
            </li>
          )
        })}
        {security && recent.length === 0 && <li className="text-[12.5px] text-ink-500">Nothing yet.</li>}
      </ul>
    </Card>
  )
}
