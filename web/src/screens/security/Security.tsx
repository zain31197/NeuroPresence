import { ImageMinus, ImagePlus, Lock, ScanFace, ShieldCheck, ShieldX, Trash2, UserRoundX, type LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'
import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { EASE_OUT } from '../../components/motion'
import { PageHeader } from '../../components/PageHeader'
import { Card } from '../../components/ui/Card'
import { ConfirmButton } from '../../components/ui/ConfirmButton'
import { Pill } from '../../components/ui/Pill'
import { Segmented } from '../../components/ui/Segmented'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, type Security as SecurityLog, type SecurityItem } from '../../lib/api'
import { cn } from '../../lib/cn'
import { dateTime } from '../../lib/format'
import { byDay, clockTime, duration, securityLine } from '../../lib/history'
import { useStudio } from '../../lib/studio'
import { useNow } from '../../lib/useNow'

type Filter = 'all' | 'check' | 'enrolment'

const ICON: Record<Extract<SecurityItem, { kind: 'enrolment' }>['action'], LucideIcon> = {
  face_verified: ScanFace,
  pose_registered: ScanFace,
  picture_enrolled: ImagePlus,
  picture_removed: ImageMinus,
  face_forgotten: UserRoundX,
}

const WEEK_SECONDS = 7 * 24 * 3600

/** Every consent check and every change to the enrolment, kept on this computer, and where the retry limit stands. */
export function Security() {
  const { status } = useStudio()
  const toast = useToast()
  const [log, setLog] = useState<SecurityLog | null>(null)
  const [read, setRead] = useState(0) // when the log was read, for the wait to count down from
  const [filter, setFilter] = useState<Filter>('all')
  const sessionState = status?.session.state
  const consentStage = status?.session.consent?.stage

  const load = useCallback(async () => {
    try {
      setLog(await api.security())
      setRead(Date.now())
    } catch {
      /* the engine is not running: the page says so below */
    }
  }, [])

  useEffect(() => {
    document.title = 'Security · NeuroPresence'
  }, [])
  // A check that ends changes both the log and the retry limit: read them again then.
  useEffect(() => {
    load()
  }, [load, sessionState, consentStage])

  const clear = async () => {
    try {
      setLog(await api.clearSecurity())
      setRead(Date.now())
      toast.done('Security log cleared')
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The log could not be cleared.')
    }
  }

  const timeline = log?.timeline ?? []
  const shown = filter === 'all' ? timeline : timeline.filter((item) => item.kind === filter)
  const count = (kind: Filter) => (kind === 'all' ? timeline.length : timeline.filter((item) => item.kind === kind).length)

  return (
    <main className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <PageHeader
        title="Security"
        actions={
          timeline.length > 0 && (
            <ConfirmButton icon={<Trash2 className="size-3.5" />} question="Clear the whole log?" confirm="Clear log" onConfirm={clear}>
              Clear log
            </ConfirmButton>
          )
        }
      >
        Every liveness check and every change to your enrolment, newest first. Only the outcome is kept: no picture of the camera and no face
        signature is stored here.
      </PageHeader>

      {!log && (
        <p className="mt-6 rounded-panel border border-line bg-surface px-4 py-3 text-[13.5px] text-ink-600">
          Reading the log from the engine. If this stays, the engine is not running.
        </p>
      )}

      {log && (
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <RetryLimit attempts={log.attempts} read={read} onOver={load} />
          <ChecksWeek timeline={timeline} />
          <Enrolled />
        </div>
      )}

      {log && (
        <>
          <div className="mt-8 flex flex-wrap items-center justify-between gap-3">
            <Segmented
              label="Show"
              value={filter}
              onChange={setFilter}
              options={[
                { value: 'all', label: 'All', count: count('all') },
                { value: 'check', label: 'Checks', count: count('check') },
                { value: 'enrolment', label: 'Enrolment', count: count('enrolment') },
              ]}
            />
            <p className="inline-flex items-center gap-1.5 text-[12.5px] text-ink-400">
              <Lock className="size-3.5" />
              Kept on this computer only
            </p>
          </div>
          {shown.length === 0 ? (
            <p className="mt-3 rounded-panel bg-ink-50 px-4 py-10 text-center text-[13.5px] text-ink-500">
              Nothing yet. The check before each camera session, and each step of enrolment, will be listed here.
            </p>
          ) : (
            <div className="mt-3 grid gap-5">
              {byDay(shown).map((day) => (
                <section key={`${filter}-${day.day}`}>
                  <h2 className="label-caps mb-2 px-1 text-ink-400">{day.day}</h2>
                  <Card className="px-4 py-2 sm:px-5">
                    <ol className="relative">
                      {day.items.map((item, index) => (
                        <TimelineItem key={`${item.kind}-${item.at}-${index}`} item={item} index={index} last={index === day.items.length - 1} />
                      ))}
                    </ol>
                  </Card>
                </section>
              ))}
            </div>
          )}
        </>
      )}
    </main>
  )
}

/** The three free attempts as three slots, and how long the next check must wait once they are used. */
function RetryLimit({ attempts, read, onOver }: { attempts: SecurityLog['attempts']; read: number; onOver: () => void }) {
  const waiting = attempts.wait_seconds > 0
  const now = useNow(waiting)
  const left = Math.max(0, attempts.wait_seconds - (now - read) / 1000)
  const over = waiting && left <= 0

  useEffect(() => {
    if (over) onOver()
  }, [over, onOver])

  const used = Math.min(attempts.failures, attempts.free)
  const tone = waiting && left > 0 ? 'critical' : attempts.failures > 0 ? 'warning' : 'good'
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="text-[13.5px] font-semibold text-ink-950">Retry limit</p>
        <Pill tone={tone}>{tone === 'critical' ? 'Waiting' : tone === 'warning' ? `${attempts.failures} not passed` : 'Clear'}</Pill>
      </div>
      <div className="mt-4 flex gap-2" aria-label={`${used} of ${attempts.free} free attempts used`}>
        {Array.from({ length: attempts.free }, (_, index) => (
          <motion.span
            key={index}
            className={cn('h-2 flex-1 rounded-full', index < used ? (waiting ? 'bg-critical' : 'bg-warning') : 'bg-ink-100')}
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            style={{ originX: 0 }}
            transition={{ duration: 0.5, ease: EASE_OUT, delay: index * 0.08 }}
          />
        ))}
      </div>
      <p className="mt-3 text-[13px] leading-snug text-ink-600">
        {waiting && left > 0 ? (
          <>
            The next check can start in <span className="font-semibold text-ink-950 tabular-nums">{duration(left)}</span>.
          </>
        ) : attempts.failures >= attempts.free ? (
          'The free attempts are used: a further check that does not pass makes the next one wait longer.'
        ) : (
          `${attempts.free - used} of ${attempts.free} free attempts left.`
        )}
      </p>
      <p className="mt-2 text-[12px] leading-snug text-ink-400">
        After three checks that do not pass, the next one waits a minute, and each one after that twice as long, up to a quarter of an hour. A
        check that passes, or an hour without one, clears them. The count survives a restart of the engine.
      </p>
    </Card>
  )
}

/** Checks of the last seven days: how many passed, as a share of all of them. */
function ChecksWeek({ timeline }: { timeline: SecurityItem[] }) {
  const since = Date.now() / 1000 - WEEK_SECONDS
  const checks = timeline.filter((item): item is Extract<SecurityItem, { kind: 'check' }> => item.kind === 'check' && item.at >= since)
  const passed = checks.filter((check) => check.passed).length
  const share = checks.length ? passed / checks.length : 0
  return (
    <Card className="p-5">
      <p className="text-[13.5px] font-semibold text-ink-950">Checks this week</p>
      <p className="display mt-3 text-[28px] leading-none font-semibold text-ink-950 tabular-nums">
        {passed}
        <span className="text-[16px] text-ink-400"> of {checks.length} passed</span>
      </p>
      <div className="mt-4 flex h-2 gap-[2px] overflow-hidden rounded-full bg-ink-100" aria-hidden="true">
        {checks.length > 0 && (
          <>
            <motion.span className="h-full rounded-full bg-good" initial={{ width: 0 }} animate={{ width: `${share * 100}%` }} transition={{ duration: 0.8, ease: EASE_OUT }} />
            {passed < checks.length && <span className="h-full flex-1 rounded-full bg-critical" />}
          </>
        )}
      </div>
      <p className="mt-3 text-[12px] leading-snug text-ink-400">
        {checks.length
          ? `${checks.length - passed} not passed. A check is not passed when an action is missed, the face is still, or it is not the verified face.`
          : 'No check in the last seven days.'}
      </p>
    </Card>
  )
}

/** What is enrolled now, and when: the face, its poses, and the meeting picture. */
function Enrolled() {
  const { status } = useStudio()
  const face = status?.enrolment.face ?? null
  const record = status?.enrolment.record ?? null
  const poses = face ? Object.values(face.poses).filter(Boolean).length : 0
  return (
    <Card className="p-5">
      <p className="text-[13.5px] font-semibold text-ink-950">Enrolled now</p>
      <ul className="mt-3 grid gap-2.5 text-[13px]">
        <Row on={!!face} label="Verified face" detail={face ? `${poses} of 5 poses` : 'Not verified'} />
        <Row on={!!record} label="Meeting picture" detail={record ? `Since ${dateTime(record.enrolled_at)}` : 'None'} />
      </ul>
      <Link to="/studio/enrolment" className="mt-4 inline-block text-[12.5px] font-semibold text-accent-700 underline-offset-4 hover:underline">
        Open Enrolment
      </Link>
    </Card>
  )
}

function Row({ on, label, detail }: { on: boolean; label: string; detail: string }) {
  return (
    <li className="flex items-center gap-2.5">
      <span className={cn('size-2 shrink-0 rounded-full', on ? 'bg-good' : 'bg-ink-300')} aria-hidden="true" />
      <span className="font-medium text-ink-900">{label}</span>
      <span className="ml-auto truncate text-right text-ink-500">{detail}</span>
    </li>
  )
}

const TONE_NODE = {
  good: 'bg-good-wash text-good-ink',
  critical: 'bg-critical-wash text-critical-ink',
  accent: 'bg-accent-50 text-accent-700',
  neutral: 'bg-ink-100 text-ink-600',
  warning: 'bg-warning-wash text-warning-ink',
  serious: 'bg-serious-wash text-serious-ink',
} as const

function TimelineItem({ item, index, last }: { item: SecurityItem; index: number; last: boolean }) {
  const line = securityLine(item)
  const Icon = item.kind === 'check' ? (item.passed ? ShieldCheck : ShieldX) : ICON[item.action]
  return (
    <motion.li
      className="relative flex gap-3.5 py-3"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: EASE_OUT, delay: Math.min(index, 10) * 0.035 }}
    >
      {/* The line joining one entry to the next. */}
      {!last && <span aria-hidden="true" className="absolute top-[44px] bottom-[-12px] left-[15px] w-px bg-line" />}
      <span className={cn('relative grid size-8 shrink-0 place-items-center rounded-full', TONE_NODE[line.tone])}>
        <Icon className="size-4" strokeWidth={2} />
      </span>
      <div className="min-w-0 flex-1 pt-1">
        <p className="text-[13.5px] font-semibold text-ink-950">{line.title}</p>
        {line.detail && <p className="mt-0.5 text-[13px] leading-snug text-ink-600">{line.detail}</p>}
        {item.kind === 'check' && item.session_id !== null && (
          <Link to={`/studio/sessions?session=${item.session_id}`} className="mt-1 inline-block text-[12.5px] font-medium text-accent-700 underline-offset-4 hover:underline">
            Open the session
          </Link>
        )}
      </div>
      <span className="shrink-0 pt-1 font-mono text-[11.5px] text-ink-400 tabular-nums">{clockTime(item.at)}</span>
    </motion.li>
  )
}
