import { History as HistoryIcon, Play, Search, Star, Trash2, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { CountUp, EASE_OUT } from '../../components/motion'
import { PageHeader } from '../../components/PageHeader'
import { buttonClass } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { ConfirmButton } from '../../components/ui/ConfirmButton'
import { Segmented } from '../../components/ui/Segmented'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, type History, type SessionRecord } from '../../lib/api'
import { dash } from '../../lib/format'
import { cn } from '../../lib/cn'
import { byDay, duration, median } from '../../lib/history'
import { useStudio } from '../../lib/studio'
import { useNow } from '../../lib/useNow'
import { SessionDrawer } from './SessionDrawer'
import { SessionRow } from './SessionRow'

type Filter = 'all' | SessionRecord['kind']

const FILTERS: { value: Filter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'camera', label: 'Camera' },
  { value: 'voice', label: 'Voice' },
  { value: 'sample', label: 'Clips' },
]

/** Every session of the account's, newest first, kept in the database: named, starred, searched, and opened in full. */
export function Sessions() {
  const { status } = useStudio()
  const toast = useToast()
  const [history, setHistory] = useState<History | null>(null)
  const [unreachable, setUnreachable] = useState(false)
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')
  const [starred, setStarred] = useState(false)
  // What the database found for the search and the star; null when neither is asked for.
  const [found, setFound] = useState<SessionRecord[] | null>(null)
  const [params, setParams] = useSearchParams()
  const open = Number(params.get('session')) || null
  const sessionState = status?.session.state

  const load = useCallback(async () => {
    try {
      setHistory(await api.history(200))
      setUnreachable(false)
    } catch {
      setUnreachable(true)
    }
  }, [])

  useEffect(() => {
    document.title = 'Sessions · NeuroPresence'
  }, [])
  // Read again whenever a session starts or ends, so the list is never behind what the engine did.
  useEffect(() => {
    load()
  }, [load, sessionState])

  const find = useCallback(async () => {
    const words = search.trim()
    if (!words && !starred) return setFound(null)
    try {
      setFound((await api.history(200, words, starred)).sessions)
    } catch {
      setUnreachable(true)
    }
  }, [search, starred])
  // The database is asked once the typing pauses; the star is asked for at once.
  useEffect(() => {
    const timer = window.setTimeout(find, search.trim() ? 220 : 0)
    return () => window.clearTimeout(timer)
  }, [find, search, sessionState])
  const reload = useCallback(() => {
    load()
    find()
  }, [load, find])

  const everything = useMemo(() => history?.sessions ?? [], [history])
  const sessions = found ?? everything
  const shown = filter === 'all' ? sessions : sessions.filter((session) => session.kind === filter)
  const days = byDay(shown.map((session) => ({ ...session, at: session.started_at })))
  const now = useNow(sessions.some((session) => session.outcome === 'running'))
  const counts = (kind: Filter) => (kind === 'all' ? sessions.length : sessions.filter((session) => session.kind === kind).length)

  const setOpen = (id: number | null) => {
    const next = new URLSearchParams(params)
    if (id === null) next.delete('session')
    else next.set('session', String(id))
    setParams(next, { replace: id === null })
  }
  const closeDrawer = useCallback(() => {
    setParams(
      (current) => {
        const next = new URLSearchParams(current)
        next.delete('session')
        return next
      },
      { replace: true },
    )
  }, [setParams])

  const star = async (session: SessionRecord, on: boolean) => {
    const mark = (list: SessionRecord[]) => list.map((item) => (item.id === session.id ? { ...item, starred: on } : item))
    setHistory((now) => now && { ...now, sessions: mark(now.sessions) })
    setFound((now) => now && mark(now))
    try {
      await api.changeSession(session.id, { starred: on })
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The star could not be saved.')
      reload()
    }
  }

  const clear = async () => {
    try {
      setHistory(await api.clearHistory())
      find()
      toast.done('History cleared. Starred sessions were kept.')
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The history could not be cleared.')
    }
  }

  return (
    <main className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <PageHeader
        title="Sessions"
        actions={
          everything.length > 0 && (
            <ConfirmButton icon={<Trash2 className="size-3.5" />} question="Clear every session without a star?" confirm="Clear history" onConfirm={clear}>
              Clear history
            </ConfirmButton>
          )
        }
      >
        Every session you have run, with how it ran and how it ended. Open one to see its figures over time, give it a name and write a note.
        Kept in your account, on this computer.
      </PageHeader>

      <Totals history={history} />

      {unreachable && !history && (
        <p className="mt-6 rounded-panel border border-line bg-surface px-4 py-3 text-[13.5px] text-ink-600">
          The engine did not answer, so the history cannot be read. Start it, and this page will fill in.
        </p>
      )}

      {history && everything.length === 0 && <Empty />}

      {everything.length > 0 && (
        <>
          <div className="mt-6 flex flex-wrap items-center gap-2.5">
            <Segmented label="Show" value={filter} onChange={setFilter} options={FILTERS.map((f) => ({ ...f, count: counts(f.value) }))} />
            <button
              type="button"
              aria-pressed={starred}
              onClick={() => setStarred(!starred)}
              className={cn(
                'inline-flex h-9 items-center gap-1.5 rounded-control border px-3 text-[13px] font-medium transition-colors duration-200',
                starred ? 'border-warning/40 bg-warning-wash text-warning-ink' : 'border-line bg-surface text-ink-600 hover:border-line-strong hover:text-ink-950',
              )}
            >
              <Star className={cn('size-3.5', starred && 'fill-warning text-warning')} />
              Starred
            </button>
            <div className="relative min-w-[200px] flex-1 sm:max-w-[320px]">
              <Search className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-ink-400" />
              <input
                type="search"
                value={search}
                maxLength={100}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search names, notes and inputs"
                aria-label="Search the sessions"
                className="h-9 w-full rounded-control border border-line bg-surface pr-8 pl-8.5 text-[13px] text-ink-950 transition-[border-color,box-shadow] outline-none placeholder:text-ink-400 hover:border-line-strong focus-visible:border-accent-500 focus-visible:shadow-[0_0_0_3px_rgb(80_98_240/0.16)] focus-visible:outline-none [&::-webkit-search-cancel-button]:hidden"
              />
              {search && (
                <button type="button" aria-label="Clear the search" onClick={() => setSearch('')} className="absolute top-1/2 right-1.5 grid size-6 -translate-y-1/2 place-items-center rounded-[7px] text-ink-400 hover:bg-ink-100 hover:text-ink-900">
                  <X className="size-3.5" />
                </button>
              )}
            </div>
            <p className="ml-auto text-[12.5px] text-ink-400" aria-live="polite">
              {shown.length} {shown.length === 1 ? 'session' : 'sessions'}
            </p>
          </div>

          <div className="mt-3 grid gap-5">
            <AnimatePresence mode="popLayout" initial={false}>
              {days.map((day) => (
                <motion.section
                  key={`${filter}-${day.day}`}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3, ease: EASE_OUT }}
                >
                  <h2 className="label-caps mb-2 px-1 text-ink-400">{day.day}</h2>
                  <Card className="divide-y divide-line overflow-hidden">
                    {day.items.map((session) => (
                      <SessionRow key={session.id} session={session} now={now} onOpen={() => setOpen(session.id)} onStar={(on) => star(session, on)} />
                    ))}
                  </Card>
                </motion.section>
              ))}
            </AnimatePresence>
            {shown.length === 0 && (
              <p className="rounded-panel bg-ink-50 px-4 py-8 text-center text-[13.5px] text-ink-500">
                {search.trim() ? `Nothing found for “${search.trim()}”.` : starred ? 'No starred sessions. Press the star beside one to keep it at hand.' : 'No sessions of this kind yet.'}
              </p>
            )}
          </div>
        </>
      )}

      <SessionDrawer id={open} targets={status?.targets ?? null} onClose={closeDrawer} onChange={reload} />
    </main>
  )
}

/** This week and all time, in one band: sessions, time live, and the typical frame rate. */
function Totals({ history }: { history: History | null }) {
  const week = history?.totals.since
  const all = history?.totals.all
  const live = (history?.sessions ?? []).filter((session) => session.kind !== 'sample' && session.outcome !== 'running')
  const fps = median(live.slice(0, 20).map((session) => session.fps))
  const items = [
    { label: 'Sessions this week', value: week?.sessions ?? null, format: (v: number) => Math.round(v).toString(), note: week ? duration(week.seconds) + ' live' : dash },
    { label: 'Sessions in all', value: all?.sessions ?? null, format: (v: number) => Math.round(v).toString(), note: all ? duration(all.seconds) + ' live' : dash },
    { label: 'Typical frame rate', value: fps, format: (v: number) => v.toFixed(1), note: fps === null ? 'after your first session' : 'fps, the last 20 sessions' },
  ]
  return (
    <Card className="mt-6 grid gap-px overflow-hidden bg-line sm:grid-cols-3">
      {items.map((item) => (
        <div key={item.label} className="bg-surface px-5 py-4">
          <p className="text-[12.5px] font-medium text-ink-500">{item.label}</p>
          <p className="display mt-1 text-[28px] leading-none font-semibold text-ink-950">
            {item.value === null ? dash : <CountUp value={item.value} format={item.format} />}
          </p>
          <p className="mt-1.5 text-[12px] text-ink-400">{item.note}</p>
        </div>
      ))}
    </Card>
  )
}

function Empty() {
  return (
    <Card className="relative mt-6 overflow-hidden px-6 py-14 text-center">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-0 left-1/2 size-[360px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgb(80_98_240/0.14),transparent_65%)]"
      />
      <motion.span
        className="relative mx-auto grid size-14 place-items-center rounded-[16px] bg-accent-50 text-accent-700"
        initial={{ scale: 0.6, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 320, damping: 20 }}
      >
        <HistoryIcon className="size-6" strokeWidth={1.8} />
      </motion.span>
      <p className="display relative mt-4 text-[18px] font-semibold text-ink-950">No sessions yet</p>
      <p className="relative mx-auto mt-1.5 max-w-[46ch] text-[13.5px] text-ink-500">
        Each session you run is kept here with its figures, its consent check and what happened in it, so you can look back at any of them.
      </p>
      <Link to="/studio/live" className={buttonClass('primary', 'md', 'relative mt-5')}>
        <Play className="size-3.5 fill-current" />
        Start a session
      </Link>
    </Card>
  )
}
