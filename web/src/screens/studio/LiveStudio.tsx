import { ArrowRight, Clapperboard, Play, RotateCcw, Square, Video } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button, buttonClass, Spinner } from '../../components/ui/Button'
import { Notice } from '../../components/ui/Notice'
import { Pill } from '../../components/ui/Pill'
import { Select } from '../../components/ui/Select'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, type Benchmark, type Session } from '../../lib/api'
import { clock } from '../../lib/format'
import { useStudio } from '../../lib/studio'
import { EnrolledPicture } from './EnrolledPicture'
import { EventsPanel } from './EventsPanel'
import { promptSecondsLeft } from './holds'
import { LatencyBreakdown } from './LatencyBreakdown'
import { MetricTiles } from './MetricTiles'
import { Monitors } from './Monitors'
import { TrackingPanel } from './TrackingPanel'

function SessionPill({ session }: { session: Session }) {
  if (session.state === 'running') {
    return (
      <span className="inline-flex h-[22px] items-center gap-1.5 rounded-full border border-line bg-surface px-2 text-[11.5px] font-semibold text-ink-950">
        <span className="size-2 animate-lamp rounded-full bg-live" />
        Live
        <span className="font-mono font-normal text-ink-500 tabular-nums">{clock(session.uptime_s)}</span>
      </span>
    )
  }
  if (session.state === 'starting') {
    return (
      <Pill tone="accent" icon={<Spinner className="size-3" />}>
        Starting
      </Pill>
    )
  }
  if (session.state === 'error') return <Pill tone="critical">Stopped on a problem</Pill>
  return (
    <Pill tone="neutral" icon={null}>
      Not running
    </Pill>
  )
}

/**
 * The identity monitor has put the still picture up. For a few seconds this asks what to do;
 * with no answer the still picture simply stays, and the way back is one button.
 */
function IdentityHold({ shownFor, busy, onResume }: { shownFor: number | null; busy: boolean; onResume: () => void }) {
  const [staying, setStaying] = useState(false)
  const left = promptSecondsLeft(shownFor, staying)
  return (
    <Notice
      className="mt-5"
      tone="critical"
      title={left > 0 ? 'The output stopped matching your picture' : 'Showing your still picture'}
      action={
        <div className="flex flex-wrap items-center gap-2">
          {left > 0 && (
            <Button size="sm" onClick={() => setStaying(true)}>
              Stay on the still picture
            </Button>
          )}
          <Button size="sm" variant="primary" icon={<RotateCcw className="size-3.5" />} busy={busy} onClick={onResume}>
            Resume reenactment
          </Button>
        </div>
      }
    >
      {left > 0
        ? `Your still picture is shown in its place. Resume with a fresh neutral pose, or it stays on the still picture in ${left} s.`
        : 'The output stopped matching your enrolled picture, and a fresh neutral pose did not bring it back. It stays on the still picture until you resume.'}
    </Notice>
  )
}

export function LiveStudio() {
  const { status, connection } = useStudio()
  const toast = useToast()
  const [input, setInput] = useState('camera:0')
  const [pending, setPending] = useState(false)
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null)

  useEffect(() => {
    document.title = 'Live Studio · NeuroPresence'
    api.latestBenchmark().then(setBenchmark, () => setBenchmark(null)) // none yet is fine
  }, [])

  const session = status?.session
  const active = session?.state === 'running' || session?.state === 'starting'
  const inputs = status?.inputs ?? []
  // A camera session animates the enrolled picture, so there has to be one. A sample clip brings its own.
  const needsPicture = !active && input.startsWith('camera') && !!status && status.enrolment.record === null
  const running = session?.state === 'running'
  const heldForIdentity = running && !!session?.holds?.identity
  const heldForDelay = running && !!session?.holds?.delay && !heldForIdentity

  const act = async (work: () => Promise<unknown>) => {
    setPending(true)
    try {
      await work()
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'Something went wrong. Try again.')
    } finally {
      setPending(false)
    }
  }

  return (
    <main className="mx-auto max-w-[1520px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="display text-[24px] leading-tight font-semibold">Live Studio</h1>
            {session && <SessionPill session={session} />}
          </div>
          <p className="mt-1 text-[13.5px] text-ink-500">Drive your enrolled picture from the camera and watch every stage respond.</p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Select
            label="Input"
            value={input}
            onValueChange={setInput}
            disabled={active || inputs.length === 0}
            groups={[
              {
                title: 'Cameras',
                options: inputs
                  .filter((option) => option.kind === 'camera')
                  .map((option) => ({ value: option.id, label: option.label, icon: <Video className="size-3.5 text-ink-500" /> })),
              },
              {
                title: 'Recorded clips',
                options: inputs
                  .filter((option) => option.kind === 'sample')
                  .map((option) => ({ value: option.id, label: option.label, icon: <Clapperboard className="size-3.5 text-ink-500" /> })),
              },
            ].filter((group) => group.options.length > 0)}
          />
          {active ? (
            <Button variant="danger" icon={<Square className="size-3.5 fill-current" />} busy={pending} onClick={() => act(api.stopSession)}>
              Stop session
            </Button>
          ) : (
            <Button
              variant="primary"
              icon={<Play className="size-3.5 fill-current" />}
              busy={pending}
              disabled={connection !== 'open' || needsPicture}
              onClick={() => act(() => api.startSession(input))}
            >
              Start session
            </Button>
          )}
        </div>
      </div>

      {connection === 'closed' && (
        <Notice className="mt-5" title="The engine is not running">
          Start it in a terminal with <code className="rounded-[5px] bg-ink-100 px-1.5 py-0.5 font-mono text-[12px]">python -m neuropresence.server</code>.
          This page reconnects by itself.
        </Notice>
      )}
      {session?.state === 'error' && (
        <Notice className="mt-5" title="The session stopped" tone="critical">
          {session.message}
        </Notice>
      )}
      {heldForIdentity && (
        <IdentityHold shownFor={status?.identity.guard?.seconds ?? null} busy={pending} onResume={() => act(api.resumeSession)} />
      )}
      {heldForDelay && (
        <Notice className="mt-5" title="The picture is arriving late">
          The delay from camera to output is above {status?.targets.end_to_end_ms} ms, so your still picture is shown: lips that move late are
          worse than a still face. Reenactment returns by itself once the delay is back under the limit.
        </Notice>
      )}
      {needsPicture && connection === 'open' && (
        <Notice
          className="mt-5"
          title="Enrol a picture first"
          action={
            <Link to="/studio/enrolment" className={buttonClass('primary', 'sm')}>
              Go to Enrolment
              <ArrowRight className="size-3.5" />
            </Link>
          }
        >
          A camera session animates your enrolled picture, and none is enrolled yet. The recorded clips run without one.
        </Notice>
      )}

      <div className="mt-5 grid gap-5">
        <Monitors />
        <MetricTiles benchmark={benchmark} />
        <TrackingPanel />
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,0.9fr)]">
          <LatencyBreakdown />
          <EventsPanel />
          <EnrolledPicture />
        </div>
      </div>
    </main>
  )
}
