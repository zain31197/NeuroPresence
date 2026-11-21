import { ChevronRight, Clapperboard, Mic, Video, type LucideIcon } from 'lucide-react'
import { LiveDot } from '../../components/fx'
import { Pill } from '../../components/ui/Pill'
import type { SessionRecord } from '../../lib/api'
import { cn } from '../../lib/cn'
import { fixed } from '../../lib/format'
import { clockTime, duration, OUTCOME } from '../../lib/history'

export const KIND_ICON: Record<SessionRecord['kind'], LucideIcon> = { camera: Video, voice: Mic, sample: Clapperboard }

/** How long a session lasted, or has lasted so far while it runs. */
export function lasted(session: SessionRecord, now: number): number {
  return session.outcome === 'running' ? Math.max(0, now / 1000 - session.started_at) : session.seconds
}

/** One past session in a list: what drove it, when, how long, how it ran and how it ended. Opens its detail. */
export function SessionRow({ session, now, onOpen, compact = false }: { session: SessionRecord; now: number; onOpen: () => void; compact?: boolean }) {
  const Icon = KIND_ICON[session.kind]
  const outcome = OUTCOME[session.outcome]
  const running = session.outcome === 'running'
  const when = running ? `since ${clockTime(session.started_at)}` : `${clockTime(session.started_at)} to ${clockTime(session.ended_at ?? session.started_at)}`
  const flags = [session.virtual_camera && 'Sent to the meeting', session.voice_in_step && 'Voice in step'].filter(Boolean) as string[]

  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex w-full items-center gap-3.5 px-4 py-3 text-left transition-colors duration-200 hover:bg-ink-50/80 focus-visible:bg-ink-50 focus-visible:outline-none sm:px-5"
    >
      <span
        className={cn(
          'relative grid size-9 shrink-0 place-items-center rounded-[11px] transition-transform duration-200 group-hover:scale-105',
          running ? 'bg-accent-50 text-accent-700' : 'bg-ink-100 text-ink-700',
        )}
      >
        <Icon className="size-4" strokeWidth={2} />
        {running && <LiveDot on className="absolute -top-0.5 -right-0.5" />}
      </span>

      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13.5px] font-semibold text-ink-950">{session.input_label}</span>
        <span className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[12.5px] text-ink-500">
          <span className="font-mono tabular-nums">{when}</span>
          <span aria-hidden="true">·</span>
          <span className="tabular-nums">{duration(lasted(session, now))}</span>
          {!compact && flags.map((flag) => (
            <span key={flag} className="hidden rounded-chip bg-ink-100 px-1.5 py-px text-[11px] font-medium text-ink-600 sm:inline">
              {flag}
            </span>
          ))}
        </span>
      </span>

      {!compact && (
        <span className="hidden shrink-0 items-center gap-4 font-mono text-[12px] text-ink-600 tabular-nums md:flex">
          <Figure value={fixed(session.fps)} unit="fps" />
          <Figure value={fixed(session.delay_ms, 0)} unit="ms" />
          <Figure value={fixed(session.identity, 2)} unit="match" />
        </span>
      )}

      {/* A fixed width, so the figures line up from row to row whatever the outcome's words. */}
      <span className="hidden shrink-0 justify-end min-[420px]:flex md:w-[158px]">
        <Pill tone={outcome.tone} icon={running ? null : undefined}>
          {outcome.label}
        </Pill>
      </span>
      <ChevronRight className="size-4 shrink-0 text-ink-300 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-ink-500" />
    </button>
  )
}

function Figure({ value, unit }: { value: string; unit: string }) {
  return (
    <span className="w-[68px] text-right">
      <span className="text-ink-900">{value}</span> <span className="text-[10.5px] text-ink-400">{unit}</span>
    </span>
  )
}
