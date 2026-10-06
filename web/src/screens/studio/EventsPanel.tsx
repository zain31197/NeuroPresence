import { Dot, OctagonAlert, TriangleAlert } from 'lucide-react'
import { Card, CardHeader } from '../../components/ui/Card'
import type { StudioEvent } from '../../lib/api'
import { timeOfDay } from '../../lib/format'
import { useStudio } from '../../lib/studio'

const ICONS: Record<StudioEvent['level'], { icon: typeof Dot; className: string; label: string }> = {
  info: { icon: Dot, className: 'text-ink-400', label: 'Note' },
  warning: { icon: TriangleAlert, className: 'text-serious-ink', label: 'Warning' },
  error: { icon: OctagonAlert, className: 'text-critical-ink', label: 'Error' },
}

/** What happened in the session, newest first: the record of every edge case. */
export function EventsPanel() {
  const { events } = useStudio()
  const newestFirst = [...events].reverse()

  return (
    <Card className="flex flex-col">
      <CardHeader title="Events" hint="Lost faces, fallbacks and changed settings, as they happen." />
      {newestFirst.length === 0 ? (
        <p className="px-5 pb-5 text-[12.5px] text-ink-500">Nothing has happened yet.</p>
      ) : (
        // The list takes whatever height the row gives the card and scrolls inside it.
        <ol className="h-0 min-h-[220px] flex-1 overflow-y-auto px-2 pb-2" aria-label="Session events, newest first">
          {newestFirst.map((event) => {
            const { icon: Icon, className, label } = ICONS[event.level]
            return (
              <li key={event.id} className="flex items-start gap-2 rounded-[8px] px-3 py-1.5 text-[13px] hover:bg-ink-50">
                <time className="w-[58px] shrink-0 pt-px font-mono text-[11.5px] text-ink-400 tabular-nums">{timeOfDay(event.at)}</time>
                <Icon className={`mt-[3px] size-3.5 shrink-0 ${className}`} strokeWidth={event.level === 'info' ? 6 : 2.25} aria-label={label} />
                <span className="min-w-0 leading-snug text-ink-800">{event.message}</span>
              </li>
            )
          })}
        </ol>
      )}
    </Card>
  )
}
