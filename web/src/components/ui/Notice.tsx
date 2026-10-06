import { Info, OctagonAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface Props {
  title: string
  /** "critical" for something that has gone wrong; "neutral" for something to know. */
  tone?: 'neutral' | 'critical'
  /** A button or link that deals with it, shown at the right. */
  action?: ReactNode
  className?: string
  children: ReactNode
}

/** A line across the top of a screen: something the person needs to know before going on. */
export function Notice({ title, tone = 'neutral', action, className, children }: Props) {
  const critical = tone === 'critical'
  const Icon = critical ? OctagonAlert : Info
  return (
    <div
      role={critical ? 'alert' : 'status'}
      className={cn(
        'flex flex-wrap items-center gap-x-4 gap-y-3 rounded-panel border px-4 py-3',
        critical ? 'border-critical/25 bg-critical-wash' : 'border-line bg-surface',
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 basis-[280px] items-start gap-3">
        <Icon className={cn('mt-0.5 size-4 shrink-0', critical ? 'text-critical-ink' : 'text-ink-500')} />
        <div className="min-w-0 text-[13.5px] leading-snug">
          <p className="font-semibold text-ink-950">{title}</p>
          <p className="mt-0.5 text-ink-700">{children}</p>
        </div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}
