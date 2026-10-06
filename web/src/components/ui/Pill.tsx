import { TriangleAlert, Check, CircleDashed, Minus, OctagonAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'

export type Tone = 'neutral' | 'good' | 'warning' | 'serious' | 'critical' | 'accent'

const tones: Record<Tone, string> = {
  neutral: 'bg-ink-100 text-ink-700',
  good: 'bg-good-wash text-good-ink',
  warning: 'bg-warning-wash text-warning-ink',
  serious: 'bg-serious-wash text-serious-ink',
  critical: 'bg-critical-wash text-critical-ink',
  accent: 'bg-accent-50 text-accent-700',
}

// A status colour never carries meaning alone: every tone has its own icon.
const icons: Record<Tone, ReactNode> = {
  neutral: <Minus className="size-3" strokeWidth={2.5} />,
  good: <Check className="size-3" strokeWidth={3} />,
  warning: <TriangleAlert className="size-3" strokeWidth={2.5} />,
  serious: <TriangleAlert className="size-3" strokeWidth={2.5} />,
  critical: <OctagonAlert className="size-3" strokeWidth={2.5} />,
  accent: <CircleDashed className="size-3" strokeWidth={2.5} />,
}

interface Props {
  tone?: Tone
  /** Replace the tone's icon, or pass null to show none. */
  icon?: ReactNode | null
  className?: string
  children: ReactNode
}

export function Pill({ tone = 'neutral', icon, className, children }: Props) {
  return (
    <span
      className={cn(
        'inline-flex h-[22px] shrink-0 items-center gap-1 rounded-full px-2 text-[11.5px] font-medium whitespace-nowrap',
        tones[tone],
        className,
      )}
    >
      {icon === undefined ? icons[tone] : icon}
      {children}
    </span>
  )
}
