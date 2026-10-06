import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/cn'

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('rounded-card border border-line bg-surface shadow-card', className)} {...rest} />
}

interface HeaderProps {
  title: string
  /** One line under the title saying what the card shows. */
  hint?: string
  action?: ReactNode
}

export function CardHeader({ title, hint, action }: HeaderProps) {
  return (
    <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-3">
      <div className="min-w-0">
        <h2 className="text-[13.5px] font-semibold text-ink-950">{title}</h2>
        {hint && <p className="mt-0.5 text-[12.5px] leading-snug text-ink-500">{hint}</p>}
      </div>
      {action}
    </div>
  )
}
