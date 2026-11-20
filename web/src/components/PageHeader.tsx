import type { ReactNode } from 'react'

/** The top of every studio screen: its name, a status beside it, one line on what it is for, and its actions. */
export function PageHeader({ title, badge, children, actions }: { title: string; badge?: ReactNode; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
      <div className="min-w-0">
        <div className="flex items-center gap-3">
          <h1 className="display text-[24px] leading-tight font-semibold">{title}</h1>
          {badge}
        </div>
        {children && <p className="mt-1 max-w-[90ch] text-[13.5px] text-ink-500">{children}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}
