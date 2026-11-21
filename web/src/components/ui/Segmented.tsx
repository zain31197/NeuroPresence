import { motion } from 'motion/react'
import { useId } from 'react'
import { cn } from '../../lib/cn'

interface Option<T extends string> {
  value: T
  label: string
  /** How many items the choice shows, beside its name. */
  count?: number
}

/** A row of choices of which one is on; the panel behind it glides to the one chosen. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: T
  onChange: (value: T) => void
  options: Option<T>[]
  label: string
  className?: string
}) {
  const id = useId()
  return (
    <div role="group" aria-label={label} className={cn('inline-flex max-w-full overflow-x-auto rounded-control border border-line bg-ink-50 p-0.5', className)}>
      {options.map((option) => {
        const on = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(option.value)}
            className={cn(
              'relative flex h-8 shrink-0 items-center gap-1.5 rounded-[8px] px-3 text-[13px] font-medium transition-colors duration-200',
              on ? 'text-ink-950' : 'text-ink-500 hover:text-ink-900',
            )}
          >
            {on && (
              <motion.span
                layoutId={`segment-${id}`}
                className="absolute inset-0 rounded-[8px] bg-surface shadow-control"
                transition={{ type: 'spring', stiffness: 500, damping: 40 }}
              />
            )}
            <span className="relative">{option.label}</span>
            {option.count !== undefined && <span className="relative font-mono text-[11px] text-ink-400 tabular-nums">{option.count}</span>}
          </button>
        )
      })}
    </div>
  )
}
