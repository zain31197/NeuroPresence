import { cn } from '../lib/cn'

/** A camera frame, the person in it, and the lamp that says they are present. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn('size-7', className)} aria-hidden="true">
      <rect x="2" y="2" width="28" height="28" rx="9" fill="var(--color-ink-950)" />
      <circle cx="15" cy="17" r="6.25" fill="none" stroke="#fff" strokeWidth="2.5" />
      <circle cx="23.5" cy="8.5" r="3" fill="var(--color-accent-500)" />
    </svg>
  )
}

export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-2.5', className)}>
      <LogoMark />
      <span className="display text-[16px] font-semibold text-ink-950">NeuroPresence</span>
    </span>
  )
}
