import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size = 'sm' | 'md' | 'lg'

const variants: Record<Variant, string> = {
  primary:
    'bg-ink-950 text-white shadow-control hover:bg-ink-800 [box-shadow:inset_0_1px_0_rgb(255_255_255/0.14),0_1px_2px_rgb(11_13_18/0.2)]',
  secondary: 'border border-line-strong bg-surface text-ink-900 shadow-control hover:bg-ink-50',
  ghost: 'text-ink-700 hover:bg-ink-100 hover:text-ink-950',
  danger: 'border border-critical/30 bg-surface text-critical-ink shadow-control hover:bg-critical-wash',
}

const sizes: Record<Size, string> = {
  sm: 'h-8 gap-1.5 px-3 text-[13px]',
  md: 'h-9 gap-2 px-3.5 text-[13.5px]',
  lg: 'h-11 gap-2 px-5 text-[15px]',
}

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  icon?: ReactNode
  /** Shows a spinner in place of the icon and blocks clicks. */
  busy?: boolean
}

/** The look of a button, for a link that should read as one. */
export function buttonClass(variant: Variant = 'secondary', size: Size = 'md', className?: string) {
  return cn(
    'inline-flex shrink-0 select-none items-center justify-center rounded-control font-medium whitespace-nowrap',
    'transition-[background-color,transform,opacity] duration-150 active:scale-[0.98]',
    'disabled:pointer-events-none disabled:opacity-45',
    variants[variant],
    sizes[size],
    className,
  )
}

export function Button({ variant = 'secondary', size = 'md', icon, busy, className, children, disabled, ...rest }: Props) {
  return (
    <button type="button" disabled={disabled || busy} className={buttonClass(variant, size, className)} {...rest}>
      {busy ? <Spinner /> : icon}
      {children}
    </button>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" className={cn('size-4 animate-spin', className)} aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14 8a6 6 0 0 0-6-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}
