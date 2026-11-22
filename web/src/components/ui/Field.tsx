import { Eye, EyeOff } from 'lucide-react'
import { motion } from 'motion/react'
import { useId, useState, type InputHTMLAttributes, type ReactNode } from 'react'
import { passwordStrength } from '../../lib/account'
import { cn } from '../../lib/cn'
import { EASE_OUT } from '../motion'

interface Props extends InputHTMLAttributes<HTMLInputElement> {
  label: string
  /** What is wrong with it, under the field, in place of the hint. */
  error?: string | null
  hint?: ReactNode
  /** Something at the right of the label, such as a link. */
  aside?: ReactNode
  /** Inside the field, at its right end. */
  trailing?: ReactNode
}

/** A labelled text field. */
export function Field({ label, error, hint, aside, trailing, className, id, ...rest }: Props) {
  const own = useId()
  const field = id ?? own
  const note = `${field}-note`
  return (
    <div className={className}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={field} className="text-[13px] font-medium text-ink-900">
          {label}
        </label>
        {aside}
      </div>
      <div className="relative mt-1.5">
        <input
          id={field}
          aria-invalid={error ? true : undefined}
          aria-describedby={error || hint ? note : undefined}
          className={cn(
            'h-11 w-full rounded-control border bg-surface px-3.5 text-[14.5px] text-ink-950 shadow-control transition-[border-color,box-shadow] duration-200 outline-none placeholder:text-ink-400',
            'focus-visible:border-accent-500 focus-visible:shadow-[0_0_0_3px_rgb(80_98_240/0.18)] focus-visible:outline-none disabled:bg-ink-50 disabled:text-ink-500',
            error ? 'border-critical' : 'border-line-strong hover:border-ink-300',
            trailing && 'pr-11',
          )}
          {...rest}
        />
        {trailing && <div className="absolute inset-y-0 right-1.5 grid place-items-center">{trailing}</div>}
      </div>
      {(error || hint) && (
        <p id={note} role={error ? 'alert' : undefined} className={cn('mt-1.5 text-[12.5px] leading-snug', error ? 'text-critical-ink' : 'text-ink-500')}>
          {error || hint}
        </p>
      )}
    </div>
  )
}

const STRENGTH = [
  { label: 'At least 8 characters', bar: 'bg-ink-200' },
  { label: 'Too easy to guess', bar: 'bg-critical' },
  { label: 'Fair. Longer is better', bar: 'bg-warning' },
  { label: 'Good', bar: 'bg-accent-500' },
  { label: 'Strong', bar: 'bg-good' },
] as const

/** A password, which can be looked at, and for a new one how hard it is to guess. */
export function PasswordField({ meter = false, ...rest }: Omit<Props, 'type' | 'trailing'> & { meter?: boolean }) {
  const [shown, setShown] = useState(false)
  const strength = passwordStrength(typeof rest.value === 'string' ? rest.value : '')
  return (
    <div>
      <Field
        {...rest}
        type={shown ? 'text' : 'password'}
        trailing={
          <button
            type="button"
            onClick={() => setShown(!shown)}
            aria-label={shown ? 'Hide the password' : 'Show the password'}
            className="grid size-8 place-items-center rounded-[8px] text-ink-500 transition-colors hover:bg-ink-100 hover:text-ink-900"
          >
            {shown ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
          </button>
        }
      />
      {meter && !rest.error && (
        <div className="mt-2 flex items-center gap-3">
          <div className="flex flex-1 gap-1" aria-hidden="true">
            {[1, 2, 3, 4].map((step) => (
              <span key={step} className="h-1 flex-1 overflow-hidden rounded-full bg-ink-100">
                <motion.span
                  className={cn('block h-full origin-left rounded-full', STRENGTH[strength].bar)}
                  initial={false}
                  animate={{ scaleX: strength >= step ? 1 : 0 }}
                  transition={{ duration: 0.3, ease: EASE_OUT, delay: step * 0.03 }}
                />
              </span>
            ))}
          </div>
          <span className="w-[148px] shrink-0 text-right text-[12px] text-ink-500" aria-live="polite">
            {STRENGTH[strength].label}
          </span>
        </div>
      )}
    </div>
  )
}
