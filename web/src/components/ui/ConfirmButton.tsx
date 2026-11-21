import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { EASE_OUT } from '../motion'
import { Button } from './Button'

interface Props {
  /** The button as it first shows. */
  children: ReactNode
  icon?: ReactNode
  /** The question asked in its place, and the words of the button that goes ahead. */
  question: string
  confirm: string
  onConfirm: () => Promise<unknown> | void
  disabled?: boolean
}

/**
 * A button for something that cannot be undone. The first press asks, in place, whether to go ahead; the
 * second does it. Escape, or eight seconds without an answer, puts the button back.
 */
export function ConfirmButton({ children, icon, question, confirm, onConfirm, disabled }: Props) {
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState(false)
  const keep = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!asking) return
    keep.current?.focus()
    const timer = window.setTimeout(() => setAsking(false), 8000)
    const escape = (event: KeyboardEvent) => event.key === 'Escape' && setAsking(false)
    window.addEventListener('keydown', escape)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('keydown', escape)
    }
  }, [asking])

  const go = async () => {
    setBusy(true)
    try {
      await onConfirm()
    } finally {
      setBusy(false)
      setAsking(false)
    }
  }

  return (
    <AnimatePresence mode="wait" initial={false}>
      {asking ? (
        <motion.div
          key="asking"
          role="group"
          aria-label={question}
          className="flex flex-wrap items-center gap-2 rounded-control border border-critical/25 bg-critical-wash py-1 pr-1 pl-3"
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.18, ease: EASE_OUT }}
        >
          <span className="text-[13px] font-medium text-critical-ink">{question}</span>
          <Button size="sm" variant="ghost" ref={keep} onClick={() => setAsking(false)}>
            Keep
          </Button>
          <Button size="sm" variant="danger" busy={busy} onClick={go}>
            {confirm}
          </Button>
        </motion.div>
      ) : (
        <motion.div key="button" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
          <Button variant="secondary" icon={icon} disabled={disabled} onClick={() => setAsking(true)}>
            {children}
          </Button>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
