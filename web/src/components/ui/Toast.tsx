import { Check, OctagonAlert, X } from 'lucide-react'
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'
import { cn } from '../../lib/cn'

type Kind = 'done' | 'problem'
interface Toast {
  id: number
  kind: Kind
  message: string
}

interface Toasts {
  /** Confirm that an action finished. */
  done: (message: string) => void
  /** Report that an action failed, in words the person can act on. */
  problem: (message: string) => void
}

const ToastContext = createContext<Toasts | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(1)

  const dismiss = useCallback((id: number) => setToasts((list) => list.filter((toast) => toast.id !== id)), [])
  const show = useCallback(
    (kind: Kind, message: string) => {
      const id = nextId.current++
      setToasts((list) => [...list.slice(-2), { id, kind, message }])
      window.setTimeout(() => dismiss(id), kind === 'problem' ? 7000 : 3500)
    },
    [dismiss],
  )
  const value = useMemo(
    () => ({ done: (message: string) => show('done', message), problem: (message: string) => show('problem', message) }),
    [show],
  )

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed right-5 bottom-5 z-50 flex w-[360px] max-w-[calc(100vw-2.5rem)] flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role={toast.kind === 'problem' ? 'alert' : 'status'}
            className="pointer-events-auto flex animate-rise items-start gap-2.5 rounded-panel border border-line bg-surface p-3 shadow-raised"
          >
            <span
              className={cn(
                'mt-px grid size-5 shrink-0 place-items-center rounded-full',
                toast.kind === 'problem' ? 'bg-critical-wash text-critical-ink' : 'bg-good-wash text-good-ink',
              )}
            >
              {toast.kind === 'problem' ? <OctagonAlert className="size-3" /> : <Check className="size-3" strokeWidth={3} />}
            </span>
            <p className="min-w-0 flex-1 text-[13px] leading-snug text-ink-800">{toast.message}</p>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss"
              className="-m-1 rounded-[6px] p-1 text-ink-400 hover:bg-ink-100 hover:text-ink-800"
            >
              <X className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast(): Toasts {
  const toasts = useContext(ToastContext)
  if (!toasts) throw new Error('useToast must be used inside <ToastProvider>.')
  return toasts
}
