import { Star, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button } from '../../components/ui/Button'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, type SessionDetail } from '../../lib/api'
import { cn } from '../../lib/cn'

/** A star that keeps a session: it is set and cleared at once, and "Clear history" leaves starred ones. */
export function StarButton({ on, onChange, className }: { on: boolean; onChange: (on: boolean) => void; className?: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? 'Remove the star' : 'Star this session'}
      onClick={(event) => {
        event.stopPropagation()
        onChange(!on)
      }}
      className={cn('grid size-8 shrink-0 place-items-center rounded-control transition-colors hover:bg-ink-100', className)}
    >
      <Star className={cn('size-4 transition-[fill,color,transform] duration-200', on ? 'scale-110 fill-warning text-warning' : 'text-ink-300 hover:text-ink-500')} />
    </button>
  )
}

/** The name and the note a person gives a session, saved as they stop writing, and deleting it. */
export function SessionNotes({ session, onChange, onDelete }: { session: SessionDetail; onChange: (session: SessionDetail) => void; onDelete: () => void }) {
  const toast = useToast()
  const [title, setTitle] = useState(session.title ?? '')
  const [note, setNote] = useState(session.note ?? '')
  const [state, setState] = useState<'idle' | 'saving' | 'saved'>('idle')
  const [asking, setAsking] = useState(false)
  const gone = useRef(false) // deleted: nothing more is saved
  const saved = useRef({ title, note })
  const typed = useRef({ title, note })
  const changed = useRef(onChange)
  useEffect(() => {
    changed.current = onChange
    typed.current = { title, note }
  })
  // Closed before the typing had paused: what was typed is saved all the same.
  useEffect(
    () => () => {
      if (gone.current || (typed.current.title === saved.current.title && typed.current.note === saved.current.note)) return
      api.changeSession(session.id, typed.current).then((reply) => changed.current(reply), () => undefined)
    },
    [session.id],
  )

  useEffect(() => {
    if (title === saved.current.title && note === saved.current.note) return
    setState('saving')
    const timer = window.setTimeout(async () => {
      try {
        const reply = await api.changeSession(session.id, { title, note })
        saved.current = { title, note }
        changed.current(reply)
        setState('saved')
      } catch (error) {
        setState('idle')
        toast.problem(error instanceof ApiError ? error.message : 'The note could not be saved.')
      }
    }, 600)
    return () => window.clearTimeout(timer)
  }, [title, note, session.id, toast])

  const remove = async () => {
    try {
      await api.deleteSession(session.id)
      gone.current = true
      toast.done('Session deleted')
      onDelete()
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The session could not be deleted.')
    }
  }

  const box =
    'w-full rounded-control border border-line bg-surface px-3 text-[13.5px] text-ink-950 transition-[border-color,box-shadow] outline-none placeholder:text-ink-400 hover:border-line-strong focus-visible:border-accent-500 focus-visible:shadow-[0_0_0_3px_rgb(80_98_240/0.16)] focus-visible:outline-none'
  return (
    <section className="rounded-panel border border-line bg-ink-50/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="label-caps text-ink-400">Your note</h3>
        <span className="text-[11.5px] text-ink-400" aria-live="polite">
          {state === 'saving' ? 'Saving' : state === 'saved' ? 'Saved' : ''}
        </span>
      </div>
      <input value={title} maxLength={80} onChange={(event) => setTitle(event.target.value)} placeholder="Name this session" aria-label="Name of the session" className={cn(box, 'mt-2.5 h-9 font-semibold')} />
      <textarea value={note} maxLength={2000} rows={3} onChange={(event) => setNote(event.target.value)} placeholder="Who was in it, what worked, what to look at" aria-label="Note" className={cn(box, 'mt-2 resize-y py-2 leading-relaxed')} />
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[12px] text-ink-500">{session.picture ? `Animated “${session.picture}”` : session.kind === 'sample' ? 'A recorded clip animated its own frame' : ''}</span>
        {session.outcome !== 'running' &&
          (asking ? (
            <span className="inline-flex items-center gap-1.5">
              <Button size="sm" variant="danger" onClick={remove}>
                Delete it
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setAsking(false)}>
                Keep
              </Button>
            </span>
          ) : (
            <Button size="sm" variant="ghost" icon={<Trash2 className="size-3.5" />} onClick={() => setAsking(true)}>
              Delete session
            </Button>
          ))}
      </div>
    </section>
  )
}
