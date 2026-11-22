import { Check, ImageIcon, Pencil, Trash2 } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useState } from 'react'
import { EASE_OUT } from '../../components/motion'
import { Button } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { useToast } from '../../components/ui/Toast'
import { bytes } from '../../lib/account'
import { api, ApiError, libraryPictureUrl, type LibraryPicture, type Library as LibraryData } from '../../lib/api'
import { cn } from '../../lib/cn'
import { dayLabel } from '../../lib/history'
import { useStudio } from '../../lib/studio'

/**
 * The account's meeting pictures, kept in the database: every picture that was enrolled, with the one in use
 * marked. Another is brought back with one press, after the same checks an upload goes through.
 */
export function Library({ version }: { version: string | null }) {
  const { refresh } = useStudio()
  const toast = useToast()
  const [library, setLibrary] = useState<LibraryData | null>(null)
  const [busy, setBusy] = useState<number | null>(null)

  const load = useCallback(() => {
    api.pictures().then(setLibrary, () => undefined)
  }, [])
  // Read again whenever another picture is enrolled or removed.
  useEffect(() => {
    load()
  }, [load, version])

  const act = async (id: number, work: () => Promise<LibraryData>, done: string) => {
    setBusy(id)
    try {
      setLibrary(await work())
      await refresh().catch(() => undefined)
      toast.done(done)
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'Something went wrong. Try again.')
    } finally {
      setBusy(null)
    }
  }

  const pictures = library?.pictures ?? []
  if (pictures.length === 0) return null
  return (
    <Card className="mt-6 overflow-hidden">
      <CardHeader
        title="Your pictures"
        hint="Every picture you have enrolled, kept in your account. Bring one back with a press; it is checked against your face again."
        action={
          <span className="shrink-0 font-mono text-[11.5px] text-ink-400 tabular-nums">
            {pictures.length} of {library?.most}
          </span>
        }
      />
      <ul className="grid gap-4 px-5 pt-1 pb-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        <AnimatePresence initial={false}>
          {pictures.map((picture) => (
            <motion.li
              key={picture.id}
              layout
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.94 }}
              transition={{ duration: 0.3, ease: EASE_OUT }}
            >
              <Tile
                picture={picture}
                busy={busy === picture.id}
                disabled={busy !== null}
                onUse={() => act(picture.id, () => api.usePicture(picture.id), `${picture.name} is your meeting picture now.`)}
                onRename={(name) => act(picture.id, () => api.renamePicture(picture.id, name), 'Renamed.')}
                onDelete={() => act(picture.id, () => api.deletePicture(picture.id), `${picture.name} deleted.`)}
              />
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </Card>
  )
}

interface TileProps {
  picture: LibraryPicture
  busy: boolean
  disabled: boolean
  onUse: () => void
  onRename: (name: string) => void
  onDelete: () => void
}

function Tile({ picture, busy, disabled, onUse, onRename, onDelete }: TileProps) {
  const [renaming, setRenaming] = useState(false)
  const [asking, setAsking] = useState(false)
  const [name, setName] = useState(picture.name)

  const save = () => {
    setRenaming(false)
    const clean = name.trim()
    if (clean && clean !== picture.name) onRename(clean)
    else setName(picture.name)
  }

  return (
    <div className={cn('group overflow-hidden rounded-panel border bg-surface transition-[border-color,box-shadow] duration-300', picture.in_use ? 'border-accent-300 shadow-[0_0_0_3px_rgb(80_98_240/0.12)]' : 'border-line hover:border-line-strong hover:shadow-card')}>
      <div className="relative aspect-square overflow-hidden bg-ink-100">
        <img src={libraryPictureUrl(picture.id)} alt={picture.name} loading="lazy" className="size-full object-cover object-[50%_35%] transition-transform duration-500 ease-out-soft group-hover:scale-[1.03]" />
        {picture.in_use && (
          <Pill tone="accent" icon={<Check className="size-3" strokeWidth={3} />} className="absolute top-2 left-2 bg-surface shadow-control">
            In use
          </Pill>
        )}
      </div>
      <div className="p-3.5">
        {renaming ? (
          <input
            autoFocus
            value={name}
            maxLength={60}
            aria-label="Name of the picture"
            onChange={(event) => setName(event.target.value)}
            onBlur={save}
            onKeyDown={(event) => {
              if (event.key === 'Enter') save()
              if (event.key === 'Escape') {
                setName(picture.name)
                setRenaming(false)
              }
            }}
            className="h-7 w-full rounded-[7px] border border-accent-400 bg-surface px-2 text-[13.5px] font-semibold text-ink-950 outline-none"
          />
        ) : (
          <button type="button" onClick={() => setRenaming(true)} className="group/name flex max-w-full items-center gap-1.5 text-left" aria-label={`Rename ${picture.name}`}>
            <span className="truncate text-[13.5px] font-semibold text-ink-950">{picture.name}</span>
            <Pencil className="size-3 shrink-0 text-ink-400 opacity-0 transition-opacity group-hover/name:opacity-100 group-focus-visible/name:opacity-100" />
          </button>
        )}
        <p className="mt-1 truncate text-[12px] text-ink-500">
          {dayLabel(picture.created_at)} · {picture.width} × {picture.height} · {bytes(picture.bytes)}
        </p>
        <p className="mt-0.5 truncate text-[12px] text-ink-500">
          {picture.face_match !== null && (
            <>
              Face match <span className="font-mono text-ink-800 tabular-nums">{picture.face_match.toFixed(2)}</span> ·{' '}
            </>
          )}
          {picture.sessions} {picture.sessions === 1 ? 'session' : 'sessions'}
        </p>
        <div className="mt-3 flex items-center gap-2">
          {asking ? (
            <>
              <Button size="sm" variant="danger" className="flex-1" disabled={disabled} onClick={() => { setAsking(false); onDelete() }}>
                Delete
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setAsking(false)}>
                Keep
              </Button>
            </>
          ) : (
            <>
              <Button size="sm" className="flex-1" icon={<ImageIcon className="size-3.5" />} busy={busy} disabled={disabled || picture.in_use} onClick={onUse}>
                {picture.in_use ? 'In use' : 'Use this'}
              </Button>
              <Button size="sm" variant="ghost" className="px-2" disabled={disabled} onClick={() => setAsking(true)} aria-label={`Delete ${picture.name}`}>
                <Trash2 className="size-3.5" />
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
