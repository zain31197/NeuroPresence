import { ChevronsUpDown, Database, LogOut, UserRound } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { EASE_OUT } from '../components/motion'
import { initials, useAccount } from '../lib/account'
import { cn } from '../lib/cn'
import { useStudio } from '../lib/studio'

/** The person, as their initials. */
export function Avatar({ name, className }: { name: string; className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        'grid size-9 shrink-0 place-items-center rounded-[11px] bg-gradient-to-br from-accent-400 to-accent-700 text-[12.5px] font-semibold tracking-wide text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.25)]',
        className,
      )}
    >
      {initials(name)}
    </span>
  )
}

/** Who is signed in, at the foot of the sidebar, with the way to their account, their data and out. */
export function UserMenu() {
  const { user, signOut } = useAccount()
  const { status } = useStudio()
  const live = status?.session.state === 'running' || status?.session.state === 'starting'
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const outside = (event: PointerEvent) => !box.current?.contains(event.target as Node) && setOpen(false)
    const escape = (event: KeyboardEvent) => event.key === 'Escape' && setOpen(false)
    window.addEventListener('pointerdown', outside)
    window.addEventListener('keydown', escape)
    return () => {
      window.removeEventListener('pointerdown', outside)
      window.removeEventListener('keydown', escape)
    }
  }, [open])

  if (!user) return null
  const item = 'flex w-full items-center gap-2.5 rounded-[9px] px-2.5 py-2 text-left text-[13.5px] font-medium text-ink-700 transition-colors hover:bg-ink-50 hover:text-ink-950'
  return (
    <div ref={box} className="relative">
      <AnimatePresence>
        {open && (
          <motion.div
            className="absolute right-0 bottom-full left-0 z-20 mb-2 rounded-panel border border-line bg-surface p-1.5 shadow-raised"
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.98 }}
            transition={{ duration: 0.16, ease: EASE_OUT }}
          >
            <Link to="/studio/account" className={item} onClick={() => setOpen(false)}>
              <UserRound className="size-4 text-ink-400" />
              Account
            </Link>
            <Link to="/studio/data" className={item} onClick={() => setOpen(false)}>
              <Database className="size-4 text-ink-400" />
              Your data
            </Link>
            <div className="my-1 h-px bg-line" />
            <button type="button" className={cn(item, 'hover:bg-critical-wash hover:text-critical-ink')} onClick={signOut}>
              <LogOut className="size-4 text-ink-400" />
              {live ? 'Stop and sign out' : 'Sign out'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <button
        type="button"
        aria-expanded={open}
        aria-label={`Account of ${user.name}`}
        onClick={() => setOpen(!open)}
        className={cn('flex w-full items-center gap-2.5 rounded-[12px] p-1.5 text-left transition-colors', open ? 'bg-ink-100' : 'hover:bg-ink-50')}
      >
        <Avatar name={user.name} />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-semibold text-ink-950">{user.name}</span>
          <span className="block truncate text-[11.5px] text-ink-500">{user.email}</span>
        </span>
        <ChevronsUpDown className="size-3.5 shrink-0 text-ink-400" />
      </button>
    </div>
  )
}
