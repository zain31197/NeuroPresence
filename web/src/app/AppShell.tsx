import { ArrowLeft, ChartNoAxesColumn, FlaskConical, Lock, MonitorPlay, Server, UserRound, type LucideIcon } from 'lucide-react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { Logo } from '../components/Logo'
import { cn } from '../lib/cn'
import { useStudio, type Connection } from '../lib/studio'

// In the order a person goes through them: a picture first, then a session that animates it.
const SCREENS: { to: string; label: string; icon: LucideIcon }[] = [
  { to: '/studio/enrolment', label: 'Enrolment', icon: UserRound },
  { to: '/studio', label: 'Live Studio', icon: MonitorPlay },
]

// Screens that are designed but not built yet. They are listed so the shape of
// the app is visible, and they cannot be opened until they exist.
const NEXT: { label: string; icon: LucideIcon }[] = [
  { label: 'Test Lab', icon: FlaskConical },
  { label: 'Benchmarks', icon: ChartNoAxesColumn },
  { label: 'System', icon: Server },
]

const ENGINE: Record<Connection, { label: string; dot: string }> = {
  open: { label: 'Engine connected', dot: 'bg-good' },
  connecting: { label: 'Connecting to the engine', dot: 'bg-warning animate-lamp' },
  closed: { label: 'Engine offline', dot: 'bg-critical' },
}

function EngineStatus({ connection, className }: { connection: Connection; className?: string }) {
  const engine = ENGINE[connection]
  return (
    <span className={cn('inline-flex items-center gap-2 text-[12.5px] text-ink-600', className)}>
      <span className={cn('size-2 rounded-full', engine.dot)} />
      {engine.label}
    </span>
  )
}

export function AppShell() {
  const { connection, status } = useStudio()
  // Until a picture is enrolled, a camera session cannot start: point at the screen that comes first.
  const startHere = status && status.enrolment.record === null ? '/studio/enrolment' : null

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-[244px] shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <Link to="/" className="flex h-16 items-center px-5" aria-label="NeuroPresence home">
          <Logo />
        </Link>

        <nav className="flex-1 overflow-y-auto px-3 pt-2" aria-label="Studio">
          <p className="label-caps px-2.5 pb-2 text-ink-400">Studio</p>
          <ul className="grid gap-0.5">
            {SCREENS.map(({ to, label, icon: Icon }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end
                  className={({ isActive }) =>
                    cn(
                      'flex h-9 items-center gap-2.5 rounded-control px-2.5 text-[13.5px] font-medium transition-colors',
                      isActive ? 'bg-ink-100 text-ink-950' : 'text-ink-600 hover:bg-ink-50 hover:text-ink-950',
                    )
                  }
                >
                  <Icon className="size-4" strokeWidth={2} />
                  {label}
                  {to === startHere && <span className="label-caps ml-auto rounded-chip bg-accent-50 px-1.5 py-0.5 text-accent-700">Start here</span>}
                </NavLink>
              </li>
            ))}
          </ul>

          <p className="label-caps px-2.5 pt-6 pb-2 text-ink-400">Coming next</p>
          <ul className="grid gap-0.5">
            {NEXT.map(({ label, icon: Icon }) => (
              <li key={label} className="flex h-9 items-center gap-2.5 px-2.5 text-[13.5px] text-ink-400" aria-disabled="true">
                <Icon className="size-4" strokeWidth={2} />
                {label}
              </li>
            ))}
          </ul>
        </nav>

        <div className="grid gap-3 border-t border-line p-4">
          <EngineStatus connection={connection} />
          <span className="inline-flex items-center gap-2 text-[12.5px] text-ink-600">
            <Lock className="size-3.5 text-ink-400" />
            Runs only on this computer
          </span>
          <Link to="/" className="inline-flex items-center gap-2 text-[12.5px] text-ink-600 hover:text-ink-950">
            <ArrowLeft className="size-3.5 text-ink-400" />
            About NeuroPresence
          </Link>
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        <header className="border-b border-line bg-surface lg:hidden">
          <div className="flex h-14 items-center justify-between px-4">
            <Link to="/" aria-label="NeuroPresence home">
              <Logo />
            </Link>
            <EngineStatus connection={connection} />
          </div>
          {/* Without the sidebar, the screens are reached from here. */}
          <nav className="flex gap-1 px-3 pb-2" aria-label="Studio">
            {SCREENS.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  cn(
                    'flex h-8 items-center gap-2 rounded-control px-2.5 text-[13px] font-medium transition-colors',
                    isActive ? 'bg-ink-100 text-ink-950' : 'text-ink-600 hover:bg-ink-50 hover:text-ink-950',
                  )
                }
              >
                <Icon className="size-3.5" strokeWidth={2} />
                {label}
              </NavLink>
            ))}
          </nav>
        </header>
        <Outlet />
      </div>
    </div>
  )
}
