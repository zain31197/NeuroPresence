import { ArrowLeft, ChartNoAxesColumn, Database, History, LayoutDashboard, Lock, MonitorPlay, Server, ShieldCheck, UserRound, type LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { Logo } from '../components/Logo'
import { DURATION, EASE_OUT } from '../components/motion'
import { useAccount } from '../lib/account'
import { cn } from '../lib/cn'
import { useStudio, type Connection } from '../lib/studio'
import { Avatar, UserMenu } from './UserMenu'

type Screen = { to: string; label: string; icon: LucideIcon }

// The studio's sidebar, in three groups. The first in the order a person goes through it: where things stand,
// a picture, then a session that animates it.
const GROUPS: { title: string; screens: Screen[] }[] = [
  {
    title: 'Studio',
    screens: [
      { to: '/studio', label: 'Overview', icon: LayoutDashboard },
      { to: '/studio/enrolment', label: 'Enrolment', icon: UserRound },
      { to: '/studio/live', label: 'Live Studio', icon: MonitorPlay },
    ],
  },
  // What has happened, kept in the account's part of the database.
  {
    title: 'Records',
    screens: [
      { to: '/studio/sessions', label: 'Sessions', icon: History },
      { to: '/studio/security', label: 'Security', icon: ShieldCheck },
      { to: '/studio/data', label: 'Your data', icon: Database },
    ],
  },
  // How the system measures up, and whether this computer is ready for it.
  {
    title: 'Insight',
    screens: [
      { to: '/studio/benchmarks', label: 'Benchmarks', icon: ChartNoAxesColumn },
      { to: '/studio/system', label: 'System', icon: Server },
    ],
  },
]

function SideLink({ to, label, icon: Icon, badge }: { to: string; label: string; icon: LucideIcon; badge?: string | null }) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        cn(
          'group relative flex h-9 items-center gap-2.5 rounded-control px-2.5 text-[13.5px] font-medium transition-[background-color,color,box-shadow] duration-200 ease-out-soft',
          isActive ? 'text-ink-950' : 'text-ink-600 hover:bg-ink-50 hover:text-ink-950',
        )
      }
    >
      {({ isActive }) => (
        <>
          {/* The place you are in: a soft panel and an ultramarine bar that glide from one link to the next. */}
          {isActive && (
            <motion.span
              layoutId="nav-active"
              className="absolute inset-0 rounded-control bg-ink-100/80 shadow-[inset_0_0_0_1px_rgb(11_13_18/0.04)]"
              transition={{ type: 'spring', stiffness: 500, damping: 40 }}
            >
              <span className="absolute top-2 bottom-2 -left-3 w-[3px] rounded-r-full bg-accent-500" />
            </motion.span>
          )}
          <Icon className="relative size-4 transition-transform duration-200 group-hover:scale-110" strokeWidth={2} />
          <span className="relative">{label}</span>
          {badge && <span className="label-caps relative ml-auto rounded-chip bg-accent-50 px-1.5 py-0.5 text-accent-700">{badge}</span>}
        </>
      )}
    </NavLink>
  )
}

const ENGINE: Record<Connection, { label: string; dot: string }> = {
  open: { label: 'Engine connected', dot: 'bg-good' },
  connecting: { label: 'Connecting to the engine', dot: 'bg-warning animate-lamp' },
  closed: { label: 'Engine offline', dot: 'bg-critical' },
}

function EngineStatus({ connection, className }: { connection: Connection; className?: string }) {
  const engine = ENGINE[connection]
  return (
    <span className={cn('inline-flex items-center gap-2 text-[12.5px] text-ink-600', className)}>
      <span className="relative grid size-2 place-items-center">
        <span className={cn('relative size-2 rounded-full', engine.dot)} />
      </span>
      {engine.label}
    </span>
  )
}

export function AppShell() {
  const { connection, status } = useStudio()
  const { user } = useAccount()
  // Until a picture is enrolled, a camera session cannot start: point at the screen that comes first.
  const startHere = status && status.enrolment.record === null ? '/studio/enrolment' : null
  const { pathname } = useLocation()

  const onAir = status?.session.state === 'running'
  // In a meeting the graphics card belongs to the picture. The endless decoration (the travelling line, the pulsing
  // dots) makes the browser redraw the page 60 times a second, which cost the drawing about 0.7 frames a second when
  // measured on 22 November 2026, so it stands still while a session runs with the virtual camera on.
  const inMeeting = onAir && !!status?.virtual_camera?.on
  useEffect(() => {
    document.documentElement.toggleAttribute('data-calm', inMeeting)
    return () => document.documentElement.removeAttribute('data-calm')
  }, [inMeeting])

  return (
    <div className="flex min-h-screen">
      {/* On air: while a session runs, a band of light travels along the top of every studio screen. */}
      {onAir && (
        <div
          aria-hidden="true"
          className={cn(
            'pointer-events-none fixed inset-x-0 top-0 z-50 h-[2px]',
            inMeeting
              ? 'bg-[linear-gradient(90deg,var(--color-live),var(--color-accent-500))]'
              : 'animate-onair bg-[linear-gradient(90deg,transparent,var(--color-live),var(--color-accent-500),transparent)] bg-[length:45%_100%] bg-no-repeat',
          )}
        />
      )}
      <aside className="sticky top-0 hidden h-screen w-[244px] shrink-0 flex-col border-r border-line bg-surface lg:flex">
        <Link to="/" className="flex h-16 items-center px-5" aria-label="NeuroPresence home">
          <Logo />
        </Link>

        <nav className="flex-1 overflow-y-auto px-3 pt-2" aria-label="Studio">
          {GROUPS.map((group, index) => (
            <div key={group.title}>
              <p className={cn('label-caps px-2.5 pb-2 text-ink-400', index > 0 && 'pt-6')}>{group.title}</p>
              <ul className="grid gap-0.5">
                {group.screens.map((screen) => (
                  <li key={screen.to}>
                    <SideLink {...screen} badge={screen.to === startHere ? 'Start here' : null} />
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="grid gap-2.5 px-5 pb-4">
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
        <div className="border-t border-line p-3">
          <UserMenu />
        </div>
      </aside>

      <div className="min-w-0 flex-1">
        <header className="border-b border-line bg-surface lg:hidden">
          <div className="flex h-14 items-center justify-between px-4">
            <Link to="/" aria-label="NeuroPresence home">
              <Logo />
            </Link>
            <div className="flex items-center gap-3">
              <EngineStatus connection={connection} className="hidden min-[420px]:inline-flex" />
              {user && (
                <Link to="/studio/account" aria-label={`Account of ${user.name}`}>
                  <Avatar name={user.name} className="size-8 rounded-[10px] text-[11px]" />
                </Link>
              )}
            </div>
          </div>
          {/* Without the sidebar, the screens are reached from here. */}
          <nav className="flex gap-1 overflow-x-auto px-3 pb-2" aria-label="Studio">
            {GROUPS.flatMap((group) => group.screens).map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end
                className={({ isActive }) =>
                  cn(
                    'flex h-8 shrink-0 items-center gap-2 rounded-control px-2.5 text-[13px] font-medium transition-colors',
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
        {/* Each screen arrives with a short rise and a soft focus, so moving between them does not snap. */}
        <motion.div
          key={pathname}
          initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: DURATION.base + 0.08, ease: EASE_OUT }}
        >
          <Outlet />
        </motion.div>
      </div>
    </div>
  )
}
