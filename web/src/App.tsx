import { MotionConfig } from 'motion/react'
import { createBrowserRouter, Navigate, RouterProvider, useLocation, useRouteError } from 'react-router-dom'
import { AppShell } from './app/AppShell'
import { Logo } from './components/Logo'
import { Spinner } from './components/ui/Button'
import { ToastProvider } from './components/ui/Toast'
import { TooltipProvider } from './components/ui/Tooltip'
import { AccountProvider, useAccount } from './lib/account'
import { StudioProvider } from './lib/studio'
import { Account } from './screens/account/Account'
import { Auth } from './screens/auth/Auth'
import { Benchmarks } from './screens/benchmarks/Benchmarks'
import { Data } from './screens/data/Data'
import { Enrolment } from './screens/enrolment/Enrolment'
import { Landing } from './screens/landing/Landing'
import { Overview } from './screens/overview/Overview'
import { Security } from './screens/security/Security'
import { Sessions } from './screens/sessions/Sessions'
import { LiveStudio } from './screens/studio/LiveStudio'
import { System } from './screens/system/System'

/** Shown if a screen fails to draw, so a fault never leaves a blank page. */
function Crash() {
  const error = useRouteError()
  return (
    <main className="grid min-h-screen place-items-center bg-canvas p-6">
      <div className="max-w-[440px] rounded-card border border-line bg-surface p-7 shadow-card">
        <h1 className="display text-[20px] font-semibold">This screen stopped working</h1>
        <p className="mt-2 text-[14px] leading-relaxed text-ink-600">
          Reload the page. If it happens again, the web app and the engine may be different versions: rebuild the app with{' '}
          <code className="rounded-[5px] bg-ink-100 px-1.5 py-0.5 font-mono text-[12.5px]">npm run build</code> and restart the engine.
        </p>
        <p className="mt-4 rounded-panel bg-ink-50 p-3 font-mono text-[12px] break-words text-ink-600">
          {error instanceof Error ? error.message : String(error)}
        </p>
        <button
          type="button"
          onClick={() => location.reload()}
          className="mt-5 inline-flex h-9 items-center rounded-control bg-ink-950 px-3.5 text-[13.5px] font-medium text-white hover:bg-ink-800"
        >
          Reload
        </button>
      </div>
    </main>
  )
}

/** While who is signed in is not known: the engine is being asked, or it is not there to ask. */
function Opening({ unreachable }: { unreachable: boolean }) {
  return (
    <main className="grid min-h-screen place-items-center bg-canvas p-6">
      <div className="flex max-w-[420px] flex-col items-center text-center">
        <Logo />
        <Spinner className="mt-7 size-5 text-ink-400" />
        {unreachable && (
          <p role="status" className="mt-5 text-[13.5px] leading-relaxed text-ink-600">
            The engine is not running. Start it in a terminal with{' '}
            <code className="rounded-[5px] bg-ink-100 px-1.5 py-0.5 font-mono text-[12px]">python -m neuropresence.server</code>. This page
            goes on by itself.
          </p>
        )}
      </div>
    </main>
  )
}

/** Everything under /studio is someone's, so it needs a person signed in; and it shares one live connection to the engine. */
function Studio() {
  const { user, first, loading, unreachable, removed } = useAccount()
  const { pathname } = useLocation()
  if (!user) {
    if (removed) return <Navigate to="/" replace />
    if (loading || unreachable) return <Opening unreachable={unreachable} />
    // A computer without an account starts with making one; signing in is one press away from there.
    return <Navigate to={`${first ? '/sign-up' : '/sign-in'}?next=${encodeURIComponent(pathname)}`} replace />
  }
  return (
    <StudioProvider>
      <ToastProvider>
        <AppShell />
      </ToastProvider>
    </StudioProvider>
  )
}

const router = createBrowserRouter([
  { path: '/', element: <Landing />, errorElement: <Crash /> },
  { path: '/sign-in', element: <Auth mode="sign-in" />, errorElement: <Crash /> },
  { path: '/sign-up', element: <Auth mode="sign-up" />, errorElement: <Crash /> },
  { path: '/reset', element: <Auth mode="reset" />, errorElement: <Crash /> },
  {
    path: '/studio',
    element: <Studio />,
    errorElement: <Crash />,
    children: [
      { index: true, element: <Overview /> },
      { path: 'live', element: <LiveStudio /> },
      { path: 'enrolment', element: <Enrolment /> },
      { path: 'sessions', element: <Sessions /> },
      { path: 'security', element: <Security /> },
      { path: 'data', element: <Data /> },
      { path: 'account', element: <Account /> },
      { path: 'benchmarks', element: <Benchmarks /> },
      { path: 'system', element: <System /> },
    ],
  },
  { path: '*', element: <Navigate to="/" replace /> },
])

export function App() {
  return (
    // People who ask their system for less motion get every animation's final state at once.
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={250}>
        <AccountProvider>
          <RouterProvider router={router} />
        </AccountProvider>
      </TooltipProvider>
    </MotionConfig>
  )
}
