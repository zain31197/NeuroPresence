import { createBrowserRouter, Navigate, RouterProvider, useRouteError } from 'react-router-dom'
import { AppShell } from './app/AppShell'
import { ToastProvider } from './components/ui/Toast'
import { TooltipProvider } from './components/ui/Tooltip'
import { StudioProvider } from './lib/studio'
import { Enrolment } from './screens/enrolment/Enrolment'
import { Landing } from './screens/landing/Landing'
import { LiveStudio } from './screens/studio/LiveStudio'

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

/** Everything under /studio shares one live connection to the engine. */
function Studio() {
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
  {
    path: '/studio',
    element: <Studio />,
    errorElement: <Crash />,
    children: [
      { index: true, element: <LiveStudio /> },
      { path: 'enrolment', element: <Enrolment /> },
    ],
  },
  { path: '*', element: <Navigate to="/" replace /> },
])

export function App() {
  return (
    <TooltipProvider delayDuration={250}>
      <RouterProvider router={router} />
    </TooltipProvider>
  )
}
