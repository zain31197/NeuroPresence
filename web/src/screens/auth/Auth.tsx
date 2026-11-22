import { ArrowLeft, ArrowRight, CircleAlert, Sparkles } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { Logo } from '../../components/Logo'
import { EASE_OUT } from '../../components/motion'
import { RecoveryCode } from '../../components/RecoveryCode'
import { Button, Spinner } from '../../components/ui/Button'
import { Field, PasswordField } from '../../components/ui/Field'
import { safeNext, useAccount } from '../../lib/account'
import { api, ApiError, type User } from '../../lib/api'
import { AuthVisual } from './AuthVisual'

type Mode = 'sign-in' | 'sign-up' | 'reset'

const TITLE: Record<Mode, string> = { 'sign-in': 'Sign in', 'sign-up': 'Create your account', reset: 'Set a new password' }

/** Signing in, making an account, and setting a new password with the recovery code. One account, on this computer. */
export function Auth({ mode }: { mode: Mode }) {
  const account = useAccount()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const next = safeNext(params.get('next'))
  const suffix = params.get('next') ? `?next=${encodeURIComponent(next)}` : ''
  // An account that was just made, or whose password was just reset: its code is shown before anything else.
  const [fresh, setFresh] = useState<{ user: User; code: string } | null>(null)

  useEffect(() => {
    document.title = `${TITLE[mode]} · NeuroPresence`
  }, [mode])

  const done = (user: User) => {
    account.enter(user)
    navigate(next, { replace: true })
  }

  if (account.user && !fresh) return <Navigate to={next} replace />
  // The first person on this computer has no account to sign in to.
  if (account.first && mode !== 'sign-up') return <Navigate to={`/sign-up${suffix}`} replace />

  return (
    <div className="grid min-h-screen bg-surface lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:grid-cols-[minmax(0,1.02fr)_minmax(0,1fr)]">
      <main className="flex min-h-screen flex-col px-5 py-6 sm:px-10">
        <div className="flex items-center justify-between">
          <Link to="/" aria-label="NeuroPresence home">
            <Logo />
          </Link>
          <Link to="/" className="inline-flex items-center gap-1.5 text-[13px] font-medium text-ink-500 transition-colors hover:text-ink-950">
            <ArrowLeft className="size-3.5" />
            Back to the site
          </Link>
        </div>

        <div className="mx-auto flex w-full max-w-[420px] flex-1 flex-col justify-center py-10">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={fresh ? 'code' : mode}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.35, ease: EASE_OUT }}
            >
              {fresh ? (
                <RecoveryCode code={fresh.code} email={fresh.user.email} onDone={() => done(fresh.user)} />
              ) : account.loading ? (
                <div className="grid place-items-center py-20 text-ink-400">
                  <Spinner className="size-6" />
                </div>
              ) : mode === 'sign-up' ? (
                <SignUp first={account.first} claims={account.claims} suffix={suffix} onDone={(user, code) => setFresh({ user, code })} />
              ) : mode === 'reset' ? (
                <Reset suffix={suffix} onDone={(user, code) => setFresh({ user, code })} />
              ) : (
                <SignIn suffix={suffix} onDone={done} />
              )}
            </motion.div>
          </AnimatePresence>
          {account.unreachable && (
            <p role="alert" className="mt-6 flex gap-2.5 rounded-panel border border-line bg-ink-50 px-3.5 py-3 text-[13px] leading-snug text-ink-700">
              <CircleAlert className="mt-px size-4 shrink-0 text-ink-500" />
              <span>
                The engine does not answer. Start it with <code className="font-mono text-[12px]">python -m neuropresence.server</code>; this page goes on by itself.
              </span>
            </p>
          )}
        </div>
        <p className="text-center text-[12px] text-ink-400">Accounts live on this computer only. Nothing is sent anywhere.</p>
      </main>
      <AuthVisual />
    </div>
  )
}

function Heading({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mb-7">
      <h1 className="display text-[30px] leading-tight font-semibold text-ink-950">{title}</h1>
      <p className="mt-2 text-[14.5px] leading-relaxed text-ink-600">{children}</p>
    </div>
  )
}

/** One request at a time, and what went wrong with it in words. */
function useSubmit() {
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const submit = (work: () => Promise<void>) => async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setProblem(null)
    try {
      await work()
    } catch (error) {
      setProblem(error instanceof ApiError ? error.message : 'Something went wrong. Try again.')
    } finally {
      setBusy(false)
    }
  }
  return { busy, problem, submit, clear: () => setProblem(null) }
}

function Problem({ children }: { children: string | null }) {
  return (
    <AnimatePresence>
      {children && (
        <motion.p
          role="alert"
          className="flex gap-2.5 overflow-hidden rounded-panel border border-critical/25 bg-critical-wash px-3.5 py-3 text-[13px] leading-snug text-critical-ink"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.25, ease: EASE_OUT }}
        >
          <CircleAlert className="mt-px size-4 shrink-0" />
          {children}
        </motion.p>
      )}
    </AnimatePresence>
  )
}

function Remember({ checked, onChange }: { checked: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-2.5 text-[13.5px] text-ink-700">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="size-4 accent-[var(--color-ink-950)]" />
      Keep me signed in on this browser
    </label>
  )
}

const inline = 'font-semibold text-accent-700 underline-offset-4 hover:underline'

function SignIn({ suffix, onDone }: { suffix: string; onDone: (user: User) => void }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(true)
  const { busy, problem, submit } = useSubmit()
  return (
    <form onSubmit={submit(async () => onDone((await api.signIn({ email, password, remember })).user))} className="grid gap-4" noValidate>
      <div>
        <Heading title="Welcome back">Sign in to your pictures, your sessions and your studio as you left it.</Heading>
        <Problem>{problem}</Problem>
      </div>
      <Field label="Email" type="email" autoComplete="username" autoFocus required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" />
      <PasswordField
        label="Password"
        autoComplete="current-password"
        required
        value={password}
        onChange={(event) => setPassword(event.target.value)}
        aside={
          <Link to={`/reset${suffix}`} className={`text-[12.5px] ${inline}`}>
            Forgot it?
          </Link>
        }
      />
      <Remember checked={remember} onChange={setRemember} />
      <Button type="submit" variant="primary" size="lg" className="mt-1 w-full" busy={busy} disabled={!email || !password}>
        Sign in
        {!busy && <ArrowRight className="size-4" />}
      </Button>
      <p className="text-center text-[13.5px] text-ink-600">
        New on this computer?{' '}
        <Link to={`/sign-up${suffix}`} className={inline}>
          Create an account
        </Link>
      </p>
    </form>
  )
}

function SignUp({ first, claims, suffix, onDone }: { first: boolean; claims: boolean; suffix: string; onDone: (user: User, code: string) => void }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [remember, setRemember] = useState(true)
  const { busy, problem, submit } = useSubmit()
  return (
    <form
      onSubmit={submit(async () => {
        const reply = await api.signUp({ email, name, password, remember })
        onDone(reply.user, reply.recovery_code)
      })}
      className="grid gap-4"
      noValidate
    >
      <div>
        <Heading title={first ? 'Create the first account' : 'Create your account'}>
          {first ? 'This computer has no account yet. Yours keeps your face, your pictures, your sessions and your settings.' : 'Your own face, pictures, sessions and settings, apart from everyone else on this computer.'}
        </Heading>
        {claims && (
          <p className="mb-4 flex gap-2.5 rounded-panel border border-accent-200 bg-accent-50 px-3.5 py-3 text-[13px] leading-snug text-accent-800">
            <Sparkles className="mt-px size-4 shrink-0" />
            What is already on this computer, the enrolled face and picture and the history, becomes yours. Nothing is lost.
          </p>
        )}
        <Problem>{problem}</Problem>
      </div>
      <Field label="Name" autoComplete="name" autoFocus required maxLength={60} value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" />
      <Field label="Email" type="email" autoComplete="username" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" hint="Only to sign in with. No email is ever sent." />
      <PasswordField label="Password" meter autoComplete="new-password" required value={password} onChange={(event) => setPassword(event.target.value)} />
      <Remember checked={remember} onChange={setRemember} />
      <Button type="submit" variant="primary" size="lg" className="mt-1 w-full" busy={busy} disabled={!name.trim() || !email || password.length < 8}>
        Create account
        {!busy && <ArrowRight className="size-4" />}
      </Button>
      {!first && (
        <p className="text-center text-[13.5px] text-ink-600">
          Have an account?{' '}
          <Link to={`/sign-in${suffix}`} className={inline}>
            Sign in
          </Link>
        </p>
      )}
    </form>
  )
}

function Reset({ suffix, onDone }: { suffix: string; onDone: (user: User, code: string) => void }) {
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [password, setPassword] = useState('')
  const { busy, problem, submit } = useSubmit()
  return (
    <form
      onSubmit={submit(async () => {
        const reply = await api.resetPassword({ email, code, password })
        onDone(reply.user, reply.recovery_code)
      })}
      className="grid gap-4"
      noValidate
    >
      <div>
        <Heading title="Set a new password">With the recovery code you were given when you made your account. You will get a new one.</Heading>
        <Problem>{problem}</Problem>
      </div>
      <Field label="Email" type="email" autoComplete="username" autoFocus required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" />
      <Field
        label="Recovery code"
        required
        autoComplete="one-time-code"
        spellCheck={false}
        value={code}
        onChange={(event) => setCode(event.target.value.toUpperCase())}
        placeholder="XXXX-XXXX-XXXX-XXXX-XXXX"
        style={{ fontFamily: 'var(--font-mono)' }}
      />
      <PasswordField label="New password" meter autoComplete="new-password" required value={password} onChange={(event) => setPassword(event.target.value)} />
      <Button type="submit" variant="primary" size="lg" className="mt-1 w-full" busy={busy} disabled={!email || !code || password.length < 8}>
        Set the new password
      </Button>
      <p className="text-center text-[13.5px] text-ink-600">
        Remembered it?{' '}
        <Link to={`/sign-in${suffix}`} className={inline}>
          Sign in
        </Link>
      </p>
    </form>
  )
}
