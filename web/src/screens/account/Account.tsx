import { CheckCircle2, Globe, KeyRound, LifeBuoy, LogOut, MonitorSmartphone, ShieldAlert, Trash2, UserRound, XCircle, type LucideIcon } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { Avatar } from '../../app/UserMenu'
import { EASE_OUT, Stagger, StaggerItem } from '../../components/motion'
import { PageHeader } from '../../components/PageHeader'
import { RecoveryCode } from '../../components/RecoveryCode'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { Field, PasswordField } from '../../components/ui/Field'
import { Pill } from '../../components/ui/Pill'
import { useToast } from '../../components/ui/Toast'
import { useAccount } from '../../lib/account'
import { api, ApiError, type Devices } from '../../lib/api'
import { clockTime, dayLabel } from '../../lib/history'

const words = (error: unknown) => (error instanceof ApiError ? error.message : 'Something went wrong. Try again.')
const when = (at: number) => `${dayLabel(at)}, ${clockTime(at)}`
/** The same inside a sentence: "today", not "Today". */
const within = (words: string) => words.replace(/^(Today|Yesterday)/, (day) => day.toLowerCase())

/** The account: who it is, its password and recovery code, where it is signed in, and deleting it. */
export function Account() {
  const { user, signOut } = useAccount()
  useEffect(() => {
    document.title = 'Account · NeuroPresence'
  }, [])
  if (!user) return null
  return (
    <main className="mx-auto max-w-[980px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <PageHeader
        title="Account"
        actions={
          <Button icon={<LogOut className="size-3.5" />} onClick={signOut}>
            Sign out
          </Button>
        }
      >
        Your account on this computer. It keeps your face, your pictures, your sessions and your settings, apart from anyone else's.
      </PageHeader>

      <Card className="relative mt-6 overflow-hidden p-6">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -top-28 -right-16 size-[320px] rounded-full bg-[radial-gradient(circle,rgb(80_98_240/0.16),transparent_65%)]"
        />
        <div className="relative flex flex-wrap items-center gap-5">
          <Avatar name={user.name} className="size-16 rounded-[20px] text-[21px]" />
          <div className="min-w-0 flex-1">
            <p className="display truncate text-[22px] leading-tight font-semibold text-ink-950">{user.name}</p>
            <p className="mt-0.5 truncate text-[14px] text-ink-600">{user.email}</p>
            <p className="mt-1.5 text-[12.5px] text-ink-400">
              Made {within(dayLabel(user.created_at))}
              {user.last_login_at !== null && ` · last signed in ${within(when(user.last_login_at))}`}
            </p>
          </div>
        </div>
      </Card>

      <Stagger className="mt-4 grid items-start gap-4 lg:grid-cols-2" gap={0.06}>
        <StaggerItem>
          <Profile />
        </StaggerItem>
        <StaggerItem>
          <Password />
        </StaggerItem>
        <StaggerItem className="lg:col-span-2">
          <SignIns />
        </StaggerItem>
        <StaggerItem>
          <Recovery />
        </StaggerItem>
        <StaggerItem>
          <Delete />
        </StaggerItem>
      </Stagger>
    </main>
  )
}

function Section({ icon: Icon, title, hint, children, tone = 'neutral' }: { icon: LucideIcon; title: string; hint: string; children: ReactNode; tone?: 'neutral' | 'critical' }) {
  return (
    <Card className={tone === 'critical' ? 'border-critical/25 p-5' : 'p-5'}>
      <div className="flex items-start gap-3">
        <span className={`grid size-9 shrink-0 place-items-center rounded-[11px] ${tone === 'critical' ? 'bg-critical-wash text-critical-ink' : 'bg-ink-100 text-ink-700'}`}>
          <Icon className="size-4" strokeWidth={2} />
        </span>
        <div className="min-w-0">
          <h2 className="text-[14px] font-semibold text-ink-950">{title}</h2>
          <p className="mt-0.5 text-[12.5px] leading-snug text-ink-500">{hint}</p>
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </Card>
  )
}

function Profile() {
  const { user, enter } = useAccount()
  const toast = useToast()
  const [name, setName] = useState(user?.name ?? '')
  const [email, setEmail] = useState(user?.email ?? '')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const newEmail = !!user && email.trim().toLowerCase() !== user.email.toLowerCase()
  const changed = !!user && (name.trim() !== user.name || newEmail)

  const save = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setProblem(null)
    try {
      const reply = await api.changeAccount({ name, ...(newEmail ? { email, password } : {}) })
      enter(reply.user)
      setEmail(reply.user.email)
      setName(reply.user.name)
      setPassword('')
      toast.done('Saved')
    } catch (error) {
      setProblem(words(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Section icon={UserRound} title="Profile" hint="Your name, and the email you sign in with.">
      <form onSubmit={save} className="grid gap-3.5" noValidate>
        <Field label="Name" autoComplete="name" maxLength={60} value={name} onChange={(event) => setName(event.target.value)} />
        <Field label="Email" type="email" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} error={newEmail ? null : problem} />
        <AnimatePresence initial={false}>
          {newEmail && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.25, ease: EASE_OUT }}>
              <PasswordField label="Your password, to change the email" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} error={problem} />
            </motion.div>
          )}
        </AnimatePresence>
        <Button type="submit" variant="primary" className="justify-self-start" busy={busy} disabled={!changed || !name.trim() || (newEmail && !password)}>
          Save
        </Button>
      </form>
    </Section>
  )
}

function Password() {
  const toast = useToast()
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const save = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setProblem(null)
    try {
      await api.changePassword({ current, new: next })
      setCurrent('')
      setNext('')
      toast.done('Password changed. Other browsers were signed out.')
    } catch (error) {
      setProblem(words(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Section icon={KeyRound} title="Password" hint="Changing it signs every other browser out.">
      <form onSubmit={save} className="grid gap-3.5" noValidate>
        <PasswordField label="Current password" autoComplete="current-password" value={current} onChange={(event) => setCurrent(event.target.value)} error={problem} />
        <PasswordField label="New password" meter autoComplete="new-password" value={next} onChange={(event) => setNext(event.target.value)} />
        <Button type="submit" variant="primary" className="justify-self-start" busy={busy} disabled={!current || next.length < 8}>
          Change password
        </Button>
      </form>
    </Section>
  )
}

function SignIns() {
  const toast = useToast()
  const [found, setFound] = useState<Devices | null>(null)
  const [busy, setBusy] = useState(false)
  const load = useCallback(() => {
    api.devices().then(setFound, () => undefined)
  }, [])
  useEffect(load, [load])
  const others = found?.devices.filter((device) => !device.this).length ?? 0

  const out = async () => {
    setBusy(true)
    try {
      setFound(await api.signOutOthers())
      toast.done('Every other browser was signed out.')
    } catch (error) {
      toast.problem(words(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Section icon={MonitorSmartphone} title="Where you are signed in" hint="Each browser holds a random token; the database keeps only its hash. The latest tries at your password are beside it.">
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <ul className="divide-y divide-line rounded-panel border border-line">
            {found?.devices.map((device, index) => (
              <li key={`${device.created_at}-${index}`} className="flex items-center gap-3 px-3.5 py-3">
                <Globe className="size-4 shrink-0 text-ink-400" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[13px] font-semibold text-ink-950">{device.device ?? 'A browser'}</p>
                  <p className="truncate text-[12px] text-ink-500">Since {within(when(device.created_at))}</p>
                </div>
                {device.this ? <Pill tone="good">This browser</Pill> : <span className="text-[12px] text-ink-400">{when(device.last_seen_at)}</span>}
              </li>
            ))}
            {!found && <li className="h-14 animate-pulse bg-ink-50" />}
          </ul>
          {others > 0 && (
            <Button size="sm" className="mt-3" busy={busy} onClick={out}>
              Sign out everywhere else
            </Button>
          )}
        </div>
        <div>
          <p className="label-caps mb-2 text-ink-400">Latest tries</p>
          <ul className="grid gap-1.5">
            {found?.attempts.map((attempt, index) => (
              <li key={`${attempt.at}-${index}`} className="flex items-center gap-2.5 text-[12.5px]">
                {attempt.succeeded ? <CheckCircle2 className="size-4 text-good" /> : <XCircle className="size-4 text-critical" />}
                <span className="text-ink-800">{attempt.succeeded ? 'Signed in' : 'Did not match'}</span>
                <span className="ml-auto font-mono text-[11.5px] text-ink-400 tabular-nums">{when(attempt.at)}</span>
              </li>
            ))}
            {found && found.attempts.length === 0 && <li className="text-[12.5px] text-ink-500">None yet.</li>}
          </ul>
          <p className="mt-3 text-[12px] leading-snug text-ink-400">After five in a row that do not match, the email waits a quarter of an hour. The count survives a restart.</p>
        </div>
      </div>
    </Section>
  )
}

function Recovery() {
  const { user } = useAccount()
  const [password, setPassword] = useState('')
  const [code, setCode] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const make = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setProblem(null)
    try {
      setCode((await api.newRecoveryCode(password)).recovery_code)
      setPassword('')
    } catch (error) {
      setProblem(words(error))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Section icon={LifeBuoy} title="Recovery code" hint="It sets a new password when you forget yours. Making a new one ends the old one.">
      {code && user ? (
        <RecoveryCode code={code} email={user.email} done="Done" onDone={() => setCode(null)} />
      ) : (
        <form onSubmit={make} className="grid gap-3.5" noValidate>
          <PasswordField label="Your password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} error={problem} />
          <Button type="submit" className="justify-self-start" busy={busy} disabled={!password}>
            Make a new code
          </Button>
        </form>
      )}
    </Section>
  )
}

function Delete() {
  const account = useAccount()
  const [asking, setAsking] = useState(false)
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const remove = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setProblem(null)
    try {
      await account.remove(password)
    } catch (error) {
      setProblem(words(error))
      setBusy(false)
    }
  }

  return (
    <Section icon={ShieldAlert} tone="critical" title="Delete the account" hint="Your face signature, pictures, sessions, notes and settings are deleted from this computer. It cannot be undone.">
      {asking ? (
        <form onSubmit={remove} className="grid gap-3.5" noValidate>
          <PasswordField label="Your password, to be sure" autoFocus autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} error={problem} />
          <div className="flex gap-2">
            <Button type="submit" variant="danger" icon={<Trash2 className="size-4" />} busy={busy} disabled={!password}>
              Delete everything
            </Button>
            <Button variant="ghost" onClick={() => setAsking(false)}>
              Keep it
            </Button>
          </div>
        </form>
      ) : (
        <Button variant="danger" icon={<Trash2 className="size-4" />} onClick={() => setAsking(true)}>
          Delete account
        </Button>
      )}
    </Section>
  )
}
