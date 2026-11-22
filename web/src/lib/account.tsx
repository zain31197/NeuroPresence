/** Who is signed in on this browser, for every page. */

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api, SIGNED_OUT_EVENT, type AccountState, type User } from './api'

interface Account extends AccountState {
  /** Nothing is known yet: the engine has not been asked, or has not answered. */
  loading: boolean
  /** The engine does not answer, so who is signed in cannot be known. */
  unreachable: boolean
  /** Signed in, or the account changed: what the engine replied with. */
  enter: (user: User) => void
  signOut: () => Promise<void>
  /** Deletes the account, and everything of it. */
  remove: (password: string) => Promise<void>
  /** The account was deleted: there is nothing to sign back in to, so the studio leaves to the site. */
  removed: boolean
  /** Ask the engine again. */
  refresh: () => Promise<void>
}

const NO_ONE: AccountState = { user: null, first: false, claims: false }
const AccountContext = createContext<Account | null>(null)

export function AccountProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AccountState | null>(null)
  const [unreachable, setUnreachable] = useState(false)
  const [removed, setRemoved] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setState(await api.account())
      setUnreachable(false)
    } catch {
      setUnreachable(true)
    }
  }, [])

  useEffect(() => {
    refresh()
    const out = () => setState((now) => (now?.user ? NO_ONE : now))
    window.addEventListener(SIGNED_OUT_EVENT, out)
    return () => window.removeEventListener(SIGNED_OUT_EVENT, out)
  }, [refresh])

  // While the engine does not answer, ask again now and then: it may only just be starting.
  useEffect(() => {
    if (!unreachable) return
    const timer = window.setInterval(refresh, 3000)
    return () => window.clearInterval(timer)
  }, [unreachable, refresh])

  const enter = useCallback((user: User) => {
    setRemoved(false)
    setState({ user, first: false, claims: false })
  }, [])
  const signOut = useCallback(async () => {
    try {
      await api.signOut()
    } finally {
      setState(NO_ONE)
    }
  }, [])

  const remove = useCallback(
    async (password: string) => {
      setRemoved(true) // before asking: the engine says "signed out" on the live connection first
      try {
        await api.deleteAccount(password)
      } catch (error) {
        setRemoved(false)
        throw error
      }
      await refresh()
    },
    [refresh],
  )

  const value = useMemo(
    () => ({ ...(state ?? NO_ONE), loading: state === null && !unreachable, unreachable, enter, signOut, remove, removed, refresh }),
    [state, unreachable, enter, signOut, remove, removed, refresh],
  )
  return <AccountContext.Provider value={value}>{children}</AccountContext.Provider>
}

export function useAccount(): Account {
  const account = useContext(AccountContext)
  if (!account) throw new Error('useAccount must be used inside <AccountProvider>.')
  return account
}

/** "Zain Shahid" as "ZS", for the round badge. */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean)
  if (words.length === 0) return '?'
  const letters = words.length === 1 ? words[0].slice(0, 2) : words[0][0] + words[words.length - 1][0]
  return letters.toUpperCase()
}

/** Where to go after signing in: the page that was asked for, and only ever one of the studio's. */
export function safeNext(next: string | null): string {
  return next && /^\/studio(\/|\?|$)/.test(next) ? next : '/studio'
}

export type Strength = 0 | 1 | 2 | 3 | 4

/** How hard a password is to guess, from 0 to 4, by its length first and its variety second. Only a hint: the engine decides. */
export function passwordStrength(password: string): Strength {
  if (!password) return 0
  const kinds = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((kind) => kind.test(password)).length
  const distinct = new Set(password).size
  if (password.length < 8 || distinct < 4) return 1
  const points = (password.length >= 10 ? 1 : 0) + (password.length >= 14 ? 1 : 0) + (kinds >= 2 ? 1 : 0) + (kinds >= 3 ? 1 : 0)
  return Math.min(4, 1 + Math.max(1, points)) as Strength
}

/** "1.4 MB", "820 KB", "0 B". */
export function bytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`
  return `${(size / 1024 / 1024).toFixed(size < 10 * 1024 * 1024 ? 1 : 0)} MB`
}
