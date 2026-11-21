import { useEffect, useState } from 'react'

/** The time now, in milliseconds, read again every `every` ms while `ticking`; for a duration or a wait that counts. */
export function useNow(ticking: boolean, every = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    setNow(Date.now())
    if (!ticking) return
    const timer = window.setInterval(() => setNow(Date.now()), every)
    return () => window.clearInterval(timer)
  }, [ticking, every])
  return now
}
