import { useEffect, useRef, useState } from 'react'

/**
 * A number that glides to each new value instead of jumping. The engine reports four times a
 * second, and a figure that snaps that often reads as noise; eased, it reads as a needle.
 * Returns the value itself when there is none yet, when it first arrives, and for people who
 * have asked for less motion.
 */
export function useEased(value: number | null | undefined, ms = 260): number | null | undefined {
  const [shown, setShown] = useState(value)
  const from = useRef(value)
  const frame = useRef(0)

  useEffect(() => {
    cancelAnimationFrame(frame.current)
    const start = from.current
    const still = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (value === null || value === undefined || start === null || start === undefined || still || start === value) {
      from.current = value
      setShown(value)
      return
    }
    const began = performance.now()
    const step = (now: number) => {
      const t = Math.min(1, (now - began) / ms)
      const eased = 1 - (1 - t) ** 3
      const next = start + (value - start) * eased
      from.current = next
      setShown(next)
      if (t < 1) frame.current = requestAnimationFrame(step)
    }
    frame.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame.current)
  }, [value, ms])

  return shown
}
