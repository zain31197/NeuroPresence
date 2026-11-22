/**
 * The app's motion, in one place. Every animation uses these timings and this easing, so the whole app
 * moves with one rhythm. Things arrive with a short rise and settle; nothing bounces. People who ask
 * their system for less motion get the final state at once (MotionConfig reducedMotion="user" in App).
 */
import { animate, motion, useInView, useReducedMotion, type Variants } from 'motion/react'
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { cn } from '../lib/cn'

/** Decelerating: quick to arrive, gentle to settle. */
export const EASE_OUT = [0.16, 1, 0.3, 1] as const
export const DURATION = { fast: 0.18, base: 0.32, slow: 0.6 } as const

const rise: Variants = {
  hidden: { opacity: 0, y: 14 },
  shown: { opacity: 1, y: 0, transition: { duration: DURATION.slow, ease: EASE_OUT } },
}

/** A block that rises into place the first time it scrolls into view. */
export function Reveal({ children, className, delay = 0, as = 'div' }: { children: ReactNode; className?: string; delay?: number; as?: 'div' | 'section' | 'li' }) {
  const Tag = motion[as]
  return (
    <Tag
      className={className}
      initial="hidden"
      whileInView="shown"
      viewport={{ once: true, margin: '0px 0px -12% 0px' }}
      variants={{ hidden: rise.hidden, shown: { ...rise.shown, transition: { duration: DURATION.slow, ease: EASE_OUT, delay } } }}
    >
      {children}
    </Tag>
  )
}

/** A group whose items arrive one after another, 60 ms apart, once in view. Wrap each item in StaggerItem. */
export function Stagger({ children, className, as = 'div', gap = 0.06 }: { children: ReactNode; className?: string; as?: 'div' | 'ul' | 'ol' | 'dl'; gap?: number }) {
  const Tag = motion[as]
  return (
    <Tag
      className={className}
      initial="hidden"
      whileInView="shown"
      viewport={{ once: true, margin: '0px 0px -10% 0px' }}
      variants={{ hidden: {}, shown: { transition: { staggerChildren: gap } } }}
    >
      {children}
    </Tag>
  )
}

export function StaggerItem({ children, className, as = 'div' }: { children: ReactNode; className?: string; as?: 'div' | 'li' }) {
  const Tag = motion[as]
  return (
    <Tag className={className} variants={rise}>
      {children}
    </Tag>
  )
}

/**
 * A figure that counts up from zero the first time it is seen. Shows the final value at once when
 * motion is reduced, and for anything that is not a finite number.
 */
export function CountUp({ value, format, className, duration = 1.1 }: { value: number; format: (value: number) => string; className?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const seen = useInView(ref, { once: true, margin: '0px 0px -10% 0px' })
  const reduced = useReducedMotion()
  const [shown, setShown] = useState(reduced || !Number.isFinite(value) ? value : 0)

  useEffect(() => {
    if (!seen || reduced || !Number.isFinite(value)) {
      setShown(value)
      return
    }
    const controls = animate(0, value, { duration, ease: EASE_OUT, onUpdate: setShown })
    return () => controls.stop()
  }, [seen, reduced, value, duration])

  return (
    <span ref={ref} className={cn('tabular-nums', className)}>
      {format(shown)}
    </span>
  )
}

/**
 * A headline whose words rise into place one after another. The text stays one readable string for screen readers.
 * wordStyle colours a word by how far along the line it is: a gradient has to be on the words themselves, because
 * each is a layer of its own while it moves, and a browser leaves such layers out of a gradient on their parent.
 */
export function WordsIn({
  text,
  className,
  delay = 0,
  wordStyle,
}: {
  text: string
  className?: string
  delay?: number
  wordStyle?: (from: number, to: number) => CSSProperties
}) {
  const words = text.split(' ')
  return (
    <span className={className} aria-label={text}>
      {words.map((word, index) => (
        <motion.span
          key={`${word}-${index}`}
          aria-hidden="true"
          className="inline-block whitespace-pre"
          style={wordStyle?.(index / words.length, (index + 1) / words.length)}
          initial={{ opacity: 0, y: '0.35em', filter: 'blur(6px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: DURATION.slow, ease: EASE_OUT, delay: delay + index * 0.045 }}
        >
          {index < words.length - 1 ? `${word} ` : word}
        </motion.span>
      ))}
    </span>
  )
}
