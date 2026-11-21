/**
 * Small effects that make the app feel alive, each doing one thing. All of them stop for people who ask their
 * system for less motion (the reduced-motion rule in app.css and MotionConfig in App).
 */
import { motion } from 'motion/react'
import { useRef, type PointerEvent, type ReactNode } from 'react'
import { cn } from '../lib/cn'

/** A card with a soft light that follows the pointer across it. */
export function SpotlightCard({ children, className }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const move = (event: PointerEvent<HTMLDivElement>) => {
    const box = ref.current?.getBoundingClientRect()
    if (!box) return
    ref.current!.style.setProperty('--spot-x', `${event.clientX - box.left}px`)
    ref.current!.style.setProperty('--spot-y', `${event.clientY - box.top}px`)
  }
  return (
    <div ref={ref} onPointerMove={move} className={cn('group/spot relative overflow-hidden', className)}>
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-0 opacity-0 transition-opacity duration-500 group-hover/spot:opacity-100"
        style={{ background: 'radial-gradient(420px circle at var(--spot-x, 50%) var(--spot-y, 0%), rgb(80 98 240 / 0.10), transparent 45%)' }}
      />
      <div className="relative z-[1] h-full">{children}</div>
    </div>
  )
}

/** A frame whose edge carries a slowly travelling band of light. */
export function GlowBorder({ children, className, radius = 22 }: { children: ReactNode; className?: string; radius?: number }) {
  return (
    <div className={cn('relative overflow-hidden p-px', className)} style={{ borderRadius: radius + 1 }}>
      <motion.div
        aria-hidden="true"
        className="absolute top-1/2 left-1/2 aspect-square w-[160%] -translate-x-1/2 -translate-y-1/2"
        style={{
          background:
            'conic-gradient(from 0deg, transparent 0deg, rgb(80 98 240 / 0.9) 50deg, rgb(159 174 255 / 0.6) 80deg, transparent 130deg, transparent 360deg)',
        }}
        animate={{ rotate: 360 }}
        transition={{ duration: 7, ease: 'linear', repeat: Infinity }}
      />
      <div aria-hidden="true" className="absolute inset-0 bg-line" style={{ borderRadius: radius + 1, opacity: 0.6 }} />
      <div className="relative" style={{ borderRadius: radius }}>
        {children}
      </div>
    </div>
  )
}

/** A row of items scrolling past without end; it stops while pointed at, and stands still for reduced motion. */
export function Marquee({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('group/marquee relative flex overflow-hidden [mask-image:linear-gradient(90deg,transparent,black_8%,black_92%,transparent)]', className)}>
      {[0, 1].map((copy) => (
        <div
          key={copy}
          aria-hidden={copy === 1 ? 'true' : undefined}
          className="flex shrink-0 animate-marquee items-center gap-3 pr-3 group-hover/marquee:[animation-play-state:paused]"
        >
          {children}
        </div>
      ))}
    </div>
  )
}

/** A status lamp that breathes: green when something is live, grey when it is not. */
export function LiveDot({ on, className }: { on: boolean; className?: string }) {
  return (
    <span className={cn('relative inline-grid size-2.5 place-items-center', className)} aria-hidden="true">
      {on && <span className="absolute inset-0 animate-ping rounded-full bg-good/60" />}
      <span className={cn('relative size-2 rounded-full', on ? 'bg-good' : 'bg-ink-300')} />
    </span>
  )
}
