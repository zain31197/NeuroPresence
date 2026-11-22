import { Database, Fingerprint, History, ImageIcon, KeyRound, Lock, NotebookPen, ShieldCheck, SlidersHorizontal, type LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'
import { LogoMark } from '../../components/Logo'
import { EASE_OUT } from '../../components/motion'

// What an account keeps. Around the mark in this order, clockwise from the top.
const KEPT: { icon: LucideIcon; label: string }[] = [
  { icon: Fingerprint, label: 'Verified face' },
  { icon: ImageIcon, label: 'Meeting pictures' },
  { icon: History, label: 'Sessions' },
  { icon: NotebookPen, label: 'Notes' },
  { icon: ShieldCheck, label: 'Checks' },
  { icon: SlidersHorizontal, label: 'Settings' },
]

const PROMISES: { icon: LucideIcon; title: string; text: string }[] = [
  { icon: Lock, title: 'On this computer only', text: 'No server anywhere knows about your account, and no email is ever sent.' },
  { icon: KeyRound, title: 'Your password is never kept', text: 'Only a salted scrypt hash of it, made to be slow to guess.' },
  { icon: Database, title: 'A real database of your own', text: 'Every row belongs to your account, and goes with it when you delete it.' },
]

/** The dark half of the sign-in pages: what an account keeps, going round the mark, and how it is kept. */
export function AuthVisual() {
  return (
    <aside className="relative hidden overflow-hidden bg-ink-950 text-white lg:flex lg:flex-col lg:justify-between lg:p-12 xl:p-16">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_70%_20%,rgb(80_98_240/0.45),transparent_70%),radial-gradient(50%_45%_at_15%_85%,rgb(59_75_224/0.3),transparent_70%)]"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 [background-image:radial-gradient(rgb(255_255_255/0.14)_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_70%_60%_at_50%_40%,black,transparent)]"
      />

      <div className="relative mx-auto grid aspect-square w-full max-w-[440px] flex-none place-items-center">
        {[0.46, 0.74, 1].map((size, index) => (
          <motion.span
            key={size}
            aria-hidden="true"
            className="absolute rounded-full border border-white/12"
            style={{ width: `${size * 100}%`, height: `${size * 100}%` }}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.9, ease: EASE_OUT, delay: 0.1 + index * 0.12 }}
          />
        ))}
        {/* A pulse leaving the mark, as a row is written. */}
        <motion.span
          aria-hidden="true"
          className="absolute size-[30%] rounded-full border border-accent-300/60"
          animate={{ scale: [1, 3.2], opacity: [0.7, 0] }}
          transition={{ duration: 3.2, ease: 'easeOut', repeat: Infinity }}
        />
        <motion.div
          className="relative grid size-[104px] place-items-center rounded-[30px] bg-white/10 ring-1 ring-white/20 backdrop-blur-md"
          initial={{ scale: 0.7, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 220, damping: 18, delay: 0.2 }}
        >
          <LogoMark className="size-14" />
        </motion.div>
        {KEPT.map(({ icon: Icon, label }, index) => {
          const turn = (index / KEPT.length) * 2 * Math.PI - Math.PI / 2
          return (
            <motion.div
              key={label}
              className="absolute"
              style={{ left: `${50 + 40 * Math.cos(turn)}%`, top: `${50 + 40 * Math.sin(turn)}%` }}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, ease: EASE_OUT, delay: 0.45 + index * 0.09 }}
            >
              <motion.span
                className="inline-flex -translate-x-1/2 -translate-y-1/2 items-center gap-2 rounded-full border border-white/15 bg-ink-900/80 px-3 py-2 text-[12.5px] font-medium whitespace-nowrap shadow-[0_12px_30px_-12px_rgb(0_0_0/0.6)] backdrop-blur"
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 4.5 + (index % 3) * 0.7, ease: 'easeInOut', repeat: Infinity, delay: index * 0.35 }}
              >
                <Icon className="size-3.5 text-accent-300" strokeWidth={2} />
                {label}
              </motion.span>
            </motion.div>
          )
        })}
      </div>

      <div className="relative mt-10">
        <p className="display max-w-[20ch] text-[30px] leading-[1.12] font-semibold">Your presence, and everything you keep of it.</p>
        <ul className="mt-7 grid gap-4">
          {PROMISES.map(({ icon: Icon, title, text }, index) => (
            <motion.li
              key={title}
              className="flex gap-3.5"
              initial={{ opacity: 0, x: 14 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, ease: EASE_OUT, delay: 0.7 + index * 0.1 }}
            >
              <span className="grid size-9 shrink-0 place-items-center rounded-[11px] bg-white/10 ring-1 ring-white/12">
                <Icon className="size-4 text-accent-300" strokeWidth={2} />
              </span>
              <span>
                <span className="block text-[14px] font-semibold">{title}</span>
                <span className="mt-0.5 block text-[13px] leading-snug text-white/65">{text}</span>
              </span>
            </motion.li>
          ))}
        </ul>
      </div>
    </aside>
  )
}
