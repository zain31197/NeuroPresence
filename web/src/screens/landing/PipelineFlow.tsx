import { Activity, Fingerprint, ScanFace, ShieldCheck, Sparkles, Video, type LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'
import { EASE_OUT } from '../../components/motion'

const ICONS: Record<string, LucideIcon> = {
  capture: ScanFace,
  motion: Activity,
  reenactment: Sparkles,
  identity: Fingerprint,
  consent: ShieldCheck,
  virtual_camera: Video,
}

const PASS_SECONDS = 3.2 // one frame's trip along the chain, drawn slower than life so the eye can follow it

/**
 * The six stages as one chain, with a pulse of light passing along it the way each camera frame does.
 * Each stage lights up as the pulse reaches it.
 */
export function PipelineFlow({ stages }: { stages: { key: string; name: string }[] }) {
  return (
    <div className="relative overflow-hidden rounded-card border border-line bg-surface px-5 pt-7 pb-6 shadow-card sm:px-8">
      {/* The line the stages sit on, and the pulse that travels along it. Wide screens only: on narrow ones the stages wrap. */}
      <div aria-hidden="true" className="absolute top-[52px] right-[8%] left-[8%] hidden h-[2px] rounded-full bg-line lg:block">
        <motion.div
          className="h-full rounded-full bg-[linear-gradient(90deg,transparent,var(--color-accent-500),transparent)] bg-[length:28%_100%] bg-no-repeat"
          animate={{ backgroundPosition: ['-30% 0', '130% 0'] }}
          transition={{ duration: PASS_SECONDS, ease: 'linear', repeat: Infinity }}
        />
      </div>
      <ol className="relative grid grid-cols-2 gap-x-4 gap-y-7 sm:grid-cols-3 lg:grid-cols-6">
        {stages.map((stage, index) => {
          const Icon = ICONS[stage.key] ?? Sparkles
          return (
            <motion.li
              key={stage.key}
              className="flex flex-col items-center text-center"
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, ease: EASE_OUT, delay: index * 0.07 }}
            >
              <span className="relative grid size-[50px] place-items-center rounded-[15px] border border-line bg-surface text-ink-900 shadow-control">
                {/* The pulse arriving: a ring that swells and fades, in turn along the chain. */}
                <motion.span
                  aria-hidden="true"
                  className="absolute inset-0 rounded-[15px] ring-2 ring-accent-400"
                  animate={{ opacity: [0, 0.9, 0], scale: [0.92, 1.12, 1.2] }}
                  transition={{ duration: 0.9, ease: 'easeOut', repeat: Infinity, repeatDelay: PASS_SECONDS - 0.9, delay: (index / Math.max(1, stages.length - 1)) * (PASS_SECONDS * 0.78) }}
                />
                <Icon className="size-[22px]" strokeWidth={1.75} />
              </span>
              <span className="label-caps mt-3 text-ink-400 tabular-nums">{String(index + 1).padStart(2, '0')}</span>
              <span className="mt-1 text-[14px] leading-snug font-semibold text-ink-950">{stage.name}</span>
            </motion.li>
          )
        })}
      </ol>
    </div>
  )
}
