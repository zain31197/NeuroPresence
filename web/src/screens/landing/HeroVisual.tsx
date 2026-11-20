import { Fingerprint, ScanFace, ShieldCheck, Video } from 'lucide-react'
import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { CountUp, EASE_OUT } from '../../components/motion'
import type { Benchmark } from '../../lib/api'
import { cn } from '../../lib/cn'
import { FaceMeshCanvas } from './FaceMeshCanvas'

/**
 * The idea in one picture, drawn as the product itself: a Live Studio window in which the face in front of
 * the camera drives the enrolled picture. The faces are an illustration; the figures under them are the
 * latest benchmark saved on this computer, or nothing when there is none.
 */
export function HeroVisual({ benchmark }: { benchmark: Benchmark | null }) {
  const summary = benchmark?.summary
  const readouts = [
    { value: summary?.fps, format: (v: number) => v.toFixed(1), unit: 'fps', label: 'frame rate' },
    { value: summary?.pipeline_ms, format: (v: number) => v.toFixed(0), unit: 'ms', label: 'per frame' },
    { value: summary?.csim_self_reenactment, format: (v: number) => v.toFixed(2), unit: 'CSIM', label: 'identity match' },
  ].filter((readout): readout is typeof readout & { value: number } => typeof readout.value === 'number')
  const identity = summary?.csim_self_reenactment

  return (
    <div className="relative">
      {/* A soft ultramarine glow behind the window, so it sits in light rather than on the page. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -inset-x-8 -inset-y-10 rounded-[48px] bg-[radial-gradient(60%_55%_at_60%_45%,rgb(80_98_240/0.22),transparent_70%)] blur-2xl"
      />
      <figure className="relative overflow-hidden rounded-[22px] border border-line bg-surface shadow-hero">
        <div className="flex h-10 items-center gap-3 border-b border-line bg-ink-50/70 px-4">
          <span className="flex gap-1.5" aria-hidden="true">
            {[0, 1, 2].map((dot) => (
              <span key={dot} className="size-2.5 rounded-full bg-ink-200" />
            ))}
          </span>
          <span className="label-caps mx-auto text-ink-500">Live Studio</span>
          <span className="inline-flex h-[22px] items-center gap-1.5 rounded-full bg-surface px-2 text-[11px] font-semibold text-ink-950 shadow-control">
            <span className="size-1.5 animate-lamp rounded-full bg-live" />
            Live
          </span>
        </div>

        <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-2 p-2.5">
          <Panel label="Camera" tone="plain">
            <FaceMeshCanvas
              className="size-full"
              meshColor="rgb(104 112 131 / 0.42)"
              contourColor="rgb(20 23 30 / 0.9)"
              label="A face drawn from tracked landmarks, turning and talking."
            />
            <ScanLine />
            <span className="absolute bottom-2.5 left-2.5 inline-flex h-[22px] items-center gap-1.5 rounded-full bg-surface/95 px-2 text-[11px] font-medium text-ink-800 shadow-control">
              <ScanFace className="size-3 text-good" />
              478 points
            </span>
          </Panel>

          <FlowArrow />

          <Panel label="Output" tone="accent">
            <FaceMeshCanvas
              className="size-full"
              meshColor="rgb(80 98 240 / 0.45)"
              contourColor="rgb(47 59 189 / 0.95)"
              delay={0.13}
              label="The enrolled picture's face following the same movement a moment later."
            />
            {/* The mark every output frame carries, where the engine puts it: the middle of the lower edge. */}
            <span className="absolute bottom-2.5 left-1/2 -translate-x-1/2 rounded-[5px] bg-ink-950/80 px-1.5 py-0.5 text-[9.5px] font-semibold tracking-wide whitespace-nowrap text-white">
              AI reenacted
            </span>
          </Panel>
        </div>

        <figcaption className="border-t border-line px-4 pt-3 pb-3.5">
          {readouts.length > 0 ? (
            <>
              <p className="label-caps text-ink-400">Latest benchmark on this computer</p>
              <dl className="mt-2.5 grid grid-cols-3 divide-x divide-line">
                {readouts.map((readout) => (
                  <div key={readout.label} className="px-3 first:pl-0 last:pr-0">
                    <dd className="flex items-baseline gap-1">
                      <CountUp value={readout.value} format={readout.format} className="display text-[21px] leading-none font-semibold text-ink-950" />
                      <span className="text-[12px] font-medium text-ink-500">{readout.unit}</span>
                    </dd>
                    <dt className="mt-1 text-[12px] text-ink-500">{readout.label}</dt>
                  </div>
                ))}
              </dl>
            </>
          ) : (
            <p className="label-caps text-ink-500">Illustration: pose, lips, expression</p>
          )}
        </figcaption>
      </figure>

      {/* What happens around the picture, as the Live Studio reports it. Hidden on narrow screens, where they would cover it. */}
      <Floating className="bottom-[34%] -left-9" delay={0.5} drift={5.5}>
        <ShieldCheck className="size-4 text-good" />
        Liveness check passed
      </Floating>
      <Floating className="top-[18%] -right-7" delay={0.75} drift={6.5}>
        <Fingerprint className="size-4 text-accent-600" />
        {typeof identity === 'number' ? (
          <>
            Identity <span className="font-mono tabular-nums">{identity.toFixed(2)}</span>
          </>
        ) : (
          'Identity checked'
        )}
      </Floating>
      <Floating className="-bottom-5 left-12" delay={1} drift={6}>
        <Video className="size-4 text-ink-700" />
        Sent to Google Meet
      </Floating>
    </div>
  )
}

function Panel({ label, tone, children }: { label: string; tone: 'plain' | 'accent'; children: ReactNode }) {
  return (
    <div className={cn('relative aspect-[4/5] overflow-hidden rounded-[16px]', tone === 'accent' ? 'bg-accent-50' : 'bg-ink-50')}>
      {children}
      <span className="label-caps absolute top-2.5 left-2.5 rounded-chip bg-surface px-2 py-1 text-ink-700 shadow-control">{label}</span>
    </div>
  )
}

/** A band of light passing over the camera picture, as the tracker reads it. */
function ScanLine() {
  return (
    <motion.div
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-0 top-0 h-full"
      initial={{ y: '-12%' }}
      animate={{ y: ['-12%', '96%'] }}
      transition={{ duration: 3.4, ease: 'easeInOut', repeat: Infinity, repeatType: 'mirror' }}
    >
      <div className="h-10 bg-[linear-gradient(180deg,transparent,rgb(90_235_120/0.16)_70%,rgb(90_235_120/0.55)_96%,transparent)]" />
    </motion.div>
  )
}

/** Frames travelling from the camera to the picture. */
function FlowArrow() {
  return (
    <div className="flex w-7 flex-col items-center gap-1.5" aria-hidden="true">
      {[0, 1, 2].map((dot) => (
        <motion.span
          key={dot}
          className="size-1.5 rounded-full bg-accent-500"
          animate={{ x: [-8, 8], opacity: [0, 1, 0] }}
          transition={{ duration: 1.4, ease: 'linear', repeat: Infinity, delay: dot * 0.22 }}
        />
      ))}
    </div>
  )
}

function Floating({ className, delay, drift, children }: { className: string; delay: number; drift: number; children: ReactNode }) {
  return (
    <motion.div
      className={cn('absolute z-10 hidden sm:block', className)}
      initial={{ opacity: 0, y: 12, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.6, ease: EASE_OUT, delay }}
    >
      <motion.div
        className="inline-flex items-center gap-2 rounded-[12px] border border-line bg-surface/95 px-3 py-2 text-[12.5px] font-medium whitespace-nowrap text-ink-900 shadow-raised backdrop-blur"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: drift, ease: 'easeInOut', repeat: Infinity, delay: delay + 0.6 }}
      >
        {children}
      </motion.div>
    </motion.div>
  )
}
