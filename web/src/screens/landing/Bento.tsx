import { Check, Clock, Fingerprint, Lock, MoveLeft, ScanFace, ShieldCheck, Tag, Timer, UserX, Volume2, type LucideIcon } from 'lucide-react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useEffect, useState, type ReactNode } from 'react'
import { SpotlightCard } from '../../components/fx'
import { EASE_OUT, Stagger, StaggerItem } from '../../components/motion'
import { cn } from '../../lib/cn'
import { FaceMeshCanvas } from './FaceMeshCanvas'

/**
 * What keeps the system safe to use, one tile each, every tile showing its idea in motion. The animations are
 * illustrations of the behaviour the engine has, not live readings.
 */
export function Bento() {
  return (
    <Stagger className="mt-12 grid grid-cols-1 gap-4 lg:grid-cols-3" gap={0.07}>
      <Tile className="lg:col-span-2" icon={ShieldCheck} title="A living person, and the right one" visual={<LivenessVisual />}>
        Before anything is animated, two actions are asked for, chosen at random, a turn of the head always among them, and your face is
        matched about four times a second all the way through. A photograph cannot turn; a stranger does not match.
      </Tile>
      <Tile icon={Fingerprint} title="Your own face only" visual={<GaugeVisual />}>
        The output is scored against your picture twice a second. A lasting drop below 0.75 takes a fresh neutral pose; if that does not
        help, your still picture is shown until you are at rest again.
      </Tile>
      <Tile icon={Tag} title="Always labelled" visual={<MarkVisual />}>
        Every frame carries "AI reenacted" in the middle of the lower edge, where a meeting's crop of the picture never reaches it.
      </Tile>
      <Tile icon={Volume2} title="Lips and voice together" visual={<SyncVisual />}>
        Your voice reaches the meeting held back by exactly as long as the picture takes, so lips and words arrive at the same moment.
      </Tile>
      <Tile icon={Timer} title="No endless retries" visual={<AttemptsVisual />}>
        Three checks that do not pass cost nothing. After that each one waits twice as long as the last: one minute, two, four.
      </Tile>
      <StaggerItem className="lg:col-span-3">
        <SpotlightCard className="rounded-card border border-line bg-ink-950 text-white shadow-card">
          <div className="flex flex-wrap items-center gap-x-10 gap-y-5 p-7">
            <span className="grid size-11 place-items-center rounded-[13px] bg-white/10 text-white">
              <Lock className="size-5" strokeWidth={1.8} />
            </span>
            <div className="min-w-0 flex-1 basis-[300px]">
              <h3 className="text-[18px] font-semibold text-white">Stays on this computer</h3>
              <p className="mt-1.5 max-w-[70ch] text-[14.5px] leading-relaxed text-ink-300">
                Camera frames, your picture and every model run here. There is no account and nothing is uploaded: the engine listens on
                this machine only.
              </p>
            </div>
            <ul className="flex flex-wrap gap-2 text-[12.5px] font-medium text-ink-200">
              {['No account', 'No upload', 'Runs on your GPU'].map((item) => (
                <li key={item} className="inline-flex items-center gap-1.5 rounded-full border border-white/15 px-3 py-1.5">
                  <Check className="size-3.5 text-good" strokeWidth={3} />
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </SpotlightCard>
      </StaggerItem>
    </Stagger>
  )
}

function Tile({ icon: Icon, title, visual, className, children }: { icon: LucideIcon; title: string; visual: ReactNode; className?: string; children: ReactNode }) {
  return (
    <StaggerItem className={className}>
      <SpotlightCard className="h-full rounded-card border border-line bg-surface shadow-card transition-shadow duration-300 hover:shadow-lift">
        <div className="flex h-full flex-col">
          <div className="relative grid h-[180px] place-items-center overflow-hidden border-b border-line bg-[radial-gradient(80%_90%_at_50%_0%,var(--color-accent-50),var(--color-surface))]">
            {visual}
          </div>
          <div className="p-6">
            <div className="flex items-center gap-2.5">
              <span className="grid size-8 place-items-center rounded-[10px] bg-ink-950 text-white">
                <Icon className="size-4" strokeWidth={1.9} />
              </span>
              <h3 className="text-[16.5px] font-semibold">{title}</h3>
            </div>
            <p className="mt-2.5 text-[14px] leading-relaxed text-ink-600">{children}</p>
          </div>
        </div>
      </SpotlightCard>
    </StaggerItem>
  )
}

/** Steps through a list on a timer; stands on the last step for reduced motion. */
function useCycle(length: number, ms: number) {
  const reduced = useReducedMotion()
  const [index, setIndex] = useState(reduced ? length - 1 : 0)
  useEffect(() => {
    if (reduced) return
    const timer = window.setInterval(() => setIndex((i) => (i + 1) % length), ms)
    return () => window.clearInterval(timer)
  }, [length, ms, reduced])
  return index
}

const PROMPTS: { icon: LucideIcon; text: string; done?: boolean }[] = [
  { icon: ScanFace, text: 'Look at the camera' },
  { icon: MoveLeft, text: 'Turn your head to your left' },
  { icon: ScanFace, text: 'Blink' },
  { icon: Check, text: 'It is you. Starting', done: true },
]

function LivenessVisual() {
  const step = useCycle(PROMPTS.length, 1700)
  const prompt = PROMPTS[step]
  return (
    <div className="flex items-center gap-4 px-4 sm:gap-8">
      <div className="relative h-[124px] w-[100px] shrink-0 overflow-hidden rounded-[14px] bg-ink-950 sm:h-[150px] sm:w-[124px]">
        <FaceMeshCanvas className="size-full" meshColor="rgb(180 186 199 / 0.35)" contourColor="rgb(233 236 241 / 0.9)" label="A face answering the prompt." />
        <motion.div
          aria-hidden="true"
          className="absolute inset-x-0 top-0 h-full"
          animate={{ y: ['-10%', '95%'] }}
          transition={{ duration: 2.4, ease: 'easeInOut', repeat: Infinity, repeatType: 'mirror' }}
        >
          <div className="h-6 bg-[linear-gradient(180deg,transparent,rgb(90_235_120/0.5))]" />
        </motion.div>
      </div>
      <div className="w-[170px] min-w-0 sm:w-[220px]">
        <p className="label-caps text-ink-400">Step {Math.min(step + 1, 3)} of 3</p>
        <AnimatePresence mode="wait">
          <motion.p
            key={step}
            className={cn('mt-2 inline-flex items-center gap-2 text-[15px] font-semibold', prompt.done ? 'text-good-ink' : 'text-ink-950')}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3, ease: EASE_OUT }}
          >
            <prompt.icon className="size-4" strokeWidth={2.2} />
            {prompt.text}
          </motion.p>
        </AnimatePresence>
        <div className="mt-3 flex gap-1.5">
          {[0, 1, 2].map((bar) => (
            <span key={bar} className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-100">
              <motion.span
                className="block h-full origin-left rounded-full bg-accent-500"
                animate={{ scaleX: step > bar ? 1 : step === bar ? [0, 1] : 0 }}
                transition={{ duration: step === bar ? 1.5 : 0.2, ease: 'linear' }}
              />
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

/** A needle that wanders where a good output scores, well above the limit it would act on. */
function GaugeVisual() {
  const reduced = useReducedMotion()
  // 0 to 1 across a half circle, centred at (100, 105); the limit at 0.75. The needle's tip is moved, not the
  // needle turned: a turned group in SVG lost its pivot and the needle vanished.
  const tip = (value: number, length = 67) => {
    const a = Math.PI * (1 - value)
    return { x: 100 + length * Math.cos(a), y: 105 - length * Math.sin(a) }
  }
  const tick = [tip(0.75, 89), tip(0.75, 71)]
  const path = [0.88, 0.94, 0.9, 0.96, 0.88].map((value) => tip(value))
  return (
    <svg viewBox="0 0 200 120" className="w-[200px]" aria-hidden="true">
      <path d="M20 105 A80 80 0 0 1 180 105" fill="none" stroke="var(--color-ink-100)" strokeWidth="12" strokeLinecap="round" />
      <path d="M156.6 48.4 A80 80 0 0 1 180 105" fill="none" stroke="var(--color-good)" strokeOpacity="0.35" strokeWidth="12" strokeLinecap="round" />
      <line x1={tick[0].x} y1={tick[0].y} x2={tick[1].x} y2={tick[1].y} stroke="var(--color-ink-700)" strokeWidth="2" strokeDasharray="3 2" />
      <text x="100" y="118" textAnchor="middle" className="fill-ink-500 font-mono text-[9px]">
        limit 0.75
      </text>
      <motion.line
        x1="100"
        y1="105"
        stroke="var(--color-accent-600)"
        strokeWidth="3"
        strokeLinecap="round"
        initial={{ x2: path[2].x, y2: path[2].y }}
        animate={reduced ? { x2: path[2].x, y2: path[2].y } : { x2: path.map((p) => p.x), y2: path.map((p) => p.y) }}
        transition={{ duration: 6, ease: 'easeInOut', repeat: Infinity }}
      />
      <circle cx="100" cy="105" r="6" fill="var(--color-ink-950)" />
    </svg>
  )
}

function MarkVisual() {
  return (
    <div className="relative h-[140px] w-[200px] overflow-hidden rounded-[12px] bg-[linear-gradient(160deg,var(--color-ink-800),var(--color-ink-950))] shadow-raised">
      <div className="absolute inset-x-0 top-2 bottom-0 mx-auto w-[110px]">
        <FaceMeshCanvas className="size-full" meshColor="rgb(159 174 255 / 0.35)" contourColor="rgb(197 207 255 / 0.95)" delay={0.6} label="A face in a meeting tile." />
      </div>
      <motion.span
        className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-[4px] bg-black/70 px-1.5 py-0.5 text-[9px] font-semibold tracking-wide whitespace-nowrap text-white"
        animate={{ opacity: [0.75, 1, 0.75] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: 'easeInOut' }}
      >
        AI reenacted
      </motion.span>
      <span className="label-caps absolute top-2 left-2 rounded-[4px] bg-white/10 px-1.5 py-0.5 text-[8.5px] text-white/80">Meeting tile</span>
    </div>
  )
}

/** The voice arriving early, then held back until it lines up with the lips. */
function SyncVisual() {
  const bars = [0.3, 0.8, 0.5, 1, 0.6, 0.35, 0.9, 0.55, 0.75, 0.4, 0.95, 0.5]
  const row = (label: string, color: string, shift: boolean) => (
    <div className="flex items-center gap-3">
      <span className="label-caps w-10 text-ink-400">{label}</span>
      <motion.div
        className="flex h-9 items-center gap-[4px]"
        animate={shift ? { x: [18, 18, 0, 0] } : undefined}
        transition={shift ? { duration: 3.2, times: [0, 0.35, 0.6, 1], ease: 'easeInOut', repeat: Infinity } : undefined}
      >
        {bars.map((height, index) => (
          <span key={index} className={cn('w-[5px] rounded-full', color)} style={{ height: `${height * 100}%` }} />
        ))}
      </motion.div>
    </div>
  )
  return (
    <div className="grid gap-2.5">
      {row('Lips', 'bg-ink-800', false)}
      {row('Voice', 'bg-accent-500', true)}
    </div>
  )
}

/** Three free tries, then waits that double. */
function AttemptsVisual() {
  const step = useCycle(7, 900)
  const items: { icon: LucideIcon; label: string; wait?: boolean }[] = [
    { icon: UserX, label: 'free' },
    { icon: UserX, label: 'free' },
    { icon: UserX, label: 'free' },
    { icon: Clock, label: '1 min', wait: true },
    { icon: Clock, label: '2 min', wait: true },
    { icon: Clock, label: '4 min', wait: true },
  ]
  return (
    <ul className="flex items-end gap-2">
      {items.map((item, index) => (
        <motion.li
          key={index}
          className="flex flex-col items-center gap-1.5"
          animate={{ opacity: step > index ? 1 : 0.25, y: step > index ? 0 : 4 }}
          transition={{ duration: 0.3, ease: EASE_OUT }}
        >
          <span className={cn('grid size-8 place-items-center rounded-full', item.wait ? 'bg-warning-wash text-warning-ink' : 'bg-ink-100 text-ink-600')}>
            <item.icon className="size-4" strokeWidth={2} />
          </span>
          <span className="font-mono text-[9.5px] text-ink-500">{item.label}</span>
        </motion.li>
      ))}
    </ul>
  )
}
