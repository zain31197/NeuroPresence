import { ArrowRight, CalendarClock, Cpu, Fingerprint, HeartPulse, Lock, Tag, Video, Webcam } from 'lucide-react'
import { motion, useMotionValueEvent, useScroll } from 'motion/react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Logo } from '../../components/Logo'
import { CountUp, EASE_OUT, Reveal, Stagger, StaggerItem, WordsIn } from '../../components/motion'
import { Pill, type Tone } from '../../components/ui/Pill'
import { api, type Benchmark, type PipelineStage } from '../../lib/api'
import { cn } from '../../lib/cn'
import { longDate } from '../../lib/format'
import { HeroVisual } from './HeroVisual'
import { Modes } from './Modes'
import { PipelineFlow } from './PipelineFlow'
import { meterPlace, resultRows, type Row } from './results'

const linkButton =
  'group inline-flex select-none items-center justify-center gap-2 rounded-control font-medium whitespace-nowrap transition-[background-color,transform,box-shadow] duration-200 active:scale-[0.98]'
const primaryLink = cn(
  linkButton,
  'bg-ink-950 text-white hover:bg-ink-800 [box-shadow:inset_0_1px_0_rgb(255_255_255/0.14),0_1px_2px_rgb(11_13_18/0.2)] hover:[box-shadow:inset_0_1px_0_rgb(255_255_255/0.14),0_10px_24px_-10px_rgb(11_13_18/0.55)]',
)
const secondaryLink = cn(linkButton, 'border border-line-strong bg-surface text-ink-900 shadow-control hover:bg-ink-50')
// The arrow of a call to action moves on a little when the button is pointed at.
const nudge = 'transition-transform duration-200 group-hover:translate-x-0.5'

export function Landing() {
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null)
  const [stages, setStages] = useState<PipelineStage[] | null>(null)
  const [enrolment, setEnrolment] = useState<PipelineStage | null>(null)
  const [modes, setModes] = useState<PipelineStage[] | null>(null)

  useEffect(() => {
    document.title = 'NeuroPresence'
    // All optional: the page reads fine without the engine, it just shows less.
    api.latestBenchmark().then(setBenchmark, () => setBenchmark(null))
    api.capabilities().then(
      (reply) => {
        setStages(reply.stages)
        setEnrolment(reply.enrolment ?? null)
        setModes(reply.modes ?? null)
      },
      () => setStages(null),
    )
  }, [])

  return (
    <div className="bg-surface">
      <Header />
      <main>
        <Hero benchmark={benchmark} />
        <WorksWith />
        <Why />
        <DrivenBy modes={modes} />
        <HowItWorks stages={stages} enrolment={enrolment} />
        <Results benchmark={benchmark} />
        <Safeguards stages={stages} />
        <Foundations />
        <Closing />
      </main>
      <Footer />
    </div>
  )
}

/* ------------------------------------------------------------------ layout */

function Shell({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn('mx-auto w-full max-w-[1200px] px-5 sm:px-8', className)}>{children}</div>
}

function SectionHeading({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <Reveal className="grid gap-x-12 gap-y-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)] lg:items-end">
      <div>
        <p className="label-caps inline-flex items-center gap-2 text-accent-700">
          <span className="h-px w-5 bg-accent-500" aria-hidden="true" />
          {eyebrow}
        </p>
        <h2 className="display mt-3 text-[clamp(28px,3.6vw,44px)] leading-[1.06] font-semibold">{title}</h2>
      </div>
      {children && <p className="max-w-[52ch] text-[16px] leading-relaxed text-ink-600">{children}</p>}
    </Reveal>
  )
}

/* ------------------------------------------------------------------ header */

function Header() {
  // Clear over the hero; once the page moves, a frosted bar with a hairline, so the links stay readable.
  const { scrollY } = useScroll()
  const [scrolled, setScrolled] = useState(false)
  useMotionValueEvent(scrollY, 'change', (y) => setScrolled(y > 12))

  return (
    <header
      className={cn(
        'sticky top-0 z-40 border-b transition-[background-color,border-color,box-shadow] duration-300',
        scrolled ? 'border-line/80 bg-surface/80 shadow-[0_8px_24px_-18px_rgb(11_13_18/0.25)] backdrop-blur-xl' : 'border-transparent bg-transparent',
      )}
    >
      <Shell className="flex h-16 items-center justify-between gap-6">
        <Link to="/" aria-label="NeuroPresence home">
          <Logo />
        </Link>
        <nav className="hidden items-center gap-1 text-[14px] font-medium text-ink-600 md:flex" aria-label="Sections">
          {[
            ['#modes', 'Modes'],
            ['#how', 'How it works'],
            ['#results', 'Results'],
            ['#safeguards', 'Safeguards'],
          ].map(([href, label]) => (
            <a key={href} href={href} className="rounded-control px-3 py-1.5 transition-colors hover:bg-ink-50 hover:text-ink-950">
              {label}
            </a>
          ))}
        </nav>
        <Link to="/studio" className={cn(primaryLink, 'h-9 px-3.5 text-[13.5px]')}>
          Open the studio
          <ArrowRight className={cn('size-3.5', nudge)} />
        </Link>
      </Shell>
    </header>
  )
}

/* -------------------------------------------------------------------- hero */

function Hero({ benchmark }: { benchmark: Benchmark | null }) {
  return (
    <section className="relative -mt-16 overflow-hidden border-b border-line pt-16">
      <Aurora />
      {/* A field of fine dots, fading out. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 [background-image:radial-gradient(var(--color-ink-200)_1px,transparent_1px)] [background-size:22px_22px] [mask-image:radial-gradient(ellipse_70%_75%_at_72%_40%,black,transparent)]"
      />
      <Shell className="relative grid items-center gap-x-14 gap-y-16 py-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.02fr)] lg:py-24">
        <div>
          <motion.a
            href="#modes"
            className="group inline-flex items-center gap-2 rounded-full border border-line bg-surface/90 py-1 pr-3 pl-1 text-[12.5px] font-medium text-ink-700 shadow-control backdrop-blur transition-colors hover:border-accent-200"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: EASE_OUT }}
          >
            <span className="rounded-full bg-accent-500 px-2 py-0.5 text-[11px] font-semibold text-white">New</span>
            Camera off? Your voice drives your picture
            <ArrowRight className={cn('size-3 text-ink-400', nudge)} />
          </motion.a>
          <h1 className="display mt-6 text-[clamp(38px,5.4vw,68px)] leading-[1.02] font-semibold">
            <WordsIn text="One good picture of you," delay={0.1} />{' '}
            <span className="bg-[linear-gradient(100deg,var(--color-accent-600),var(--color-accent-400)_55%,var(--color-ink-950))] bg-clip-text text-transparent">
              <WordsIn text="driven live by your own face." delay={0.3} />
            </span>
          </h1>
          <motion.p
            className="mt-6 max-w-[50ch] text-[17px] leading-relaxed text-ink-600"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: EASE_OUT, delay: 0.55 }}
          >
            NeuroPresence animates a single enrolled picture of you with your live head pose, lip movement and expressions, and sends
            it to your meeting as an ordinary camera. It runs on your own GPU, and nothing leaves your computer.
          </motion.p>
          <motion.div
            className="mt-8 flex flex-wrap items-center gap-3"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: EASE_OUT, delay: 0.68 }}
          >
            <Link to="/studio" className={cn(primaryLink, 'h-12 px-5 text-[15px]')}>
              Open the studio
              <ArrowRight className={cn('size-4', nudge)} />
            </Link>
            <a href="#how" className={cn(secondaryLink, 'h-12 px-5 text-[15px]')}>
              See how it works
            </a>
          </motion.div>
          <motion.ul
            className="mt-9 flex flex-wrap gap-x-6 gap-y-2.5 text-[13.5px] text-ink-600"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.85 }}
          >
            {[
              [Cpu, 'Runs on your own GPU'],
              [Lock, 'Nothing leaves your computer'],
              [Fingerprint, 'Your own face only'],
            ].map(([Icon, text]) => {
              const I = Icon as typeof Cpu
              return (
                <li key={text as string} className="inline-flex items-center gap-2">
                  <I className="size-4 text-accent-600" strokeWidth={1.9} />
                  {text as string}
                </li>
              )
            })}
          </motion.ul>
        </div>
        <motion.div
          initial={{ opacity: 0, y: 24, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.9, ease: EASE_OUT, delay: 0.2 }}
        >
          <HeroVisual benchmark={benchmark} />
        </motion.div>
      </Shell>
    </section>
  )
}

/** Two slow lights drifting behind the hero. */
function Aurora() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      <motion.div
        className="absolute -top-40 right-[-8%] size-[620px] rounded-full bg-[radial-gradient(circle,rgb(80_98_240/0.20),transparent_65%)] blur-2xl"
        animate={{ x: [0, -60, 0], y: [0, 40, 0], scale: [1, 1.08, 1] }}
        transition={{ duration: 18, ease: 'easeInOut', repeat: Infinity }}
      />
      <motion.div
        className="absolute top-[30%] left-[-12%] size-[520px] rounded-full bg-[radial-gradient(circle,rgb(90_235_120/0.12),transparent_65%)] blur-2xl"
        animate={{ x: [0, 70, 0], y: [0, -30, 0] }}
        transition={{ duration: 22, ease: 'easeInOut', repeat: Infinity }}
      />
    </div>
  )
}

/* -------------------------------------------------------------- works with */

function WorksWith() {
  return (
    <section className="border-b border-line bg-canvas/60 py-8">
      <Shell className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4 text-center">
        <p className="label-caps text-ink-400">Shows up as a camera in</p>
        <Stagger className="flex flex-wrap items-center justify-center gap-x-8 gap-y-3" gap={0.08}>
          {['Google Meet', 'Zoom', 'any app that picks a camera'].map((name, index) => (
            <StaggerItem key={name} className="inline-flex items-center gap-2 text-[16px] font-semibold text-ink-700">
              <Video className="size-4 text-ink-400" strokeWidth={1.9} />
              <span className={index === 2 ? 'font-medium text-ink-500' : 'display'}>{name}</span>
            </StaggerItem>
          ))}
        </Stagger>
      </Shell>
    </section>
  )
}

/* --------------------------------------------------------------------- why */

const REASONS = [
  {
    icon: Webcam,
    title: 'A bad angle',
    text: 'A laptop on a low table films you from below. Your enrolled picture is framed properly, once, and stays that way.',
  },
  {
    icon: HeartPulse,
    title: 'A rough day',
    text: 'Unwell, tired or recovering. You are present and taking part; the picture people see is the one you chose.',
  },
  {
    icon: CalendarClock,
    title: 'Back-to-back calls',
    text: 'By the sixth call the light has changed and so have you. Your picture has not.',
  },
]

function Why() {
  return (
    <section className="border-b border-line py-20 lg:py-28">
      <Shell>
        <SectionHeading eyebrow="What it is for" title="For the calls where the camera works against you.">
          A virtual background changes what is behind you. It does nothing about how you are framed, lit or feeling. NeuroPresence
          recomposes you, and only you.
        </SectionHeading>
        <Stagger className="mt-12 grid gap-4 md:grid-cols-3">
          {REASONS.map(({ icon: Icon, title, text }) => (
            <StaggerItem
              key={title}
              className="group rounded-card border border-line bg-surface p-7 shadow-card transition-[box-shadow,transform] duration-300 hover:-translate-y-0.5 hover:shadow-lift"
            >
              <span className="grid size-10 place-items-center rounded-[12px] bg-accent-50 text-accent-700 transition-colors duration-300 group-hover:bg-accent-500 group-hover:text-white">
                <Icon className="size-5" strokeWidth={1.8} />
              </span>
              <h3 className="mt-5 text-[17px] font-semibold">{title}</h3>
              <p className="mt-2 text-[14.5px] leading-relaxed text-ink-600">{text}</p>
            </StaggerItem>
          ))}
        </Stagger>
        <Reveal className="mt-6 max-w-[78ch] text-[14.5px] leading-relaxed text-ink-600">
          <span className="font-semibold text-ink-950">What it is not for:</span> appearing present when you are not. Your live face, or
          your live voice, drives every frame. When neither is there, the output stops moving and shows your enrolled picture, still.
        </Reveal>
      </Shell>
    </section>
  )
}

/* --------------------------------------------------------------- driven by */

function DrivenBy({ modes }: { modes: PipelineStage[] | null }) {
  return (
    <section id="modes" className="scroll-mt-16 border-b border-line bg-canvas py-20 lg:py-28">
      <Shell>
        <SectionHeading eyebrow="Two ways to drive it" title="Your face when the camera is on. Your voice when it is off.">
          Both start with the same check at the camera, both carry the same mark, and both reach the meeting the same way.
        </SectionHeading>
        <Modes modes={modes} />
      </Shell>
    </section>
  )
}

/* ------------------------------------------------------------ how it works */

const STATUS: Record<PipelineStage['status'], { tone: Tone; label: string }> = {
  working: { tone: 'good', label: 'Working' },
  partial: { tone: 'warning', label: 'Partly built' },
  planned: { tone: 'neutral', label: 'Planned' },
}

// Shown when the engine is not running, without any claim about what is built.
const STAGE_OUTLINE = [
  { key: 'capture', name: 'Capture', summary: 'Reads the camera and finds the face and its 478 landmarks in every frame.' },
  { key: 'motion', name: 'Motion encoding', summary: 'Turns the tracked face into head pose, lip and expression signals.' },
  { key: 'reenactment', name: 'Reenactment', summary: 'Animates the enrolled picture with those signals and blends it back into the frame.' },
  { key: 'identity', name: 'Identity', summary: 'Checks that the output still looks like the enrolled person.' },
  { key: 'consent', name: 'Consent and disclosure', summary: "Animates only the enrolled user's own face and marks the output as synthetic." },
  { key: 'virtual_camera', name: 'Virtual camera', summary: 'Delivers the output to meeting apps as an ordinary camera.' },
]

// The same, for the step that comes before the stages.
const ENROLMENT_OUTLINE =
  'Your face is verified live with the camera, and only its signature is kept. Then you upload the picture you want people to see: it is checked for quality, and accepted only if it shows the same face.'

function HowItWorks({ stages, enrolment }: { stages: PipelineStage[] | null; enrolment: PipelineStage | null }) {
  const list = stages ?? STAGE_OUTLINE.map((stage) => ({ ...stage, status: null, note: null }))
  return (
    <section id="how" className="scroll-mt-16 border-b border-line py-20 lg:py-28">
      <Shell>
        <SectionHeading eyebrow="How it works" title="Six stages, from the camera to the call.">
          You enrol one picture, once. After that, each camera frame goes through the chain before the next one starts. What is
          built and what is still to come is reported by the engine itself, so this page cannot claim more than the code does.
        </SectionHeading>
        <Reveal className="mt-12">
          <PipelineFlow stages={list} />
        </Reveal>
        <Reveal className="mt-4 flex flex-wrap items-center gap-x-8 gap-y-4 rounded-card border border-line bg-surface px-6 py-5 shadow-card">
          <p className="label-caps w-full text-ink-400 sm:w-[132px]">Once, before the first call</p>
          <div className="min-w-0 flex-1 basis-[300px]">
            <h3 className="text-[17px] font-semibold">Enrolment</h3>
            <p className="mt-1 max-w-[70ch] text-[14.5px] leading-relaxed text-ink-600">{enrolment?.summary ?? ENROLMENT_OUTLINE}</p>
          </div>
          {enrolment && <Pill tone={STATUS[enrolment.status].tone}>{STATUS[enrolment.status].label}</Pill>}
        </Reveal>
        <Stagger as="ol" className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {list.map((stage, index) => (
            <StaggerItem
              as="li"
              key={stage.key}
              className="flex flex-col rounded-card border border-line bg-surface p-6 shadow-card transition-shadow duration-300 hover:shadow-lift"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="grid size-8 place-items-center rounded-full bg-ink-950 font-mono text-[12.5px] font-medium text-white tabular-nums">
                  {index + 1}
                </span>
                {stage.status && <Pill tone={STATUS[stage.status].tone}>{STATUS[stage.status].label}</Pill>}
              </div>
              <h3 className="mt-5 text-[17px] font-semibold">{stage.name}</h3>
              <p className="mt-1.5 text-[14.5px] leading-relaxed text-ink-600">{stage.summary}</p>
              {stage.note && <p className="mt-3 border-t border-line pt-3 text-[13px] leading-snug text-ink-500">{stage.note}</p>}
            </StaggerItem>
          ))}
        </Stagger>
      </Shell>
    </section>
  )
}

/* ----------------------------------------------------------------- results */

function Results({ benchmark }: { benchmark: Benchmark | null }) {
  const rows = benchmark ? resultRows(benchmark).filter((row) => row.value !== null && row.value !== undefined) : []
  const met = rows.filter((row) => row.met(row.value as number)).length

  return (
    <section id="results" className="scroll-mt-16 border-b border-line bg-canvas py-20 lg:py-28">
      <Shell>
        <SectionHeading eyebrow="Results" title="Measured, not promised.">
          Every figure below is read from the latest benchmark saved on this computer and set beside the target the project
          committed to. Where a target is missed, it says so.
        </SectionHeading>

        {benchmark && rows.length > 0 ? (
          <Reveal className="mt-12 overflow-hidden rounded-card border border-line bg-surface shadow-card">
            <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line bg-ink-50/60 px-6 py-3.5">
              <p className="text-[14px] font-semibold text-ink-950">
                {met} of {rows.length} on target
              </p>
              <p className="font-mono text-[11.5px] text-ink-500">
                {[longDate(benchmark.date), benchmark.machine.gpu, benchmark.clips ? `${benchmark.clips} clips` : null]
                  .filter(Boolean)
                  .join('  ·  ')}
              </p>
            </div>
            <Stagger as="ul" gap={0.05}>
              {rows.map((row) => (
                <ResultRow key={row.name} row={row} />
              ))}
            </Stagger>
          </Reveal>
        ) : (
          <Reveal className="mt-12 rounded-card border border-dashed border-line-strong bg-surface p-8">
            <p className="text-[15px] font-medium text-ink-950">No benchmark has been run on this computer yet.</p>
            <p className="mt-1.5 text-[14px] text-ink-600">
              Run <code className="rounded-[5px] bg-ink-50 px-1.5 py-0.5 font-mono text-[12.5px] shadow-control">python scripts/benchmark.py</code>{' '}
              and reload this page to see the measured figures here.
            </p>
          </Reveal>
        )}
        <Reveal className="mt-5 max-w-[86ch] text-[13.5px] leading-relaxed text-ink-500">
          The benchmark drives the pipeline with recorded sample clips, one stage after another, so it can be repeated exactly. A
          live session tracks the next frame while it draws this one, and runs faster. Flicker and head jitter are ratios against
          the same measurement on the real video, where 1× means as steady as real video.
        </Reveal>
      </Shell>
    </section>
  )
}

function ResultRow({ row }: { row: Row }) {
  const value = row.value as number
  const ok = row.met(value)
  const { fill, target } = meterPlace(row, value)
  return (
    <StaggerItem
      as="li"
      className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-6 gap-y-2 border-t border-line px-6 py-4 first:border-t-0 md:grid-cols-[minmax(0,1.3fr)_7.5rem_minmax(0,1fr)_7rem]"
    >
      <div className="min-w-0">
        <p className="text-[15px] font-medium text-ink-950">{row.name}</p>
        <p className="mt-0.5 truncate text-[13px] text-ink-500">{row.about}</p>
      </div>
      <p className="display text-right text-[20px] font-semibold text-ink-950 md:text-left">
        <CountUp value={value} format={row.format} />
      </p>
      {/* The figure against its target: the mark is the target, the bar is the figure. */}
      <div className="col-span-2 md:col-span-1">
        <div className="relative h-2 overflow-hidden rounded-full bg-ink-100" role="img" aria-label={`${row.format(value)} against a target of ${row.target}`}>
          <motion.div
            className={cn('h-full origin-left rounded-full', ok ? 'bg-good' : 'bg-serious')}
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: fill }}
            viewport={{ once: true }}
            transition={{ duration: 1.1, ease: EASE_OUT, delay: 0.15 }}
          />
          <span className="absolute inset-y-0 w-[2px] bg-ink-950" style={{ left: `calc(${target * 100}% - 1px)` }} />
        </div>
        <p className="mt-1.5 font-mono text-[11.5px] text-ink-500">Target {row.target}</p>
      </div>
      <div className="col-span-2 justify-self-start md:col-span-1 md:justify-self-end">
        <Pill tone={ok ? 'good' : 'serious'}>{ok ? 'On target' : 'Off target'}</Pill>
      </div>
    </StaggerItem>
  )
}

/* -------------------------------------------------------------- safeguards */

function Safeguards({ stages }: { stages: PipelineStage[] | null }) {
  const consent = stages?.find((stage) => stage.key === 'consent')?.status ?? null
  // Written as a commitment until the engine reports the stage as working.
  const built = consent === 'working'
  const items = [
    {
      icon: Fingerprint,
      title: 'Your own face only',
      text: built
        ? 'Before a session starts you are asked for two quick actions, chosen at random, and the face at the camera is matched with your verified face all the way through. No match, no animation.'
        : 'A session will start only if the live face matches the enrolled picture. No match, no animation.',
      status: consent,
    },
    {
      icon: Tag,
      title: 'Always labelled',
      text: built
        ? 'Every output frame carries a visible "AI reenacted" mark, so the people you are talking to know what they are seeing.'
        : 'Every output frame will carry a visible synthetic-media mark, so the people you are talking to know what they are seeing.',
      status: consent,
    },
    {
      icon: Lock,
      title: 'Stays on this computer',
      text: 'Camera frames, your picture and every model run locally. The app listens on this machine only, and there is no account.',
      status: 'working' as const,
    },
  ]
  return (
    <section id="safeguards" className="scroll-mt-16 border-b border-line py-20 lg:py-28">
      <Shell>
        <SectionHeading eyebrow="Safeguards" title="Designed to be used on yourself, and to say what it is.">
          Face reenactment can be misused to impersonate someone. The answer here is in the design, not in a policy: the system is
          restricted to its user's own likeness and discloses itself.
        </SectionHeading>
        <Stagger className="mt-12 grid gap-4 md:grid-cols-3">
          {items.map(({ icon: Icon, title, text, status }) => (
            <StaggerItem
              key={title}
              className="group flex flex-col rounded-card border border-line bg-surface p-6 shadow-card transition-[box-shadow,transform] duration-300 hover:-translate-y-0.5 hover:shadow-lift"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="grid size-10 place-items-center rounded-[12px] bg-ink-950 text-white transition-transform duration-300 group-hover:scale-105">
                  <Icon className="size-5" strokeWidth={1.75} />
                </span>
                {status && <Pill tone={STATUS[status].tone}>{STATUS[status].label}</Pill>}
              </div>
              <h3 className="mt-5 text-[17px] font-semibold">{title}</h3>
              <p className="mt-1.5 text-[14.5px] leading-relaxed text-ink-600">{text}</p>
            </StaggerItem>
          ))}
        </Stagger>
      </Shell>
    </section>
  )
}

/* ------------------------------------------------------------- foundations */

const FOUNDATIONS = [
  { name: 'LivePortrait', role: 'Reenactment core', source: 'Guo et al., 2024' },
  { name: 'Ditto', role: 'Speech to motion, camera off', source: 'Ant Group, 2025' },
  { name: 'MediaPipe', role: 'Face tracking', source: 'Google' },
  { name: 'ArcFace', role: 'Identity embedding', source: 'InsightFace' },
]

function Foundations() {
  return (
    <section className="border-b border-line bg-canvas py-20 lg:py-28">
      <Shell>
        <SectionHeading eyebrow="Foundations" title="Built on open research.">
          The networks are other people's published work. This project's own contribution is around them: a live pipeline fast
          enough on a consumer GPU, the safeguards, the meeting output, and honest measurement.
        </SectionHeading>
        <Stagger as="dl" className="mt-12 grid gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {FOUNDATIONS.map((item) => (
            <StaggerItem key={item.name} className="bg-surface p-6 transition-colors duration-300 hover:bg-ink-50/60">
              <dt className="display text-[20px] font-semibold text-ink-950">{item.name}</dt>
              <dd className="mt-1.5 text-[14px] text-ink-700">{item.role}</dd>
              <dd className="mt-3 font-mono text-[11.5px] text-ink-500">{item.source}</dd>
            </StaggerItem>
          ))}
        </Stagger>
        <Reveal className="mt-5 max-w-[86ch] text-[13.5px] leading-relaxed text-ink-500">
          Several of these components are released for research or non-commercial use only, so NeuroPresence is a research
          prototype and is not licensed for commercial deployment.
        </Reveal>
      </Shell>
    </section>
  )
}

/* ----------------------------------------------------------------- closing */

function Closing() {
  return (
    <section className="py-20 lg:py-28">
      <Shell>
        <Reveal className="relative isolate overflow-hidden rounded-[28px] bg-ink-950 px-8 py-12 sm:px-14 sm:py-16">
          <motion.div
            aria-hidden="true"
            className="absolute -top-32 -right-24 -z-10 size-[460px] rounded-full bg-[radial-gradient(circle,rgb(80_98_240/0.55),transparent_65%)] blur-2xl"
            animate={{ x: [0, -40, 0], y: [0, 30, 0] }}
            transition={{ duration: 14, ease: 'easeInOut', repeat: Infinity }}
          />
          <div
            aria-hidden="true"
            className="absolute inset-0 -z-10 [background-image:radial-gradient(rgb(255_255_255/0.08)_1px,transparent_1px)] [background-size:22px_22px] [mask-image:linear-gradient(90deg,transparent,black)]"
          />
          <div className="flex flex-wrap items-center justify-between gap-x-10 gap-y-8">
            <div>
              <h2 className="display text-[clamp(28px,3.6vw,44px)] leading-tight font-semibold text-white">See it run.</h2>
              <p className="mt-3 max-w-[46ch] text-[16px] leading-relaxed text-ink-300">
                Open the studio, enrol your picture, and watch every stage respond to your face.
              </p>
            </div>
            <Link to="/studio" className={cn(linkButton, 'h-12 bg-white px-5 text-[15px] text-ink-950 hover:bg-ink-100')}>
              Open the studio
              <ArrowRight className={cn('size-4', nudge)} />
            </Link>
          </div>
        </Reveal>
      </Shell>
    </section>
  )
}

/* ------------------------------------------------------------------ footer */

function Footer() {
  return (
    <footer className="border-t border-line bg-canvas py-12">
      <Shell className="grid gap-x-12 gap-y-8 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <Logo />
          <p className="mt-4 max-w-[40ch] text-[13.5px] leading-relaxed text-ink-600">
            Identity-preserving, latency-bounded neural face reenactment for consent-gated professional telepresence.
          </p>
        </div>
        <div>
          <p className="label-caps text-ink-400">Project</p>
          <p className="mt-3 text-[13.5px] leading-relaxed text-ink-700">
            Final Year Project, BS Data Science
            <br />
            FAST School of Computing, NUCES Islamabad
            <br />
            2026 to 2027
          </p>
        </div>
        <div>
          <p className="label-caps text-ink-400">Team</p>
          <p className="mt-3 text-[13.5px] leading-relaxed text-ink-700">
            Zain Shahid
            <br />
            Muhammad Talha Arshad
            <br />
            Sana Ullah Farooqi
          </p>
          <p className="mt-3 text-[13.5px] text-ink-500">Supervised by Mr. Muhammad Aamir Gulzar</p>
        </div>
      </Shell>
    </footer>
  )
}
