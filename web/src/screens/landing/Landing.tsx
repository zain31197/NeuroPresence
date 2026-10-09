import { ArrowRight, Fingerprint, Lock, Tag } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Logo } from '../../components/Logo'
import { Pill, type Tone } from '../../components/ui/Pill'
import { api, type Benchmark, type PipelineStage } from '../../lib/api'
import { cn } from '../../lib/cn'
import { longDate } from '../../lib/format'
import { HeroVisual } from './HeroVisual'
import { resultRows } from './results'

const linkButton =
  'inline-flex select-none items-center justify-center gap-2 rounded-control font-medium whitespace-nowrap transition-[background-color,transform] duration-150 active:scale-[0.98]'
const primaryLink = cn(
  linkButton,
  'bg-ink-950 text-white hover:bg-ink-800 [box-shadow:inset_0_1px_0_rgb(255_255_255/0.14),0_1px_2px_rgb(11_13_18/0.2)]',
)
const secondaryLink = cn(linkButton, 'border border-line-strong bg-surface text-ink-900 shadow-control hover:bg-ink-50')

export function Landing() {
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null)
  const [stages, setStages] = useState<PipelineStage[] | null>(null)
  const [enrolment, setEnrolment] = useState<PipelineStage | null>(null)

  useEffect(() => {
    document.title = 'NeuroPresence'
    // Both are optional: the page reads fine without the engine, it just shows less.
    api.latestBenchmark().then(setBenchmark, () => setBenchmark(null))
    api.capabilities().then(
      (reply) => {
        setStages(reply.stages)
        setEnrolment(reply.enrolment ?? null)
      },
      () => setStages(null),
    )
  }, [])

  return (
    <div className="bg-surface">
      <Header />
      <main>
        <Hero benchmark={benchmark} />
        <Why />
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
    <div className="grid gap-x-12 gap-y-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)] lg:items-end">
      <div>
        <p className="label-caps text-accent-700">{eyebrow}</p>
        <h2 className="display mt-3 text-[clamp(28px,3.6vw,42px)] leading-[1.08] font-semibold">{title}</h2>
      </div>
      {children && <p className="max-w-[52ch] text-[16px] leading-relaxed text-ink-600">{children}</p>}
    </div>
  )
}

/* ------------------------------------------------------------------ header */

function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-line/70 bg-surface/85 backdrop-blur-md">
      <Shell className="flex h-16 items-center justify-between gap-6">
        <Link to="/" aria-label="NeuroPresence home">
          <Logo />
        </Link>
        <nav className="hidden items-center gap-7 text-[14px] font-medium text-ink-600 md:flex" aria-label="Sections">
          <a href="#how" className="hover:text-ink-950">
            How it works
          </a>
          <a href="#results" className="hover:text-ink-950">
            Results
          </a>
          <a href="#safeguards" className="hover:text-ink-950">
            Safeguards
          </a>
        </nav>
        <Link to="/studio" className={cn(primaryLink, 'h-9 px-3.5 text-[13.5px]')}>
          Open the studio
          <ArrowRight className="size-3.5" />
        </Link>
      </Shell>
    </header>
  )
}

/* -------------------------------------------------------------------- hero */

function Hero({ benchmark }: { benchmark: Benchmark | null }) {
  return (
    <section className="relative overflow-hidden border-b border-line">
      {/* A field of fine dots, fading out: the page's only texture. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 [background-image:radial-gradient(var(--color-ink-200)_1px,transparent_1px)] [background-size:22px_22px] [mask-image:radial-gradient(ellipse_70%_75%_at_72%_40%,black,transparent)]"
      />
      <Shell className="relative grid items-center gap-x-14 gap-y-12 py-16 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.02fr)] lg:py-24">
        <div className="animate-rise">
          <p className="inline-flex items-center gap-2 rounded-full border border-line bg-surface py-1 pr-3 pl-1.5 text-[12.5px] font-medium text-ink-700 shadow-control">
            <span className="grid size-5 place-items-center rounded-full bg-ink-100 text-ink-700">
              <Lock className="size-3" />
            </span>
            Research prototype. Runs on your own computer.
          </p>
          <h1 className="display mt-6 text-[clamp(38px,5.4vw,66px)] leading-[1.02] font-semibold">
            One good picture of you, driven live by your own face.
          </h1>
          <p className="mt-6 max-w-[50ch] text-[17px] leading-relaxed text-ink-600">
            NeuroPresence animates a single enrolled picture of you with your live head pose, lip movement and expressions. It
            runs on your own GPU, and nothing leaves your computer.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <Link to="/studio" className={cn(primaryLink, 'h-12 px-5 text-[15px]')}>
              Open the studio
              <ArrowRight className="size-4" />
            </Link>
            <a href="#how" className={cn(secondaryLink, 'h-12 px-5 text-[15px]')}>
              See how it works
            </a>
          </div>
        </div>
        <div className="animate-rise [animation-delay:120ms]">
          <HeroVisual benchmark={benchmark} />
        </div>
      </Shell>
    </section>
  )
}

/* --------------------------------------------------------------------- why */

const REASONS = [
  {
    title: 'A bad angle',
    text: 'A laptop on a low table films you from below. Your enrolled picture is framed properly, once, and stays that way.',
  },
  {
    title: 'A rough day',
    text: 'Unwell, tired or recovering. You are present and taking part; the picture people see is the one you chose.',
  },
  {
    title: 'Back-to-back calls',
    text: 'By the sixth call the light has changed and so have you. Your picture has not.',
  },
]

function Why() {
  return (
    <section className="border-b border-line py-20 lg:py-24">
      <Shell>
        <SectionHeading eyebrow="What it is for" title="For the calls where the camera works against you.">
          A virtual background changes what is behind you. It does nothing about how you are framed, lit or feeling. NeuroPresence
          recomposes you, and only you.
        </SectionHeading>
        <div className="mt-12 grid gap-px overflow-hidden rounded-card border border-line bg-line md:grid-cols-3">
          {REASONS.map((reason) => (
            <div key={reason.title} className="bg-surface p-7">
              <h3 className="text-[17px] font-semibold">{reason.title}</h3>
              <p className="mt-2 text-[14.5px] leading-relaxed text-ink-600">{reason.text}</p>
            </div>
          ))}
        </div>
        <p className="mt-6 max-w-[78ch] text-[14.5px] leading-relaxed text-ink-600">
          <span className="font-semibold text-ink-950">What it is not for:</span> appearing present when you are not. Your live face
          drives every frame. When it leaves the camera, the output stops moving and shows your enrolled picture, still.
        </p>
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
    <section id="how" className="scroll-mt-16 border-b border-line bg-canvas py-20 lg:py-24">
      <Shell>
        <SectionHeading eyebrow="How it works" title="Six stages, from the camera to the call.">
          You enrol one picture, once. After that, each camera frame goes through the chain before the next one starts. What is
          built and what is still to come is reported by the engine itself, so this page cannot claim more than the code does.
        </SectionHeading>
        <div className="mt-12 flex flex-wrap items-center gap-x-8 gap-y-4 rounded-card border border-line bg-surface px-6 py-5 shadow-card">
          <p className="label-caps w-full text-ink-400 sm:w-[132px]">Once, before the first call</p>
          <div className="min-w-0 flex-1 basis-[300px]">
            <h3 className="text-[17px] font-semibold">Enrolment</h3>
            <p className="mt-1 max-w-[70ch] text-[14.5px] leading-relaxed text-ink-600">{enrolment?.summary ?? ENROLMENT_OUTLINE}</p>
          </div>
          {enrolment && <Pill tone={STATUS[enrolment.status].tone}>{STATUS[enrolment.status].label}</Pill>}
        </div>
        <ol className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {list.map((stage, index) => (
            <li key={stage.key} className="flex flex-col rounded-card border border-line bg-surface p-6 shadow-card">
              <div className="flex items-center justify-between gap-3">
                <span className="grid size-8 place-items-center rounded-full bg-ink-950 font-mono text-[12.5px] font-medium text-white tabular-nums">
                  {index + 1}
                </span>
                {stage.status && <Pill tone={STATUS[stage.status].tone}>{STATUS[stage.status].label}</Pill>}
              </div>
              <h3 className="mt-5 text-[17px] font-semibold">{stage.name}</h3>
              <p className="mt-1.5 text-[14.5px] leading-relaxed text-ink-600">{stage.summary}</p>
              {stage.note && <p className="mt-3 border-t border-line pt-3 text-[13px] leading-snug text-ink-500">{stage.note}</p>}
            </li>
          ))}
        </ol>
      </Shell>
    </section>
  )
}

/* ----------------------------------------------------------------- results */

function Results({ benchmark }: { benchmark: Benchmark | null }) {
  const rows = benchmark ? resultRows(benchmark).filter((row) => row.value !== null && row.value !== undefined) : []
  const met = rows.filter((row) => row.met(row.value as number)).length

  return (
    <section id="results" className="scroll-mt-16 border-b border-line py-20 lg:py-24">
      <Shell>
        <SectionHeading eyebrow="Results" title="Measured, not promised.">
          Every figure below is read from the latest benchmark saved on this computer and set beside the target the project
          committed to. Where a target is missed, it says so.
        </SectionHeading>

        {benchmark && rows.length > 0 ? (
          <div className="mt-12 overflow-hidden rounded-card border border-line bg-surface shadow-card">
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
            <ul>
              {rows.map((row) => {
                const ok = row.met(row.value as number)
                return (
                  <li
                    key={row.name}
                    className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-6 gap-y-1 border-t border-line px-6 py-4 first:border-t-0 md:grid-cols-[minmax(0,1.5fr)_7.5rem_10rem_7rem]"
                  >
                    <div className="min-w-0">
                      <p className="text-[15px] font-medium text-ink-950">{row.name}</p>
                      <p className="mt-0.5 truncate text-[13px] text-ink-500">{row.about}</p>
                    </div>
                    <p className="display text-right text-[20px] font-semibold text-ink-950 tabular-nums md:text-left">
                      {row.format(row.value as number)}
                    </p>
                    <p className="font-mono text-[12.5px] text-ink-500">
                      <span className="md:hidden">Target </span>
                      {row.target}
                    </p>
                    <div className="justify-self-end">
                      <Pill tone={ok ? 'good' : 'serious'}>{ok ? 'On target' : 'Off target'}</Pill>
                    </div>
                  </li>
                )
              })}
            </ul>
          </div>
        ) : (
          <div className="mt-12 rounded-card border border-dashed border-line-strong bg-ink-50 p-8">
            <p className="text-[15px] font-medium text-ink-950">No benchmark has been run on this computer yet.</p>
            <p className="mt-1.5 text-[14px] text-ink-600">
              Run <code className="rounded-[5px] bg-surface px-1.5 py-0.5 font-mono text-[12.5px] shadow-control">python scripts/benchmark.py</code>{' '}
              and reload this page to see the measured figures here.
            </p>
          </div>
        )}
        <p className="mt-5 max-w-[86ch] text-[13.5px] leading-relaxed text-ink-500">
          The benchmark drives the pipeline with recorded sample clips, so it can be repeated exactly. Flicker and head jitter are
          ratios against the same measurement on the real video, where 1× means as steady as real video.
        </p>
      </Shell>
    </section>
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
        ? 'Every output frame carries a visible synthetic-media mark, so the people you are talking to know what they are seeing.'
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
    <section id="safeguards" className="scroll-mt-16 border-b border-line bg-canvas py-20 lg:py-24">
      <Shell>
        <SectionHeading eyebrow="Safeguards" title="Designed to be used on yourself, and to say what it is.">
          Face reenactment can be misused to impersonate someone. The answer here is in the design, not in a policy: the system is
          restricted to its user's own likeness and discloses itself.
        </SectionHeading>
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {items.map(({ icon: Icon, title, text, status }) => (
            <div key={title} className="flex flex-col rounded-card border border-line bg-surface p-6 shadow-card">
              <div className="flex items-center justify-between gap-3">
                <span className="grid size-10 place-items-center rounded-[12px] bg-ink-100 text-ink-900">
                  <Icon className="size-5" strokeWidth={1.75} />
                </span>
                {status && <Pill tone={STATUS[status].tone}>{STATUS[status].label}</Pill>}
              </div>
              <h3 className="mt-5 text-[17px] font-semibold">{title}</h3>
              <p className="mt-1.5 text-[14.5px] leading-relaxed text-ink-600">{text}</p>
            </div>
          ))}
        </div>
      </Shell>
    </section>
  )
}

/* ------------------------------------------------------------- foundations */

const FOUNDATIONS = [
  { name: 'LivePortrait', role: 'Reenactment core', source: 'Guo et al., 2024' },
  { name: 'MediaPipe', role: 'Face tracking', source: 'Google' },
  { name: 'ArcFace', role: 'Identity embedding', source: 'InsightFace' },
  { name: 'RAFT', role: 'Optical flow, for evaluation', source: 'Teed and Deng, 2020' },
]

function Foundations() {
  return (
    <section className="border-b border-line py-20 lg:py-24">
      <Shell>
        <SectionHeading eyebrow="Foundations" title="Built on open research.">
          The networks are other people's published work. This project's own contribution is around them: making the pipeline
          fast enough on a consumer GPU, measuring it honestly, and the safeguards.
        </SectionHeading>
        <dl className="mt-12 grid gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
          {FOUNDATIONS.map((item) => (
            <div key={item.name} className="bg-surface p-6">
              <dt className="display text-[20px] font-semibold text-ink-950">{item.name}</dt>
              <dd className="mt-1.5 text-[14px] text-ink-700">{item.role}</dd>
              <dd className="mt-3 font-mono text-[11.5px] text-ink-500">{item.source}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-5 max-w-[86ch] text-[13.5px] leading-relaxed text-ink-500">
          Several of these components are released for research or non-commercial use only, so NeuroPresence is a research
          prototype and is not licensed for commercial deployment.
        </p>
      </Shell>
    </section>
  )
}

/* ----------------------------------------------------------------- closing */

function Closing() {
  return (
    <section className="py-20 lg:py-24">
      <Shell>
        <div className="flex flex-wrap items-center justify-between gap-x-10 gap-y-6 rounded-[26px] bg-ink-950 px-8 py-10 sm:px-12 sm:py-12">
          <div>
            <h2 className="display text-[clamp(26px,3.2vw,36px)] leading-tight font-semibold text-white">See it run.</h2>
            <p className="mt-2 max-w-[46ch] text-[16px] leading-relaxed text-ink-300">
              Open the studio, enrol your picture, and watch every stage respond to your face.
            </p>
          </div>
          <Link to="/studio" className={cn(linkButton, 'h-12 bg-white px-5 text-[15px] text-ink-950 hover:bg-ink-100')}>
            Open the studio
            <ArrowRight className="size-4" />
          </Link>
        </div>
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
          <p className="mt-3 text-[13.5px] text-ink-500">Supervised by Muhammad Aamir Gulzar</p>
        </div>
      </Shell>
    </footer>
  )
}
