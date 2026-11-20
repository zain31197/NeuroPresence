import { Camera, Check, Copy, OctagonAlert, RotateCcw, TriangleAlert, type LucideIcon } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { EASE_OUT, Stagger, StaggerItem } from '../../components/motion'
import { PageHeader } from '../../components/PageHeader'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { Select } from '../../components/ui/Select'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, type CheckLevel, type SystemCheck, type SystemReport } from '../../lib/api'
import { cn } from '../../lib/cn'
import { useStudio } from '../../lib/studio'

const LEVEL: Record<CheckLevel, { icon: LucideIcon; tone: string; ring: string; label: string }> = {
  ok: { icon: Check, tone: 'bg-good-wash text-good-ink', ring: 'var(--color-good)', label: 'Ready' },
  warn: { icon: TriangleAlert, tone: 'bg-warning-wash text-warning-ink', ring: 'var(--color-warning)', label: 'Look at this' },
  fail: { icon: OctagonAlert, tone: 'bg-critical-wash text-critical-ink', ring: 'var(--color-critical)', label: 'Fix this' },
}

/** A fix that is a command to type is shown as one, with a button that copies it. */
const isCommand = (fix: string) => /^(python|pip|cd |npm)/.test(fix)

export function System() {
  const { status } = useStudio()
  const toast = useToast()
  const [report, setReport] = useState<SystemReport | null>(null)
  const [checking, setChecking] = useState(false)
  const [camera, setCamera] = useState('camera:0')
  const [cameraResult, setCameraResult] = useState<SystemCheck | null>(null)
  const [testing, setTesting] = useState(false)
  const cameras = (status?.inputs ?? []).filter((input) => input.kind === 'camera')

  const check = async () => {
    setChecking(true)
    try {
      setReport(await api.system())
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The engine did not answer.')
    } finally {
      setChecking(false)
    }
  }
  const testCamera = async () => {
    setTesting(true)
    setCameraResult(null)
    try {
      setCameraResult(await api.testCamera(camera))
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'The camera could not be tested.')
    } finally {
      setTesting(false)
    }
  }

  useEffect(() => {
    document.title = 'System · NeuroPresence'
    check()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <main className="mx-auto max-w-[1180px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      <PageHeader
        title="System"
        actions={
          <Button icon={<RotateCcw className="size-3.5" />} busy={checking} onClick={check}>
            Check again
          </Button>
        }
      >
        Is this computer ready for a session? The same checks as scripts/preflight.py, run by the engine. Open it before a call or a
        demonstration.
      </PageHeader>

      <Verdict report={report} checking={checking} />

      <Card className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3 p-5">
        <span className="grid size-10 shrink-0 place-items-center rounded-[12px] bg-ink-100 text-ink-800">
          <Camera className="size-5" strokeWidth={1.8} />
        </span>
        <div className="min-w-0 flex-1 basis-[260px]">
          <p className="text-[14px] font-semibold text-ink-950">Camera</p>
          <p className="mt-0.5 text-[13px] text-ink-500">
            Opens the camera for two seconds and counts the pictures it really sends. A session draws no faster than that.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {cameras.length > 0 && (
            <Select
              label="Camera"
              value={camera}
              onValueChange={setCamera}
              groups={[{ options: cameras.map((input) => ({ value: input.id, label: input.label })) }]}
            />
          )}
          <Button variant="primary" busy={testing} disabled={status?.session.state === 'running'} onClick={testCamera}>
            Test camera
          </Button>
        </div>
        <AnimatePresence>
          {cameraResult && (
            <motion.div
              className="w-full"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3, ease: EASE_OUT }}
            >
              <ul className="rounded-panel border border-line">
                <CheckRow check={cameraResult} />
              </ul>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {report && (
        <Stagger className="mt-4 grid gap-4 lg:grid-cols-2" gap={0.08}>
          {report.groups.map((group) => (
            <StaggerItem key={group.key} className={cn(group.checks.length > 3 && 'lg:row-span-2')}>
              <Card className="h-full overflow-hidden">
                <div className="flex items-center justify-between border-b border-line bg-ink-50/60 px-5 py-3">
                  <h2 className="text-[14px] font-semibold text-ink-950">{group.title}</h2>
                  <GroupPill checks={group.checks} />
                </div>
                <ul className="divide-y divide-line">
                  {group.checks.map((item) => (
                    <CheckRow key={item.name} check={item} />
                  ))}
                </ul>
              </Card>
            </StaggerItem>
          ))}
        </Stagger>
      )}
    </main>
  )
}

/** The answer in one look: a ring that draws itself in the colour of the worst finding. */
function Verdict({ report, checking }: { report: SystemReport | null; checking: boolean }) {
  const level: CheckLevel | null = report ? (report.failed ? 'fail' : report.warned ? 'warn' : 'ok') : null
  const title = !report
    ? 'Checking this computer'
    : report.failed
      ? `${report.failed} ${report.failed === 1 ? 'thing' : 'things'} to fix before a session`
      : report.warned
        ? 'Ready for a session'
        : 'Ready for a session'
  const note = !report
    ? 'The checks run side by side and take a few seconds.'
    : report.failed
      ? 'Each one below says what to do.'
      : report.warned
        ? `${report.warned} ${report.warned === 1 ? 'thing is' : 'things are'} worth a look; none of them stops a session.`
        : 'Every check passed.'
  const Icon = level ? LEVEL[level].icon : Check
  return (
    <Card className="relative mt-6 overflow-hidden p-6 sm:p-7">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 -right-16 size-[320px] rounded-full opacity-60 blur-2xl transition-colors duration-700"
        style={{ background: `radial-gradient(circle, ${level ? LEVEL[level].ring : 'var(--color-accent-300)'}33, transparent 65%)` }}
      />
      <div className="relative flex flex-wrap items-center gap-6">
        <div className="relative grid size-[84px] shrink-0 place-items-center">
          <svg viewBox="0 0 84 84" className="absolute inset-0 size-full -rotate-90" aria-hidden="true">
            <circle cx="42" cy="42" r="37" fill="none" stroke="var(--color-ink-100)" strokeWidth="6" />
            {report ? (
              <motion.circle
                key={level}
                cx="42"
                cy="42"
                r="37"
                fill="none"
                stroke={LEVEL[level!].ring}
                strokeWidth="6"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.9, ease: EASE_OUT }}
              />
            ) : (
              <motion.circle
                cx="42"
                cy="42"
                r="37"
                fill="none"
                stroke="var(--color-accent-500)"
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray="60 200"
                animate={{ rotate: 360 }}
                style={{ originX: '50%', originY: '50%' }}
                transition={{ duration: 1.1, ease: 'linear', repeat: Infinity }}
              />
            )}
          </svg>
          <AnimatePresence mode="wait">
            {report && (
              <motion.span
                key={level}
                className={cn('grid size-11 place-items-center rounded-full', LEVEL[level!].tone)}
                initial={{ scale: 0.4, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 380, damping: 22, delay: 0.5 }}
              >
                <Icon className="size-6" strokeWidth={2.4} />
              </motion.span>
            )}
          </AnimatePresence>
        </div>
        <div className="min-w-0 flex-1">
          <p className="display text-[24px] leading-tight font-semibold text-ink-950">{title}</p>
          <p className="mt-1 text-[14px] text-ink-600">{note}</p>
          {report && (
            <p className="mt-2 font-mono text-[11.5px] text-ink-400">
              checked {report.checked_at.replace('T', ' ')} · {(report.took_ms / 1000).toFixed(1)} s{checking ? ' · checking again' : ''}
            </p>
          )}
        </div>
      </div>
    </Card>
  )
}

function GroupPill({ checks }: { checks: SystemCheck[] }) {
  const failed = checks.filter((c) => c.level === 'fail').length
  const warned = checks.filter((c) => c.level === 'warn').length
  if (failed) return <Pill tone="critical">{failed} to fix</Pill>
  if (warned) return <Pill tone="warning">{warned} to look at</Pill>
  return <Pill tone="good">All ready</Pill>
}

function CheckRow({ check }: { check: SystemCheck }) {
  const level = LEVEL[check.level]
  const Icon = level.icon
  const toast = useToast()
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(check.fix)
      toast.done('Copied')
    } catch {
      /* the command stays on screen to copy by hand */
    }
  }
  return (
    <li className="flex gap-3.5 px-5 py-3.5">
      <motion.span
        className={cn('mt-0.5 grid size-6 shrink-0 place-items-center rounded-full', level.tone)}
        initial={{ scale: 0.5, opacity: 0 }}
        whileInView={{ scale: 1, opacity: 1 }}
        viewport={{ once: true }}
        transition={{ type: 'spring', stiffness: 420, damping: 24 }}
        aria-label={level.label}
      >
        <Icon className="size-3.5" strokeWidth={2.6} />
      </motion.span>
      <div className="min-w-0 flex-1">
        <p className="text-[13.5px] font-semibold text-ink-950">{check.name}</p>
        <p className="mt-0.5 text-[13px] leading-snug break-words text-ink-600">{check.detail}</p>
        {check.fix && check.level !== 'ok' && (
          isCommand(check.fix) ? (
            <div className="mt-2 flex items-center gap-2">
              <code className="min-w-0 flex-1 truncate rounded-[7px] bg-ink-950 px-2.5 py-1.5 font-mono text-[12px] text-ink-100">{check.fix}</code>
              <Button size="sm" variant="ghost" icon={<Copy className="size-3.5" />} onClick={copy} aria-label={`Copy: ${check.fix}`}>
                Copy
              </Button>
            </div>
          ) : (
            <p className="mt-1.5 text-[12.5px] font-medium text-ink-800">{check.fix}</p>
          )
        )}
      </div>
    </li>
  )
}
