import { ArrowRight, AudioLines, Webcam, type LucideIcon } from 'lucide-react'
import { motion } from 'motion/react'
import { SpotlightCard } from '../../components/fx'
import { Stagger, StaggerItem } from '../../components/motion'
import { Pill, type Tone } from '../../components/ui/Pill'
import type { PipelineStage } from '../../lib/api'
import { FaceMeshCanvas } from './FaceMeshCanvas'

const STATUS: Record<PipelineStage['status'], { tone: Tone; label: string }> = {
  working: { tone: 'good', label: 'Working' },
  partial: { tone: 'warning', label: 'Partly built' },
  planned: { tone: 'neutral', label: 'Planned' },
}

// Shown when the engine is not running, without any claim about what is built.
const OUTLINE: Pick<PipelineStage, 'key' | 'name' | 'summary'>[] = [
  { key: 'camera', name: 'Camera on', summary: 'Your face at the camera drives your picture: head pose, lips, eyes and expression, as you move.' },
  { key: 'voice', name: 'Camera off', summary: 'Your voice drives your picture: a speech to motion model turns what you say into lips, head and blinks.' },
]

const ICONS: Record<string, LucideIcon> = { camera: Webcam, voice: AudioLines }

/** The two ways the picture is driven, each with what the engine says about it. */
export function Modes({ modes }: { modes: PipelineStage[] | null }) {
  const list: (Pick<PipelineStage, 'key' | 'name' | 'summary'> & Partial<PipelineStage>)[] = modes ?? OUTLINE
  return (
    <Stagger className="mt-12 grid gap-4 md:grid-cols-2">
      {list.map((mode) => {
        const Icon = ICONS[mode.key] ?? Webcam
        return (
          <StaggerItem key={mode.key}>
            <SpotlightCard className="group flex h-full flex-col rounded-card border border-line bg-surface shadow-card transition-shadow duration-300 hover:shadow-lift">
            <div className="relative grid h-[220px] place-items-center overflow-hidden border-b border-line bg-[radial-gradient(70%_80%_at_50%_30%,var(--color-accent-50),var(--color-surface))]">
              {mode.key === 'voice' ? <VoiceVisual /> : <CameraVisual />}
            </div>
            <div className="flex flex-1 flex-col p-6">
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2.5">
                  <span className="grid size-9 place-items-center rounded-[11px] bg-ink-950 text-white">
                    <Icon className="size-[18px]" strokeWidth={1.9} />
                  </span>
                  <h3 className="display text-[19px] font-semibold">{mode.name}</h3>
                </span>
                {mode.status && <Pill tone={STATUS[mode.status].tone}>{STATUS[mode.status].label}</Pill>}
              </div>
              <p className="mt-3 text-[15px] leading-relaxed text-ink-700">{mode.summary}</p>
              {mode.note && <p className="mt-4 border-t border-line pt-3 text-[13px] leading-snug text-ink-500">{mode.note}</p>}
            </div>
            </SpotlightCard>
          </StaggerItem>
        )
      })}
    </Stagger>
  )
}

/** The face at the camera, read point by point. */
function CameraVisual() {
  return (
    <div className="relative h-[180px] w-[150px]">
      <FaceMeshCanvas className="size-full" meshColor="rgb(80 98 240 / 0.35)" contourColor="rgb(47 59 189 / 0.9)" label="A face drawn from its landmarks, turning and talking." />
      <span aria-hidden="true" className="absolute inset-0 rounded-[14px] border border-dashed border-accent-300" />
      {[
        'top-0 left-0 border-t-2 border-l-2 rounded-tl-[10px]',
        'top-0 right-0 border-t-2 border-r-2 rounded-tr-[10px]',
        'bottom-0 left-0 border-b-2 border-l-2 rounded-bl-[10px]',
        'bottom-0 right-0 border-r-2 border-b-2 rounded-br-[10px]',
      ].map((corner) => (
        <span key={corner} aria-hidden="true" className={`absolute size-5 border-accent-500 ${corner}`} />
      ))}
    </div>
  )
}

/** Speech becoming movement: a voice's level, and the face it drives. */
function VoiceVisual() {
  const bars = [0.35, 0.7, 1, 0.55, 0.85, 0.4, 0.95, 0.6, 0.3, 0.75, 0.5, 0.9]
  return (
    <div className="flex items-center gap-6" aria-hidden="true">
      <div className="flex h-24 items-center gap-[5px]">
        {bars.map((height, index) => (
          <motion.span
            key={index}
            className="w-[5px] origin-center rounded-full bg-accent-500"
            style={{ height: `${height * 100}%` }}
            animate={{ scaleY: [0.25, 1, 0.45, 0.9, 0.25] }}
            transition={{ duration: 1.6 + (index % 4) * 0.18, ease: 'easeInOut', repeat: Infinity, delay: index * 0.07 }}
          />
        ))}
      </div>
      <motion.span className="text-accent-500" animate={{ x: [0, 4, 0] }} transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}>
        <ArrowRight className="size-5" />
      </motion.span>
      <div className="h-[170px] w-[140px]">
        <FaceMeshCanvas className="size-full" meshColor="rgb(104 112 131 / 0.3)" contourColor="rgb(20 23 30 / 0.85)" delay={0.4} label="A face moving with the voice." />
      </div>
    </div>
  )
}
