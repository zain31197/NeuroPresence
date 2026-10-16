import { Mic, Minus, Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Card } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { Select } from '../../components/ui/Select'
import { Switch } from '../../components/ui/Switch'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError } from '../../lib/api'
import { useStudio } from '../../lib/studio'
import { meterShare, offsetLabel, voiceLine } from './virtualMicrophone'

const STEP_MS = 10
const LIMIT_MS = 200

const stepButton =
  'grid size-7 place-items-center rounded-control border border-line-strong bg-surface text-ink-700 shadow-control transition-colors hover:bg-ink-50 disabled:pointer-events-none disabled:opacity-45'

/**
 * The switch that sends the voice to meeting applications, held back by as long as the picture takes.
 * With it: which microphone is read, how loud it is, and a fine adjustment for the delays the meeting
 * application adds on its own, which the engine cannot measure.
 */
export function VirtualMicrophoneBar() {
  const { status } = useStudio()
  const toast = useToast()
  const mic = status?.virtual_microphone
  // What the person just chose, shown at once; the engine's own answer replaces it a moment later.
  const [chosen, setChosen] = useState<boolean | null>(null)
  const [offset, setOffset] = useState<number | null>(null)
  useEffect(() => {
    if (chosen !== null && mic?.on === chosen) setChosen(null)
  }, [mic?.on, chosen])
  useEffect(() => {
    if (offset !== null && mic?.offset_ms === offset) setOffset(null)
  }, [mic?.offset_ms, offset])

  const line = voiceLine(mic)
  if (!mic || !line) return null

  const change = async (patch: { on?: boolean; microphone?: string; offset_ms?: number }) => {
    try {
      await api.changeVirtualMicrophone(patch)
    } catch (error) {
      setChosen(null)
      setOffset(null)
      toast.problem(error instanceof ApiError ? error.message : 'The virtual microphone could not be changed.')
    }
  }
  const on = chosen ?? mic.on
  const shownOffset = offset ?? mic.offset_ms
  const nudge = (by: number) => {
    const next = Math.max(-LIMIT_MS, Math.min(LIMIT_MS, shownOffset + by))
    setOffset(next)
    void change({ offset_ms: next })
  }

  // The switch stays beside the title at any width; the controls go below the words on a narrow screen.
  return (
    <Card className="flex flex-wrap items-center gap-x-4 gap-y-3 px-5 py-3.5">
      <span className="order-1 grid size-9 shrink-0 place-items-center rounded-full bg-ink-50 text-ink-700 ring-1 ring-line">
        <Mic className="size-4" />
      </span>
      <div className="order-2 min-w-0 flex-1 basis-[200px]">
        <div className="flex items-center gap-2">
          <label htmlFor="virtual-microphone" className="cursor-pointer text-[13.5px] font-semibold text-ink-950">
            Voice in step
          </label>
          <Pill tone={line.tone}>{line.state}</Pill>
        </div>
        <p className="mt-0.5 text-[12.5px] leading-snug text-ink-600">{line.text}</p>
      </div>

      {mic.microphones.length > 0 && (
        <div className="order-4 flex w-full flex-wrap items-center gap-x-4 gap-y-2 pl-[52px] lg:order-3 lg:w-auto lg:pl-0">
          <div className="flex items-center gap-2.5">
            <Select
              label="Microphone"
              className="h-8 max-w-[260px] min-w-[200px] text-[12.5px]"
              value={mic.microphone ?? ''}
              onValueChange={(name) => void change({ microphone: name })}
              groups={[{ title: 'Microphone', options: mic.microphones.map((name) => ({ value: name, label: name })) }]}
            />
            <span
              className="h-1.5 w-12 overflow-hidden rounded-full bg-ink-100"
              role="meter"
              aria-label="Microphone level"
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={on ? mic.level : 0}
            >
              <span
                className="block h-full rounded-full bg-accent-500 transition-[width] duration-200 ease-out-soft"
                style={{ width: `${meterShare(on ? mic.level : 0) * 100}%` }}
              />
            </span>
          </div>
          <div
            className="flex items-center gap-1"
            title="Words before lips: raise it. Lips before words: lower it. The meeting app adds delays of its own that cannot be measured from here."
          >
            <span className="mr-1 text-[12px] text-ink-500">Adjust</span>
            <button type="button" aria-label="Hold the voice back less" disabled={shownOffset <= -LIMIT_MS} onClick={() => nudge(-STEP_MS)} className={stepButton}>
              <Minus className="size-3.5" />
            </button>
            <span className="w-[58px] text-center font-mono text-[12.5px] text-ink-900 tabular-nums" aria-live="polite">
              {offsetLabel(shownOffset)}
            </span>
            <button type="button" aria-label="Hold the voice back more" disabled={shownOffset >= LIMIT_MS} onClick={() => nudge(STEP_MS)} className={stepButton}>
              <Plus className="size-3.5" />
            </button>
          </div>
        </div>
      )}

      <div className="order-3 flex lg:order-4">
        <Switch
          id="virtual-microphone"
          label="Voice in step"
          checked={on}
          onCheckedChange={(next) => {
            setChosen(next)
            void change({ on: next })
          }}
        />
      </div>
    </Card>
  )
}
