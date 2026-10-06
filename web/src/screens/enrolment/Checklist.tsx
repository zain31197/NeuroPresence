import { Check as Tick, Info, Minus, X } from 'lucide-react'
import type { ReactNode } from 'react'
import { Tooltip } from '../../components/ui/Tooltip'
import type { Check, CheckKey, EnrolmentLimits } from '../../lib/api'
import { cn } from '../../lib/cn'
import { splitHint } from '../../lib/format'

/** The seven checks in their fixed order, so the list can be drawn before the first frame has been judged. */
const LABELS: [CheckKey, string][] = [
  ['face', 'One face in view'],
  ['facing', 'Facing the camera'],
  ['size', 'Face large enough'],
  ['framing', 'Whole head in frame'],
  ['light', 'Enough light'],
  ['sharp', 'Sharp'],
  ['expression', 'Relaxed face, eyes open'],
]

/** The readings shown as numbers: the ones that depend on the camera and the room. */
const READOUT: Partial<Record<CheckKey, (value: number) => string>> = {
  size: (value) => `${Math.round(value)} px`,
  light: (value) => `${Math.round(value)}`,
  sharp: (value) => `${Math.round(value)}`,
}

/** What a check asks for, with the limit it applies. */
function about(key: CheckKey, limits: EnrolmentLimits | null): string {
  switch (key) {
    case 'face':
      return 'Exactly one face must be in the picture: yours.'
    case 'facing':
      return limits
        ? `Head turned at most ${limits.max_yaw_deg}° to either side, chin at most ${limits.max_pitch_deg}° up or down, and tilted at most ${limits.max_roll_deg}°.`
        : 'The head faces the camera, level and upright.'
    case 'size':
      return limits
        ? `At least ${limits.min_face_height_px} px from forehead to chin, and at most ${Math.round(limits.max_face_height_share * 100)}% of the picture's height. From ${limits.good_face_height_px} px the output is sharper still.`
        : 'The face needs enough pixels: the output can only be as sharp as the picture.'
    case 'framing':
      return 'Room above the forehead for the hair, below the chin and on both sides, with the face near the middle.'
    case 'light':
      return limits
        ? `The brightest parts of the face reach ${limits.min_bright_level} or more on a scale of 255, without burning out to white. Skin tone does not decide this.`
        : 'The face is lit, without burning out to white. Skin tone does not decide this.'
    case 'sharp':
      return limits
        ? `Fine detail on the face, scored ${limits.min_sharpness} or more. From ${limits.good_sharpness} the output itself still counts as sharp.`
        : 'Fine detail on the face: no blur from movement or focus.'
    case 'expression':
      return 'Mouth closed, eyes open, no broad smile. This is the face people see whenever tracking is lost.'
  }
}

type State = 'passed' | 'fault' | 'unmeasured' | 'pending'

function stateOf(check: Check | undefined): State {
  if (!check) return 'pending'
  if (check.passed) return 'passed'
  // With no single face in view, the other checks cannot be measured: they fail without a fault of their own.
  return check.hint ? 'fault' : 'unmeasured'
}

// A status colour never carries meaning alone: each state has its own mark and a spoken label.
const LAMPS: Record<State, { className: string; mark: ReactNode; label: string }> = {
  passed: { className: 'bg-good-wash text-good-ink', mark: <Tick className="size-3" strokeWidth={3} />, label: 'Passed' },
  fault: { className: 'bg-serious-wash text-serious-ink', mark: <X className="size-3" strokeWidth={3} />, label: 'Needs fixing' },
  unmeasured: { className: 'bg-ink-100 text-ink-400', mark: <Minus className="size-3" strokeWidth={3} />, label: 'Not measured' },
  pending: { className: 'bg-ink-100 text-ink-300', mark: <Minus className="size-3" strokeWidth={3} />, label: 'Waiting for the camera' },
}

interface Props {
  /** null before the first frame has been judged. */
  checks: Check[] | null
  limits: EnrolmentLimits | null
  /**
   * Show under each check what is wrong with it, or what would make it better.
   * Off for the live camera, where rows that grow and shrink as the person moves
   * would make the list jump; the one thing to fix is shown on the picture instead.
   */
  detail: boolean
  /**
   * Include what to do about a fault ("Move closer to the camera."), not only what it is.
   * Off for an uploaded picture, which nobody is sitting in front of.
   */
  advice?: boolean
}

export function Checklist({ checks, limits, detail, advice = true }: Props) {
  const found = new Map(checks?.map((check) => [check.key, check]))
  return (
    <ul className="divide-y divide-line" aria-label="Enrolment checks">
      {LABELS.map(([key, fallback]) => {
        const check = found.get(key)
        const state = stateOf(check)
        const lamp = LAMPS[state]
        const readout = check && typeof check.value === 'number' ? READOUT[key]?.(check.value) : undefined
        const said = detail && check ? (check.passed ? check.tip : check.hint) : undefined
        const note = said && !advice ? splitHint(said)[0] || said : said
        return (
          <li key={key} className="flex items-start gap-3 py-2.5">
            <span className={cn('mt-px grid size-5 shrink-0 place-items-center rounded-full transition-colors duration-200', lamp.className)}>
              {lamp.mark}
              <span className="sr-only">{lamp.label}:</span>
            </span>
            <div className="min-w-0 flex-1">
              <p className={cn('flex items-center gap-1.5 text-[13.5px] font-medium', state === 'passed' || state === 'fault' ? 'text-ink-900' : 'text-ink-500')}>
                {check?.label ?? fallback}
                <Tooltip content={about(key, limits)}>
                  <button type="button" aria-label={`About the check: ${check?.label ?? fallback}`} className="shrink-0 rounded-full text-ink-300 hover:text-ink-600">
                    <Info className="size-3.5" />
                  </button>
                </Tooltip>
              </p>
              {note && <p className={cn('mt-0.5 text-[12.5px] leading-snug', check?.passed ? 'text-ink-500' : 'text-ink-700')}>{note}</p>}
            </div>
            {readout && <span className="shrink-0 pt-0.5 font-mono text-[11.5px] text-ink-500 tabular-nums">{readout}</span>}
          </li>
        )
      })}
    </ul>
  )
}
