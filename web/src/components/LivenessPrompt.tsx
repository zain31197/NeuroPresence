import { ArrowLeft, ArrowRight, Eye, Laugh, ScanFace } from 'lucide-react'
import type { ReactNode } from 'react'
import type { ConsentPrompt, LivenessAction } from '../lib/api'
import { cn } from '../lib/cn'
import { Spinner } from './ui/Button'
import { missNote, mouthShare, stepLine, timeLeftShare, turnSide } from './liveness'

const ICON: Record<LivenessAction, ReactNode> = {
  blink: <Eye className="size-5" />,
  turn_left: <ArrowLeft className="size-5" />,
  turn_right: <ArrowRight className="size-5" />,
  open_mouth: <Laugh className="size-5" />,
}

/**
 * The liveness prompt, drawn over a mirrored camera picture: what to do now, in words large enough
 * to read while looking at the camera, how long is left, and how far it has got: for a turn of the
 * head a marker on the side to turn to, for an open mouth a ring around the sign. An action that
 * was not seen in time is asked for once more, with a line on what was missing. Put it inside the
 * box that holds the picture.
 */
export function LivenessPrompt({ prompt }: { prompt: ConsentPrompt }) {
  const share = timeLeftShare(prompt)
  const side = turnSide(prompt.action)
  const step = stepLine(prompt)
  const mouth = mouthShare(prompt)
  const note = missNote(prompt)
  const icon =
    prompt.stage === 'confirming' ? <Spinner className="size-5" /> : prompt.action ? ICON[prompt.action] : <ScanFace className="size-5" />
  return (
    <>
      {side && <TurnMarker side={side} progress={prompt.progress ?? 0} />}
      <div className="pointer-events-none absolute inset-x-3 bottom-3 flex justify-center">
        <div
          role="status"
          aria-live="assertive"
          className="w-full max-w-[520px] overflow-hidden rounded-panel bg-black/75 text-white shadow-raised backdrop-blur-sm"
        >
          <div className="flex items-center gap-3 px-3.5 py-3">
            {/* For an open mouth the ring around the sign fills as the mouth opens. */}
            <span
              className="grid size-11 shrink-0 place-items-center rounded-full transition-[background] duration-150"
              style={{
                background:
                  mouth === null ? 'transparent' : `conic-gradient(var(--color-tracking) ${mouth * 360}deg, rgb(255 255 255 / 0.28) 0deg)`,
              }}
            >
              <span className={cn('grid size-9 place-items-center rounded-full', prompt.action ? 'bg-white text-ink-950' : 'bg-white/15 text-white')}>
                {icon}
              </span>
            </span>
            <div className="min-w-0 flex-1">
              <p className="label-caps text-white/65">Checking that it is you{step && ` · ${step}`}</p>
              <p className="mt-0.5 text-[15px] leading-snug font-semibold sm:text-[18px]">{prompt.prompt}</p>
              {note && <p className="mt-0.5 text-[12.5px] leading-snug text-warning">{note}</p>}
            </div>
            {share !== null && prompt.seconds_left !== null && (
              <span className="shrink-0 font-mono text-[13px] text-white/80 tabular-nums">{prompt.seconds_left.toFixed(1)} s</span>
            )}
          </div>
          {/* The time left for this action. Kept in place between actions so the box does not jump. */}
          <div className="h-1 bg-white/15">
            <div
              className={cn('h-full bg-tracking transition-[width] duration-300 ease-linear', share === null && 'opacity-0')}
              style={{ width: `${(share ?? 1) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </>
  )
}

/** Which way to turn, at that side of the picture, with a ring that fills as the turn is seen. */
function TurnMarker({ side, progress }: { side: 'left' | 'right'; progress: number }) {
  const turned = Math.min(1, Math.max(0, progress))
  const Arrow = side === 'left' ? ArrowLeft : ArrowRight
  return (
    <div
      aria-hidden="true"
      className={cn('pointer-events-none absolute inset-y-0 grid w-[22%] place-items-center pb-16', side === 'left' ? 'left-0' : 'right-0')}
    >
      <div
        className="grid size-[72px] place-items-center rounded-full shadow-raised transition-[background] duration-150"
        style={{ background: `conic-gradient(var(--color-tracking) ${turned * 360}deg, rgb(255 255 255 / 0.28) 0deg)` }}
      >
        <div className="grid size-[60px] place-items-center rounded-full bg-black/75 text-white backdrop-blur-sm">
          <Arrow className={cn('size-7', turned < 1 && 'animate-pulse')} strokeWidth={2.5} />
        </div>
      </div>
    </div>
  )
}
