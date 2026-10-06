import { Check, ScanFace, TriangleAlert, UserRound, VideoOff } from 'lucide-react'
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { Empty, Monitor, paint, Waiting } from '../../components/Monitor'
import { Spinner } from '../../components/ui/Button'
import { Pill } from '../../components/ui/Pill'
import { candidatePictureUrl, enrolledPictureUrl, type Candidate, type EnrolmentRecord, type FaceBox, type Preview } from '../../lib/api'
import { cn } from '../../lib/cn'
import { splitHint } from '../../lib/format'
import { useStudio } from '../../lib/studio'
import { outlineEllipse } from './outline'

/** What the Enrolment screen is showing: the open camera, a picture to review, the enrolled picture, or nothing yet. */
export type Mode = 'camera' | 'review' | 'enrolled' | 'empty'

interface Size {
  width: number
  height: number
}

interface Props {
  mode: Mode
  preview: Preview | null
  candidate: Candidate | null
  record: EnrolmentRecord | null
  /** Seconds left before the picture is taken, or null when not counting down. */
  count: number | null
  taking: boolean
  /** Goes up by one each time a picture has been taken, to play the flash. */
  shots: number
  /** What to offer when the camera could not be opened. */
  retry: ReactNode
}

/** The one large picture of the Enrolment screen. */
export function Stage({ mode, preview, candidate, record, count, taking, shots, retry }: Props) {
  const { onFrame } = useStudio()
  const canvas = useRef<HTMLCanvasElement>(null)
  const [size, setSize] = useState<Size | null>(null)
  const running = preview?.state === 'running'

  useEffect(() => {
    let decoding = false
    return onFrame(async (header, cameraJpeg) => {
      if (header.kind !== 'preview' || decoding) return // skip a frame rather than queue it behind the one being drawn
      decoding = true
      try {
        const bitmap = await createImageBitmap(cameraJpeg)
        paint(canvas.current, bitmap)
        const { width, height } = bitmap
        bitmap.close()
        setSize((shown) => (shown && shown.width === width && shown.height === height ? shown : { width, height }))
      } finally {
        decoding = false
      }
    })
  }, [onFrame])

  useEffect(() => {
    if (!running) setSize(null) // so a camera opened later never starts on a frame from before
  }, [running])

  const camera = mode === 'camera'
  const showing = camera && running && size !== null
  // The frame keeps its own shape inside the 16:9 screen, so the outline drawn over it lines up with the picture.
  const shape: CSSProperties | undefined = size
    ? { aspectRatio: `${size.width} / ${size.height}`, ...(size.width / size.height >= 16 / 9 ? { width: '100%' } : { height: '100%' }) }
    : undefined

  const reviewing = mode === 'review' && candidate !== null
  const enrolled = mode === 'enrolled' && record !== null

  return (
    <Monitor
      label={camera ? 'Camera' : enrolled ? 'Enrolled picture' : 'Picture'}
      detail={showing ? 'Mirrored' : reviewing ? (candidate.origin === 'camera' ? 'Just taken' : 'Uploaded') : undefined}
      lit={showing || reviewing || enrolled}
      chip={
        showing && preview ? (
          <Progress preview={preview} />
        ) : reviewing ? (
          candidate.passed ? (
            <Pill tone="good">Passed every check</Pill>
          ) : (
            <Pill tone="serious">Cannot be enrolled</Pill>
          )
        ) : undefined
      }
      footer={
        reviewing ? (
          <span>{`${candidate.width} × ${candidate.height}`}</span>
        ) : enrolled ? (
          <span>{`${record.width} × ${record.height}`}</span>
        ) : undefined
      }
    >
      {/* The canvas stays in place in every mode, so a retake shows the camera at once. */}
      <div className={cn('absolute inset-0 grid place-items-center', !showing && 'hidden')}>
        <div className="relative max-h-full max-w-full" style={shape}>
          <canvas ref={canvas} role="img" aria-label="Camera preview, mirrored" className="size-full -scale-x-100" />
          {size && preview?.outline && <Outline size={size} box={preview.outline} ready={preview.ready} />}
        </div>
      </div>

      {camera &&
        !showing &&
        (preview?.state === 'error' ? (
          <Empty icon={<VideoOff className="size-5" />} title="The camera could not be opened" hint={preview.message} action={retry} />
        ) : (
          <Waiting message={preview?.message || 'Waiting for the camera'} />
        ))}
      {reviewing && (
        <img
          src={candidatePictureUrl(candidate.id)}
          alt={candidate.origin === 'camera' ? 'The picture just taken' : 'The uploaded picture'}
          className="absolute inset-0 size-full object-contain"
        />
      )}
      {enrolled && <img src={enrolledPictureUrl(record.id)} alt="Your enrolled picture" className="absolute inset-0 size-full object-contain" />}
      {mode === 'empty' && (
        <Empty
          icon={<UserRound className="size-5" />}
          title="No picture enrolled yet"
          hint="Open the camera and follow the checks, or upload a picture you already have."
        />
      )}

      {showing && preview && <Guidance preview={preview} taking={taking} place="over" />}
      {showing && count !== null && <Countdown count={count} />}
      {shots > 0 && <div key={shots} aria-hidden="true" className="pointer-events-none absolute inset-0 animate-flash bg-white" />}
    </Monitor>
  )
}

/**
 * Where the face should be: an outline over the camera picture, with the rest dimmed a little.
 * The engine says where (a face that fills it passes the size and framing checks, and is as
 * large as they allow). It is drawn in the picture's own pixels, so it sits in the same place
 * whatever size the screen is.
 */
function Outline({ size, box, ready }: { size: Size; box: FaceBox; ready: boolean }) {
  const { width, height } = size
  const { cx, cy, rx, ry } = outlineEllipse(box, width, height)
  const ellipse = `M${cx - rx} ${cy}a${rx} ${ry} 0 1 0 ${2 * rx} 0a${rx} ${ry} 0 1 0 ${-2 * rx} 0Z`
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="pointer-events-none absolute inset-0 size-full" aria-hidden="true">
      <path d={`M0 0H${width}V${height}H0Z ${ellipse}`} fillRule="evenodd" fill="rgb(13 15 20 / 0.3)" />
      <path d={ellipse} fill="none" stroke="rgb(13 15 20 / 0.4)" strokeWidth="5" vectorEffect="non-scaling-stroke" />
      <path
        d={ellipse}
        fill="none"
        stroke={ready ? 'var(--color-tracking)' : 'rgb(255 255 255 / 0.9)'}
        strokeWidth="2.5"
        vectorEffect="non-scaling-stroke"
        className="transition-[stroke] duration-200"
      />
    </svg>
  )
}

function Progress({ preview }: { preview: Preview }) {
  if (!preview.checks) return null
  if (preview.ready) return <Pill tone="good">Ready</Pill>
  const passed = preview.checks.filter((check) => check.passed).length
  return (
    <Pill tone="neutral" icon={<ScanFace className="size-3" />} className="bg-white text-ink-800">
      <span className="tabular-nums">
        {passed} of {preview.checks.length}
      </span>{' '}
      checks
    </Pill>
  )
}

/**
 * The one thing to do next. On the picture itself, and large enough to read from where the
 * person is sitting: they are looking at the camera, not at the list beside it. On a narrow
 * screen the picture is too small to carry it, so the same line goes under the picture instead.
 */
export function Guidance({ preview, taking, place }: { preview: Preview; taking: boolean; place: 'over' | 'under' }) {
  if (!preview.checks) return null
  const over = place === 'over'
  const quiet = over ? 'text-white/75' : 'text-ink-600'
  let mark: ReactNode
  let line: ReactNode
  let tip = ''
  if (taking) {
    mark = <Spinner className="mt-0.5 size-4 shrink-0" />
    line = <span className="font-semibold">Hold still.</span>
  } else if (preview.hint) {
    const [problem, fix] = splitHint(preview.hint)
    mark = <TriangleAlert className={cn('mt-0.5 size-4 shrink-0', over ? 'text-serious' : 'text-serious-ink')} strokeWidth={2.25} />
    line = (
      <>
        {problem && <span className={quiet}>{problem} </span>}
        <span className="font-semibold">{fix}</span>
      </>
    )
  } else if (preview.ready) {
    mark = <Check className={cn('mt-0.5 size-4 shrink-0', over ? 'text-tracking' : 'text-good-ink')} strokeWidth={3} />
    line = (
      <>
        <span className="font-semibold">Ready.</span> <span className={quiet}>Look at the camera and take the picture.</span>
      </>
    )
    tip = preview.tip
  } else {
    mark = <ScanFace className={cn('mt-0.5 size-4 shrink-0', over ? 'text-white/70' : 'text-ink-500')} />
    line = <span className="font-semibold">Hold still.</span>
  }
  const words = (
    <div className="min-w-0">
      <p className={cn('leading-snug', over ? 'text-[14.5px]' : 'text-[13.5px]')}>{line}</p>
      {tip && <p className={cn('mt-0.5 text-[12.5px] leading-snug', over ? 'text-white/70' : 'text-ink-500')}>{tip}</p>}
    </div>
  )
  if (!over) {
    return (
      <div role="status" className="mx-1 mt-2 flex items-start gap-2.5 rounded-panel bg-ink-50 px-3 py-2.5 text-ink-950 sm:hidden">
        {mark}
        {words}
      </div>
    )
  }
  return (
    <div className="pointer-events-none absolute inset-x-3 bottom-3 hidden justify-center sm:flex">
      <div role="status" className="flex max-w-[620px] items-start gap-2.5 rounded-panel bg-black/70 px-3.5 py-2.5 text-white shadow-raised backdrop-blur-sm">
        {mark}
        {words}
      </div>
    </div>
  )
}

function Countdown({ count }: { count: number }) {
  return (
    // Beside the face, not over it: the person still needs to see that they are holding the pose.
    <div role="timer" aria-live="assertive" className="pointer-events-none absolute inset-y-0 left-0 grid w-[27%] place-items-center pb-10">
      <div className="text-center">
        <p
          key={count}
          className="display animate-count text-[clamp(64px,9vw,128px)] leading-none font-semibold text-white tabular-nums [text-shadow:0_4px_32px_rgb(0_0_0/0.65)]"
        >
          {count}
        </p>
        <p className="mt-2 text-[13px] font-medium text-white/90 [text-shadow:0_1px_8px_rgb(0_0_0/0.7)]">Look at the camera</p>
      </div>
    </div>
  )
}
