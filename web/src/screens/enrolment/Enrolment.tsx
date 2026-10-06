import { ArrowRight, Camera, RefreshCw, Square, Trash2, Upload, Video, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type ChangeEvent, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Monitor, Waiting } from '../../components/Monitor'
import { Button, buttonClass } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { Notice } from '../../components/ui/Notice'
import { Pill } from '../../components/ui/Pill'
import { Select } from '../../components/ui/Select'
import { useToast } from '../../components/ui/Toast'
import { Tooltip } from '../../components/ui/Tooltip'
import { api, ApiError, STOP_PREVIEW_PATH, type Candidate, type EnrolmentGuide, type EnrolmentLimits, type EnrolmentRecord, type Preview } from '../../lib/api'
import { dateTime, splitHint } from '../../lib/format'
import { useStudio } from '../../lib/studio'
import { Checklist } from './Checklist'
import { Evidence } from './Evidence'
import { Guidance, Stage, type Mode } from './Stage'

/** The one request in flight, so its button can show that it is working and the others wait. */
type Work = 'camera' | 'take' | 'upload' | 'confirm' | 'discard' | 'remove' | 'session'

const COUNTDOWN_SECONDS = 3

const valueOf = (record: EnrolmentRecord | null, key: string) => {
  const value = record?.checks.find((check) => check.key === key)?.value
  return typeof value === 'number' ? value : null
}

export function Enrolment() {
  const { status, connection, refresh } = useStudio()
  const toast = useToast()
  const fileInput = useRef<HTMLInputElement>(null)
  const [camera, setCamera] = useState('camera:0')
  const [work, setWork] = useState<Work | null>(null)
  const [count, setCount] = useState<number | null>(null)
  const [shots, setShots] = useState(0)
  const [guide, setGuide] = useState<EnrolmentGuide | null>(null)

  useEffect(() => {
    document.title = 'Enrolment · NeuroPresence'
    api.enrolmentGuide().then(setGuide, () => setGuide(null)) // without it the screen still works, it just explains less
  }, [])

  const preview = status?.enrolment.preview ?? null
  const candidate = status?.enrolment.candidate ?? null
  const record = status?.enrolment.record ?? null
  const session = status?.session
  const sessionActive = session?.state === 'running' || session?.state === 'starting'
  const cameraOpen = preview?.state === 'running' || preview?.state === 'starting'
  const cameraFailed = preview?.state === 'error'
  const mode: Mode = candidate ? 'review' : cameraOpen || cameraFailed ? 'camera' : record ? 'enrolled' : 'empty'
  const cameras = (status?.inputs ?? []).filter((input) => input.kind === 'camera')
  const offline = connection !== 'open'
  const busy = work !== null

  /** Run one request, then read the new state at once. Says so if it was refused. */
  const run = useCallback(
    async (kind: Work, action: () => Promise<unknown>, done?: string) => {
      setWork(kind)
      try {
        await action()
        await refresh()
        if (done) toast.done(done)
        return true
      } catch (error) {
        toast.problem(error instanceof ApiError ? error.message : 'Something went wrong. Try again.')
        await refresh().catch(() => undefined)
        return false
      } finally {
        setWork(null)
      }
    },
    [refresh, toast],
  )

  const openCamera = () => run('camera', () => api.startPreview(camera))
  const closeCamera = () => run('camera', api.stopPreview)
  const changeCamera = (input: string) => {
    setCamera(input)
    if (cameraOpen || cameraFailed) {
      void run('camera', async () => {
        await api.stopPreview()
        await api.startPreview(input)
      })
    }
  }
  const take = useCallback(async () => {
    if (await run('take', api.takePicture)) setShots((taken) => taken + 1)
  }, [run])
  const chooseFile = () => fileInput.current?.click()
  const onFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = '' // so choosing the same file again still fires
    if (file) void run('upload', () => api.uploadPicture(file))
  }

  // The count of three before a picture: time to look up from the button to the camera.
  useEffect(() => {
    if (count === null) return
    if (count === 0) {
      setCount(null)
      void take()
      return
    }
    const timer = window.setTimeout(() => setCount((left) => (left === null ? null : left - 1)), 1000)
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setCount(null)
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('keydown', onKey)
    }
  }, [count, take])

  useEffect(() => {
    if (mode !== 'camera') setCount(null)
  }, [mode])

  // Leaving this screen, or closing the tab, must not leave the camera filming.
  const open = useRef(false)
  useEffect(() => {
    open.current = cameraOpen
  }, [cameraOpen])
  useEffect(() => {
    const close = () => {
      if (open.current) navigator.sendBeacon(STOP_PREVIEW_PATH)
    }
    window.addEventListener('pagehide', close)
    return () => {
      window.removeEventListener('pagehide', close)
      close()
    }
  }, [])

  const heading = (
    <div className="min-w-0">
      <div className="flex items-center gap-3">
        <h1 className="display text-[24px] leading-tight font-semibold">Enrolment</h1>
        {status &&
          (record ? (
            <Pill tone="good">Picture enrolled</Pill>
          ) : (
            <Pill tone="neutral" icon={null}>
              No picture yet
            </Pill>
          ))}
      </div>
      <p className="mt-1 text-[13.5px] text-ink-500">
        The one picture of you that every output frame is made from. It is checked before it is kept, and it stays on this computer.
      </p>
    </div>
  )
  const engineDown = connection === 'closed' && (
    <Notice className="mt-5" title="The engine is not running">
      Start it in a terminal with <code className="rounded-[5px] bg-ink-100 px-1.5 py-0.5 font-mono text-[12px]">python -m neuropresence.server</code>.
      This page reconnects by itself.
    </Notice>
  )

  if (!status) {
    return (
      <main className="mx-auto max-w-[1520px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
        {heading}
        {engineDown}
        <Card className="mt-5 p-2">
          <Monitor label="Picture" lit={false} className="mx-auto max-w-[960px]">
            <Waiting message="Connecting to the engine" />
          </Monitor>
        </Card>
      </main>
    )
  }

  const chooser = (
    <Select
      label="Camera"
      value={camera}
      onValueChange={changeCamera}
      disabled={offline || busy || count !== null || cameras.length === 0}
      groups={[{ options: cameras.map((option) => ({ value: option.id, label: option.label, icon: <Video className="size-3.5 text-ink-500" /> })) }]}
    />
  )
  const uploadButton = (label: string, variant: 'primary' | 'secondary' = 'secondary') => (
    <Button variant={variant} icon={<Upload className="size-4" />} busy={work === 'upload'} disabled={offline || busy} onClick={chooseFile}>
      {label}
    </Button>
  )
  const cameraButton = (label: string, variant: 'primary' | 'secondary') => (
    <Tooltip content={sessionActive ? 'Stop the live session first: the camera can only be open in one place.' : 'Opens the camera with the checks running. Nothing is kept until you take a picture.'}>
      <span>
        <Button variant={variant} icon={<Camera className="size-4" />} busy={work === 'camera'} disabled={offline || busy || sessionActive} onClick={openCamera}>
          {label}
        </Button>
      </span>
    </Tooltip>
  )

  let left: ReactNode = chooser
  let right: ReactNode
  if (mode === 'review' && candidate) {
    // What is wrong with an uploaded picture, without the advice meant for someone sitting at the camera.
    const fault = splitHint(candidate.hint)[0] || candidate.hint || 'This picture did not pass every check.'
    left = (
      <p className="min-w-0 flex-1 basis-[200px] text-[13px] leading-snug text-ink-600">
        {work === 'confirm'
          ? 'Preparing the picture. The first time, the animation model has to load, which takes a few seconds.'
          : !candidate.passed
            ? `${fault} Choose another picture, or take one with the camera.`
            : candidate.origin === 'camera'
              ? 'Not mirrored: this is how others will see you.'
              : 'An uploaded picture goes through the same checks as one from the camera.'}
      </p>
    )
    right = (
      <>
        {candidate.origin === 'camera' ? (
          <Button icon={<RefreshCw className="size-4" />} busy={work === 'discard'} disabled={busy} onClick={() => run('discard', api.discardCandidate)}>
            Retake
          </Button>
        ) : (
          <>
            <Button variant="ghost" busy={work === 'discard'} disabled={busy} onClick={() => run('discard', api.discardCandidate)}>
              Discard
            </Button>
            {uploadButton(candidate.passed ? 'Choose another' : 'Choose another picture', candidate.passed ? 'secondary' : 'primary')}
          </>
        )}
        {candidate.passed && (
          <Button variant="primary" busy={work === 'confirm'} disabled={offline || busy} onClick={() => run('confirm', api.confirmPicture, 'Picture enrolled.')}>
            Use this picture
          </Button>
        )}
      </>
    )
  } else if (mode === 'camera') {
    left = (
      <div className="flex flex-wrap items-center gap-2">
        {chooser}
        <Button variant="ghost" disabled={offline || busy || count !== null} onClick={closeCamera}>
          Close camera
        </Button>
      </div>
    )
    right = cameraFailed ? (
      uploadButton('Upload a picture')
    ) : count !== null ? (
      <Button icon={<X className="size-4" />} onClick={() => setCount(null)}>
        Cancel
      </Button>
    ) : (
      <Tooltip content={preview?.ready ? 'Counts down from three, then keeps the sharpest frame with your eyes open.' : 'Follow the hint on the picture until every check passes.'}>
        <span>
          <Button
            variant="primary"
            icon={<Camera className="size-4" />}
            busy={work === 'take'}
            disabled={offline || busy || !preview?.ready}
            onClick={() => setCount(COUNTDOWN_SECONDS)}
          >
            Take picture
          </Button>
        </span>
      </Tooltip>
    )
  } else if (mode === 'enrolled') {
    right = (
      <>
        {uploadButton('Upload a picture')}
        {cameraButton('Take a new picture', 'secondary')}
      </>
    )
  } else {
    right = (
      <>
        {uploadButton('Upload a picture')}
        {cameraButton('Open the camera', 'primary')}
      </>
    )
  }

  return (
    <main className="mx-auto max-w-[1520px] px-4 py-6 sm:px-6 lg:px-8 lg:py-7">
      {heading}
      {engineDown}
      {sessionActive && (
        <Notice
          className="mt-5"
          title="A live session is running"
          action={
            <Button size="sm" icon={<Square className="size-3 fill-current" />} busy={work === 'session'} disabled={busy} onClick={() => run('session', api.stopSession)}>
              Stop session
            </Button>
          }
        >
          {session?.source === 'sample'
            ? 'Stop it to take or replace your picture.'
            : 'Stop it to take a new picture with the camera. An uploaded picture can replace the enrolled one while it runs.'}
        </Notice>
      )}

      <div className="@container mt-5">
        <div className="grid gap-5 @4xl:grid-cols-[minmax(0,1.62fr)_minmax(320px,1fr)]">
          <Card className="self-start p-2">
            <Stage
              mode={mode}
              preview={preview}
              candidate={candidate}
              record={record}
              count={count}
              taking={work === 'take'}
              shots={shots}
              retry={
                <Button size="sm" variant="primary" busy={work === 'camera'} disabled={offline || busy} onClick={openCamera}>
                  Try again
                </Button>
              }
            />
            {mode === 'camera' && preview?.state === 'running' && <Guidance preview={preview} taking={work === 'take'} place="under" />}
            <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3 px-3 pt-3.5 pb-2">
              {left}
              <div className="ml-auto flex flex-wrap items-center justify-end gap-2">{right}</div>
            </div>
          </Card>

          {mode === 'review' && candidate ? (
            <ReviewPanel candidate={candidate} limits={guide?.limits ?? null} />
          ) : mode === 'camera' ? (
            <CameraPanel preview={preview} limits={guide?.limits ?? null} onUpload={chooseFile} disabled={offline || busy || count !== null} />
          ) : record ? (
            <EnrolledPanel
              record={record}
              removing={work === 'remove'}
              disabled={offline || busy}
              onRemove={() => run('remove', api.removeEnrolment, 'Enrolled picture removed.')}
            />
          ) : (
            <StartPanel />
          )}
        </div>
      </div>

      <Evidence guide={guide} yours={valueOf(record, 'size')} />
      <input ref={fileInput} type="file" accept="image/png,image/jpeg" className="sr-only" tabIndex={-1} aria-hidden="true" onChange={onFile} />
    </main>
  )
}

/* ------------------------------------------------------------------ panels */

function CameraPanel({ preview, limits, onUpload, disabled }: { preview: Preview | null; limits: EnrolmentLimits | null; onUpload: () => void; disabled: boolean }) {
  return (
    <Card className="flex flex-col">
      <CardHeader title="Checks" hint="Every one has to pass before a picture can be taken." />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <Checklist checks={preview?.state === 'running' ? preview.checks : null} limits={limits} detail={false} />
        <p className="mt-auto border-t border-line pt-3.5 text-[12.5px] leading-relaxed text-ink-500">
          The picture is taken after a count of three, and the sharpest frame with your eyes open is kept. Already have a good
          picture?{' '}
          <button type="button" disabled={disabled} onClick={onUpload} className="font-medium text-ink-900 underline decoration-ink-300 underline-offset-2 hover:decoration-ink-900 disabled:opacity-45">
            Upload it instead
          </button>
          .
        </p>
      </div>
    </Card>
  )
}

function ReviewPanel({ candidate, limits }: { candidate: Candidate; limits: EnrolmentLimits | null }) {
  return (
    <Card className="flex flex-col">
      <CardHeader
        title={candidate.passed ? 'Review the picture' : 'This picture cannot be enrolled'}
        hint={
          candidate.passed
            ? 'It passed every check. Keep it, or try for a better one.'
            : 'One or more checks failed. Each says what is wrong.'
        }
      />
      <div className="px-5 pb-5">
        <Checklist checks={candidate.checks} limits={limits} detail advice={candidate.origin === 'camera'} />
      </div>
    </Card>
  )
}

function EnrolledPanel({ record, removing, disabled, onRemove }: { record: EnrolmentRecord; removing: boolean; disabled: boolean; onRemove: () => void }) {
  const [asking, setAsking] = useState(false)
  const face = valueOf(record, 'size')
  const sharpness = valueOf(record, 'sharp')
  // For an uploaded picture, what could be better without the advice meant for someone at the camera.
  const tips = record.checks.flatMap((check) =>
    check.passed && check.tip ? [record.origin === 'camera' ? check.tip : splitHint(check.tip)[0] || check.tip] : [],
  )
  const facts = [
    ['Enrolled', dateTime(record.enrolled_at)],
    ['From', record.origin === 'camera' ? 'The camera' : 'An uploaded file'],
    ['Picture', `${record.width} × ${record.height} px`],
    face !== null && ['Face', `${Math.round(face)} px tall`],
    sharpness !== null && ['Sharpness', `${Math.round(sharpness)}`],
    ['Face signature', record.has_signature ? 'Saved, for the identity check' : 'Not saved: the identity model is not installed'],
  ].filter((fact): fact is [string, string] => fact !== false)

  return (
    <Card className="flex flex-col">
      <CardHeader title="Your enrolled picture" hint="A camera session animates this picture. It passed every check." />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <dl className="divide-y divide-line text-[13px]">
          {facts.map(([name, value]) => (
            <div key={name} className="flex items-baseline justify-between gap-4 py-2">
              <dt className="shrink-0 text-ink-500">{name}</dt>
              <dd className="min-w-0 text-right text-ink-900 tabular-nums">{value}</dd>
            </div>
          ))}
        </dl>

        {tips.length > 0 && (
          <div className="mt-3 rounded-control bg-ink-50 px-3.5 py-3">
            <p className="label-caps text-ink-500">Could be better</p>
            <ul className="mt-1.5 grid gap-1 text-[12.5px] leading-snug text-ink-700">
              {tips.map((tip) => (
                <li key={tip}>{tip}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-auto grid gap-2 pt-5">
          <Link to="/studio" className={buttonClass('primary', 'md', 'w-full')}>
            Go to Live Studio
            <ArrowRight className="size-3.5" />
          </Link>
          {asking ? (
            <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 rounded-control border border-critical/25 bg-critical-wash px-3 py-2">
              <p className="text-[12.5px] leading-snug text-ink-800">Remove the picture and its face signature from this computer?</p>
              <div className="ml-auto flex gap-2">
                <Button size="sm" disabled={removing} onClick={() => setAsking(false)}>
                  Keep it
                </Button>
                <Button size="sm" variant="danger" busy={removing} onClick={onRemove}>
                  Remove
                </Button>
              </div>
            </div>
          ) : (
            <Button variant="ghost" icon={<Trash2 className="size-4" />} disabled={disabled} onClick={() => setAsking(true)}>
              Remove picture
            </Button>
          )}
        </div>
      </div>
    </Card>
  )
}

const STEPS = [
  { title: 'Open the camera', text: 'Sit facing it with light on your face, and fill the outline with your face, from forehead to chin.' },
  { title: 'Follow the checks', text: 'Seven checks run on every frame. The picture tells you the one thing to fix next.' },
  { title: 'Take and keep the picture', text: 'After a count of three the sharpest frame is kept. Review it, then use it or retake it.' },
]

function StartPanel() {
  return (
    <Card className="flex flex-col">
      <CardHeader title="How enrolment works" hint="It takes about a minute, once." />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <ol className="grid gap-4">
          {STEPS.map((step, index) => (
            <li key={step.title} className="flex gap-3">
              <span className="grid size-6 shrink-0 place-items-center rounded-full bg-ink-950 font-mono text-[11.5px] font-medium text-white tabular-nums">
                {index + 1}
              </span>
              <div className="min-w-0">
                <p className="text-[13.5px] font-medium text-ink-900">{step.title}</p>
                <p className="mt-0.5 text-[12.5px] leading-snug text-ink-600">{step.text}</p>
              </div>
            </li>
          ))}
        </ol>
        <div className="mt-auto pt-5">
          <p className="border-t border-line pt-3.5 text-[12.5px] leading-relaxed text-ink-500">
            The picture and its face signature are kept as files in this project's data folder. Nothing is uploaded, and
            removing the picture deletes both.
          </p>
        </div>
      </div>
    </Card>
  )
}
