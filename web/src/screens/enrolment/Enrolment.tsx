import { ArrowRight, Camera, Check, RefreshCw, Square, Trash2, Upload, UserRoundX, Video, X } from 'lucide-react'
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
import {
  api,
  ApiError,
  POSE_KEYS,
  STOP_PREVIEW_PATH,
  type Candidate,
  type EnrolmentGuide,
  type EnrolmentLimits,
  type EnrolmentRecord,
  type FaceIdentity,
  type Preview,
} from '../../lib/api'
import { cn } from '../../lib/cn'
import { dateTime, splitHint } from '../../lib/format'
import { useStudio } from '../../lib/studio'
import { Checklist } from './Checklist'
import { Evidence } from './Evidence'
import { PoseCapture } from './PoseCapture'
import { Guidance, Stage, type Mode } from './Stage'

/**
 * Enrolment has two steps. First the face is verified live with the camera: that picture is
 * used only to make a face signature and is not stored. Then the picture people will see in
 * meetings is uploaded, and it is accepted only if its face matches that signature.
 */

/** The one request in flight, so its button can show that it is working and the others wait. */
type Work = 'camera' | 'take' | 'upload' | 'confirm' | 'discard' | 'remove' | 'forget' | 'session'

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
  const face = status?.enrolment.face ?? null
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
          ) : face ? (
            <Pill tone="accent" icon={<Check className="size-3" strokeWidth={3} />}>
              Face verified
            </Pill>
          ) : (
            <Pill tone="neutral" icon={null}>
              Not started
            </Pill>
          ))}
      </div>
      <p className="mt-1 text-[13.5px] text-ink-500">
        Verify your face with the camera, register how far it turns, then upload the picture people will see. It is used only if it
        shows the same face.
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

  // Step one and a half, between verifying the face and uploading the meeting picture: four more
  // pictures, turned to each side and tilted up and down (see PoseCapture). Its own camera and
  // candidate state share the same preview/candidate the rest of this screen uses, so this branch
  // owns the whole page while it runs, rather than risk two panels both trying to read them.
  //
  // Gated on there being no meeting picture yet: once one is enrolled, poses are not forced on
  // a person who already finished enrolling before this step existed, or who simply has not
  // registered them. Re-uploading a new picture still requires them (see runtime.py: check_upload).
  if (face && record === null && !POSE_KEYS.every((pose) => face.poses[pose])) {
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
            Stop it to register a pose: the camera can only be open in one place.
          </Notice>
        )}
        <div className="mt-5">
          <PoseCapture face={face} camera={camera} disabled={offline || sessionActive} />
        </div>
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
  // An upload is compared with the verified face, so there has to be one first.
  const uploadButton = (label: string, variant: 'primary' | 'secondary' = 'secondary') => (
    <Tooltip content={face ? 'It goes through the quality checks and is compared with your verified face.' : 'Verify your face with the camera first. An uploaded picture is compared with it.'}>
      <span>
        <Button variant={variant} icon={<Upload className="size-4" />} busy={work === 'upload'} disabled={offline || busy || !face} onClick={chooseFile}>
          {label}
        </Button>
      </span>
    </Tooltip>
  )
  const cameraButton = (label: string, variant: 'primary' | 'secondary') => (
    <Tooltip content={sessionActive ? 'Stop the live session first: the camera can only be open in one place.' : 'Opens the camera with the checks running. Only a face signature is kept from the picture.'}>
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
    const fromCamera = candidate.origin === 'camera'
    // What is wrong with an uploaded picture, without the advice meant for someone sitting at the camera.
    const fault = splitHint(candidate.hint)[0] || candidate.hint || 'This picture did not pass every check.'
    left = (
      <p className="min-w-0 flex-1 basis-[200px] text-[13px] leading-snug text-ink-600">
        {work === 'confirm' && !fromCamera
          ? 'Preparing the picture. The first time, the animation model has to load, which takes a few seconds.'
          : fromCamera
            ? 'Only a face signature is kept from this picture: the numbers that say who you are. The picture itself is not stored.'
            : candidate.passed
              ? 'It passed the quality checks and shows the same person as your verified face.'
              : `${fault} Choose another picture.`}
      </p>
    )
    right = fromCamera ? (
      <>
        <Button icon={<RefreshCw className="size-4" />} busy={work === 'discard'} disabled={busy} onClick={() => run('discard', api.discardCandidate)}>
          Retake
        </Button>
        <Button variant="primary" busy={work === 'confirm'} disabled={offline || busy} onClick={() => run('confirm', api.confirmPicture, 'Face verified. Only its signature was kept.')}>
          Confirm my face
        </Button>
      </>
    ) : (
      <>
        <Button variant="ghost" busy={work === 'discard'} disabled={busy} onClick={() => run('discard', api.discardCandidate)}>
          Discard
        </Button>
        {uploadButton(candidate.passed ? 'Choose another' : 'Choose another picture', candidate.passed ? 'secondary' : 'primary')}
        {candidate.passed && (
          <Button variant="primary" busy={work === 'confirm'} disabled={offline || busy} onClick={() => run('confirm', api.confirmPicture, 'Meeting picture enrolled.')}>
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
    right = cameraFailed ? null : count !== null ? (
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
        {cameraButton('Verify face again', 'secondary')}
        {uploadButton('Upload another picture')}
      </>
    )
  } else {
    // Nothing to show yet: the next step is the primary one.
    right = face ? (
      <>
        {cameraButton('Verify face again', 'secondary')}
        {uploadButton('Upload a picture', 'primary')}
      </>
    ) : (
      cameraButton('Open the camera', 'primary')
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
            ? 'Stop it to verify your face or change your picture.'
            : 'Stop it to verify your face again. An uploaded picture can replace the meeting picture while it runs.'}
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
              empty={
                face
                  ? { title: 'Your face is verified', hint: 'Now upload the picture you want people to see in meetings. It is used only if it shows the same face.' }
                  : { title: 'Start by verifying your face', hint: 'Open the camera and follow the checks. Only a face signature is kept from that picture.' }
              }
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
            <CameraPanel preview={preview} limits={guide?.limits ?? null} />
          ) : record ? (
            <EnrolledPanel
              record={record}
              face={face}
              limits={guide?.limits ?? null}
              work={work}
              disabled={offline || busy}
              onRemove={() => run('remove', api.removeEnrolment, 'Meeting picture removed.')}
              onForget={() => run('forget', api.forgetFace, 'Face signature and meeting picture removed.')}
            />
          ) : (
            <StepsPanel face={face} forgetting={work === 'forget'} disabled={offline || busy} onForget={() => run('forget', api.forgetFace, 'Face signature removed.')} />
          )}
        </div>
      </div>

      <Evidence guide={guide} yours={valueOf(record, 'size')} />
      <input ref={fileInput} type="file" accept="image/png,image/jpeg" className="sr-only" tabIndex={-1} aria-hidden="true" onChange={onFile} />
    </main>
  )
}

/* ------------------------------------------------------------------ panels */

function CameraPanel({ preview, limits }: { preview: Preview | null; limits: EnrolmentLimits | null }) {
  return (
    <Card className="flex flex-col">
      <CardHeader title="Checks" hint="Every one has to pass before the picture can be taken." />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <Checklist checks={preview?.state === 'running' ? preview.checks : null} limits={limits} detail={false} />
        <p className="mt-auto border-t border-line pt-3.5 text-[12.5px] leading-relaxed text-ink-500">
          This picture is used only to make your face signature and is not stored. It is taken after a count of three, as the
          sharpest frame with your eyes open. The picture people see in meetings is uploaded in the next step.
        </p>
      </div>
    </Card>
  )
}

function ReviewPanel({ candidate, limits }: { candidate: Candidate; limits: EnrolmentLimits | null }) {
  const fromCamera = candidate.origin === 'camera'
  return (
    <Card className="flex flex-col">
      <CardHeader
        title={!candidate.passed ? 'This picture cannot be used' : fromCamera ? 'Is this you?' : 'Review the picture'}
        hint={
          !candidate.passed
            ? 'One or more checks failed. Each says what is wrong.'
            : fromCamera
              ? 'It passed every check. Confirm it to make your face signature.'
              : 'It passed every check, and it shows the same person as your verified face.'
        }
      />
      <div className="px-5 pb-5">
        <Checklist checks={candidate.checks} limits={limits} detail advice={fromCamera} />
      </div>
    </Card>
  )
}

/** A button that asks once more before doing something that cannot be undone. */
function Ask({ label, question, confirm, icon, busy, disabled, onConfirm }: { label: string; question: string; confirm: string; icon: ReactNode; busy: boolean; disabled: boolean; onConfirm: () => void }) {
  const [asking, setAsking] = useState(false)
  if (!asking) {
    return (
      <Button variant="ghost" icon={icon} disabled={disabled} onClick={() => setAsking(true)}>
        {label}
      </Button>
    )
  }
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 rounded-control border border-critical/25 bg-critical-wash px-3 py-2">
      <p className="min-w-0 flex-1 basis-[180px] text-[12.5px] leading-snug text-ink-800">{question}</p>
      <div className="ml-auto flex gap-2">
        <Button size="sm" disabled={busy} onClick={() => setAsking(false)}>
          Keep it
        </Button>
        <Button size="sm" variant="danger" busy={busy} onClick={onConfirm}>
          {confirm}
        </Button>
      </div>
    </div>
  )
}

interface EnrolledProps {
  record: EnrolmentRecord
  face: FaceIdentity | null
  limits: EnrolmentLimits | null
  work: Work | null
  disabled: boolean
  onRemove: () => void
  onForget: () => void
}

function EnrolledPanel({ record, face, limits, work, disabled, onRemove, onForget }: EnrolledProps) {
  const faceHeight = valueOf(record, 'size')
  const sharpness = valueOf(record, 'sharp')
  // For an uploaded picture, what could be better without the advice meant for someone at the camera.
  const tips = record.checks.flatMap((check) => (check.passed && check.tip ? [splitHint(check.tip)[0] || check.tip] : []))
  const match =
    typeof record.match === 'number'
      ? `${record.match.toFixed(2)}${limits ? ` (${limits.same_person_csim} or more is the same person)` : ''}`
      : 'Not compared: enrolled before this check existed'
  const facts = [
    ['Face verified', face ? dateTime(face.verified_at) : 'Not yet'],
    ['Picture enrolled', dateTime(record.enrolled_at)],
    ['Match with your face', match],
    ['Picture', `${record.width} × ${record.height} px`],
    faceHeight !== null && ['Face in the picture', `${Math.round(faceHeight)} px tall`],
    sharpness !== null && ['Sharpness', `${Math.round(sharpness)}`],
  ].filter((fact): fact is [string, string] => fact !== false)

  return (
    <Card className="flex flex-col">
      <CardHeader title="Your meeting picture" hint="A camera session animates this picture. It shows the face you verified." />
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
          <Ask
            label="Remove picture"
            question="Remove the meeting picture? Your verified face stays, so you can upload another."
            confirm="Remove"
            icon={<Trash2 className="size-4" />}
            busy={work === 'remove'}
            disabled={disabled}
            onConfirm={onRemove}
          />
          <Ask
            label="Forget my face"
            question="Remove your face signature and the meeting picture from this computer?"
            confirm="Forget"
            icon={<UserRoundX className="size-4" />}
            busy={work === 'forget'}
            disabled={disabled}
            onConfirm={onForget}
          />
        </div>
      </div>
    </Card>
  )
}

function StepsPanel({ face, forgetting, disabled, onForget }: { face: FaceIdentity | null; forgetting: boolean; disabled: boolean; onForget: () => void }) {
  const posesDone = face !== null && POSE_KEYS.every((pose) => face.poses[pose])
  const steps = [
    {
      title: 'Verify your face',
      text: face
        ? `Done on ${dateTime(face.verified_at)}. Only the face signature was kept.`
        : 'Open the camera and follow the checks. The picture is used only to make a face signature, and is not stored.',
      done: face !== null,
    },
    {
      title: 'Register your pose range',
      text: 'Five more pictures: facing the camera, turned to each side, and tilted up and down. A richer face signature, and the reenacted head fitted to how far yours actually moves.',
      done: posesDone,
    },
    {
      title: 'Upload your meeting picture',
      text: 'The picture people will see: well lit, with the background you want. It is accepted only if it shows the same face.',
      done: false,
    },
    { title: 'Go to the Live Studio', text: 'A camera session animates the meeting picture with your live face.', done: false },
  ]
  return (
    <Card className="flex flex-col">
      <CardHeader title="How enrolment works" hint="Once, it takes a few minutes." />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <ol className="grid gap-4">
          {steps.map((step, index) => (
            <li key={step.title} className="flex gap-3">
              <span
                className={cn(
                  'grid size-6 shrink-0 place-items-center rounded-full font-mono text-[11.5px] font-medium tabular-nums',
                  step.done ? 'bg-good-wash text-good-ink' : 'bg-ink-950 text-white',
                )}
              >
                {step.done ? <Check className="size-3.5" strokeWidth={3} aria-label="Done" /> : index + 1}
              </span>
              <div className="min-w-0">
                <p className="text-[13.5px] font-medium text-ink-900">{step.title}</p>
                <p className="mt-0.5 text-[12.5px] leading-snug text-ink-600">{step.text}</p>
              </div>
            </li>
          ))}
        </ol>
        <div className="mt-auto grid gap-2 pt-5">
          <p className="border-t border-line pt-3.5 text-[12.5px] leading-relaxed text-ink-500">
            The face signature and the meeting picture are kept as files in this project's data folder. Nothing is uploaded
            anywhere else.
          </p>
          {face && (
            <Ask
              label="Forget my face"
              question="Remove your face signature from this computer?"
              confirm="Forget"
              icon={<UserRoundX className="size-4" />}
              busy={forgetting}
              disabled={disabled}
              onConfirm={onForget}
            />
          )}
        </div>
      </div>
    </Card>
  )
}
