import { Camera, Check as Tick } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Button } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError, POSE_KEYS, POSE_LABEL, type FaceIdentity, type PoseKey } from '../../lib/api'
import { cn } from '../../lib/cn'
import { useStudio } from '../../lib/studio'
import { Checklist, POSE_LABELS } from './Checklist'
import { Stage } from './Stage'

/**
 * Between verifying the face and uploading the meeting picture: the user registers four more
 * pictures of themselves, turned to each side and tilted up and down. Two things come of it
 * (see neuropresence/enrolment/checks.py and reenactment/engine.py): a richer face signature,
 * so a meeting picture or a live face at a slight angle is still recognised; and a measured
 * range for how far this particular person's head actually turns, which replaces the fixed,
 * one-person guess the reenactment engine otherwise clamps motion to.
 *
 * Uploading is blocked until all four are registered (see runtime.py: check_upload).
 */

const COUNTDOWN_SECONDS = 3

const POSE_HINT: Record<PoseKey, string> = {
  front: 'Face the camera directly, relaxed, the same as your first picture.',
  left: 'Turn your head to one side, as far as comfortable, and hold it.',
  right: 'Now turn to the other side, as far as comfortable, and hold it.',
  up: 'Tilt your head back, looking up, and hold it.',
  down: 'Tilt your head down, chin toward your chest, and hold it.',
}

type Work = 'camera' | 'take' | 'confirm' | 'discard'

interface Props {
  face: FaceIdentity
  camera: string
  disabled: boolean
}

export function PoseCapture({ face, camera, disabled }: Props) {
  const { status, connection, refresh } = useStudio()
  const toast = useToast()
  const [work, setWork] = useState<Work | null>(null)
  const [count, setCount] = useState<number | null>(null)
  const [opening, setOpening] = useState<PoseKey | null>(null)

  const remaining = POSE_KEYS.filter((pose) => !face.poses[pose])
  const current = remaining[0] ?? null

  const preview = status?.enrolment.preview ?? null
  const candidate = status?.enrolment.candidate ?? null
  const forThisPose = candidate?.origin === `pose:${current}`
  const cameraOpen = (preview?.state === 'running' || preview?.state === 'starting') && preview.pose === current
  const cameraFailed = preview?.state === 'error' && opening === current
  const offline = connection !== 'open'
  const busy = work !== null

  const run = useCallback(
    async (kind: Work, action: () => Promise<unknown>) => {
      setWork(kind)
      try {
        await action()
        await refresh()
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

  const openCamera = (pose: PoseKey) => {
    setOpening(pose)
    void run('camera', () => api.startPosePreview(pose, camera))
  }
  const take = useCallback(async () => {
    await run('take', api.takePicture)
  }, [run])
  const confirm = () => run('confirm', api.confirmPicture).then(() => setOpening(null))
  const retake = () => run('discard', api.discardCandidate)

  // The count of three before a picture: time to look up from the button to the camera.
  useEffect(() => {
    if (count === null) return
    if (count === 0) {
      setCount(null)
      void take()
      return
    }
    const timer = window.setTimeout(() => setCount((left) => (left === null ? null : left - 1)), 1000)
    return () => window.clearTimeout(timer)
  }, [count, take])

  if (current === null) return null // every pose is registered: nothing more to show here

  const mode = forThisPose && candidate ? 'review' : cameraOpen || cameraFailed ? 'camera' : 'empty'

  return (
    <Card className="self-start p-2">
      <CardHeader
        title="Register your pose range"
        hint="Five more pictures: facing the camera, turned to each side, and tilted up and down. Used for a richer face signature, and to fit how far the reenacted head is allowed to move to how far yours actually does."
      />
      <div className="flex flex-wrap items-center gap-1.5 px-3 pb-2.5">
        {POSE_KEYS.map((pose) => {
          const done = Boolean(face.poses[pose])
          const active = pose === current
          return (
            <span
              key={pose}
              className={cn(
                'label-caps flex items-center gap-1 rounded-chip px-2 py-1',
                done ? 'bg-good-wash text-good-ink' : active ? 'bg-ink-950 text-white' : 'bg-ink-100 text-ink-500',
              )}
            >
              {done && <Tick className="size-3" strokeWidth={3} />}
              {POSE_LABEL[pose]}
            </span>
          )
        })}
      </div>

      <Stage
        mode={mode}
        preview={cameraOpen || cameraFailed ? preview : null}
        candidate={forThisPose ? candidate : null}
        record={null}
        count={count}
        taking={work === 'take'}
        shots={0}
        empty={{ title: POSE_LABEL[current], hint: POSE_HINT[current] }}
        retry={
          <Button size="sm" variant="primary" busy={work === 'camera'} disabled={offline || busy} onClick={() => openCamera(current)}>
            Try again
          </Button>
        }
      />

      {mode === 'review' && candidate && (
        <div className="px-5 pt-3.5">
          <Checklist checks={candidate.checks} limits={null} detail labels={POSE_LABELS} />
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-3 px-3 pt-3.5 pb-2">
        <p className="min-w-0 flex-1 basis-[200px] text-[13px] leading-snug text-ink-600">
          {mode === 'camera' ? POSE_HINT[current] : mode === 'empty' ? 'Opens the camera with the checks running, as before.' : null}
        </p>
        <div className="ml-auto flex flex-wrap items-center justify-end gap-2">
          {mode === 'review' ? (
            <>
              <Button variant="ghost" busy={work === 'discard'} disabled={disabled || busy} onClick={retake}>
                Retake
              </Button>
              <Button variant="primary" busy={work === 'confirm'} disabled={offline || busy || !candidate?.passed} onClick={confirm}>
                Use this
              </Button>
            </>
          ) : mode === 'camera' ? (
            <Button
              variant="primary"
              icon={<Camera className="size-4" />}
              busy={work === 'take'}
              disabled={offline || busy || !preview?.ready || count !== null}
              onClick={() => setCount(COUNTDOWN_SECONDS)}
            >
              Take picture
            </Button>
          ) : (
            <Button variant="primary" icon={<Camera className="size-4" />} busy={work === 'camera'} disabled={disabled || busy} onClick={() => openCamera(current)}>
              Open the camera
            </Button>
          )}
        </div>
      </div>
      {remaining.length < POSE_KEYS.length && (
        <div className="px-5 pb-4">
          <Pill tone="neutral">
            {POSE_KEYS.length - remaining.length} of {POSE_KEYS.length} registered
          </Pill>
        </div>
      )}
    </Card>
  )
}
