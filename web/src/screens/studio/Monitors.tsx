import { ImageOff, Pause, RotateCcw, ScanFace, Users, VideoOff } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Empty, Monitor, paint, Waiting } from '../../components/Monitor'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { useToast } from '../../components/ui/Toast'
import { Tooltip } from '../../components/ui/Tooltip'
import { api, ApiError, enrolledPictureUrl, type TrackStatus } from '../../lib/api'
import { cn } from '../../lib/cn'
import { useStudio } from '../../lib/studio'
import { FeatureSwitches } from './FeatureSwitches'

interface FrameInfo {
  live: boolean
  status: TrackStatus
  width: number
  height: number
}

const signed = (angle: number) => `${angle >= 0 ? '+' : '−'}${Math.abs(angle).toFixed(1)}°`

/** The camera and the output made from it, side by side and always from the same instant. */
export function Monitors() {
  const { status, onFrame } = useStudio()
  const toast = useToast()
  const [resetting, setResetting] = useState(false)
  const camera = useRef<HTMLCanvasElement>(null)
  const output = useRef<HTMLCanvasElement>(null)
  const [frame, setFrame] = useState<FrameInfo | null>(null)

  const session = status?.session
  const running = session?.state === 'running'
  const starting = session?.state === 'starting'
  const record = status?.enrolment.record ?? null
  // A sample clip animates one of its own frames, so its still picture is not the enrolled one.
  const still = session?.source === 'sample' ? 'Still picture' : 'Enrolled picture'

  useEffect(() => {
    let decoding = false
    return onFrame(async (header, cameraJpeg, outputJpeg) => {
      if (header.kind !== 'live' || !outputJpeg) return // a frame of the enrolment preview is not for this screen
      if (decoding) return // still drawing the previous pair: skip this one rather than queue it
      decoding = true
      try {
        const [cameraBitmap, outputBitmap] = await Promise.all([createImageBitmap(cameraJpeg), createImageBitmap(outputJpeg)])
        paint(camera.current, cameraBitmap)
        paint(output.current, outputBitmap)
        const next: FrameInfo = { live: header.live, status: header.status, width: outputBitmap.width, height: outputBitmap.height }
        cameraBitmap.close()
        outputBitmap.close()
        setFrame((shown) =>
          shown && shown.live === next.live && shown.status === next.status && shown.width === next.width && shown.height === next.height
            ? shown
            : next,
        )
      } finally {
        decoding = false
      }
    })
  }, [onFrame])

  useEffect(() => {
    if (!running) setFrame(null)
  }, [running])

  const showing = running && frame !== null
  const pose = session?.tracking?.pose_deg

  const resetNeutral = async () => {
    setResetting(true)
    try {
      await api.resetNeutral()
      toast.done('Neutral pose reset.')
    } catch (error) {
      toast.problem(error instanceof ApiError ? error.message : 'Something went wrong. Try again.')
    } finally {
      setResetting(false)
    }
  }

  return (
    <Card className="p-2">
      <div className="grid gap-2 md:grid-cols-2">
        <Monitor
          label="Camera"
          detail={running ? session?.input : undefined}
          lit={showing}
          chip={showing && <TrackingChip status={frame.status} />}
          footer={
            showing &&
            pose && (
              <span className="flex gap-3">
                <span>yaw {signed(pose[0])}</span>
                <span>pitch {signed(pose[1])}</span>
                <span>roll {signed(pose[2])}</span>
              </span>
            )
          }
        >
          <canvas ref={camera} role="img" aria-label="Camera preview" className={cn('size-full object-contain', !showing && 'hidden')} />
          {!showing &&
            (starting ? (
              <Waiting message={session?.message} />
            ) : (
              <Empty icon={<VideoOff className="size-5" />} title="The camera is off" hint="Start a session to see yourself here." />
            ))}
        </Monitor>

        <Monitor
          label="Output"
          lit={showing || (!starting && record !== null)}
          chip={
            showing ? (
              frame.live ? (
                <LiveChip />
              ) : (
                <Pill tone="neutral" icon={<Pause className="size-3" />} className="bg-white text-ink-800">
                  {still}
                </Pill>
              )
            ) : (
              !starting &&
              record && (
                <Pill tone="neutral" icon={null} className="bg-white text-ink-800">
                  Enrolled picture
                </Pill>
              )
            )
          }
          footer={showing && <span>{`${frame.width} × ${frame.height}`}</span>}
        >
          <canvas ref={output} role="img" aria-label="Reenacted output" className={cn('size-full object-contain', !showing && 'hidden')} />
          {!showing &&
            (starting ? (
              <Waiting message={session?.message} />
            ) : record ? (
              <img src={enrolledPictureUrl(record.id)} alt="Your enrolled picture" className="size-full object-contain" />
            ) : (
              <Empty
                icon={<ImageOff className="size-5" />}
                title="No picture enrolled yet"
                hint="A camera session animates your enrolled picture, so enrolling one comes first."
              />
            ))}
        </Monitor>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 px-3 pt-3.5 pb-2">
        <FeatureSwitches />
        <Tooltip content="Makes your current pose the starting point that all movement is measured from.">
          <span>
            <Button size="sm" icon={<RotateCcw className="size-3.5" />} disabled={!running} busy={resetting} onClick={resetNeutral}>
              Reset neutral pose
            </Button>
          </span>
        </Tooltip>
      </div>
    </Card>
  )
}

function LiveChip() {
  return (
    <span className="inline-flex h-[22px] items-center gap-1.5 rounded-full bg-white px-2 text-[11.5px] font-semibold text-ink-950">
      <span className="size-2 animate-lamp rounded-full bg-live" />
      Live
    </span>
  )
}

function TrackingChip({ status }: { status: TrackStatus }) {
  if (status === 'ok') {
    return (
      <Pill tone="neutral" icon={<ScanFace className="size-3" />} className="bg-white text-ink-800">
        Face tracked
      </Pill>
    )
  }
  return status === 'no_face' ? (
    <Pill tone="serious" icon={<VideoOff className="size-3" />}>
      No face in view
    </Pill>
  ) : (
    <Pill tone="serious" icon={<Users className="size-3" />}>
      More than one face
    </Pill>
  )
}
