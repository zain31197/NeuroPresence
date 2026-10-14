import { Cast } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Card } from '../../components/ui/Card'
import { Pill } from '../../components/ui/Pill'
import { Switch } from '../../components/ui/Switch'
import { useToast } from '../../components/ui/Toast'
import { api, ApiError } from '../../lib/api'
import { useStudio } from '../../lib/studio'
import { cameraLine } from './virtualCamera'

/**
 * The switch that offers the output to meeting applications as a camera, and one line on what
 * they see right now: nothing, a plain card, or the output of the running session.
 */
export function VirtualCameraBar() {
  const { status } = useStudio()
  const toast = useToast()
  const camera = status?.virtual_camera
  // What the person just chose, shown at once; the engine's own answer replaces it a moment later.
  const [chosen, setChosen] = useState<boolean | null>(null)
  useEffect(() => {
    if (chosen !== null && camera?.on === chosen) setChosen(null)
  }, [camera?.on, chosen])

  const line = cameraLine(camera)
  if (!camera || !line) return null

  const toggle = async (on: boolean) => {
    setChosen(on)
    try {
      await api.setVirtualCamera(on)
    } catch (error) {
      setChosen(null)
      toast.problem(error instanceof ApiError ? error.message : 'The virtual camera could not be changed.')
    }
  }

  return (
    <Card className="flex flex-wrap items-center gap-x-4 gap-y-3 px-5 py-3.5">
      <span className="grid size-9 shrink-0 place-items-center rounded-full bg-ink-50 text-ink-700 ring-1 ring-line">
        <Cast className="size-4" />
      </span>
      <div className="min-w-0 flex-1 basis-[320px]">
        <div className="flex items-center gap-2">
          <label htmlFor="virtual-camera" className="cursor-pointer text-[13.5px] font-semibold text-ink-950">
            Virtual camera
          </label>
          <Pill tone={line.tone}>{line.state}</Pill>
        </div>
        <p className="mt-0.5 text-[12.5px] leading-snug text-ink-600">{line.text}</p>
      </div>
      <Switch id="virtual-camera" label="Virtual camera" checked={chosen ?? camera.on} onCheckedChange={(on) => void toggle(on)} />
    </Card>
  )
}
