import { Card } from '../../components/ui/Card'
import { Sparkline } from '../../components/viz/Sparkline'
import { dash } from '../../lib/format'
import { useStudio } from '../../lib/studio'

const CAPACITY = 240 // one minute at four readings a second
const INTERVAL_S = 0.25

const degrees = (angle: number) => `${angle >= 0 ? '+' : '−'}${Math.abs(angle).toFixed(0)}°`
const share = (value: number) => value.toFixed(2)

/**
 * The signal that drives the picture: where the head points and how far the mouth is open,
 * as the tracker reads them from the camera. A steady head should draw flat lines here.
 */
export function TrackingPanel() {
  const { status, history } = useStudio()
  const tracking = status?.session.tracking
  const pose = tracking?.pose_deg ?? null
  const mouth = tracking?.mouth_open ?? null

  const signals = [
    { label: 'Head turn', about: 'Left and right', now: pose ? degrees(pose[0]) : null, values: history.map((sample) => sample.yaw), format: degrees },
    { label: 'Head nod', about: 'Up and down', now: pose ? degrees(pose[1]) : null, values: history.map((sample) => sample.pitch), format: degrees },
    { label: 'Mouth opening', about: 'From closed, 0, to wide open, 1', now: mouth !== null ? share(mouth) : null, values: history.map((sample) => sample.mouth), format: share },
  ]

  return (
    <Card className="@container overflow-hidden">
      <div className="px-5 pt-4 pb-3">
        <h2 className="text-[13.5px] font-semibold text-ink-950">Tracking</h2>
        <p className="mt-0.5 text-[12.5px] leading-snug text-ink-500">
          The signal that drives your picture, as the tracker reads it from the camera. Turn on the tracking overlay to see the
          face points and the window the face crop is cut from.
        </p>
      </div>
      {/* Hairlines between the signals are the gaps of this grid showing the line colour. */}
      <div className="grid grid-cols-1 gap-px border-t border-line bg-line @2xl:grid-cols-3">
        {signals.map((signal) => (
          <div key={signal.label} className="min-w-0 bg-surface px-5 py-4">
            <div className="flex items-baseline justify-between gap-3">
              <p className="text-[12.5px] font-medium text-ink-600">{signal.label}</p>
              <p className="font-mono text-[13px] text-ink-950 tabular-nums">{signal.now ?? dash}</p>
            </div>
            <p className="text-[11.5px] text-ink-400">{signal.about}</p>
            <div className="mt-2">
              <Sparkline values={signal.values} capacity={CAPACITY} interval={INTERVAL_S} format={signal.format} label={signal.label} />
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
