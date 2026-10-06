import { Card, CardHeader } from '../../components/ui/Card'
import { FaceSizeChart } from '../../components/viz/FaceSizeChart'
import type { EnrolmentGuide, EnrolmentStudy } from '../../lib/api'
import { detailAt, largestFace, signed, span } from './study'

const capital = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)

/** The measurements the checks were set from, so the screen can show its reasons and not only its rules. */
export function Evidence({ guide, yours }: { guide: EnrolmentGuide | null; yours: number | null }) {
  if (!guide) return null
  const { study, limits } = guide

  return (
    <section className="mt-9" aria-labelledby="evidence-title">
      <h2 id="evidence-title" className="display text-[18px] font-semibold">
        Why these checks
      </h2>
      <p className="mt-1 max-w-[76ch] text-[13.5px] leading-relaxed text-ink-500">
        Every output frame is made from the enrolled picture, so a fault in it shows in all of them. The limits come from
        measurements on this project's own pipeline.
      </p>

      {study ? (
        <div className="mt-4 grid gap-5 lg:grid-cols-[minmax(0,1.12fr)_minmax(0,1fr)]">
          <FaceSize study={study} minPx={limits.min_face_height_px} goodPx={limits.good_face_height_px} yours={yours} />
          <FaultCosts study={study} />
        </div>
      ) : (
        <Card className="mt-4 px-5 py-4">
          <p className="text-[13.5px] font-medium text-ink-950">The study behind the checks has not been run on this computer.</p>
          <p className="mt-1 text-[13px] text-ink-600">
            Run <code className="rounded-[5px] bg-ink-100 px-1.5 py-0.5 font-mono text-[12px]">python scripts/study_enrolment.py</code> and
            reload this page to see its results here. It takes about 15 minutes.
          </p>
        </Card>
      )}
    </section>
  )
}

function FaceSize({ study, minPx, goodPx, yours }: { study: EnrolmentStudy; minPx: number; goodPx: number; yours: number | null }) {
  const { pictures, reference_px } = study.face_size
  const atMinimum = detailAt(pictures, minPx)
  const largest = largestFace(pictures)
  const atLargest = detailAt(pictures, largest)
  return (
    <Card className="flex flex-col">
      <CardHeader
        title="A larger face gives a sharper output"
        hint="Fine detail in the output face, for the same picture enrolled at different sizes."
      />
      <div className="px-5 pb-5">
        <FaceSizeChart pictures={pictures} referencePx={reference_px} minPx={minPx} goodPx={goodPx} yours={yours} />
        {atMinimum && atLargest && (
          <p className="mt-4 border-t border-line pt-3 text-[12.5px] leading-relaxed text-ink-600">
            At the {minPx} px minimum the output has {span(atMinimum)} of the detail it has at {reference_px} px. At {largest} px it
            has {span(atLargest)} times as much. So the picture is stored at full size, and the checks ask you to come closer
            while there is room.
          </p>
        )}
      </div>
    </Card>
  )
}

function FaultCosts({ study }: { study: EnrolmentStudy }) {
  const worst = Math.max(...study.faults.map((fault) => -fault.csim_change), 0)
  return (
    <Card className="flex flex-col">
      <CardHeader
        title="What a fault in the picture costs"
        hint={`The same clip animated from a faulty picture of itself and from a good one. ${study.clips} sample clips.`}
      />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="label-caps text-left text-ink-400">
              <th scope="col" className="pb-2 font-medium">
                Picture
              </th>
              <th scope="col" colSpan={2} className="pb-2 font-medium">
                Identity match
              </th>
              <th scope="col" className="pb-2 text-right font-medium">
                Detail kept
              </th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            <tr className="border-t border-line">
              <th scope="row" className="py-2 pr-3 text-left font-medium whitespace-nowrap text-ink-950">
                Good picture
              </th>
              <td className="w-[22%] py-2 sm:w-[34%]" />
              <td className="py-2 pr-3 pl-3 text-right font-mono text-[12.5px] text-ink-900">{study.good_picture.csim.toFixed(3)}</td>
              <td className="py-2 text-right font-mono text-[12.5px] text-ink-900">{study.good_picture.detail_kept.toFixed(2)}</td>
            </tr>
            {study.faults.map((fault) => (
              <tr key={fault.name} className="border-t border-line">
                <th scope="row" className="py-2 pr-3 text-left font-normal whitespace-nowrap text-ink-800">
                  {capital(fault.name)}
                  {fault.clips < study.clips && (
                    <span className="block text-[11.5px] leading-tight text-ink-400">
                      found in {fault.clips} of {study.clips} clips
                    </span>
                  )}
                </th>
                <td className="w-[22%] py-2 sm:w-[34%]">
                  {/* A thin bar from zero: how much of the worst loss this fault costs. */}
                  <div
                    aria-hidden="true"
                    className="h-2 rounded-r-[4px] bg-series-1"
                    style={{ width: `${worst > 0 ? Math.max(0, (-fault.csim_change / worst) * 100) : 0}%`, minWidth: fault.csim_change < 0 ? 2 : 0 }}
                  />
                </td>
                <td className="py-2 pr-3 pl-3 text-right font-mono text-[12.5px] text-ink-900">{signed(fault.csim_change)}</td>
                <td className="py-2 text-right font-mono text-[12.5px] text-ink-900">{fault.detail_kept.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-auto border-t border-line pt-3 text-[12.5px] leading-relaxed text-ink-600">
          Identity match is how alike the output face and the real frame of the same instant are (1.0 is identical); the faults
          are given as the change from the good picture. Detail kept is the share of the real frame's fine detail that survives.
        </p>
      </div>
    </Card>
  )
}
