import { ArrowRight, UserRound } from 'lucide-react'
import { Link } from 'react-router-dom'
import { buttonClass } from '../../components/ui/Button'
import { Card, CardHeader } from '../../components/ui/Card'
import { enrolledPictureUrl } from '../../lib/api'
import { dateTime } from '../../lib/format'
import { useStudio } from '../../lib/studio'

/** The picture a camera session animates, and the way to the screen where it is taken and replaced. */
export function EnrolledPicture() {
  const { status } = useStudio()
  const record = status?.enrolment.record ?? null
  const sample = status?.session.source === 'sample'
  const face = record?.checks.find((check) => check.key === 'size')?.value

  return (
    <Card className="flex flex-col">
      <CardHeader title="Enrolled picture" hint="The one picture of you that every output frame is made from." />
      <div className="flex flex-1 flex-col px-5 pb-5">
        <div className="relative aspect-video overflow-hidden rounded-panel bg-sunken">
          {record ? (
            <>
              <img src={enrolledPictureUrl(record.id)} alt="Your enrolled picture" className="size-full object-contain" />
              <span className="absolute bottom-2 left-2 rounded-chip bg-black/55 px-1.5 py-0.5 font-mono text-[10.5px] text-white/90 tabular-nums backdrop-blur-sm">
                {record.width} × {record.height}
              </span>
            </>
          ) : (
            <div className="absolute inset-0 grid place-items-center text-center">
              <div>
                <span className="mx-auto grid size-9 place-items-center rounded-full bg-surface text-ink-500 shadow-control">
                  <UserRound className="size-4" />
                </span>
                <p className="mt-2 text-[12.5px] text-ink-500">None yet</p>
              </div>
            </div>
          )}
        </div>

        {record ? (
          <dl className="mt-4 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[12.5px]">
            <dt className="text-ink-500">Enrolled</dt>
            <dd className="truncate text-ink-900">{dateTime(record.enrolled_at)}</dd>
            <dt className="text-ink-500">From</dt>
            <dd className="text-ink-900">{record.origin === 'camera' ? 'The camera' : 'An uploaded file'}</dd>
            {typeof face === 'number' && (
              <>
                <dt className="text-ink-500">Face</dt>
                <dd className="text-ink-900 tabular-nums">{Math.round(face)} px tall</dd>
              </>
            )}
          </dl>
        ) : (
          <p className="mt-4 text-[12.5px] leading-snug text-ink-500">
            A camera session needs one. Sample clips run without it: each animates one of its own frames.
          </p>
        )}
        {sample && record && (
          <p className="mt-3 rounded-[8px] bg-ink-50 px-3 py-2 text-[12.5px] leading-snug text-ink-700">
            This session animates a frame of the sample clip, not your picture.
          </p>
        )}

        <div className="mt-auto pt-4">
          <Link to="/studio/enrolment" className={buttonClass('secondary', 'md', 'w-full')}>
            {record ? 'Manage in Enrolment' : 'Enrol a picture'}
            <ArrowRight className="size-3.5" />
          </Link>
        </div>
      </div>
    </Card>
  )
}
