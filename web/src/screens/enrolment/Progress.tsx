import { Check } from 'lucide-react'
import { motion } from 'motion/react'
import { Link } from 'react-router-dom'
import { EASE_OUT } from '../../components/motion'
import { POSE_KEYS, type EnrolmentRecord, type FaceIdentity } from '../../lib/api'
import { cn } from '../../lib/cn'

const STEPS = ['Verify your face', 'Register your poses', 'Upload your picture', 'Go live'] as const

/** Which steps are done, from what the engine has stored: the face, its five poses, and the meeting picture. */
export function stepsDone(face: FaceIdentity | null, record: EnrolmentRecord | null): number {
  if (!face) return 0
  const poses = POSE_KEYS.every((pose) => face.poses[pose])
  if (record) return 3 // a picture enrolled before the pose step existed still counts: it can go live
  return poses ? 2 : 1
}

/** Where enrolment stands, in four steps: done ones ticked, the current one lit, the line between them filling. */
export function Progress({ face, record }: { face: FaceIdentity | null; record: EnrolmentRecord | null }) {
  const done = stepsDone(face, record)
  return (
    <nav aria-label="Enrolment steps" className="mt-5 rounded-card border border-line bg-surface px-5 py-4 shadow-card">
      <ol className="relative grid grid-cols-4 gap-2">
        {/* The line behind the steps, and how much of it is done. */}
        <div aria-hidden="true" className="absolute top-[15px] right-[12.5%] left-[12.5%] h-[2px] rounded-full bg-ink-100">
          <motion.div
            className="h-full origin-left rounded-full bg-accent-500"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: Math.min(done, STEPS.length - 1) / (STEPS.length - 1) }}
            transition={{ duration: 0.8, ease: EASE_OUT }}
          />
        </div>
        {STEPS.map((title, index) => {
          const state = index < done ? 'done' : index === done ? 'current' : 'todo'
          return (
            <li key={title} className="relative flex flex-col items-center text-center" aria-current={state === 'current' ? 'step' : undefined}>
              <span
                className={cn(
                  'relative grid size-8 place-items-center rounded-full font-mono text-[12px] font-semibold tabular-nums transition-colors duration-300',
                  state === 'done' && 'bg-accent-500 text-white',
                  state === 'current' && 'bg-ink-950 text-white',
                  state === 'todo' && 'border border-line-strong bg-surface text-ink-400',
                )}
              >
                {state === 'current' && (
                  <motion.span
                    aria-hidden="true"
                    className="absolute inset-0 rounded-full ring-2 ring-ink-950"
                    animate={{ scale: [1, 1.45], opacity: [0.45, 0] }}
                    transition={{ duration: 1.6, ease: 'easeOut', repeat: Infinity }}
                  />
                )}
                {state === 'done' ? (
                  <motion.span initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: 'spring', stiffness: 420, damping: 22 }}>
                    <Check className="size-4" strokeWidth={3} aria-label="Done" />
                  </motion.span>
                ) : (
                  index + 1
                )}
              </span>
              {index === STEPS.length - 1 && state === 'current' ? (
                // Everything is in place: the last step is where to go next.
                <Link to="/studio" className="mt-2 text-[12.5px] leading-tight font-semibold text-accent-700 underline-offset-4 hover:underline">
                  {title}
                </Link>
              ) : (
                <span className={cn('mt-2 text-[12.5px] leading-tight font-medium', state === 'todo' ? 'text-ink-400' : 'text-ink-900')}>{title}</span>
              )}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
