import type { ReactNode } from 'react'
import { cn } from '../lib/cn'
import { Spinner } from './ui/Button'

/** Draw a decoded frame onto a canvas, resizing the canvas to the frame when its size changes. */
export function paint(canvas: HTMLCanvasElement | null, bitmap: ImageBitmap) {
  if (!canvas) return
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width
    canvas.height = bitmap.height
  }
  canvas.getContext('2d')?.drawImage(bitmap, 0, 0)
}

interface MonitorProps {
  label: string
  detail?: string | null
  /** A monitor that is showing a picture is dark, like a screen; an empty one is a light placeholder. */
  lit: boolean
  chip?: ReactNode
  footer?: ReactNode
  /** Which lower corner the footer sits in. The output carries its disclosure mark in the lower left, so its footer goes right. */
  footerAt?: 'left' | 'right'
  /** The picture is live right now: the screen carries a soft red halo, like a tally light. */
  live?: boolean
  className?: string
  children: ReactNode
}

/** A 16:9 screen with a label in its corner: the frame around every picture in the studio. */
export function Monitor({ label, detail, lit, chip, footer, footerAt = 'left', live, className, children }: MonitorProps) {
  return (
    <div
      className={cn(
        'relative aspect-video overflow-hidden rounded-panel transition-[background-color,box-shadow] duration-500 ease-out-soft',
        lit ? 'bg-stage shadow-screen ring-1 ring-black/10' : 'border border-dashed border-line-strong bg-ink-50',
        lit && live && 'shadow-[0_0_0_1.5px_rgb(229_72_77/0.55),0_18px_44px_-20px_rgb(229_72_77/0.5)]',
        className,
      )}
    >
      {children}
      {/* A little shade at the top and bottom edge, so the labels read on any picture. */}
      {lit && <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-gradient-to-b from-black/25 via-transparent to-black/20" />}
      <div className="absolute top-3 left-3 flex items-center gap-1.5">
        <span className={cn('label-caps rounded-chip px-2 py-1', lit ? 'bg-black/65 text-white' : 'bg-surface text-ink-600 shadow-control')}>
          {label}
        </span>
        {detail && <span className="label-caps rounded-chip bg-black/65 px-2 py-1 text-white/80">{detail}</span>}
      </div>
      {chip && <div className="absolute top-3 right-3">{chip}</div>}
      {footer && (
        <div
          className={cn(
            'absolute bottom-3 rounded-chip bg-black/65 px-2 py-1 font-mono text-[11px] text-white/90 tabular-nums',
            footerAt === 'right' ? 'right-3' : 'left-3',
          )}
        >
          {footer}
        </div>
      )}
    </div>
  )
}

interface EmptyProps {
  icon: ReactNode
  title: string
  hint: string
  /** What to do about it: a button or two. */
  action?: ReactNode
}

/** What a monitor shows when it has no picture: what is missing and how to get it. */
export function Empty({ icon, title, hint, action }: EmptyProps) {
  return (
    <div className="absolute inset-0 grid place-items-center p-6 text-center">
      <div className="max-w-[300px]">
        <span className="mx-auto grid size-11 place-items-center rounded-full bg-surface text-ink-500 shadow-card ring-1 ring-line">{icon}</span>
        <p className="mt-3 text-[13.5px] font-medium text-ink-900">{title}</p>
        <p className="mt-1 text-[12.5px] leading-snug text-ink-500">{hint}</p>
        {action && <div className="mt-4 flex flex-wrap items-center justify-center gap-2">{action}</div>}
      </div>
    </div>
  )
}

export function Waiting({ message }: { message?: string }) {
  return (
    <div className="absolute inset-0 grid place-items-center p-6 text-center" role="status">
      <div>
        <Spinner className="mx-auto size-5 text-ink-500" />
        <p className="mt-3 text-[13.5px] font-medium text-ink-900">{message || 'Starting'}</p>
      </div>
    </div>
  )
}
