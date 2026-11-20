import type { Benchmark } from '../../lib/api'

/** One line of the results table on the landing page: a measured figure and its target. */
export interface Row {
  name: string
  about: string
  value: number | null | undefined
  format: (value: number) => string
  target: string
  /** True if the measured value meets the target. */
  met: (value: number) => boolean
  /** The target as a number, and whether more is better ('up') or less is ('down'), for the meter beside the figure. */
  goal: number
  better: 'up' | 'down'
}

export function resultRows(benchmark: Benchmark): Row[] {
  const s = benchmark.summary
  const t = { fps: 24, render_ms: 42, end_to_end_ms: 150, vram_gb: 8, csim: 0.8, ...benchmark.targets }
  return [
    { name: 'Frame rate', about: 'Output frames per second', value: s.fps, format: (v) => `${v.toFixed(1)} fps`, target: `≥ ${t.fps} fps`, met: (v) => v >= t.fps, goal: t.fps, better: 'up' },
    { name: 'Render time', about: 'GPU time of the reenactment stage per frame', value: s.render_ms, format: (v) => `${v.toFixed(0)} ms`, target: `≤ ${t.render_ms} ms`, met: (v) => v <= t.render_ms, goal: t.render_ms, better: 'down' },
    { name: 'Whole pipeline', about: 'Tracking, motion, render and compose per frame', value: s.pipeline_ms, format: (v) => `${v.toFixed(0)} ms`, target: `≤ ${t.end_to_end_ms} ms`, met: (v) => v <= t.end_to_end_ms, goal: t.end_to_end_ms, better: 'down' },
    { name: 'GPU memory', about: 'Peak memory the pipeline holds', value: s.peak_vram_gb, format: (v) => `${v.toFixed(2)} GB`, target: `≤ ${t.vram_gb} GB`, met: (v) => v <= t.vram_gb, goal: t.vram_gb, better: 'down' },
    { name: 'Identity match', about: 'Similarity of the output face to the enrolled picture (CSIM)', value: s.csim_self_reenactment, format: (v) => v.toFixed(2), target: `≥ ${t.csim.toFixed(2)}`, met: (v) => v >= t.csim, goal: t.csim, better: 'up' },
    { name: 'Lag', about: 'How far the output trails the driving face', value: s.lag_frames === null || s.lag_frames === undefined ? s.lag_frames : Math.abs(s.lag_frames), format: (v) => `${Math.abs(v).toFixed(1)} frames`, target: 'under 0.5 frames', met: (v) => Math.abs(v) < 0.5, goal: 0.5, better: 'down' },
    { name: 'Flicker', about: 'Unexplained change between frames, against real video', value: s.warping_error_ratio, format: (v) => `${v.toFixed(2)}×`, target: '≤ 1× real video', met: (v) => v <= 1, goal: 1, better: 'down' },
    { name: 'Head jitter', about: 'Trembling of the head, against real video', value: s.jitter_ratio, format: (v) => `${v.toFixed(2)}×`, target: '≤ 1× real video', met: (v) => v <= 1, goal: 1, better: 'down' },
  ]
}

/**
 * Where a figure and its target sit on a meter from 0 to a little past the target, as shares of its width:
 * the target always at the same place, so rows can be compared at a glance. A figure off the scale is drawn
 * at its end.
 */
export function meterPlace(row: Pick<Row, 'goal' | 'better'>, value: number): { fill: number; target: number } {
  const span = row.better === 'up' ? row.goal * 1.25 : row.goal * 1.6
  const target = row.goal / span
  return { fill: Math.max(0, Math.min(1, Math.abs(value) / span)), target }
}
