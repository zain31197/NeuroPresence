import { describe, expect, it } from 'vitest'
import type { Benchmark } from '../../../lib/api'
import { resultRows } from '../results'

// The baseline measured on 6 October 2026, as scripts/benchmark.py saved it.
const baseline: Benchmark = {
  file: 'benchmark_baseline_windows_rtx5050.json',
  date: '2026-10-06T22:52:14',
  machine: { gpu: 'NVIDIA GeForce RTX 5050' },
  clips: 8,
  summary: {
    fps: 7.464,
    render_ms: 94.469,
    pipeline_ms: 134.035,
    peak_vram_gb: 1.17,
    csim_self_reenactment: 0.896,
    csim_real_video: 0.85,
    pose_error_deg: 0.536,
    mouth_opening_correlation: 0.962,
    lag_frames: -0.005,
    warping_error_ratio: 0.892,
    jitter_ratio: 1.492,
  },
  targets: { fps: 24, render_ms: 42, end_to_end_ms: 150, vram_gb: 8, csim: 0.8 },
}

const verdicts = (benchmark: Benchmark) =>
  Object.fromEntries(resultRows(benchmark).map((row) => [row.name, row.met(row.value as number)]))

describe('resultRows', () => {
  it('judges the baseline honestly: speed and jitter are off target, the rest is on', () => {
    expect(verdicts(baseline)).toEqual({
      'Frame rate': false,
      'Render time': false,
      'Whole pipeline': true,
      'GPU memory': true,
      'Identity match': true,
      Lag: true,
      Flicker: true,
      'Head jitter': false,
    })
  })

  it('shows each figure with its unit', () => {
    const shown = Object.fromEntries(resultRows(baseline).map((row) => [row.name, row.format(row.value as number)]))
    expect(shown['Frame rate']).toBe('7.5 fps')
    expect(shown['Render time']).toBe('94 ms')
    expect(shown['Identity match']).toBe('0.90')
    expect(shown['Lag']).toBe('0.0 frames')
    expect(shown['Head jitter']).toBe('1.49×')
  })

  it('judges against the targets saved with the benchmark, not fixed numbers', () => {
    const easier = { ...baseline, targets: { ...baseline.targets, fps: 5 } }
    expect(verdicts(easier)['Frame rate']).toBe(true)
  })

  it('treats a lead as lag too', () => {
    const early = { ...baseline, summary: { ...baseline.summary, lag_frames: -1.2 } }
    expect(verdicts(early)['Lag']).toBe(false)
  })
})
