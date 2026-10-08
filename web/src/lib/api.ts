/** The shapes the Python backend sends, and the calls the app makes to it. */

export type SessionState = 'idle' | 'starting' | 'running' | 'error'
export type TrackStatus = 'ok' | 'no_face' | 'multiple_faces'
export type StageKey = 'tracker' | 'motion' | 'render' | 'compose'

export interface Metrics {
  fps: number | null
  pipeline_ms: number | null
  end_to_end_ms: number | null
  /** GPU time of the reenactment stage per reenacted frame. */
  render_ms: number | null
  /** Where an average frame spends its time; these add up to pipeline_ms. */
  stages_ms: Record<StageKey, number> | null
  dropped_share: number | null
  live_share: number | null
}

export interface Tracking {
  status: TrackStatus
  pose_deg: [number, number, number] | null
  mouth_open: number | null
}

export interface Session {
  state: SessionState
  message: string
  input: string | null
  /** What the output is made from: the enrolled picture, or a sample clip's own frame. */
  source: 'enrolment' | 'sample' | null
  uptime_s: number | null
  frames: number
  metrics: Metrics | null
  tracking: Tracking | null
  /** Why the still picture is shown although a face is in view, if it is. */
  holds?: { identity: boolean; delay: boolean }
}

/** What the identity monitor is doing about the score (see identity/guard.py in the engine). */
export interface IdentityGuard {
  /** The "Identity fallback" switch. */
  enabled: boolean
  /** 'anchoring': a fresh neutral pose was taken. 'fallback': the still picture is shown until the person resumes. */
  state: 'steady' | 'dipping' | 'anchoring' | 'fallback'
  /** How long the still picture has been shown, in seconds. Null unless the state is 'fallback'. */
  seconds: number | null
  /** The mean of the last seconds that the last decision was taken on. */
  mean: number | null
  /** Below this mean the output no longer counts as matching the picture. */
  low: number
}

/** 'identity' is the extra check an uploaded picture gets: does it show the verified face?
 * 'pose' is the one check a pose capture gets instead of 'facing'/'framing'/'expression' (see checks.py). */
export type CheckKey = 'face' | 'facing' | 'size' | 'framing' | 'light' | 'sharp' | 'expression' | 'identity' | 'pose'
export type PoseKey = 'front' | 'left' | 'right' | 'up' | 'down'
export type PictureOrigin = 'camera' | 'upload' | `pose:${PoseKey}`
/** 'front' first: the pose a person is actually in for most of a meeting, so the one most worth
 * a second reference sample. The other four also calibrate how far this person's head turns. */
export const POSE_KEYS: PoseKey[] = ['front', 'left', 'right', 'up', 'down']
export const POSE_LABEL: Record<PoseKey, string> = {
  front: 'Facing the camera',
  left: 'Turned to one side',
  right: 'Turned to the other side',
  up: 'Tilted up',
  down: 'Tilted down',
}

/** One registered pose: when it was captured, and the angle measured (yaw for left/right, pitch for up/down). */
export interface RegisteredPose {
  captured_at: string
  angle_deg: number
}

/** One enrolment check and what it found (neuropresence/enrolment/checks.py). */
export interface Check {
  key: CheckKey
  label: string
  passed: boolean
  /** What is wrong and what to change. Empty when the check passes, or when it could not be measured. */
  hint: string
  /** The measurement the verdict rests on, where there is one. */
  value: number | null
  /** For a check that passes: what would make the picture better still. */
  tip?: string
}

/** Who the user is: the face verified live with the camera. Only its signature is kept, never the picture.
 * poses: the richer signature, turned to each side and tilted up/down (see enrolment/checks.py), also
 * used to calibrate how far the reenactment engine lets this person's own head move. */
export interface FaceIdentity {
  id: string
  verified_at: string // local time, ISO format
  checks: Check[]
  poses: Partial<Record<PoseKey, RegisteredPose>>
}

/** The meeting picture's record. The picture itself is fetched from enrolledPictureUrl. */
export interface EnrolmentRecord {
  id: string
  enrolled_at: string // local time, ISO format
  origin: PictureOrigin
  width: number
  height: number
  checks: Check[]
  /** Whether a face signature was saved with it, for the identity check. */
  has_signature: boolean
  /** How alike its face is to the verified face (CSIM). Absent on a picture enrolled before that check existed. */
  match?: number | null
}

/** A picture that has been checked and is waiting to be kept or discarded. */
export interface Candidate {
  id: number
  origin: PictureOrigin
  width: number
  height: number
  checks: Check[]
  passed: boolean
  hint: string
  tip: string
}

/** A box in a camera picture, as shares of the picture's width and height. */
export interface FaceBox {
  x: number
  y: number
  w: number
  h: number
}

/** The camera preview on the Enrolment screen, with the checks run on its newest frame. */
export interface Preview {
  state: SessionState
  message: string
  input: string | null
  /** null for the ordinary frontal capture; which pose is being registered otherwise. */
  pose: PoseKey | null
  checks: Check[] | null
  /** The one thing to fix first. Empty when every check passes. */
  hint: string
  tip: string
  /** Every check has held steadily: a picture can be taken now. */
  ready: boolean
  /** Where the face should be: a face that fills this box passes the size and framing checks with room to spare. */
  outline?: FaceBox | null
  taking: boolean
}

export interface Enrolment {
  /** Step one: the face, verified with the camera. */
  face: FaceIdentity | null
  /** Step two: the meeting picture, uploaded and matched against that face. */
  record: EnrolmentRecord | null
  candidate: Candidate | null
  preview: Preview
}

export interface Feature {
  key: string
  label: string
  description: string
  enabled: boolean
}

export interface InputOption {
  id: string
  kind: 'camera' | 'sample'
  label: string
}

export interface Targets {
  fps: number
  render_ms: number
  end_to_end_ms: number
  vram_gb: number
  csim: number
}

export interface Status {
  session: Session
  enrolment: Enrolment
  identity: { available: boolean; csim: number | null; reason: string | null; guard?: IdentityGuard }
  features: Feature[]
  gpu: { name: string; total_gb: number; allocated_gb: number; peak_gb: number } | null
  targets: Targets
  inputs: InputOption[]
}

export interface StudioEvent {
  id: number
  at: number // seconds since 1970
  level: 'info' | 'warning' | 'error'
  message: string
}

export interface FrameHeader {
  id: number
  /** "live": a camera frame with the output made from it. "preview": a camera frame alone, on the Enrolment screen. */
  kind: 'live' | 'preview'
  live: boolean
  status: TrackStatus
  camera_bytes: number
  output_bytes: number
}

export interface PipelineStage {
  key: string
  name: string
  summary: string
  status: 'working' | 'partial' | 'planned'
  note: string | null
}

export interface Benchmark {
  file: string
  date: string
  machine: { os?: string; gpu?: string; python?: string; torch?: string }
  clips: number | null
  summary: {
    fps: number | null
    render_ms: number | null
    pipeline_ms: number | null
    peak_vram_gb: number | null
    csim_self_reenactment: number | null
    csim_real_video: number | null
    pose_error_deg: number | null
    mouth_opening_correlation: number | null
    lag_frames: number | null
    warping_error_ratio: number | null
    jitter_ratio: number | null
  }
  targets: Partial<Targets>
}

/** The numbers the enrolment checks compare against. */
export interface EnrolmentLimits {
  max_picture_px: number
  min_face_height_px: number
  good_face_height_px: number
  max_face_height_share: number
  max_yaw_deg: number
  max_pitch_deg: number
  max_roll_deg: number
  min_sharpness: number
  good_sharpness: number
  min_bright_level: number
  /** Two faces at least this alike (CSIM) count as the same person. */
  same_person_csim: number
}

/** What scripts/study_enrolment.py measured, in the shape the Enrolment screen draws. */
export interface EnrolmentStudy {
  clips: number
  good_picture: { csim: number; detail_kept: number }
  /** Faulty source pictures against a good one of the same clip, the costliest first. */
  faults: { name: string; clips: number; csim_change: number; detail_kept: number }[]
  face_size: {
    /** Detail is given as a multiple of what a face this tall yields. */
    reference_px: number
    pictures: { name: string; points: { face_height_px: number; detail: number }[] }[]
  }
}

export interface EnrolmentGuide {
  limits: EnrolmentLimits
  study: EnrolmentStudy | null
}

/** A request the backend refused. `message` is written for the person using the app. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new ApiError('The engine is not reachable. Start it with: python -m neuropresence.server', 0)
  }
  if (!response.ok) {
    let message = `The request failed (${response.status}).`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch {
      /* the reply had no JSON body; keep the generic message */
    }
    throw new ApiError(message, response.status)
  }
  return response.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const STOP_PREVIEW_PATH = '/api/enrolment/preview/stop'

export const api = {
  status: () => request<Status>('/api/status'),
  capabilities: () => request<{ stages: PipelineStage[]; enrolment?: PipelineStage }>('/api/capabilities'),
  latestBenchmark: () => request<Benchmark>('/api/benchmarks/latest'),

  enrolmentGuide: () => request<EnrolmentGuide>('/api/enrolment/guide'),
  startPreview: (input: string) => request<Status>('/api/enrolment/preview/start', { method: 'POST', ...json({ input }) }),
  /** Opens the camera to register one pose; /take and /confirm are the same calls as the frontal capture. */
  startPosePreview: (pose: PoseKey, input: string) =>
    request<Status>(`/api/enrolment/pose/${pose}/start`, { method: 'POST', ...json({ input }) }),
  stopPreview: () => request<Status>(STOP_PREVIEW_PATH, { method: 'POST' }),
  /** Takes the best frame of a short burst. Refused with the thing to fix if no frame passes the checks. */
  takePicture: () => request<Candidate>('/api/enrolment/take', { method: 'POST' }),
  /** The picture becomes the candidate even if a check fails, so the result can be shown. */
  uploadPicture: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Candidate>('/api/enrolment/upload', { method: 'POST', body: form })
  },
  /**
   * Keeps the candidate: a camera picture as the face signature (the picture is dropped), an uploaded
   * one as the meeting picture. The first upload loads the animation model, which takes a few seconds.
   */
  confirmPicture: () => request<{ face: FaceIdentity | null; record: EnrolmentRecord | null }>('/api/enrolment/confirm', { method: 'POST' }),
  discardCandidate: () => request<{ ok: boolean }>('/api/enrolment/candidate', { method: 'DELETE' }),
  /** Removes the meeting picture. The verified face stays. */
  removeEnrolment: () => request<{ ok: boolean }>('/api/enrolment', { method: 'DELETE' }),
  /** Removes the face signature and the meeting picture with it. */
  forgetFace: () => request<{ ok: boolean }>('/api/enrolment/face', { method: 'DELETE' }),

  startSession: (input: string) => request<Status>('/api/session/start', { method: 'POST', ...json({ input }) }),
  stopSession: () => request<Status>('/api/session/stop', { method: 'POST' }),
  resetNeutral: () => request<{ ok: boolean }>('/api/session/neutral', { method: 'POST' }),
  /** Go live again after the identity monitor put the still picture up. A fresh neutral pose is taken. */
  resumeSession: () => request<Status>('/api/session/resume', { method: 'POST' }),
  setFeature: (key: string, enabled: boolean) =>
    request<Feature>(`/api/features/${encodeURIComponent(key)}`, { method: 'PATCH', ...json({ enabled }) }),
}

/** Address of the enrolled picture. Its id changes with every enrolment, so the browser refetches it. */
export const enrolledPictureUrl = (id: string) => `/api/enrolment/picture?v=${encodeURIComponent(id)}`
/** Address of the picture waiting to be kept or discarded. */
export const candidatePictureUrl = (id: number) => `/api/enrolment/candidate/picture?v=${id}`
