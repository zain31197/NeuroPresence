/** The live connection to the engine: status, events and frames over one WebSocket. */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, type FrameHeader, type Status, type StudioEvent } from './api'
import { parseFrame } from './frames'

export type Connection = 'connecting' | 'open' | 'closed'

/** One reading of the live figures, kept for the trend lines. */
export interface Sample {
  fps: number | null
  endToEnd: number | null
  render: number | null
  csim: number | null
  /** Head turn and nod in degrees, and mouth opening from 0 to 1, as the tracker reads them. */
  yaw: number | null
  pitch: number | null
  mouth: number | null
}

/** `output` is null for a frame of the enrolment preview, which has none. */
type FrameListener = (header: FrameHeader, camera: Blob, output: Blob | null) => void

interface Studio {
  connection: Connection
  status: Status | null
  events: StudioEvent[]
  /** Up to the last minute of readings, oldest first. Empty unless a session is running. */
  history: Sample[]
  /** Be called for every frame the engine sends. Returns the function that unsubscribes. */
  onFrame: (listener: FrameListener) => () => void
  /** Read the status now, without waiting for the next push. Call it after an action that changes it. */
  refresh: () => Promise<void>
}

const HISTORY_SAMPLES = 240 // one minute at four status messages a second
const EVENT_LIMIT = 100

const StudioContext = createContext<Studio | null>(null)

export function StudioProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState<Connection>('connecting')
  const [status, setStatus] = useState<Status | null>(null)
  const [events, setEvents] = useState<StudioEvent[]>([])
  const [history, setHistory] = useState<Sample[]>([])
  const listeners = useRef(new Set<FrameListener>())

  useEffect(() => {
    let socket: WebSocket | null = null
    let retry: number | undefined
    let closed = false
    let attempt = 0

    const connect = () => {
      setConnection('connecting')
      const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
      socket = new WebSocket(`${scheme}://${location.host}/api/stream`)
      socket.binaryType = 'arraybuffer'
      let first = true

      socket.onopen = () => {
        attempt = 0
        setConnection('open')
      }
      socket.onmessage = (message) => {
        if (typeof message.data !== 'string') {
          const [header, camera, output] = parseFrame(message.data as ArrayBuffer)
          listeners.current.forEach((listener) => listener(header, camera, output))
          return
        }
        const update: { status: Status; events: StudioEvent[] } = JSON.parse(message.data)
        setStatus(update.status)
        // A new connection replays the engine's whole event list, so start from it.
        const replay = first
        first = false
        if (replay || update.events.length > 0) {
          setEvents((seen) => [...(replay ? [] : seen), ...update.events].slice(-EVENT_LIMIT))
        }
        const { session, identity } = update.status
        setHistory((samples) => {
          if (session.state !== 'running' || !session.metrics) return samples.length ? [] : samples
          // "?? null" so a figure an older engine does not send is a gap, not a crash.
          const sample: Sample = {
            fps: session.metrics.fps ?? null,
            endToEnd: session.metrics.end_to_end_ms ?? null,
            render: session.metrics.render_ms ?? null,
            csim: identity.csim ?? null,
            yaw: session.tracking?.pose_deg?.[0] ?? null,
            pitch: session.tracking?.pose_deg?.[1] ?? null,
            mouth: session.tracking?.mouth_open ?? null,
          }
          return [...samples, sample].slice(-HISTORY_SAMPLES)
        })
      }
      socket.onclose = () => {
        if (closed) return
        setConnection('closed')
        attempt += 1
        retry = window.setTimeout(connect, Math.min(5000, 500 * attempt))
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry)
      socket?.close()
    }
  }, [])

  const onFrame = useCallback((listener: FrameListener) => {
    listeners.current.add(listener)
    return () => {
      listeners.current.delete(listener)
    }
  }, [])

  const refresh = useCallback(async () => {
    setStatus(await api.status())
  }, [])

  const value = useMemo(
    () => ({ connection, status, events, history, onFrame, refresh }),
    [connection, status, events, history, onFrame, refresh],
  )
  return <StudioContext.Provider value={value}>{children}</StudioContext.Provider>
}

export function useStudio(): Studio {
  const studio = useContext(StudioContext)
  if (!studio) throw new Error('useStudio must be used inside <StudioProvider>.')
  return studio
}
