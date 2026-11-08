import { useState, useCallback, useRef, useEffect } from 'react'
import { FrameType, type GenerationParams, type Frame } from '@/types'
import { getIdToken } from '@/lib/auth'

const log = (...args: unknown[]) => console.debug(`[zimg ${new Date().toISOString().slice(11, 23)}]`, ...args)

interface UseWebSocketReturn {
  connect: (params: GenerationParams, wsUrl?: string) => void
  disconnect: () => void
  frames: Frame[]
  isConnected: boolean
  isGenerating: boolean
  error: string | null
  clearError: () => void
  requestId: string | null
  currentFrame: Frame | null
  intermediateUrl: string
  serverElapsed: number | null
}

/**
 * WebSocket hook for real-time image generation.
 *
 * Intermediate frames update a ref (no re-render) and use
 * requestAnimationFrame to update the display efficiently.
 * Only the final frame triggers a React state update.
 */
export function useWebSocket(): UseWebSocketReturn {
  const [frames, setFrames] = useState<Frame[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [requestId, setRequestId] = useState<string | null>(null)
  const [currentFrame, setCurrentFrame] = useState<Frame | null>(null)
  const [intermediateUrl, setIntermediateUrl] = useState('')
  const [serverElapsed, setServerElapsed] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const pendingUrlRef = useRef<string>('')
  const rafRef = useRef<number>(0)
  const stepCountRef = useRef(0)
  const connectingRef = useRef(false)
  const generationRef = useRef(0)

  const connect = useCallback((params: GenerationParams, wsUrl?: string) => {
    // Prevent rapid double-calls (React state is async, ref is synchronous)
    if (connectingRef.current) {
      log(' connect blocked — already connecting')
      return
    }
    connectingRef.current = true
    log(' connect called', { width: params.width, height: params.height, backend: params.backend, denoise: params.denoise_viz })

    // Close previous connection if still open. Its close event fires
    // asynchronously — tag this attempt with a generation id so that stale
    // event can never clobber the state of the NEW connection started below.
    const generation = ++generationRef.current
    if (wsRef.current) {
      log(' closing previous WebSocket')
      wsRef.current.close()
      wsRef.current = null
    }

    // Reset state and immediately mark as generating to prevent double-clicks
    setIsGenerating(true)
    setFrames([])
    setError(null)
    setRequestId(null)
    setCurrentFrame(null)
    setIntermediateUrl('')
    setServerElapsed(null)
    pendingUrlRef.current = ''
    stepCountRef.current = 0

    // Exchange Cognito token for a short-lived HMAC ticket, then open WS
    const token = getIdToken()
    let url = wsUrl || ''

    if (!url) {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      url = `${protocol}//${window.location.host}/ws/generate`
    }

    const openWs = (ticket: string | null) => {
      const separator = url.includes('?') ? '&' : '?'
      const fullUrl = ticket ? `${url}${separator}ticket=${ticket}` : url

      log(' opening WebSocket...')
      const ws = new WebSocket(fullUrl)
      const t0 = performance.now()
      let firstFrameLogged = false

      ws.onopen = () => {
        setIsConnected(true)
        ws.send(JSON.stringify({ action: 'generate', ...params }))
        log(` ws_open=${((performance.now() - t0) / 1000).toFixed(2)}s — sent generate request`)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (!data.type) {
            log(' msg: unknown (no type field)', Object.keys(data))
            return
          }
          log(' msg:', data.type, data.frame_type || '')

          if (data.type === 'request_id') {
            log(` request_id=${data.request_id} backend=${data.backend} endpoint=${data.endpoint}`)
            setRequestId(data.request_id)
          } else if (data.type === 'frame') {
            if (data.frame_type === 'final') {
              log(` final_frame=${((performance.now() - t0) / 1000).toFixed(2)}s url=${data.url?.substring(0, 60)}...`)
              const frame: Frame = {
                frame_type: FrameType.FINAL,
                step_index: data.step || 0,
                payload: new Uint8Array(0),
                metadata: { url: data.url },
              }
              setFrames((prev) => [...prev, frame])
              setCurrentFrame(frame)
              setIntermediateUrl('')
            } else {
              if (!firstFrameLogged) {
                log(` first_intermediate=${((performance.now() - t0) / 1000).toFixed(2)}s`)
                firstFrameLogged = true
              }
              stepCountRef.current += 1
              const frame: Frame = {
                frame_type: FrameType.INTERMEDIATE,
                step_index: data.step || stepCountRef.current,
                payload: new Uint8Array(0),
                metadata: { url: data.url },
              }
              setFrames((prev) => [...prev, frame])
              pendingUrlRef.current = data.url
              if (!rafRef.current) {
                rafRef.current = requestAnimationFrame(() => {
                  rafRef.current = 0
                  setIntermediateUrl(pendingUrlRef.current)
                })
              }
            }
          } else if (data.type === 'complete') {
            if (data.elapsed_s != null) {
              setServerElapsed(data.elapsed_s)
            }
            console.debug(
              `[zimg] complete=${((performance.now() - t0) / 1000).toFixed(2)}s` +
              ` lambda=${data.lambda_elapsed_s ?? '?'}s endpoint=${data.endpoint}`
            )
            connectingRef.current = false
            setIsGenerating(false)
          } else if (data.type === 'error') {
            log(` error: ${data.message}`)
            setError(data.message || 'Generation error')
            connectingRef.current = false
            setIsGenerating(false)
          }
        } catch {
          // Non-JSON message — ignore
        }
      }

      ws.onerror = (e) => {
        log(' ws_error', e)
        if (generation !== generationRef.current) return
        setError('WebSocket error')
        connectingRef.current = false
        setIsGenerating(false)
      }

      ws.onclose = (e) => {
        log(` ws_close code=${e.code} reason=${e.reason}`)
        if (generation !== generationRef.current) return
        setIsConnected(false)
        connectingRef.current = false
        setIsGenerating(false)
      }

      wsRef.current = ws
    }

    // Fetch short-lived ticket, then open WebSocket
    if (token) {
      fetch('/api/ws-ticket', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((res) => {
          if (!res.ok) throw new Error(`ticket fetch failed: ${res.status}`)
          return res.json()
        })
        .then((data) => openWs(data.ticket))
        .catch((err) => {
          log(' ticket exchange failed', err)
          setError('Failed to obtain WebSocket ticket')
          connectingRef.current = false
          setIsGenerating(false)
        })
    } else {
      openWs(null)
    }
  }, [])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setIsGenerating(false)
  }, [])

  useEffect(() => {
    return () => {
      disconnect()
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [disconnect])

  return {
    connect,
    disconnect,
    frames,
    isConnected,
    isGenerating,
    error,
    clearError: useCallback(() => setError(null), []),
    requestId,
    currentFrame,
    intermediateUrl,
    serverElapsed,
  }
}
