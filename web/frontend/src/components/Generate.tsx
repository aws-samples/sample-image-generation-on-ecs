import { useState, useEffect, useRef } from 'react'
import { Download, X, Sparkles, ChevronDown, ChevronUp } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { useWebSocket } from '@/hooks/useWebSocket.ts'
import { useGenerateState } from '@/contexts/GenerateContext'
import { FrameType } from '@/types'
import { getIdToken } from '@/lib/auth'

export default function Generate() {
  const { prompt, setPrompt, seedMode, setSeedMode, seed, setSeed, backend, setBackend, width, setWidth, height, setHeight, denoiseViz, setDenoiseViz } = useGenerateState()
  const queryClient = useQueryClient()
  const [wsUrl, setWsUrl] = useState('')
  const [gpuEnabled, setGpuEnabled] = useState(true)
  const [inferentiaEnabled, setInferentiaEnabled] = useState(true)
  const [gpuMaxPixels, setGpuMaxPixels] = useState(1048576)
  const [gpuMinDimension, setGpuMinDimension] = useState(256)
  const [gpuDimensionStep, setGpuDimensionStep] = useState(8)
  const [endpointStatus, setEndpointStatus] = useState<Record<string, string | boolean>>({})
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [enhanceError, setEnhanceError] = useState('')
  const [showLightbox, setShowLightbox] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [finalTime, setFinalTime] = useState<number | null>(null)
  const [wasCancelled, setWasCancelled] = useState(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const { connect, disconnect, frames, isGenerating, error, clearError, currentFrame, intermediateUrl, serverElapsed, requestId } = useWebSocket()

  useEffect(() => {
    console.debug(`[zimg:ui ${new Date().toISOString().slice(11,23)}] isGenerating changed:`, isGenerating)
    if (isGenerating) {
      setElapsed(0)
      setFinalTime(null)
      const start = Date.now()
      timerRef.current = setInterval(() => setElapsed(Date.now() - start), 50)
    } else {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
      if (elapsed > 0) setFinalTime(elapsed)
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [isGenerating])

  useEffect(() => {
    const preview = new URLSearchParams(window.location.search).get('preview')

    fetch('/api/config')
      .then((r) => r.json())
      .then((data) => {
        if (data.websocket_url) setWsUrl(data.websocket_url)

        let gpu = data.gpu_enabled ?? true
        let inf = data.inferentia_enabled ?? true
        if (preview === 'gpu') { gpu = true; inf = false }
        else if (preview === 'neuron') { gpu = false; inf = true }
        else if (preview === 'both') { gpu = true; inf = true }

        setGpuEnabled(gpu)
        setInferentiaEnabled(inf)
        const defaultBe = gpu && !inf ? 'gpu' : 'neuron'
        setBackend(defaultBe)

        if (!gpu && backend === 'gpu') setBackend('neuron')
        if (!inf && backend === 'neuron') setBackend('gpu')
        if (data.gpu_max_pixels) setGpuMaxPixels(data.gpu_max_pixels)
        if (data.gpu_min_dimension) setGpuMinDimension(data.gpu_min_dimension)
        if (data.gpu_dimension_step) setGpuDimensionStep(data.gpu_dimension_step)
      })
      .catch(() => {})
    const fetchStatus = () => {
      const token = getIdToken()
      fetch('/api/endpoint-status', { headers: token ? { Authorization: `Bearer ${token}` } : {} })
        .then((r) => r.json())
        .then(setEndpointStatus)
        .catch(() => {})
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 30_000)
    return () => clearInterval(interval)
  }, [])

  const configLoaded = wsUrl !== ''
  const selectedEndpointKey = backend === 'gpu' ? 'gpu' : `neuron_${width}`
  const epStatus = endpointStatus[selectedEndpointKey]
  const endpointReady = epStatus === true || epStatus === 'ready'
  const endpointOffline = epStatus === 'offline'
  const endpointUnknown = epStatus === undefined

  const pixels = width * height
  const pixelBudgetOk = backend !== 'gpu' || pixels <= gpuMaxPixels
  const dimStepOk = backend !== 'gpu' || (width % gpuDimensionStep === 0 && height % gpuDimensionStep === 0)
  const dimMinOk = backend !== 'gpu' || (width >= gpuMinDimension && height >= gpuMinDimension)
  const resolutionValid = pixelBudgetOk && dimStepOk && dimMinOk
  const currentStep = Math.min(frames.filter((f) => f.frame_type === FrameType.INTERMEDIATE).length, 8)
  const hasFinal = frames.some((f) => f.frame_type === FrameType.FINAL)
  const finalFrame = frames.find((f) => f.frame_type === FrameType.FINAL)
  const finalUrl = (finalFrame?.metadata?.url as string) || ''

  useEffect(() => {
    if (hasFinal) queryClient.invalidateQueries({ queryKey: ['gallery'] })
  }, [hasFinal])

  useEffect(() => {
    console.debug(`[zimg:ui ${new Date().toISOString().slice(11,23)}] render state`, { frames: frames.length, currentStep, hasFinal, displayUrl: displayUrl?.substring(0, 50), requestId: requestId?.substring(0, 8), isGenerating, error })
  }, [frames, isGenerating, error, requestId])
  const displayUrl = finalUrl || intermediateUrl || (currentFrame?.metadata?.url as string) || ''

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.debug(`[zimg:ui ${new Date().toISOString().slice(11,23)}] handleSubmit`, { prompt: prompt.trim().substring(0, 30), configLoaded, isGenerating, endpointReady })
    if (!prompt.trim() || !configLoaded || isGenerating) {
      console.debug(`[zimg:ui ${new Date().toISOString().slice(11,23)}] handleSubmit blocked`, { hasPrompt: !!prompt.trim(), configLoaded, isGenerating })
      return
    }
    setWasCancelled(false)
    console.debug(`[zimg:ui ${new Date().toISOString().slice(11,23)}] calling connect`, { width, height, backend, denoiseViz })
    connect(
      {
        prompt: prompt.trim(),
        seed: seedMode === 'fixed' && seed ? parseInt(seed) : Math.floor(Math.random() * 2147483647),
        width,
        height,
        backend,
        denoise_viz: denoiseViz,
      },
      wsUrl
    )
  }

  const handleDownload = () => {
    if (!finalUrl) return
    const a = document.createElement('a')
    a.href = finalUrl
    a.download = `zimage-${Date.now()}.png`
    a.target = '_blank'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleEnhance = async () => {
    if (!prompt.trim() || isEnhancing) return
    setIsEnhancing(true)
    setEnhanceError('')
    try {
      const token = getIdToken()
      if (!token) {
        setEnhanceError('Please sign in to use prompt enhancement.')
        return
      }
      const resp = await fetch('/api/enhance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ prompt: prompt.trim() }),
      })
      if (!resp.ok) {
        const errBody = await resp.json().catch(() => null)
        throw new Error(errBody?.error || `Enhancement failed (${resp.status})`)
      }
      const data = await resp.json()
      if (data.enhanced_prompt) setPrompt(data.enhanced_prompt)
    } catch (err) {
      setEnhanceError(err instanceof Error ? err.message : 'Enhancement failed. Please try again.')
    } finally {
      setIsEnhancing(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="grid grid-cols-1 lg:grid-cols-[340px_minmax(0,540px)] gap-10 items-start justify-center">
        {/* Controls */}
        <div className="space-y-5">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Prompt */}
            <div>
              <label htmlFor="prompt-input" className="block text-sm font-semibold text-muted mb-2">
                What do you want to create?
              </label>
              <textarea
                id="prompt-input"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A dreamy landscape with floating islands..."
                className="input-field h-44 resize-none"
                disabled={isGenerating || isEnhancing}
              />
              <button
                type="button"
                onClick={handleEnhance}
                disabled={!prompt.trim() || isGenerating || isEnhancing}
                className="mt-3 w-full inline-flex items-center justify-center gap-2 px-4 py-2.5
                         text-sm font-semibold text-accent bg-accent-light border border-accent/20
                         rounded-full shadow-soft
                         hover:bg-accent/10 hover:border-accent/40 hover:shadow-card
                         disabled:opacity-40 disabled:pointer-events-none
                         transition-all duration-200"
              >
                {isEnhancing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-accent border-t-transparent" />
                    Enhancing prompt...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    Enhance Prompt
                  </>
                )}
              </button>
              {enhanceError && (
                <p className="mt-2 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-center">
                  {enhanceError}
                </p>
              )}
            </div>

            {/* Resolution */}
            <div>
              <span className="block text-sm font-semibold text-muted mb-2">Resolution</span>
              {backend === 'neuron' ? (
                <div className="segmented-control">
                  {[512, 768, 1024].map((res) => (
                    <button
                      key={res}
                      type="button"
                      onClick={() => { setWidth(res); setHeight(res); clearError() }}
                      disabled={isGenerating}
                      className={`segmented-btn ${width === res && height === res ? 'segmented-btn-active' : ''}`}
                    >
                      {res}
                    </button>
                  ))}
                </div>
              ) : (
                <>
                  <select
                    value={`${width}x${height}`}
                    onChange={(e) => {
                      const [w, h] = e.target.value.split('x').map(Number)
                      setWidth(w); setHeight(h); clearError()
                    }}
                    disabled={isGenerating}
                    className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-sm text-ink"
                  >
                    <option value="512x512">Square Small — 512 × 512</option>
                    <option value="768x768">Square Medium — 768 × 768</option>
                    <option value="1024x1024">Square Large — 1024 × 1024</option>
                    <option value="1280x720">Landscape — 1280 × 720</option>
                    <option value="720x1280">Portrait — 720 × 1280</option>
                    <option value="1024x768">Landscape 4:3 — 1024 × 768</option>
                    <option value="768x1024">Portrait 3:4 — 768 × 1024</option>
                  </select>
                  {!resolutionValid && (
                    <p className="mt-1.5 text-xs text-red-600">
                      {!pixelBudgetOk && `${pixels.toLocaleString()} px exceeds max ${gpuMaxPixels.toLocaleString()} px`}
                      {!dimStepOk && `Dimensions must be divisible by ${gpuDimensionStep}`}
                      {!dimMinOk && `Minimum dimension is ${gpuMinDimension}px`}
                    </p>
                  )}
                </>
              )}
            </div>

            {/* Endpoint status indicator */}
            <div className="flex items-center gap-2 text-xs">
              <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-medium ${
                endpointReady ? 'bg-emerald-50 text-emerald-700'
                : endpointOffline ? 'bg-red-50 text-red-700'
                : endpointUnknown ? 'bg-slate-100 text-slate-500'
                : 'bg-amber-50 text-amber-700'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${
                  endpointReady ? 'bg-emerald-500'
                  : endpointOffline ? 'bg-red-500'
                  : endpointUnknown ? 'bg-slate-400'
                  : 'bg-amber-500 animate-pulse'
                }`} />
                {endpointReady ? 'Ready' : endpointOffline ? 'Offline' : endpointUnknown ? 'Checking...' : 'Warming up'}
              </span>
            </div>

            {/* Advanced settings (collapsible) */}
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-sm font-medium text-muted hover:text-ink transition-colors"
            >
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              Advanced Settings
            </button>

            {showAdvanced && (
              <div className="space-y-4 pl-2 border-l-2 border-accent-light">
                {/* Engine */}
                {gpuEnabled && inferentiaEnabled && (
                  <div>
                    <span className="block text-sm font-semibold text-muted mb-2">Engine</span>
                    <div className="segmented-control">
                      {(['neuron', 'gpu'] as const).map((opt) => (
                        <button
                          key={opt}
                          type="button"
                          onClick={() => {
                            setBackend(opt)
                            // Neuron only supports square 512/768/1024 — GPU allows
                            // arbitrary (incl. non-square) resolutions, so a GPU pick
                            // may not match any Neuron button. Reset so the pill
                            // selector always has an active option.
                            if (opt === 'neuron' && !(width === height && [512, 768, 1024].includes(width))) {
                              setWidth(1024); setHeight(1024)
                            }
                            clearError()
                          }}
                          disabled={isGenerating}
                          className={`segmented-btn ${backend === opt ? 'segmented-btn-active' : ''}`}
                        >
                          {opt === 'neuron' ? 'Neuron' : 'GPU'}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Seed */}
                <div>
                  <span className="block text-sm font-semibold text-muted mb-2">Seed</span>
                  <div className="segmented-control">
                    {(['random', 'fixed'] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setSeedMode(mode)}
                        disabled={isGenerating}
                        className={`segmented-btn capitalize ${seedMode === mode ? 'segmented-btn-active' : ''}`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                  {seedMode === 'fixed' && (
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      placeholder="Enter seed"
                      className="input-field mt-2"
                      disabled={isGenerating}
                    />
                  )}
                </div>

                {/* Denoise toggle */}
                <div>
                  <span className="block text-sm font-semibold text-muted mb-2">Denoising</span>
                  <div className="segmented-control">
                    <button type="button" onClick={() => setDenoiseViz(true)} disabled={isGenerating} className={`segmented-btn ${denoiseViz ? 'segmented-btn-active' : ''}`}>On</button>
                    <button type="button" onClick={() => setDenoiseViz(false)} disabled={isGenerating} className={`segmented-btn ${!denoiseViz ? 'segmented-btn-active' : ''}`}>Off</button>
                  </div>
                </div>
              </div>
            )}

            {/* Submit + Download row */}
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={isGenerating || !prompt.trim() || !configLoaded || !endpointReady || !resolutionValid}
                className="btn-primary flex-1"
              >
                {isGenerating ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    {!requestId ? 'Connecting...' : denoiseViz && currentStep > 0 ? `Generating... (${currentStep}/8)` : 'Generating...'}
                  </span>
                ) : (
                  'Generate'
                )}
              </button>
              <button
                type="button"
                onClick={handleDownload}
                disabled={!hasFinal || isGenerating}
                className="btn-secondary flex items-center justify-center gap-2 px-4"
              >
                <Download className="w-4 h-4" />
              </button>
            </div>

            {isGenerating && (
              <button type="button" onClick={() => { console.debug(`[zimg:ui ${new Date().toISOString().slice(11,23)}] cancel clicked`); setWasCancelled(true); disconnect() }} className="btn-secondary w-full flex items-center justify-center gap-2">
                <X className="w-4 h-4" />
                Cancel
              </button>
            )}

            {!endpointReady && !isGenerating && (
              <p className={`text-xs rounded-lg px-3 py-2 text-center ${
                endpointOffline ? 'text-red-600 bg-red-50 border border-red-200'
                : endpointUnknown ? 'text-slate-600 bg-slate-50 border border-slate-200'
                : 'text-amber-600 bg-amber-50 border border-amber-200'
              }`}>
                {endpointOffline
                  ? 'This endpoint is offline (scaled to 0). Scale it up from the Admin console.'
                  : endpointUnknown
                  ? 'Checking endpoint status...'
                  : 'This endpoint is still warming up. Please wait or try a different engine/resolution.'}
              </p>
            )}
          </form>

          {error && (
            <div className="p-4 bg-red-50 border border-red-100 rounded-2xl">
              <p className="text-sm text-red-600 font-medium">{error}</p>
            </div>
          )}
        </div>

        {/* Image */}
        <div className="space-y-4">
          <div
            className={`card overflow-hidden flex items-center justify-center aspect-square ${
              hasFinal && !isGenerating ? 'cursor-pointer hover:shadow-glow transition-shadow duration-300' : ''
            }`}
            onClick={() => hasFinal && !isGenerating && setShowLightbox(true)}
          >
            {displayUrl ? (
              <img
                src={displayUrl}
                alt={hasFinal ? 'Generated image' : `Step ${currentStep}`}
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-muted/40 text-center p-12 text-lg font-medium">
                {isGenerating ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="animate-spin rounded-full h-8 w-8 border-2 border-muted/30 border-t-accent" />
                    <span>{!requestId ? 'Connecting...' : 'Generating image...'}</span>
                  </div>
                ) : 'Your creation will appear here'}
              </div>
            )}
          </div>

          {/* Timer */}
          {(isGenerating || finalTime !== null) && (
            <p className="text-center text-sm font-mono text-muted">
              {isGenerating ? `${(elapsed / 1000).toFixed(1)}s` : wasCancelled ? 'Cancelled' : `Created in ${serverElapsed != null ? serverElapsed.toFixed(2) : (finalTime! / 1000).toFixed(2)}s`}
            </p>
          )}

          {/* Progress */}
          {isGenerating && denoiseViz && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs font-medium text-muted">
                <span>Step {currentStep} / 8</span>
                <span>{Math.round((currentStep / 8) * 100)}%</span>
              </div>
              <div className="w-full bg-accent-light rounded-full h-1.5">
                <div
                  className="bg-accent h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${(currentStep / 8) * 100}%` }}
                />
              </div>
            </div>
          )}

        </div>
      </div>

      {/* Lightbox */}
      {showLightbox && finalUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink/80 backdrop-blur-md p-6"
          onClick={() => setShowLightbox(false)}
        >
          <div className="max-w-[90vw] max-h-[90vh] flex flex-col items-center" onClick={(e) => e.stopPropagation()}>
            <img src={finalUrl} alt="Generated image" className="max-w-full max-h-[80vh] object-contain rounded-3xl shadow-card" />
            <div className="mt-5 text-center">
              <p className="text-white/50 text-xs font-semibold uppercase tracking-wider">
                {width}&times;{height} &middot; {backend}
              </p>
              <p className="mt-2 text-white/90 text-sm max-w-lg">{prompt}</p>
            </div>
            <button onClick={() => setShowLightbox(false)} className="mt-5 btn-ghost text-white/70 border border-white/20">
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
