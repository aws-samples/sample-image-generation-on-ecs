import { useState } from 'react'
import type { SfnExecution } from '@/types/admin'
import { startSfn, stopSfn } from '@/lib/admin-api'

interface Props {
  executions: SfnExecution[]
  onRefresh: () => void
}

export default function SfnPanel({ executions, onRefresh }: Props) {
  const [loading, setLoading] = useState(false)

  const running = executions.find(e => e.status === 'RUNNING')

  const handleStart = async () => {
    setLoading(true)
    try {
      await startSfn()
      onRefresh()
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    if (!confirm('Stop the running compile execution?')) return
    setLoading(true)
    try {
      await stopSfn()
      onRefresh()
    } finally {
      setLoading(false)
    }
  }

  const STATUS_BADGE: Record<string, { bg: string; text: string }> = {
    RUNNING: { bg: 'bg-blue-50', text: 'text-blue-700' },
    SUCCEEDED: { bg: 'bg-emerald-50', text: 'text-emerald-700' },
    FAILED: { bg: 'bg-red-50', text: 'text-red-700' },
    TIMED_OUT: { bg: 'bg-amber-50', text: 'text-amber-700' },
    ABORTED: { bg: 'bg-slate-100', text: 'text-slate-600' },
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-ink">Compile (SFN)</h3>
        <div className="flex gap-2">
          {running ? (
            <button
              onClick={handleStop}
              disabled={loading}
              className="px-3 py-1 text-sm bg-red-50 text-red-700 rounded hover:bg-red-100"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={loading}
              className="px-3 py-1 text-sm bg-green-50 text-green-700 rounded hover:bg-green-100"
            >
              Start Compile
            </button>
          )}
        </div>
      </div>

      <div className="space-y-2 max-h-48 overflow-y-auto">
        {executions.map(exec => {
          const badge = STATUS_BADGE[exec.status] || { bg: 'bg-slate-100', text: 'text-slate-500' }
          return (
            <div key={exec.execution_arn} className="flex items-center justify-between text-sm py-1.5 border-b border-border/50 last:border-0">
              <span className="text-ink font-mono truncate">{exec.name}</span>
              <span className={`inline-flex px-2 py-0.5 text-xs font-semibold rounded-full ${badge.bg} ${badge.text}`}>
                {exec.status}
              </span>
              <span className="text-muted text-xs">
                {new Date(exec.start_date).toLocaleString()}
              </span>
            </div>
          )
        })}
        {executions.length === 0 && <p className="text-muted text-sm">No executions.</p>}
      </div>
    </div>
  )
}
