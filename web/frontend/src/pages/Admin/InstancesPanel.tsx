import { useState } from 'react'
import type { EndpointInstances, TargetInstance } from '@/types/admin'
import { scaleEndpoint } from '@/lib/admin-api'

const HEALTH_BADGE: Record<string, { bg: string; text: string; dot: string }> = {
  healthy: { bg: 'bg-emerald-50', text: 'text-emerald-700', dot: 'bg-emerald-500' },
  unhealthy: { bg: 'bg-red-50', text: 'text-red-700', dot: 'bg-red-500' },
  draining: { bg: 'bg-amber-50', text: 'text-amber-700', dot: 'bg-amber-500' },
  initial: { bg: 'bg-blue-50', text: 'text-blue-700', dot: 'bg-blue-500' },
  unused: { bg: 'bg-slate-100', text: 'text-slate-500', dot: 'bg-slate-400' },
}


function HealthBadge({ state }: { state: string }) {
  const badge = HEALTH_BADGE[state] || HEALTH_BADGE.unused
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-medium rounded-full ${badge.bg} ${badge.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${badge.dot}`} />
      {state}
    </span>
  )
}

function TargetRow({ target }: { target: TargetInstance }) {
  return (
    <tr className="border-b border-border/30 last:border-0">
      <td className="px-4 py-2 text-xs font-mono text-ink">{target.id}</td>
      <td className="px-4 py-2 text-xs text-muted">{target.availability_zone}</td>
      <td className="px-4 py-2"><HealthBadge state={target.health_state} /></td>
      <td className="px-4 py-2 text-xs text-muted">{target.reason || '—'}</td>
    </tr>
  )
}

interface ScaleState {
  desired: number
  min: number
  max: number
}

interface Props {
  instances: EndpointInstances[]
  onRefresh?: () => void
}

export default function InstancesPanel({ instances, onRefresh }: Props) {
  const [scales, setScales] = useState<Record<string, ScaleState>>(() => {
    const init: Record<string, ScaleState> = {}
    for (const ep of instances) {
      init[ep.resolution] = {
        desired: ep.desired_count,
        min: ep.desired_count,
        max: 10,
      }
    }
    return init
  })
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState<string | null>(null)

  if (instances.length === 0) {
    return <p className="text-muted text-sm">No instance data available.</p>
  }

  function updateScale(resolution: string, field: keyof ScaleState, value: number) {
    setScales(prev => {
      const current = prev[resolution]
      const updated = { ...current, [field]: value }
      if (field === 'desired') {
        updated.min = value
        if (value > updated.max) updated.max = value
      } else if (field === 'min' && value > updated.max) {
        updated.max = value
      } else if (field === 'max' && value < updated.min) {
        updated.min = value
      }
      return { ...prev, [resolution]: updated }
    })
  }


  async function handleApplyAll() {
    setLoading(true)
    try {
      const minWait = new Promise(r => setTimeout(r, 3000))
      await Promise.all([
        minWait,
        ...instances.map(ep => {
          const s = scales[ep.resolution]
          if (!s) return Promise.resolve()
          return scaleEndpoint(ep.resolution, { min_capacity: s.min, max_capacity: s.max })
        }),
      ])
      onRefresh?.()
    } finally {
      setLoading(false)
    }
  }

  const totalRunning = instances.reduce((sum, ep) => sum + ep.running_count, 0)
  const totalDesired = instances.reduce((sum, ep) => sum + ep.desired_count, 0)

  return (
    <div className={`card overflow-hidden ${loading ? 'opacity-60 pointer-events-none' : ''}`}>
      <div className="px-5 py-4 border-b border-border flex items-center justify-between">
        <h3 className="text-base font-bold">Instance Monitoring</h3>
        <span className="text-xs text-muted">
          {loading ? <span className="text-blue-600 font-semibold">Applying changes...</span> : (
            <>Total: <span className="font-semibold text-ink">{totalRunning}/{totalDesired}</span> running</>
          )}
        </span>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface-raised border-b border-border">
            <th className="text-left px-5 py-2.5 text-xs font-semibold text-muted">Endpoint</th>
            <th className="text-center px-3 py-2.5 text-xs font-semibold text-muted">Running</th>
            <th className="text-center px-3 py-2.5 text-xs font-semibold text-muted">Desired</th>
            <th className="text-center px-3 py-2.5 text-xs font-semibold text-muted">Min</th>
            <th className="text-center px-3 py-2.5 text-xs font-semibold text-muted">Max</th>
            <th className="text-center px-3 py-2.5 text-xs font-semibold text-muted">Health</th>
          </tr>
        </thead>
        <tbody>
          {instances.map(ep => {
            const s = scales[ep.resolution] || { desired: ep.desired_count, min: ep.desired_count, max: 10 }
            const healthyCount = ep.targets.filter(t => t.health_state === 'healthy').length
            const label = ep.resolution === 'gpu' ? 'GPU' : `${ep.resolution}px`
            const isExpanded = expanded === ep.resolution

            return (
              <>
                <tr
                  key={ep.resolution}
                  className="border-b border-border hover:bg-surface-raised/50 cursor-pointer"
                  onClick={() => setExpanded(isExpanded ? null : ep.resolution)}
                >
                  <td className="px-5 py-3">
                    <span className="font-semibold text-ink">{label}</span>
                    <span className="ml-2 text-xs text-muted font-mono">{ep.name}</span>
                  </td>
                  <td className="text-center px-3 py-3 text-sm font-bold text-ink">{ep.running_count}</td>
                  <td className="text-center px-3 py-3" onClick={e => e.stopPropagation()}>
                    <input
                      type="number" min={0} value={s.desired} disabled={loading}
                      onChange={e => updateScale(ep.resolution, 'desired', Number(e.target.value))}
                      className="w-[50px] px-1.5 py-0.5 text-xs text-center border border-border rounded bg-surface font-bold disabled:opacity-50"
                    />
                  </td>
                  <td className="text-center px-3 py-3" onClick={e => e.stopPropagation()}>
                    <input
                      type="number" min={0} value={s.min} disabled={loading}
                      onChange={e => updateScale(ep.resolution, 'min', Number(e.target.value))}
                      className="w-[50px] px-1.5 py-0.5 text-xs text-center border border-border rounded bg-surface disabled:opacity-50"
                    />
                  </td>
                  <td className="text-center px-3 py-3" onClick={e => e.stopPropagation()}>
                    <input
                      type="number" min={1} value={s.max} disabled={loading}
                      onChange={e => updateScale(ep.resolution, 'max', Number(e.target.value))}
                      className="w-[50px] px-1.5 py-0.5 text-xs text-center border border-border rounded bg-surface disabled:opacity-50"
                    />
                  </td>
                  <td className="text-center px-3 py-3">
                    <span className="text-sm font-bold text-ink">{healthyCount}/{ep.targets.length}</span>
                  </td>
                </tr>
                {isExpanded && ep.targets.length > 0 && (
                  <tr key={`${ep.resolution}-detail`}>
                    <td colSpan={6} className="p-0">
                      <table className="w-full bg-surface-raised/30">
                        <thead>
                          <tr className="border-b border-border/50">
                            <th className="text-left px-6 py-1.5 text-[10px] font-semibold text-muted">IP</th>
                            <th className="text-left px-4 py-1.5 text-[10px] font-semibold text-muted">AZ</th>
                            <th className="text-left px-4 py-1.5 text-[10px] font-semibold text-muted">Health</th>
                            <th className="text-left px-4 py-1.5 text-[10px] font-semibold text-muted">Reason</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ep.targets.map(t => <TargetRow key={t.id} target={t} />)}
                        </tbody>
                      </table>
                    </td>
                  </tr>
                )}
              </>
            )
          })}
        </tbody>
      </table>

      <div className="px-5 py-3 border-t border-border flex justify-end">
        <button
          onClick={handleApplyAll}
          disabled={loading}
          className="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Applying...' : 'Apply Changes'}
        </button>
      </div>
    </div>
  )
}
