import { useEffect, useState, useCallback } from 'react'
import type { EndpointStatus, AlarmState, PipelineStage, SfnExecution, EndpointInstances } from '@/types/admin'
import { getEndpoints, getAlarms, getPipelineStatus, getSfnExecutions, getInstances } from '@/lib/admin-api'
import EndpointCard from './EndpointCard'
import AlarmPanel from './AlarmPanel'
import InstancesPanel from './InstancesPanel'
import MetricsPanel from './MetricsPanel'
import PipelineView from './PipelineView'
import SfnPanel from './SfnPanel'

export default function AdminDashboard() {
  const [endpoints, setEndpoints] = useState<EndpointStatus[]>([])
  const [alarms, setAlarms] = useState<AlarmState[]>([])
  const [pipeline, setPipeline] = useState<PipelineStage[]>([])
  const [sfn, setSfn] = useState<SfnExecution[]>([])
  const [instances, setInstances] = useState<EndpointInstances[]>([])
  const [endpointReady, setEndpointReady] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [ep, al, pl, sf, inst] = await Promise.all([
        getEndpoints(),
        getAlarms(),
        getPipelineStatus(),
        getSfnExecutions(),
        getInstances(),
      ])
      setEndpoints(ep)
      setAlarms(al)
      setPipeline(pl)
      setSfn(sf)
      setInstances(inst)
      setError(null)
      // Also fetch probe-based readiness
      fetch('/api/endpoint-status', {
        headers: { Authorization: `Bearer ${(await import('@/lib/auth')).getIdToken() || ''}` },
      })
        .then((r) => r.json())
        .then(setEndpointReady)
        .catch(() => {})
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load admin data')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, 5_000)
    return () => clearInterval(interval)
  }, [refresh])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-6 w-6 border-2 border-accent border-t-transparent" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-5">
        <p className="text-red-600 font-medium">{error}</p>
        <button onClick={refresh} className="mt-2 text-sm text-accent font-medium hover:underline">Retry</button>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-1 h-8 rounded-full bg-accent" />
          <h2 className="font-display text-3xl font-extrabold tracking-tight text-ink">Admin</h2>
        </div>
        <button onClick={refresh} className="btn-secondary text-sm">
          Refresh
        </button>
      </div>

      {/* Pipeline + SFN side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PipelineView stages={pipeline} />
        <SfnPanel executions={sfn} onRefresh={refresh} />
      </div>

      {/* Instance Monitoring */}
      <InstancesPanel instances={instances} onRefresh={refresh} />

      {/* Endpoint Metrics Charts */}
      <MetricsPanel endpoints={endpoints} />

      {/* Alarms table */}
      <AlarmPanel alarms={alarms} />

      {/* Endpoints */}
      <div>
        <h3 className="text-lg font-bold mb-4">Endpoints</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {endpoints.map(ep => (
            <EndpointCard key={ep.name} endpoint={ep} onRefresh={refresh} ready={endpointReady[ep.name.includes('gpu') ? 'gpu' : `neuron_${ep.name.split('-').pop()}`] ?? null} />
          ))}
        </div>
      </div>
    </div>
  )
}
