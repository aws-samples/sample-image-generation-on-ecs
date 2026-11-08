import { useState } from 'react'
import type { EndpointStatus } from '@/types/admin'
import { scaleEndpoint, deleteEndpoint } from '@/lib/admin-api'

const STATUS_COLORS: Record<string, string> = {
  InService: 'bg-emerald-50 text-emerald-700',
  Creating: 'bg-blue-50 text-blue-700',
  Updating: 'bg-amber-50 text-amber-700',
  Failed: 'bg-red-50 text-red-700',
  Deleting: 'bg-slate-100 text-slate-600',
  NotFound: 'bg-slate-100 text-slate-500',
}

const BORDER_COLORS: Record<string, string> = {
  InService: 'border-l-emerald-500',
  Creating: 'border-l-blue-500',
  Updating: 'border-l-amber-500',
  Failed: 'border-l-red-500',
  Deleting: 'border-l-slate-400',
  NotFound: 'border-l-slate-300',
}

interface Props {
  endpoint: EndpointStatus
  onRefresh: () => void
  ready: boolean | null
}

export default function EndpointCard({ endpoint, onRefresh, ready }: Props) {
  const [showScale, setShowScale] = useState(false)
  const [min, setMin] = useState(endpoint.current_instance_count)
  const [max, setMax] = useState(endpoint.desired_instance_count)
  const [loading, setLoading] = useState(false)

  const handleScale = async () => {
    setLoading(true)
    try {
      await scaleEndpoint(endpoint.name, { min_capacity: min, max_capacity: max })
      setShowScale(false)
      onRefresh()
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async () => {
    if (!confirm(`Delete endpoint ${endpoint.name}? This cannot be undone.`)) return
    setLoading(true)
    try {
      await deleteEndpoint(endpoint.name)
      onRefresh()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={`bg-surface-raised rounded-lg shadow-soft border-l-4 ${BORDER_COLORS[endpoint.status] || 'border-l-slate-300'} p-6`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-ink">{endpoint.name}</h3>
        <div className="flex items-center gap-2">
          {endpoint.status === 'InService' && ready !== null && (
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${ready ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>
              {ready ? 'Ready' : 'Loading'}
            </span>
          )}
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${STATUS_COLORS[endpoint.status] || 'bg-gray-100'}`}>
            {endpoint.status}
          </span>
        </div>
      </div>

      <dl className="grid grid-cols-2 gap-2 text-sm">
        <dt className="text-muted">Instance Type</dt>
        <dd className="text-ink">{endpoint.instance_type || '—'}</dd>
        <dt className="text-muted">Current</dt>
        <dd className="text-ink">{endpoint.current_instance_count}</dd>
        <dt className="text-muted">Desired</dt>
        <dd className="text-ink">{endpoint.desired_instance_count}</dd>
      </dl>

      {endpoint.status === 'InService' && (
        <div className="mt-4 flex gap-2">
          <button
            onClick={() => setShowScale(!showScale)}
            className="px-3 py-1 text-sm bg-blue-50 text-blue-700 rounded hover:bg-blue-100"
          >
            Scale
          </button>
          <button
            onClick={handleDelete}
            disabled={loading}
            className="px-3 py-1 text-sm bg-red-50 text-red-700 rounded hover:bg-red-100"
          >
            Delete
          </button>
        </div>
      )}

      {showScale && (
        <div className="mt-3 p-3 bg-gray-50 rounded space-y-2">
          <div className="flex gap-4">
            <label className="text-sm">
              Min: <input type="number" min={0} value={min} onChange={e => setMin(+e.target.value)}
                className="w-16 ml-1 px-2 py-1 border rounded" />
            </label>
            <label className="text-sm">
              Max: <input type="number" min={1} value={max} onChange={e => setMax(+e.target.value)}
                className="w-16 ml-1 px-2 py-1 border rounded" />
            </label>
          </div>
          <button
            onClick={handleScale}
            disabled={loading}
            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            {loading ? 'Updating...' : 'Apply'}
          </button>
        </div>
      )}
    </div>
  )
}
