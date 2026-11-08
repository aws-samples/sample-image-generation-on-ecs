import { useEffect, useState, useCallback, useRef } from 'react'
import {
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Scatter,
} from 'recharts'
import type { EndpointStatus, EndpointMetricsTimeSeries } from '@/types/admin'
import { getEndpointMetricsTimeSeries } from '@/lib/admin-api'

interface Props {
  endpoints: EndpointStatus[]
}

interface ChartDataPoint {
  time: string
  label: string
  latency: number | null
  invocations: number | null
  errors5xx: number | null
}

function formatRelativeTime(isoStr: string): string {
  const diff = Date.now() - new Date(isoStr).getTime()
  const minutes = Math.round(diff / 60_000)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.round(hours / 24)}d ago`
}

function buildChartData(metrics: EndpointMetricsTimeSeries): ChartDataPoint[] {
  const { latency, invocations, errors5xx } = metrics

  // Merge all timestamps and sort ascending
  const allTimestamps = new Set([
    ...latency.timestamps,
    ...invocations.timestamps,
    ...errors5xx.timestamps,
  ])
  const sorted = [...allTimestamps].sort(
    (a, b) => new Date(a).getTime() - new Date(b).getTime(),
  )

  // Build lookup maps
  const latencyMap = new Map(latency.timestamps.map((t, i) => [t, latency.values[i]]))
  const invocMap = new Map(invocations.timestamps.map((t, i) => [t, invocations.values[i]]))
  const errMap = new Map(errors5xx.timestamps.map((t, i) => [t, errors5xx.values[i]]))

  return sorted.map((ts) => ({
    time: ts,
    label: formatRelativeTime(ts),
    latency: latencyMap.has(ts) ? (latencyMap.get(ts)! / 1000) : null, // convert us to ms
    invocations: invocMap.get(ts) ?? null,
    errors5xx: errMap.get(ts) ?? null,
  }))
}

function EndpointChart({ endpoint }: { endpoint: EndpointStatus }) {
  const [metrics, setMetrics] = useState<EndpointMetricsTimeSeries | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchMetrics = useCallback(async () => {
    try {
      const data = await getEndpointMetricsTimeSeries(endpoint.name, 6)
      setMetrics(data)
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load metrics')
    } finally {
      setLoading(false)
    }
  }, [endpoint.name])

  useEffect(() => {
    fetchMetrics()
    intervalRef.current = setInterval(fetchMetrics, 60_000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [fetchMetrics])

  if (loading) {
    return (
      <div className="card p-4 flex items-center justify-center h-[200px]">
        <div className="animate-spin rounded-full h-4 w-4 border-2 border-accent border-t-transparent" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-4 h-[200px] flex items-center justify-center">
        <p className="text-red-500 text-xs">{error}</p>
      </div>
    )
  }

  if (!metrics) return null

  const chartData = buildChartData(metrics)

  if (chartData.length === 0) {
    return (
      <div className="card p-4 h-[200px] flex flex-col">
        <h4 className="text-sm font-semibold text-ink truncate mb-2">
          {endpoint.name.replace(/^dev-zimg-/, '')}
        </h4>
        <div className="flex-1 flex items-center justify-center">
          <p className="text-muted text-xs">No data available</p>
        </div>
      </div>
    )
  }

  // Show ~6 tick labels across the x-axis
  const tickInterval = Math.max(1, Math.floor(chartData.length / 6))

  return (
    <div className="card p-4 h-[200px] flex flex-col">
      <h4 className="text-sm font-semibold text-ink truncate mb-1">
        {endpoint.name.replace(/^dev-zimg-/, '')}
      </h4>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10 }}
              stroke="var(--color-muted, #94a3b8)"
              interval={tickInterval}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              yAxisId="latency"
              orientation="left"
              tick={{ fontSize: 10 }}
              stroke="var(--color-muted, #94a3b8)"
              tickLine={false}
              axisLine={false}
              width={32}
              tickFormatter={(v: number) => `${Math.round(v)}`}
            />
            <YAxis
              yAxisId="invocations"
              orientation="right"
              tick={{ fontSize: 10 }}
              stroke="var(--color-muted, #94a3b8)"
              tickLine={false}
              axisLine={false}
              width={28}
              hide
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-surface, #1e293b)',
                border: '1px solid var(--color-border, #334155)',
                borderRadius: '6px',
                fontSize: '11px',
              }}
              labelFormatter={(label) => String(label)}
              formatter={(value, name) => {
                const v = Number(value)
                if (name === 'latency') return [`${v.toFixed(1)} ms`, 'Latency (p99)']
                if (name === 'invocations') return [`${Math.round(v)}`, 'Invocations']
                if (name === 'errors5xx') return [`${Math.round(v)}`, '5xx Errors']
                return [String(value), String(name)]
              }}
            />
            <Area
              yAxisId="invocations"
              type="monotone"
              dataKey="invocations"
              fill="var(--color-accent, #6366f1)"
              fillOpacity={0.1}
              stroke="none"
              connectNulls
            />
            <Line
              yAxisId="latency"
              type="monotone"
              dataKey="latency"
              stroke="var(--color-accent, #6366f1)"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
            <Scatter
              yAxisId="invocations"
              dataKey="errors5xx"
              fill="#ef4444"
              shape="circle"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-3 mt-1 text-[10px] text-muted">
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-0.5 rounded bg-accent inline-block" /> Latency (ms)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm bg-accent/10 inline-block border border-accent/30" /> Invocations
        </span>
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500 inline-block" /> 5xx
        </span>
      </div>
    </div>
  )
}

export default function MetricsPanel({ endpoints }: Props) {
  if (endpoints.length === 0) {
    return <p className="text-muted text-sm">No endpoints available.</p>
  }

  return (
    <div>
      <h3 className="text-lg font-bold mb-4">Endpoint Metrics</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {endpoints.map((ep) => (
          <EndpointChart key={ep.name} endpoint={ep} />
        ))}
      </div>
    </div>
  )
}
