import type { AlarmState } from '@/types/admin'

const STATE_BADGE: Record<string, { bg: string; text: string; label: string }> = {
  OK: { bg: 'bg-emerald-50', text: 'text-emerald-700', label: 'OK' },
  ALARM: { bg: 'bg-red-50', text: 'text-red-700', label: 'Alarm' },
  INSUFFICIENT_DATA: { bg: 'bg-slate-100', text: 'text-slate-500', label: 'No data' },
}

interface Props {
  alarms: AlarmState[]
}

export default function AlarmPanel({ alarms }: Props) {
  if (alarms.length === 0) {
    return <p className="text-muted text-sm">No alarms configured.</p>
  }

  const grouped = alarms.reduce<Record<string, AlarmState[]>>((acc, alarm) => {
    const key = alarm.endpoint || 'other'
    ;(acc[key] ??= []).push(alarm)
    return acc
  }, {})

  return (
    <div className="card overflow-hidden">
      <div className="px-5 py-4 border-b border-border">
        <h3 className="text-base font-bold">Alarms</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-surface-raised">
              <th className="text-left px-5 py-2.5 font-semibold text-muted">Endpoint</th>
              <th className="text-left px-5 py-2.5 font-semibold text-muted">Metric</th>
              <th className="text-left px-5 py-2.5 font-semibold text-muted">Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(grouped).map(([endpoint, groupAlarms]) =>
              groupAlarms.map((alarm, i) => {
                const badge = STATE_BADGE[alarm.state] || STATE_BADGE.INSUFFICIENT_DATA
                return (
                  <tr key={alarm.name} className="border-b border-border/50 last:border-0">
                    {i === 0 && (
                      <td className="px-5 py-2.5 font-medium text-ink align-top" rowSpan={groupAlarms.length}>
                        {endpoint}
                      </td>
                    )}
                    <td className="px-5 py-2.5 text-muted">{alarm.metric}</td>
                    <td className="px-5 py-2.5">
                      <span className={`inline-flex px-2 py-0.5 text-xs font-semibold rounded-full ${badge.bg} ${badge.text}`}>
                        {badge.label}
                      </span>
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
