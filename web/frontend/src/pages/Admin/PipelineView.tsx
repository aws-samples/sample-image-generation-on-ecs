import type { PipelineStage } from '@/types/admin'

const STATUS_STYLES: Record<string, { dot: string; text: string }> = {
  Succeeded: { dot: 'bg-emerald-500', text: 'text-emerald-700' },
  InProgress: { dot: 'bg-accent animate-pulse', text: 'text-accent' },
  Failed: { dot: 'bg-red-500', text: 'text-red-700' },
  Stopped: { dot: 'bg-amber-500', text: 'text-amber-700' },
}

interface Props {
  stages: PipelineStage[]
}

export default function PipelineView({ stages }: Props) {
  if (stages.length === 0) {
    return <p className="text-muted text-sm">No pipeline data.</p>
  }

  return (
    <div className="card p-5">
      <h3 className="text-base font-bold mb-4">Pipeline</h3>
      <div className="space-y-0">
        {stages.map((stage, i) => {
          const style = STATUS_STYLES[stage.status || ''] || { dot: 'bg-border', text: 'text-muted' }
          return (
            <div key={stage.stage} className="flex items-start gap-3">
              {/* Vertical line + dot */}
              <div className="flex flex-col items-center">
                <span className={`w-3 h-3 rounded-full ${style.dot} shrink-0 mt-0.5`} />
                {i < stages.length - 1 && <div className="w-0.5 h-6 bg-border" />}
              </div>
              {/* Label */}
              <div className="pb-3">
                <span className="text-sm font-medium text-ink">{stage.stage}</span>
                <span className={`ml-2 text-xs font-medium ${style.text}`}>{stage.status}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
