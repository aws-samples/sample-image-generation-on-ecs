export interface EndpointStatus {
  name: string
  status: 'InService' | 'Creating' | 'Updating' | 'Failed' | 'Deleting' | 'NotFound'
  instance_type: string
  current_instance_count: number
  desired_instance_count: number
}

export interface EndpointMetrics {
  invocations: number[]
  latency_p99_ms: number[]
  errors_4xx: number[]
  errors_5xx: number[]
  timestamps: string[]
}

export interface MetricTimeSeries {
  timestamps: string[]
  values: number[]
}

export interface EndpointMetricsTimeSeries {
  latency: MetricTimeSeries
  invocations: MetricTimeSeries
  errors5xx: MetricTimeSeries
}

export interface AlarmState {
  name: string
  state: 'OK' | 'ALARM' | 'INSUFFICIENT_DATA'
  endpoint: string
  metric: string
}

export interface ScaleRequest {
  min_capacity: number
  max_capacity: number
}

export interface SfnExecution {
  execution_arn: string
  name: string
  status: 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'TIMED_OUT' | 'ABORTED'
  start_date: string
  stop_date: string | null
}

export interface PipelineStage {
  stage: string
  status: string | null
}

export interface TargetInstance {
  id: string
  availability_zone: string
  health_state: 'healthy' | 'unhealthy' | 'draining' | 'initial' | 'unused'
  reason: string | null
  active_connections: number | null
}

export interface EndpointInstances {
  name: string
  resolution: string
  running_count: number
  desired_count: number
  targets: TargetInstance[]
}
