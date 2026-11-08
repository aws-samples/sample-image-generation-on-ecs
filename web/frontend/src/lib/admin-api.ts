/** Admin API client. Shares the bearer-token + refresh-on-401 interceptors with api.ts. */

import type {
  AlarmState,
  EndpointInstances,
  EndpointMetrics,
  EndpointMetricsTimeSeries,
  EndpointStatus,
  PipelineStage,
  ScaleRequest,
  SfnExecution,
} from '@/types/admin'

import { createAuthedAxios } from './http'

const adminApi = createAuthedAxios('/admin/api')

export async function getEndpoints(): Promise<EndpointStatus[]> {
  const { data } = await adminApi.get('/endpoints')
  return Array.isArray(data) ? data : []
}

export async function getEndpointMetrics(name: string): Promise<EndpointMetrics> {
  const { data } = await adminApi.get(`/endpoints/${name}/metrics`)
  return data
}

export async function getEndpointMetricsTimeSeries(
  endpoint: string,
  hours: number = 6,
): Promise<EndpointMetricsTimeSeries> {
  const { data } = await adminApi.get(`/metrics`, { params: { endpoint, hours } })
  return data
}

export async function scaleEndpoint(name: string, req: ScaleRequest): Promise<void> {
  await adminApi.post(`/endpoints/${name}/scale`, req)
}

export async function deleteEndpoint(name: string): Promise<void> {
  await adminApi.delete(`/endpoints/${name}`)
}

export async function getAlarms(): Promise<AlarmState[]> {
  const { data } = await adminApi.get('/alarms')
  return Array.isArray(data) ? data : []
}

export async function getPipelineStatus(): Promise<PipelineStage[]> {
  const { data } = await adminApi.get('/pipeline')
  return Array.isArray(data) ? data : []
}

export async function getSfnExecutions(): Promise<SfnExecution[]> {
  const { data } = await adminApi.get('/sfn')
  return Array.isArray(data) ? data : []
}

export async function startSfn(): Promise<{ execution_arn: string; name: string }> {
  const { data } = await adminApi.post('/sfn/start')
  return data
}

export async function stopSfn(): Promise<{ status: string }> {
  const { data } = await adminApi.post('/sfn/stop')
  return data
}

export async function getInstances(): Promise<EndpointInstances[]> {
  const { data } = await adminApi.get('/instances')
  return Array.isArray(data) ? data : []
}
