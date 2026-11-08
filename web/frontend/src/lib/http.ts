/**
 * Shared axios factory with bearer-token + 401-refresh-and-retry interceptors.
 *
 * Used by api.ts (general API) and admin-api.ts (admin actions). Eliminates
 * the duplicate interceptor blocks (A-5, Q-25) and adds automatic token
 * refresh on 401 (Q-16).
 */

import axios, { AxiosError, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios'
import { clearTokens, getAuthConfig, getIdToken, refreshTokens } from './auth'

interface RetryableConfig extends InternalAxiosRequestConfig {
  _retry?: boolean
}

let _refreshInFlight: Promise<boolean> | null = null

/** Coalesce concurrent refresh attempts so we don't spam Cognito. */
async function refreshOnce(): Promise<boolean> {
  if (!_refreshInFlight) {
    _refreshInFlight = (async () => {
      try {
        const cfg = await getAuthConfig()
        return await refreshTokens(cfg)
      } finally {
        _refreshInFlight = null
      }
    })()
  }
  return _refreshInFlight
}

export function createAuthedAxios(baseURL: string): AxiosInstance {
  const instance = axios.create({
    baseURL,
    headers: { 'Content-Type': 'application/json' },
  })

  // Attach bearer token to every request (when present).
  instance.interceptors.request.use((config) => {
    const token = getIdToken()
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
  })

  // On 401, try a single refresh + retry; on second failure, clear tokens.
  instance.interceptors.response.use(
    (resp) => resp,
    async (error: AxiosError) => {
      const original = error.config as RetryableConfig | undefined
      if (!original || error.response?.status !== 401 || original._retry) {
        return Promise.reject(error)
      }
      original._retry = true
      const ok = await refreshOnce()
      if (!ok) {
        clearTokens()
        return Promise.reject(error)
      }
      const newToken = getIdToken()
      if (newToken) original.headers.Authorization = `Bearer ${newToken}`
      return instance(original)
    },
  )

  return instance
}
