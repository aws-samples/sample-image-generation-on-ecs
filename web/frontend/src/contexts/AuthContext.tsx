/**
 * React context providing authentication state.
 *
 * Manages the PKCE flow: on /auth/callback, exchanges code for tokens
 * directly with Cognito (no backend round-trip). Auto-refreshes tokens
 * before expiry. Refreshes on tab focus to handle laptop wake.
 */

import { useQueryClient } from '@tanstack/react-query'
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'

import {
  clearTokens,
  getAuthConfig,
  getIdToken,
  handleCallback,
  isAdmin as checkAdmin,
  isAuthenticated as checkAuth,
  login,
  logout,
  refreshTokens,
} from '@/lib/auth'

interface AuthConfig {
  clientId: string
  domain: string
  region: string
  userPoolId: string
}

interface AuthContextType {
  isAuthenticated: boolean
  isAdmin: boolean
  isLoading: boolean
  token: string | null
  authError: string | null
  doLogin: () => void
  doLogout: () => void
}

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  isAdmin: false,
  isLoading: true,
  token: null,
  authError: null,
  doLogin: () => {},
  doLogout: () => {},
})

export function useAuth() {
  return useContext(AuthContext)
}

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const [isAuth, setIsAuth] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [token, setToken] = useState<string | null>(null)
  const [config, setConfig] = useState<AuthConfig | null>(null)
  const [authError, setAuthError] = useState<string | null>(null)

  // Load auth config on mount
  useEffect(() => {
    getAuthConfig()
      .then(setConfig)
      .catch((e) => {
        console.error('Failed to load auth config', e)
        setAuthError('Auth service unavailable')
        setIsLoading(false)
      })
  }, [])

  // Handle PKCE callback (?code=...&state=...) on /auth/callback
  useEffect(() => {
    if (!config) return

    const url = new URL(window.location.href)
    const isCallback = url.pathname === '/auth/callback'
    const code = url.searchParams.get('code')
    const state = url.searchParams.get('state')

    if (isCallback && code) {
      handleCallback(code, state).then((success) => {
        if (success) {
          setIsAuth(true)
          setToken(getIdToken())
        } else {
          setAuthError('Login failed — please try again')
        }
        setIsLoading(false)
        // Navigate to the Generate page (router-aware so the route actually changes).
        navigate('/', { replace: true })
      })
      return
    }

    // Existing session
    if (checkAuth()) {
      setIsAuth(true)
      setToken(getIdToken())
    }
    setIsLoading(false)
  }, [config])

  // Proactive token refresh every 50 minutes (id/access tokens last 60).
  useEffect(() => {
    if (!isAuth || !config) return
    const interval = setInterval(async () => {
      const ok = await refreshTokens(config)
      if (ok) {
        setToken(getIdToken())
      } else {
        setIsAuth(false)
        setToken(null)
        clearTokens()
      }
    }, 50 * 60 * 1000)
    return () => clearInterval(interval)
  }, [isAuth, config])

  // Refresh on tab focus to handle laptop wake / long backgrounding.
  useEffect(() => {
    if (!isAuth || !config) return
    const onVisibility = async () => {
      if (document.visibilityState !== 'visible') return
      // Only refresh if token is near expiry — getIdToken returns null then.
      if (!getIdToken()) {
        const ok = await refreshTokens(config)
        if (ok) setToken(getIdToken())
        else {
          setIsAuth(false)
          setToken(null)
          clearTokens()
        }
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [isAuth, config])

  const doLogin = useCallback(() => {
    if (config) login(config)
  }, [config])

  const doLogout = useCallback(() => {
    // Clear React Query cache so the next user doesn't see this user's data flash.
    queryClient.clear()
    if (config) logout(config)
  }, [config, queryClient])

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated: isAuth,
        isAdmin: isAuth && checkAdmin(),
        isLoading,
        token,
        authError,
        doLogin,
        doLogout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}
