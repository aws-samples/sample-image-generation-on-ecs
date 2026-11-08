/**
 * Cognito auth — PKCE OAuth2 flow with localStorage token storage.
 *
 *   1. login() generates PKCE verifier+challenge, redirects to Cognito.
 *   2. Cognito redirects back to /auth/callback?code=...&state=...
 *   3. handleCallback() exchanges code+verifier for tokens directly with Cognito.
 *   4. Tokens stored in localStorage.
 *   5. logout() revokes refresh token, clears storage, redirects to Cognito.
 */

import {
  exchangeCodeForTokens,
  generateCodeChallenge,
  generateCodeVerifier,
  popPkce,
  refreshAccessToken,
  revokeRefreshToken,
  stashPkce,
} from './cognito'

const TOKEN_KEYS = {
  id: 'zimage_id_token',
  access: 'zimage_access_token',
  refresh: 'zimage_refresh_token',
  expiry: 'zimage_token_expiry',
}

interface AuthConfig {
  userPoolId: string
  clientId: string
  domain: string
  region: string
}

let _configCache: AuthConfig | null = null

/** Read Cognito config from <meta> tag injected at deploy time. */
export async function getAuthConfig(): Promise<AuthConfig> {
  if (_configCache) return _configCache
  const meta = document.querySelector('meta[name="cognito-config"]')
  if (!meta) {
    throw new Error(
      'Cognito config meta tag missing. The deploy script must inject ' +
        '<meta name="cognito-config" data-client-id=... data-domain=... /> in index.html.',
    )
  }
  _configCache = {
    userPoolId: meta.getAttribute('data-user-pool-id') || '',
    clientId: meta.getAttribute('data-client-id') || '',
    domain: meta.getAttribute('data-domain') || '',
    region: meta.getAttribute('data-region') || 'us-east-1',
  }
  return _configCache
}

/** Redirect to Cognito Hosted UI with PKCE challenge. */
export async function login(config: { clientId: string; domain: string }): Promise<void> {
  const verifier = generateCodeVerifier(64)
  const challenge = await generateCodeChallenge(verifier)
  const state = crypto.randomUUID()
  stashPkce(verifier, state)

  const redirectUri = `${window.location.origin}/auth/callback`
  const url =
    `https://${config.domain}/login?` +
    `client_id=${config.clientId}&` +
    `response_type=code&` +
    `scope=openid+email+profile&` +
    `redirect_uri=${encodeURIComponent(redirectUri)}&` +
    `state=${state}&` +
    `code_challenge=${challenge}&` +
    `code_challenge_method=S256`

  window.location.href = url
}

/**
 * Exchange the authorization code for tokens after Cognito redirect.
 *
 * Returns true on success, false if state mismatch / verifier missing /
 * Cognito rejects the exchange.
 */
export async function handleCallback(code: string, returnedState: string | null): Promise<boolean> {
  const stash = popPkce()
  if (!stash) {
    console.error('PKCE: no verifier in sessionStorage — login flow not started here?')
    return false
  }
  if (stash.state !== returnedState) {
    console.error('PKCE: state mismatch — possible CSRF attempt')
    return false
  }

  const config = await getAuthConfig()
  const redirectUri = `${window.location.origin}/auth/callback`
  try {
    const tokens = await exchangeCodeForTokens(
      config.domain,
      config.clientId,
      code,
      stash.verifier,
      redirectUri,
    )
    storeTokens(tokens)
    return true
  } catch (e) {
    console.error('Token exchange failed:', e)
    return false
  }
}

function storeTokens(tokens: {
  id_token: string
  access_token?: string
  refresh_token?: string
  expires_in?: number
}) {
  localStorage.setItem(TOKEN_KEYS.id, tokens.id_token)
  if (tokens.access_token) localStorage.setItem(TOKEN_KEYS.access, tokens.access_token)
  if (tokens.refresh_token) localStorage.setItem(TOKEN_KEYS.refresh, tokens.refresh_token)
  const expiresIn = tokens.expires_in || 3600
  localStorage.setItem(TOKEN_KEYS.expiry, (Date.now() + expiresIn * 1000).toString())
}

export function clearTokens(): void {
  Object.values(TOKEN_KEYS).forEach((key) => localStorage.removeItem(key))
}

/** Revoke refresh token, clear storage, redirect to Cognito logout. */
export async function logout(config: { clientId: string; domain: string }): Promise<void> {
  const refreshToken = getRefreshToken()
  if (refreshToken) {
    try {
      await revokeRefreshToken(config.domain, config.clientId, refreshToken)
    } catch (e) {
      console.warn('Refresh-token revocation failed (continuing logout):', e)
    }
  }
  clearTokens()
  const logoutUri = encodeURIComponent(window.location.origin + '/')
  window.location.href = `https://${config.domain}/logout?client_id=${config.clientId}&logout_uri=${logoutUri}`
}

/** Get valid ID token, or null if expired/missing. */
export function getIdToken(): string | null {
  const token = localStorage.getItem(TOKEN_KEYS.id)
  if (!token) return null
  const expiry = parseInt(localStorage.getItem(TOKEN_KEYS.expiry) || '0', 10)
  if (Date.now() > expiry - 60000) return null
  return token
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(TOKEN_KEYS.refresh)
}

export function isAuthenticated(): boolean {
  return getIdToken() !== null
}

export function isAdmin(): boolean {
  const token = getIdToken()
  if (!token) return false
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    const groups: string[] = payload['cognito:groups'] || []
    return groups.includes('admins')
  } catch {
    return false
  }
}

/** Refresh tokens via refresh_token grant. Returns true on success. */
export async function refreshTokens(config: { clientId: string; domain: string }): Promise<boolean> {
  const refresh = getRefreshToken()
  if (!refresh) return false
  try {
    const tokens = await refreshAccessToken(config.domain, config.clientId, refresh)
    storeTokens(tokens)
    return true
  } catch {
    return false
  }
}
