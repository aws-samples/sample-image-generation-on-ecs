/**
 * Cognito PKCE OAuth2 helpers.
 *
 * PKCE (Proof Key for Code Exchange, RFC 7636) lets a public SPA use the
 * authorization code flow without a backend mediating the token exchange:
 *
 *   1. SPA generates random code_verifier (43–128 chars).
 *   2. SPA computes code_challenge = base64url(sha256(code_verifier)).
 *   3. SPA sends code_challenge to /authorize.
 *   4. SPA sends code + code_verifier to /oauth2/token.
 *   5. Cognito verifies sha256(verifier) == challenge → returns tokens.
 *
 * Tokens never transit our backend, so we can drop the server-side
 * /auth/callback route (S-7, A-2).
 */

const VERIFIER_KEY = 'zimage_pkce_verifier'
const STATE_KEY = 'zimage_oauth_state'

/** Cryptographically random URL-safe string of length 43–128. */
export function generateCodeVerifier(length = 64): string {
  const bytes = new Uint8Array(length)
  crypto.getRandomValues(bytes)
  return base64UrlEncode(bytes).slice(0, length)
}

/** SHA-256 hash of the verifier, base64url-encoded. */
export async function generateCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder()
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(verifier))
  return base64UrlEncode(new Uint8Array(digest))
}

/** Stash verifier + state in sessionStorage for the round-trip to Cognito. */
export function stashPkce(verifier: string, state: string): void {
  sessionStorage.setItem(VERIFIER_KEY, verifier)
  sessionStorage.setItem(STATE_KEY, state)
}

/** Pop verifier + state on callback; returns null if either is missing. */
export function popPkce(): { verifier: string; state: string } | null {
  const verifier = sessionStorage.getItem(VERIFIER_KEY)
  const state = sessionStorage.getItem(STATE_KEY)
  sessionStorage.removeItem(VERIFIER_KEY)
  sessionStorage.removeItem(STATE_KEY)
  if (!verifier || !state) return null
  return { verifier, state }
}

interface TokenResponse {
  id_token: string
  access_token?: string
  refresh_token?: string
  expires_in?: number
}

/** Exchange the auth code for tokens via Cognito's /oauth2/token. */
export async function exchangeCodeForTokens(
  domain: string,
  clientId: string,
  code: string,
  verifier: string,
  redirectUri: string,
): Promise<TokenResponse> {
  const body = new URLSearchParams({
    grant_type: 'authorization_code',
    client_id: clientId,
    code,
    redirect_uri: redirectUri,
    code_verifier: verifier,
  })
  const resp = await fetch(`https://${domain}/oauth2/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })
  if (!resp.ok) {
    throw new Error(`Token exchange failed: ${resp.status} ${await resp.text()}`)
  }
  return resp.json()
}

/** Refresh tokens using the refresh_token. */
export async function refreshAccessToken(
  domain: string,
  clientId: string,
  refreshToken: string,
): Promise<TokenResponse> {
  const body = new URLSearchParams({
    grant_type: 'refresh_token',
    client_id: clientId,
    refresh_token: refreshToken,
  })
  const resp = await fetch(`https://${domain}/oauth2/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: body.toString(),
  })
  if (!resp.ok) throw new Error(`Refresh failed: ${resp.status}`)
  return resp.json()
}

/** Revoke a refresh token (logout cleanup). */
export async function revokeRefreshToken(
  domain: string,
  clientId: string,
  refreshToken: string,
): Promise<void> {
  await fetch(`https://${domain}/oauth2/revoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ token: refreshToken, client_id: clientId }).toString(),
  })
}

// ── helpers ─────────────────────────────────────────────────────────────

function base64UrlEncode(bytes: Uint8Array): string {
  let str = ''
  for (let i = 0; i < bytes.length; i++) str += String.fromCharCode(bytes[i])
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
