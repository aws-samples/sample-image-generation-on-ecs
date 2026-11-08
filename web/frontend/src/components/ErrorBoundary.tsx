/**
 * Minimal React error boundary.
 *
 * Wraps a route subtree and renders a fallback UI when a render error
 * escapes. Without this a single bad render blanks the entire SPA.
 */

import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: (error: Error, retry: () => void) => ReactNode
  onError?: (error: Error, info: React.ErrorInfo) => void
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info)
    this.props.onError?.(error, info)
  }

  retry = () => this.setState({ error: null })

  render() {
    if (this.state.error) {
      if (this.props.fallback) return this.props.fallback(this.state.error, this.retry)
      return (
        <div className="rounded border border-red-300 bg-red-50 p-4 text-red-800">
          <h2 className="mb-2 text-lg font-semibold">Something went wrong</h2>
          <p className="mb-3 text-sm">{this.state.error.message}</p>
          <button
            type="button"
            onClick={this.retry}
            className="rounded bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-700"
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
