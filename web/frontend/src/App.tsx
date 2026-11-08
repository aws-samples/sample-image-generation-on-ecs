import { Link, Route, Routes, useLocation } from 'react-router-dom'

import Gallery from './components/Gallery.tsx'
import Generate from './components/Generate.tsx'
import { ErrorBoundary } from './components/ErrorBoundary'
import { useAuth } from './contexts/AuthContext'
import { GenerateProvider } from './contexts/GenerateContext'
import AdminDashboard from './pages/Admin'

function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  const location = useLocation()
  const active = location.pathname === to
  return (
    <Link
      to={to}
      className={`px-4 py-2 text-sm font-semibold rounded-full transition-all duration-200 ${
        active
          ? 'bg-accent text-white shadow-soft'
          : 'text-muted hover:text-ink hover:bg-accent-light'
      }`}
    >
      {children}
    </Link>
  )
}

function App() {
  const { isAuthenticated, isAdmin, isLoading, doLogin, doLogout, authError } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-accent font-semibold text-lg">Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {authError && (
        <div className="bg-red-50 border-b border-red-100 px-4 py-2 text-sm text-red-600 font-medium">
          {authError}
        </div>
      )}

      <nav className="sticky top-0 z-40 bg-canvas/80 backdrop-blur-md border-b border-border/50">
        <div className="w-full px-6 lg:px-10">
          <div className="flex justify-between h-16 items-center">
            <Link to="/" className="font-display text-xl font-extrabold tracking-tight text-accent">
              Z-Image
            </Link>
            <div className="flex items-center gap-1">
              {isAuthenticated && (
                <>
                  <NavLink to="/">Create</NavLink>
                  <NavLink to="/gallery">Gallery</NavLink>
                  {isAdmin && <NavLink to="/admin">Admin</NavLink>}
                  <div className="w-px h-5 bg-border mx-2" />
                  <button
                    onClick={doLogout}
                    className="btn-ghost text-sm"
                  >
                    Sign out
                  </button>
                </>
              )}
              {!isAuthenticated && (
                <button onClick={doLogin} className="btn-primary text-sm">
                  Sign in
                </button>
              )}
            </div>
          </div>
        </div>
      </nav>

      <main className="w-full px-6 lg:px-10 py-10">
        {isAuthenticated ? (
          <GenerateProvider>
            <Routes>
              <Route path="/" element={<ErrorBoundary><Generate /></ErrorBoundary>} />
              <Route path="/gallery" element={<ErrorBoundary><Gallery /></ErrorBoundary>} />
              <Route path="/admin" element={<ErrorBoundary><AdminDashboard /></ErrorBoundary>} />
            </Routes>
          </GenerateProvider>
        ) : (
          <div className="text-center py-24">
            <h2 className="font-display text-5xl font-extrabold tracking-tight mb-4 text-accent">
              Create stunning images
            </h2>
            <p className="text-muted text-lg mb-10 max-w-lg mx-auto">
              Generate beautiful visuals in seconds.
            </p>
            <button onClick={doLogin} className="btn-primary text-base px-8 py-3.5">
              Get started
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
