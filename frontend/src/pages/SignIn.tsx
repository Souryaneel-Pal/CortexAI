import { useState } from 'react'
import { useNavigate, Navigate } from 'react-router-dom'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import { loginUser } from '../lib/api'

export function SignIn() {
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const token = sessionStorage.getItem('auth_token')
  if (token) {
    return <Navigate to="/dashboard" replace />
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      const response = await loginUser({ email, password })
      sessionStorage.setItem('auth_token', response.token)
      sessionStorage.setItem('auth_user', JSON.stringify(response.user))
      navigate('/dashboard')
    } catch (err) {
      // `request()` throws with the raw status line, e.g.
      // `POST /api/auth/login failed (401): {"detail":"..."}`. Surface a clean
      // message for the two cases a user can act on, and keep the raw text for
      // anything else so a real backend fault stays diagnosable.
      const raw = err instanceof Error ? err.message : String(err)
      if (raw.includes('(401)')) {
        setError('Incorrect user ID or password.')
      } else if (err instanceof TypeError) {
        setError('Cannot reach the CortexAI API. Start it with `uvicorn src.api.main:app`.')
      } else {
        setError(raw)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface px-margin-mobile py-xl md:px-margin-desktop">
      <div className="relative w-full max-w-md overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-2 md:p-xl">
        {/* Brand */}
        <div className="mb-xl flex flex-col items-center text-center gap-sm">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary-container/20">
            <MaterialIcon name="psychology" filled className="text-primary text-3xl" />
          </div>
          <div>
            <h1 className="font-headline-lg text-headline-lg font-bold text-primary">CortexAI</h1>
            <p className="font-label-sm text-label-sm text-on-surface-variant">Multimodal Psychiatric Decision Support</p>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-md">
          {error && (
            <div
              role="alert"
              className="flex items-start gap-xs rounded-lg border border-error/40 bg-error/10 px-md py-sm text-error"
            >
              <MaterialIcon name="error" className="mt-[2px] text-[16px] text-error" />
              <span className="font-label-sm text-label-sm">{error}</span>
            </div>
          )}

          <label className="flex flex-col gap-xs">
            <span className="font-label-sm text-label-sm text-on-surface-variant">Clinician ID / Email</span>
            <div className="flex items-center gap-xs rounded border border-outline-variant bg-surface px-sm py-xs focus-within:border-primary focus-within:ring-1 focus-within:ring-primary">
              <MaterialIcon name="email" className="text-outline" />
              <input
                // `text`, not `email`: the field accepts a clinician ID as well
                // as an address (the Admin account signs in as `admin`), and
                // `type="email"` made the browser block submission for anything
                // without an "@" -- the request never reached the backend.
                type="text"
                autoComplete="username"
                required
                placeholder="admin  or  julian.vance@cortex.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full border-none bg-transparent outline-none font-body-sm text-body-sm text-on-surface"
              />
            </div>
          </label>

          <label className="flex flex-col gap-xs">
            <span className="font-label-sm text-label-sm text-on-surface-variant">Password</span>
            <div className="flex items-center gap-xs rounded border border-outline-variant bg-surface px-sm py-xs focus-within:border-primary focus-within:ring-1 focus-within:ring-primary">
              <MaterialIcon name="lock" className="text-outline" />
              <input
                type="password"
                autoComplete="current-password"
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full border-none bg-transparent outline-none font-body-sm text-body-sm text-on-surface"
              />
            </div>
          </label>

          <button
            type="submit"
            disabled={loading}
            className="mt-md flex w-full items-center justify-center gap-xs rounded-lg bg-primary py-sm px-md font-label-md text-label-md text-on-primary transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <>
                <MaterialIcon name="progress_activity" className="animate-spin" />
                Signing In...
              </>
            ) : (
              <>
                <MaterialIcon name="login" />
                Sign In
              </>
            )}
          </button>
        </form>

        {/* Info Footnote */}
        <div className="mt-xl border-t border-outline-variant pt-md text-center">
          <p className="font-label-sm text-[10px] text-label-sm leading-relaxed text-on-surface-variant">
            Access restricted to authorized clinical staff. All login attempts and clinical actions are logged in the SQLite audit system.
          </p>
        </div>
      </div>
    </div>
  )
}
