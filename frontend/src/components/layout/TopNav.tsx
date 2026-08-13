/**
 * Primary navigation, as a top bar.
 *
 * Replaces the 288px fixed sidebar: on a dashboard that is mostly wide charts
 * and tables, a vertical rail costs a fifth of the canvas to show five links.
 * Moving them into the header returns that width to the data.
 *
 * The bar also carries live system state (checkpoints + local LLM), which used
 * to be buried in a card on one page — it belongs where it is visible before a
 * clinician starts an assessment, not after.
 */
import { useEffect, useRef, useState } from 'react'
import { Link, NavLink, useNavigate } from 'react-router-dom'
import { MaterialIcon } from '../ui/MaterialIcon'
import { NAV_ITEMS, SETTINGS_ITEM } from './navItems'
import { useAssessment } from '../../lib/assessmentContext'
import { useTheme } from '../../lib/theme'
import type { UserInfo } from '../../lib/api'

function readUser(): UserInfo | null {
  try {
    const raw = sessionStorage.getItem('auth_user')
    return raw ? (JSON.parse(raw) as UserInfo) : null
  } catch {
    return null
  }
}

// Written out in full rather than interpolated: Tailwind's JIT scans source
// text for literal class names, so `bg-${tone}` compiles to nothing.
const STATUS_TONE = {
  ok: {
    wrap: 'border-secondary/30 bg-secondary/10',
    dot: 'bg-secondary text-secondary',
    text: 'text-secondary',
  },
  degraded: {
    wrap: 'border-tertiary/30 bg-tertiary/10',
    dot: 'bg-tertiary text-tertiary',
    text: 'text-tertiary',
  },
  offline: {
    wrap: 'border-error/30 bg-error/10',
    dot: 'bg-error text-error',
    text: 'text-error',
  },
} as const

/** Compact traffic-light for the backend + local model stack. */
function SystemStatus() {
  const { backendReachable, health } = useAssessment()
  const reasoning = health?.reasoning

  const offline = backendReachable === false
  const degraded = !offline && reasoning != null && (!reasoning.ollama_reachable || !reasoning.llm_available)
  const tone = STATUS_TONE[offline ? 'offline' : degraded ? 'degraded' : 'ok']
  const label = offline ? 'API offline' : degraded ? 'LLM offline' : 'All systems live'

  const detail = offline
    ? 'Cannot reach the CortexAI API.'
    : (reasoning?.detail ??
      `Checkpoints ${health?.is_demo_untrained_model ? 'not loaded' : 'loaded'} · ${reasoning?.llm_model ?? 'llama3.1'} · retrieval: ${reasoning?.retrieval_backend ?? 'unknown'}`)

  return (
    <div title={detail} className={`hidden items-center gap-xs rounded-full border px-sm py-1 lg:flex ${tone.wrap}`}>
      <span className={`animate-pulse-dot h-1.5 w-1.5 rounded-full ${tone.dot}`} />
      <span className={`font-label-sm text-label-sm ${tone.text}`}>{label}</span>
    </div>
  )
}

export function TopNav({ onOpenCommand }: { onOpenCommand: () => void }) {
  const navigate = useNavigate()
  const { theme, toggle } = useTheme()
  const [menuOpen, setMenuOpen] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const user = readUser()
  const isAdmin = user?.role === 'Admin'

  // Settings is Admin-only, matching the backend's 403 on PUT /api/settings.
  const items = isAdmin ? [...NAV_ITEMS, SETTINGS_ITEM] : NAV_ITEMS

  useEffect(() => {
    if (!menuOpen) return
    const onClick = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [menuOpen])

  function signOut() {
    sessionStorage.removeItem('auth_token')
    sessionStorage.removeItem('auth_user')
    navigate('/')
  }

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    [
      'relative rounded-full px-md py-1.5 font-label-md text-label-md transition-colors',
      isActive
        ? 'bg-primary/10 text-primary'
        : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface',
    ].join(' ')

  return (
    <header className="glass sticky top-0 z-40 border-b border-outline-variant/60">
      <div className="mx-auto flex h-16 w-full max-w-canvas items-center gap-md px-margin-mobile md:px-margin-desktop">
        {/* Brand */}
        <Link to="/dashboard" className="flex shrink-0 items-center gap-sm">
          <span className="ai-gradient-bg flex h-9 w-9 items-center justify-center rounded-lg text-white shadow-ai-glow">
            <MaterialIcon name="neurology" filled className="text-[20px]" />
          </span>
          <span className="hidden flex-col leading-none sm:flex">
            <span className="font-headline-sm text-headline-sm font-bold tracking-tight text-on-surface">CortexAI</span>
            <span className="font-label-sm text-[10px] uppercase tracking-[0.18em] text-on-surface-variant">
              Clinical Console
            </span>
          </span>
        </Link>

        {/* Primary nav */}
        <nav className="ml-md hidden items-center gap-1 md:flex">
          {items.map((item) => (
            <NavLink key={item.id} to={item.path ?? '#'} className={linkClass}>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex-1" />

        <SystemStatus />

        {/* Command palette trigger */}
        <button
          type="button"
          onClick={onOpenCommand}
          className="hidden items-center gap-sm rounded-full border border-outline-variant bg-surface-container-lowest/70 px-md py-1.5 text-on-surface-variant transition-colors hover:border-primary/40 hover:text-on-surface sm:flex"
          aria-label="Open command palette"
        >
          <MaterialIcon name="search" className="text-[18px]" />
          <span className="font-label-sm text-label-sm">Search</span>
          <kbd className="rounded border border-outline-variant bg-surface-container px-1.5 font-label-sm text-[10px] text-on-surface-variant">
            ⌘K
          </kbd>
        </button>

        <button
          type="button"
          onClick={toggle}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          className="rounded-full p-2 text-on-surface-variant transition-colors hover:bg-surface-container hover:text-on-surface"
        >
          <MaterialIcon name={theme === 'dark' ? 'light_mode' : 'dark_mode'} className="text-[20px]" />
        </button>

        {/* Account */}
        <div className="relative" ref={menuRef}>
          <button
            type="button"
            onClick={() => setMenuOpen((v) => !v)}
            className="flex items-center gap-sm rounded-full border border-outline-variant bg-surface-container-lowest/70 py-1 pl-1 pr-sm transition-colors hover:border-primary/40"
          >
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/15 font-label-sm text-label-sm text-primary">
              {(user?.name ?? 'U').slice(0, 1).toUpperCase()}
            </span>
            <span className="hidden text-left leading-tight lg:block">
              <span className="block font-label-sm text-label-sm text-on-surface">{user?.name ?? 'Signed in'}</span>
              <span className="block font-label-sm text-[10px] uppercase tracking-wider text-on-surface-variant">
                {user?.role ?? '—'}
              </span>
            </span>
            <MaterialIcon name="expand_more" className="text-[18px] text-on-surface-variant" />
          </button>

          {menuOpen && (
            <div className="panel animate-fade absolute right-0 top-full z-50 mt-sm w-56 rounded-lg p-xs">
              <div className="px-sm py-sm">
                <p className="font-label-md text-label-md text-on-surface">{user?.name ?? 'Signed in'}</p>
                <p className="truncate font-label-sm text-label-sm text-on-surface-variant">{user?.email ?? ''}</p>
              </div>
              <div className="rule-fade my-xs" />
              {isAdmin && (
                <Link
                  to="/settings"
                  onClick={() => setMenuOpen(false)}
                  className="flex items-center gap-sm rounded px-sm py-sm font-label-md text-label-md text-on-surface-variant transition-colors hover:bg-surface-container hover:text-on-surface"
                >
                  <MaterialIcon name="tune" className="text-[18px]" /> Model Settings
                </Link>
              )}
              <button
                type="button"
                onClick={signOut}
                className="flex w-full items-center gap-sm rounded px-sm py-sm font-label-md text-label-md text-error transition-colors hover:bg-error/10"
              >
                <MaterialIcon name="logout" className="text-[18px]" /> Sign out
              </button>
            </div>
          )}
        </div>

        {/* Mobile nav toggle */}
        <button
          type="button"
          onClick={() => setMobileOpen((v) => !v)}
          className="rounded-full p-2 text-on-surface-variant transition-colors hover:bg-surface-container md:hidden"
          aria-label="Toggle navigation"
        >
          <MaterialIcon name={mobileOpen ? 'close' : 'menu'} />
        </button>
      </div>

      {mobileOpen && (
        <nav className="animate-fade border-t border-outline-variant/60 px-margin-mobile py-sm md:hidden">
          {items.map((item) => (
            <NavLink
              key={item.id}
              to={item.path ?? '#'}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-sm rounded-lg px-sm py-sm font-label-md text-label-md transition-colors ${
                  isActive ? 'bg-primary/10 text-primary' : 'text-on-surface-variant hover:bg-surface-container'
                }`
              }
            >
              <MaterialIcon name={item.icon} className="text-[20px]" />
              {item.label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  )
}
