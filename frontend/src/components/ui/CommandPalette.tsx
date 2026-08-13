/**
 * ⌘K command palette.
 *
 * With navigation moved into the header, the fastest route between pages is
 * the keyboard rather than the mouse — and a clinician mid-assessment should
 * not have to hunt for "Analytics". Also exposes the actions that have no
 * home in the nav (theme, sign out, jump to the current result).
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { MaterialIcon } from './MaterialIcon'
import { NAV_ITEMS, SETTINGS_ITEM } from '../layout/navItems'
import { useTheme } from '../../lib/theme'
import { useAssessment } from '../../lib/assessmentContext'

interface Command {
  id: string
  label: string
  hint?: string
  icon: string
  run: () => void
}

export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate()
  const { theme, toggle } = useTheme()
  const { hasLiveResult, clear } = useAssessment()
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const isAdmin = (() => {
    try {
      return JSON.parse(sessionStorage.getItem('auth_user') ?? '{}')?.role === 'Admin'
    } catch {
      return false
    }
  })()

  const commands = useMemo<Command[]>(() => {
    const go = (path: string) => () => {
      onClose()
      navigate(path)
    }
    const items: Command[] = NAV_ITEMS.map((n) => ({
      id: n.id,
      label: n.label,
      hint: 'Navigate',
      icon: n.icon,
      run: go(n.path ?? '/dashboard'),
    }))
    if (isAdmin) {
      items.push({
        id: SETTINGS_ITEM.id,
        label: SETTINGS_ITEM.label,
        hint: 'Admin',
        icon: SETTINGS_ITEM.icon,
        run: go('/settings'),
      })
    }
    items.push({
      id: 'theme',
      label: theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
      hint: 'Appearance',
      icon: theme === 'dark' ? 'light_mode' : 'dark_mode',
      run: () => {
        toggle()
        onClose()
      },
    })
    if (hasLiveResult) {
      items.push({
        id: 'clear',
        label: 'Clear current assessment',
        hint: 'Session',
        icon: 'delete_sweep',
        run: () => {
          clear()
          onClose()
        },
      })
    }
    items.push({
      id: 'signout',
      label: 'Sign out',
      hint: 'Session',
      icon: 'logout',
      run: () => {
        sessionStorage.removeItem('auth_token')
        sessionStorage.removeItem('auth_user')
        onClose()
        navigate('/')
      },
    })
    return items
  }, [navigate, onClose, theme, toggle, hasLiveResult, clear, isAdmin])

  const results = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return commands
    return commands.filter((c) => c.label.toLowerCase().includes(q) || c.hint?.toLowerCase().includes(q))
  }, [commands, query])

  useEffect(() => {
    if (open) {
      setQuery('')
      setCursor(0)
      // Focus after paint, or the input isn't mounted yet.
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  useEffect(() => {
    setCursor(0)
  }, [query])

  if (!open) return null

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape') return onClose()
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setCursor((c) => (c + 1) % Math.max(results.length, 1))
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      setCursor((c) => (c - 1 + results.length) % Math.max(results.length, 1))
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      results[cursor]?.run()
    }
  }

  return (
    <div
      className="animate-fade fixed inset-0 z-[60] flex items-start justify-center bg-on-surface/40 p-margin-mobile pt-[12vh] backdrop-blur-sm"
      onMouseDown={onClose}
      role="presentation"
    >
      <div
        className="panel animate-rise w-full max-w-xl overflow-hidden rounded-xl"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        <div className="flex items-center gap-sm border-b border-outline-variant/60 px-md">
          <MaterialIcon name="search" className="text-outline" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Jump to a page or run a command…"
            className="w-full bg-transparent py-md font-body-md text-body-md text-on-surface outline-none placeholder:text-on-surface-variant"
          />
          <kbd className="rounded border border-outline-variant bg-surface-container px-1.5 font-label-sm text-[10px] text-on-surface-variant">
            ESC
          </kbd>
        </div>

        <ul className="max-h-[46vh] overflow-y-auto p-xs">
          {results.length === 0 && (
            <li className="px-md py-lg text-center font-body-sm text-body-sm text-on-surface-variant">
              Nothing matches “{query}”.
            </li>
          )}
          {results.map((command, i) => (
            <li key={command.id}>
              <button
                type="button"
                onMouseEnter={() => setCursor(i)}
                onClick={command.run}
                className={`flex w-full items-center gap-sm rounded-lg px-md py-sm text-left transition-colors ${
                  i === cursor ? 'bg-primary/10 text-primary' : 'text-on-surface-variant hover:bg-surface-container'
                }`}
              >
                <MaterialIcon name={command.icon} className="text-[20px]" />
                <span className="flex-1 font-label-md text-label-md">{command.label}</span>
                {command.hint && (
                  <span className="font-label-sm text-[10px] uppercase tracking-wider text-on-surface-variant">
                    {command.hint}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
