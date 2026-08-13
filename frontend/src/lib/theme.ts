/**
 * Theme control.
 *
 * Flips a single `dark` class on <html>; every colour token in index.css
 * responds to it, so no component needs `dark:` variants.
 *
 * Defaults to the OS preference and only pins a choice once the user makes
 * one — so someone on a dark desktop gets a dark console without asking, but
 * an explicit choice is never overridden later.
 */
import { useCallback, useEffect, useState } from 'react'

export type ThemeMode = 'light' | 'dark'

const STORAGE_KEY = 'cortexai.theme'

function systemPrefersDark(): boolean {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

export function resolveInitialTheme(): ThemeMode {
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored === 'light' || stored === 'dark') return stored
  return systemPrefersDark() ? 'dark' : 'light'
}

export function applyTheme(mode: ThemeMode): void {
  document.documentElement.classList.toggle('dark', mode === 'dark')
  // Keeps the browser's own UI (form controls, scrollbars) in step.
  document.documentElement.style.colorScheme = mode
}

export function useTheme() {
  const [theme, setTheme] = useState<ThemeMode>(() =>
    typeof window === 'undefined' ? 'light' : resolveInitialTheme(),
  )

  useEffect(() => {
    applyTheme(theme)
  }, [theme])

  // Follow the OS while the user has expressed no preference of their own.
  useEffect(() => {
    if (localStorage.getItem(STORAGE_KEY)) return
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (e: MediaQueryListEvent) => setTheme(e.matches ? 'dark' : 'light')
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  const toggle = useCallback(() => {
    setTheme((current) => {
      const next: ThemeMode = current === 'dark' ? 'light' : 'dark'
      localStorage.setItem(STORAGE_KEY, next)
      return next
    })
  }, [])

  return { theme, toggle }
}
