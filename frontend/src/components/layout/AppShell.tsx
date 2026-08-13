import { useEffect, useState, type ReactNode } from 'react'
import { TopNav } from './TopNav'
import { CommandPalette } from '../ui/CommandPalette'
import { MaterialIcon } from '../ui/MaterialIcon'

interface AppShellProps {
  children: ReactNode
  title?: string
  /** Short line under the title. */
  subtitle?: string
  /** Small caps label above the title, for section context. */
  eyebrow?: string
  /** Right-aligned actions in the page header. */
  actions?: ReactNode
  showSearch?: boolean
  searchPlaceholder?: string
  /** Clinical Report uses a document-style full-bleed canvas instead of the standard padded one. */
  bareMain?: boolean
}

/**
 * App frame.
 *
 * Navigation lives in the header rather than a fixed rail, so the content
 * canvas is the full window width — worth ~288px on every page, which is the
 * difference between a chart being readable and being cramped.
 */
export function AppShell({
  children,
  title,
  subtitle,
  eyebrow,
  actions,
  bareMain = false,
}: AppShellProps) {
  const [commandOpen, setCommandOpen] = useState(false)

  // ⌘K / Ctrl-K anywhere in the app.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCommandOpen((v) => !v)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="flex min-h-screen flex-col bg-background text-on-background">
      <TopNav onOpenCommand={() => setCommandOpen(true)} />
      <CommandPalette open={commandOpen} onClose={() => setCommandOpen(false)} />

      {bareMain ? (
        children
      ) : (
        <main className="mx-auto w-full max-w-canvas flex-1 p-margin-mobile md:px-margin-desktop md:py-xl">
          {title && <PageHeader title={title} subtitle={subtitle} eyebrow={eyebrow} actions={actions} />}
          <div className="animate-rise">{children}</div>
        </main>
      )}

      <footer className="mx-auto w-full max-w-canvas px-margin-mobile pb-lg md:px-margin-desktop">
        <div className="rule-fade mb-md" />
        <p className="flex items-center gap-xs font-label-sm text-label-sm text-on-surface-variant">
          <MaterialIcon name="verified_user" className="text-[14px]" />
          Decision support for qualified professionals — not a diagnosis. Always route people to human care.
        </p>
      </footer>
    </div>
  )
}

/**
 * Page header.
 *
 * The old pages opened with a plain 32px heading, which gave every screen the
 * same flat entry. This gives the title a gradient tail, an optional eyebrow
 * for context, and a slot for the page's primary action, so the top of each
 * page states where you are and what you can do.
 */
function PageHeader({
  title,
  subtitle,
  eyebrow,
  actions,
}: {
  title: string
  subtitle?: string
  eyebrow?: string
  actions?: ReactNode
}) {
  return (
    <header className="animate-rise mb-xl flex flex-col gap-md md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        {eyebrow && (
          <p className="mb-xs flex items-center gap-xs font-label-sm text-label-sm uppercase tracking-[0.18em] text-primary">
            <span className="h-1 w-1 rounded-full bg-primary" />
            {eyebrow}
          </p>
        )}
        <h1 className="headline-gradient font-headline-lg-mobile text-headline-lg-mobile md:font-headline-lg md:text-headline-lg">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-xs max-w-2xl font-body-md text-body-md leading-relaxed text-on-surface-variant">
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-sm">{actions}</div>}
    </header>
  )
}
