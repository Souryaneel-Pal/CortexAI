import type { ReactNode } from 'react'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

interface AppShellProps {
  children: ReactNode
  title?: string
  showSearch?: boolean
  searchPlaceholder?: string
  /** Most pages constrain to the 1440px canvas with standard margins; Clinical Report uses a document-style full-bleed canvas instead. */
  bareMain?: boolean
}

export function AppShell({ children, title, showSearch, searchPlaceholder, bareMain = false }: AppShellProps) {
  return (
    <div className="flex min-h-screen bg-background text-on-background">
      <Sidebar />
      <div className="flex min-h-screen flex-1 flex-col md:ml-72">
        <TopBar title={title} showSearch={showSearch} searchPlaceholder={searchPlaceholder} />
        {bareMain ? (
          children
        ) : (
          <main className="mx-auto w-full max-w-canvas flex-1 p-margin-mobile md:p-margin-desktop">{children}</main>
        )}
      </div>
    </div>
  )
}
