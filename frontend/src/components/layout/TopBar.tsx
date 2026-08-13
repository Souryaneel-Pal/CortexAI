import { MaterialIcon } from '../ui/MaterialIcon'

interface TopBarProps {
  title?: string
  showSearch?: boolean
  searchPlaceholder?: string
}

/** Shared sticky/glassmorphism top app bar (DESIGN.md: `backdrop-filter: blur(12px)` on nav bars). */
export function TopBar({ title, showSearch = false, searchPlaceholder = 'Search patients, reports...' }: TopBarProps) {
  return (
    <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-outline-variant bg-surface/80 px-gutter shadow-level-1 backdrop-blur-nav">
      <div className="flex flex-1 items-center gap-md">
        <span className="font-headline-sm text-headline-sm font-bold text-on-surface md:hidden">CortexAI</span>
        {title && <h2 className="hidden font-headline-sm text-headline-sm font-bold text-on-surface md:block">{title}</h2>}
        {showSearch && (
          <div className="relative hidden max-w-md flex-1 sm:block">
            <MaterialIcon name="search" className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" />
            <input
              type="text"
              placeholder={searchPlaceholder}
              className="w-full rounded-lg border border-outline-variant bg-surface-container-lowest py-2 pl-10 pr-4 font-body-sm text-body-sm outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
            />
          </div>
        )}
      </div>
      <div className="flex items-center gap-md">
        <button
          type="button"
          className="rounded-full p-2 text-on-surface-variant transition-all hover:bg-surface-container focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          aria-label="Notifications"
        >
          <MaterialIcon name="notifications" />
        </button>
        <button
          type="button"
          className="rounded-full p-2 text-on-surface-variant transition-all hover:bg-surface-container focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
          aria-label="Switch account"
        >
          <MaterialIcon name="swap_horiz" />
        </button>
        <div className="ml-sm h-8 w-8 overflow-hidden rounded-full md:hidden">
          <img
            alt="Clinician Profile"
            className="h-full w-full object-cover"
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuBAGkiXXpZ0uCt5YZEklcKNvqfLz79WTXbiCQCQ6qv6A8FBDH6We2OYXODIlFhidPLVV7VqW4na0CVK5oWVOKGwOpSo7R57H5MH5RqjNVbGuIfLuVZOSjcyCJ31GzWGKbz2ct1CSPrzIJszCBJsH4WQFplP2ce90I0xzMpXK47Sx64AYd4rhpCtn4D5bXUEzwj5u5Z6RzOeoIF7nbSq1p1A4GFU-riWrGQ2Y6a3OvvVMMUvwzKQjlD3"
          />
        </div>
      </div>
    </header>
  )
}
