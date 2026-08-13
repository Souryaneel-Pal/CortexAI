import { Link, NavLink } from 'react-router-dom'
import { MaterialIcon } from '../ui/MaterialIcon'
import { NAV_ITEMS, SETTINGS_ITEM } from './navItems'
import { DECISION_SUPPORT_SIDEBAR } from '../../lib/responsibleAI'

/**
 * Shared side navigation — consolidated from the near-duplicate `<nav>`
 * markup independently generated on each of the six Stitch pages (brand
 * block, primary CTA, nav list, profile footer). See navItems.ts for why
 * "Results" / "Settings" are inert here.
 */
export function Sidebar() {
  const userJson = sessionStorage.getItem('auth_user')
  const user = userJson ? JSON.parse(userJson) : null
  const isAdmin = user && user.role === 'Admin'

  return (
    <nav
      aria-label="Primary"
      className="hidden md:flex h-screen w-72 flex-col fixed left-0 top-0 z-50 border-r border-outline-variant bg-surface p-lg gap-sm"
    >
      {/* Brand */}
      <Link to="/dashboard" className="mb-xl flex items-center gap-md px-sm">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary-container/20">
          <MaterialIcon name="psychology" filled className="text-primary" />
        </div>
        <div className="flex flex-col">
          <span className="font-headline-md text-headline-md font-bold text-primary">CortexAI</span>
          <span className="font-label-sm text-label-sm text-on-surface-variant">Senior Psychiatrist</span>
        </div>
      </Link>

      {/* Primary CTA */}
      <Link
        to="/assessment/new"
        className="mb-md flex w-full items-center justify-center gap-sm rounded-lg bg-primary py-sm px-md font-label-md text-label-md text-on-primary transition-colors hover:bg-on-primary-fixed-variant"
      >
        <MaterialIcon name="add_circle" filled />
        New Assessment
      </Link>

      {/* Nav links */}
      <div className="flex flex-1 flex-col gap-xs overflow-y-auto">
        {NAV_ITEMS.map((item) =>
          item.path ? (
            <NavLink
              key={item.id}
              to={item.path}
              end={item.path === '/dashboard'}
              className={({ isActive }) =>
                `flex items-center gap-md rounded-lg px-md py-sm font-label-md text-label-md transition-colors duration-200 ${
                  isActive
                    ? 'bg-surface-container-high font-bold text-primary'
                    : 'text-on-surface-variant hover:bg-surface-container-low hover:text-primary'
                }`
              }
            >
              <MaterialIcon name={item.icon} />
              <span>{item.label}</span>
            </NavLink>
          ) : (
            <span
              key={item.id}
              title="Not part of this conversion pass"
              className="flex cursor-not-allowed items-center gap-md rounded-lg px-md py-sm font-label-md text-label-md text-on-surface-variant/50"
            >
              <MaterialIcon name={item.icon} />
              <span>{item.label}</span>
            </span>
          ),
        )}
        
        {isAdmin && SETTINGS_ITEM.path && (
          <NavLink
            to={SETTINGS_ITEM.path}
            className={({ isActive }) =>
              `mt-auto flex items-center gap-md rounded-lg px-md py-sm font-label-md text-label-md transition-colors duration-200 ${
                isActive
                  ? 'bg-surface-container-high font-bold text-primary'
                  : 'text-on-surface-variant hover:bg-surface-container-low hover:text-primary'
              }`
            }
          >
            <MaterialIcon name={SETTINGS_ITEM.icon} />
            <span>{SETTINGS_ITEM.label}</span>
          </NavLink>
        )}
      </div>

      {/* Decision-support framing — present on every page, per PROJECT_PLAN.md */}
      <div className="mt-sm rounded-lg bg-surface-container-low p-md">
        <div className="mb-1 flex items-center gap-xs">
          <MaterialIcon name="verified_user" className="text-[16px] text-secondary" />
          <span className="font-label-sm text-label-sm font-bold text-on-surface">Decision support</span>
        </div>
        <p className="font-label-sm text-label-sm leading-relaxed text-on-surface-variant">
          {DECISION_SUPPORT_SIDEBAR}
        </p>
      </div>

      {/* User Profile */}
      <div className="mt-sm flex items-center justify-between border-t border-outline-variant pt-lg">
        <div className="flex items-center gap-md">
          <img
            alt={user?.name || "Dr. Julian Vance"}
            className="h-10 w-10 rounded-full object-cover"
            src="https://lh3.googleusercontent.com/aida-public/AB6AXuBQzNLcxqLRYAI0qAsSjVNGHWEhs85aLpxcnrjkMaHlENhJ-hEiJzXec1RjRwObFfegHcHJlJJmcXd9gxfX_ed2q_o69r7X_3E5mNf-XcFhB6UqRlFRKB3drOUhQeuXg-79wG_ZRDrjw64yWyrHv8rfQZHThuqc8uveb75B_XAa_Ogi2RdVUZYDFKqnhKvw-WTN9GJJHW7mRillDqMxfbLk_SoEkHE9cir2jDLUUEDKEgoInwgwa7l6"
          />
          <div>
            <p className="font-label-md text-label-md font-bold text-on-surface truncate max-w-[120px]">
              {user?.name || "Dr. Julian Vance"}
            </p>
            <p className="font-label-sm text-label-sm text-on-surface-variant">
              {user?.role || "Clinician"}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => {
            sessionStorage.clear()
            window.location.href = '/'
          }}
          title="Sign Out"
          className="flex h-8 w-8 items-center justify-center rounded-lg border border-outline-variant hover:bg-surface-container text-on-surface-variant hover:text-error transition-colors"
        >
          <MaterialIcon name="logout" className="text-[18px]" />
        </button>
      </div>
    </nav>
  )
}
