export interface NavItem {
  id: string
  label: string
  icon: string
  path?: string
}

/**
 * Shared sidebar nav, consolidated from the (near-identical, independently
 * generated) sidebars across all six Stitch exports. "Results" and
 * "Settings" appear in every source page's nav but have no corresponding
 * approved page in this conversion pass, so they render as inert
 * (non-navigating) items rather than being invented.
 */
export const NAV_ITEMS: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', icon: 'dashboard', path: '/dashboard' },
  { id: 'new-assessment', label: 'New Assessment', icon: 'add_circle', path: '/assessment/new' },
  { id: 'results', label: 'Results', icon: 'fact_check', path: '/results' },
  { id: 'reports', label: 'Reports', icon: 'assessment', path: '/reports' },
  { id: 'analytics', label: 'Analytics', icon: 'analytics', path: '/analytics' },
]

export const SETTINGS_ITEM: NavItem = { id: 'settings', label: 'Settings', icon: 'settings', path: '/settings' }
