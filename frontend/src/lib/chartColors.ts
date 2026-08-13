/**
 * Hex values mirrored from tailwind.config.ts / DESIGN.md for use inside
 * Recharts SVG props, which need literal color values rather than
 * Tailwind utility classes.
 *
 * DESIGN.md "Data Visualization": charts use the Primary, Secondary and
 * Tertiary palette exclusively. Tertiary (violet) is reserved for
 * AI-generated/derived series (e.g. NLP-derived emotion frequency).
 * Error red is used only for severity/status encoding, matching how the
 * approved static designs already use it (severe-risk dots, alert KPIs).
 */
export const chartColors = {
  primary: '#006591',
  primaryContainer: '#0ea5e9',
  secondary: '#006b5f',
  secondaryContainer: '#6df5e1',
  tertiary: '#6d3bd7',
  tertiaryContainer: '#a986ff',
  error: '#ba1a1a',
  outlineVariant: '#bec8d2',
  onSurfaceVariant: '#3e4850',
} as const

export const severityColor: Record<'Healthy' | 'Mild' | 'Moderate' | 'Severe', string> = {
  Healthy: chartColors.secondary,
  Mild: chartColors.primaryContainer,
  Moderate: chartColors.tertiary,
  Severe: chartColors.error,
}
