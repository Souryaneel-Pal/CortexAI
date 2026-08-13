import type { Severity } from '../../types'

const DOT_CLASS: Record<Severity, string> = {
  Healthy: 'bg-secondary',
  Mild: 'bg-primary-container',
  Moderate: 'bg-tertiary',
  Severe: 'bg-error',
}

export function SeverityDot({ severity }: { severity: Severity }) {
  return (
    <span className="flex items-center gap-2">
      <span className={`h-2 w-2 rounded-full ${DOT_CLASS[severity]}`} />
      <span>{severity}</span>
    </span>
  )
}

const STATUS_CLASS: Record<string, string> = {
  'AI Reviewing': 'bg-surface-variant text-on-surface',
  Completed: 'bg-secondary/10 text-secondary border border-secondary/20',
}

export function StatusChip({ status }: { status: string }) {
  return (
    <span
      className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${STATUS_CLASS[status] ?? 'bg-surface-variant text-on-surface'}`}
    >
      {status}
    </span>
  )
}
