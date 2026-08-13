import { MaterialIcon } from './MaterialIcon'

/**
 * Marks a metric, chart or table as illustrative sample data rather than a
 * real computed result — required anywhere numbers are shown ahead of real
 * model output (see PROJECT_PLAN.md "Known constraint, logged up front").
 */
export function SampleDataBadge({ className = '' }: { className?: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border border-outline-variant bg-surface-container-low px-2 py-0.5 font-label-sm text-label-sm text-on-surface-variant ${className}`.trim()}
      title="Illustrative sample data — not a real patient result."
    >
      <MaterialIcon name="info" className="text-[13px]" />
      Sample data
    </span>
  )
}
