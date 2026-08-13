import type { ReactNode } from 'react'
import { MaterialIcon } from './MaterialIcon'

/**
 * DESIGN.md "Specialized Components / Insight Banner": light Violet tint
 * background with a subtle 1px border, reserved for AI-generated content.
 */
export function InsightBanner({
  icon = 'psychology',
  title,
  children,
  className = '',
}: {
  icon?: string
  title: string
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-xl border border-tertiary/20 bg-surface p-lg shadow-level-1 ai-glow ${className}`.trim()}
    >
      <div className="absolute left-0 top-0 h-full w-1 bg-tertiary" />
      <div className="flex items-start gap-md">
        <MaterialIcon name={icon} filled className="text-3xl text-tertiary" />
        <div>
          <h3 className="font-headline-sm text-headline-sm text-on-surface mb-sm">{title}</h3>
          <div className="font-body-md text-body-md leading-relaxed text-on-surface-variant">{children}</div>
        </div>
      </div>
    </div>
  )
}
