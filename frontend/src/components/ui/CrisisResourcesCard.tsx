import { CRISIS_RESOURCES, CRISIS_SURFACING_NOTE } from '../../lib/responsibleAI'
import { MaterialIcon } from './MaterialIcon'

/**
 * Crisis/helpline resource surfacing — shown wherever severe-distress
 * indicators co-occur (PROJECT_PLAN.md P4/P6, non-negotiable).
 */
export function CrisisResourcesCard() {
  return (
    <div className="rounded-xl border border-error/30 bg-error-container/30 p-lg">
      <div className="mb-sm flex items-center gap-sm">
        <MaterialIcon name="emergency" className="text-error" />
        <h3 className="font-headline-sm text-headline-sm text-on-surface">Support &amp; Crisis Resources</h3>
      </div>
      <p className="font-body-sm text-body-sm text-on-surface-variant mb-md">{CRISIS_SURFACING_NOTE}</p>
      <ul className="flex flex-col gap-xs">
        {CRISIS_RESOURCES.map((r) => (
          <li key={r.name} className="font-body-sm text-body-sm text-on-surface">
            <span className="font-medium">{r.name}</span>
            <span className="text-on-surface-variant"> — {r.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
