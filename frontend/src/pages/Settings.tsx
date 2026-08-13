import { useEffect, useState } from 'react'
import { Navigate } from 'react-router-dom'
import { AppShell } from '../components/layout/AppShell'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import { getSettings, putSettings, type SettingsState } from '../lib/api'

export function Settings() {
  const userJson = sessionStorage.getItem('auth_user')
  const user = userJson ? JSON.parse(userJson) : null

  // Restrict to Admin
  if (!user || user.role !== 'Admin') {
    return <Navigate to="/dashboard" replace />
  }

  const [settings, setSettings] = useState<SettingsState>({
    uncertainty_threshold: 0.60,
    mdi_threshold: 0.50,
    ignore_face: false,
    ignore_speech: false,
    ignore_tabular: false,
  })

  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getSettings()
      .then((data) => {
        setSettings(data)
        setLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Failed to fetch settings')
        setLoading(false)
      })
  }, [])

  async function handleSave() {
    setSaving(true)
    setSuccess(false)
    setError(null)
    try {
      await putSettings(settings)
      setSuccess(true)
      setTimeout(() => setSuccess(false), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings')
    } finally {
      setSaving(false)
    }
  }

  function updateField<K extends keyof SettingsState>(key: K, value: SettingsState[K]) {
    setSettings((s) => ({ ...s, [key]: value }))
  }

  if (loading) {
    return (
      <AppShell>
        <div className="flex h-[400px] items-center justify-center">
          <div className="flex flex-col items-center gap-md">
            <MaterialIcon name="progress_activity" className="animate-spin text-4xl text-primary" />
            <span className="font-label-md text-label-md text-on-surface-variant">Loading settings...</span>
          </div>
        </div>
      </AppShell>
    )
  }

  return (
    <AppShell>
      <div className="flex flex-col gap-xs mb-xl">
        <h1 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface">
          Clinical Settings Control Panel
        </h1>
        <p className="max-w-2xl font-body-md text-body-md text-on-surface-variant">
          Calibrate distress indexing sensitivity, uncertainty deferrals, and sensor stream masking parameters.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-lg lg:grid-cols-3">
        {/* Sliders Card */}
        <section className="flex flex-col gap-lg rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1 lg:col-span-2">
          <div>
            <h2 className="mb-xs font-headline-sm text-headline-sm text-on-surface">Clinical Decision Guardrails</h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Adjust criteria for human-in-the-loop triggers and clinical index flags.
            </p>
          </div>

          <div className="flex flex-col gap-xl">
            {/* Uncertainty Gate */}
            <div className="flex flex-col gap-sm">
              <div className="flex items-center justify-between">
                <span className="font-label-md text-label-md font-bold text-on-surface">
                  Uncertainty Deferral Threshold
                </span>
                <span className="font-label-sm text-label-sm rounded bg-primary-container/20 px-sm py-xs text-primary font-bold">
                  {Math.round(settings.uncertainty_threshold * 100)}%
                </span>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                If the model's confidence is below this value, the prediction will be flagged for human review (deferred) instead of being accepted automatically.
              </p>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.uncertainty_threshold}
                onChange={(e) => updateField('uncertainty_threshold', parseFloat(e.target.value))}
                className="h-2 w-full cursor-pointer rounded-lg bg-surface-container accent-primary"
              />
              <div className="flex justify-between font-label-sm text-[10px] text-outline">
                <span>0% (Always Accept)</span>
                <span>50%</span>
                <span>100% (Always Defer)</span>
              </div>
            </div>

            {/* MDI Sensitivity */}
            <div className="flex flex-col gap-sm">
              <div className="flex items-center justify-between">
                <span className="font-label-md text-label-md font-bold text-on-surface">
                  Masked-Distress Index (MDI) Sensitivity
                </span>
                <span className="font-label-sm text-label-sm rounded bg-primary-container/20 px-sm py-xs text-primary font-bold">
                  {Math.round(settings.mdi_threshold * 100)}%
                </span>
              </div>
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Minimum contradiction score between facial appearance (calm) and physical/vocal metrics (high stress) to trigger a contradiction alert.
              </p>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={settings.mdi_threshold}
                onChange={(e) => updateField('mdi_threshold', parseFloat(e.target.value))}
                className="h-2 w-full cursor-pointer rounded-lg bg-surface-container accent-primary"
              />
              <div className="flex justify-between font-label-sm text-[10px] text-outline">
                <span>0% (Flag All)</span>
                <span>50%</span>
                <span>100% (Never Flag)</span>
              </div>
            </div>
          </div>
        </section>

        {/* Toggles Card */}
        <section className="flex flex-col gap-lg rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1 lg:col-span-1">
          <div>
            <h2 className="mb-xs font-headline-sm text-headline-sm text-on-surface">Sensor Modality Masking</h2>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Selectively mask data streams to evaluate model fallback performance.
            </p>
          </div>

          <div className="flex flex-col gap-md">
            {(
              [
                ['ignore_face', 'Ignore Facial Video', 'videocam', 'Disable camera analysis feed'],
                ['ignore_speech', 'Ignore Speech Audio', 'mic', 'Deactivate voice feature extraction'],
                ['ignore_tabular', 'Ignore Tabular Metrics', 'favorite', 'Bypass physical and self-report features'],
              ] as const
            ).map(([field, label, icon, desc]) => (
              <label
                key={field}
                className="flex cursor-pointer items-start gap-md rounded-lg border border-outline-variant/40 bg-surface-container-low p-md transition-colors hover:bg-surface-container"
              >
                <input
                  type="checkbox"
                  checked={settings[field]}
                  onChange={(e) => updateField(field, e.target.checked)}
                  className="mt-xs h-4 w-4 rounded border-outline bg-surface text-primary focus:ring-primary"
                />
                <div className="flex flex-1 flex-col gap-2">
                  <div className="flex items-center gap-xs font-label-md text-label-md font-bold text-on-surface">
                    <MaterialIcon name={icon} className="text-primary text-[18px]" />
                    {label}
                  </div>
                  <span className="font-body-sm text-body-sm text-on-surface-variant">{desc}</span>
                </div>
              </label>
            ))}
          </div>
        </section>
      </div>

      {/* Action Footer */}
      <div className="mt-xl flex flex-col items-start gap-lg border-t border-outline-variant pt-lg md:flex-row md:items-center md:justify-between">
        <div className="mr-auto flex max-w-md flex-1 flex-col gap-xs">
          {success && (
            <div
              role="alert"
              className="flex items-center gap-xs rounded-lg border border-secondary-container bg-secondary-container/10 px-md py-sm text-on-secondary-container"
            >
              <MaterialIcon name="check_circle" className="text-secondary" />
              <span className="font-label-sm text-label-sm font-bold">Configuration saved successfully.</span>
            </div>
          )}
          {error && (
            <div
              role="alert"
              className="flex items-center gap-xs rounded-lg border border-error/40 bg-error/10 px-md py-sm text-error"
            >
              <MaterialIcon name="error" className="text-error" />
              <span className="font-label-sm text-label-sm">{error}</span>
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-sm rounded-lg bg-primary px-3xl py-md font-label-md text-label-md text-on-primary transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {saving ? (
            <>
              <MaterialIcon name="progress_activity" className="animate-spin" />
              Saving Settings...
            </>
          ) : (
            <>
              <MaterialIcon name="save" />
              Save Configuration
            </>
          )}
        </button>
      </div>
    </AppShell>
  )
}
