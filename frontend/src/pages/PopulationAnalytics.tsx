import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AppShell } from '../components/layout/AppShell'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import { SampleDataBadge } from '../components/ui/SampleDataBadge'
import { getAnalytics, type AnalyticsData } from '../lib/api'
import { chartColors } from '../lib/chartColors'
import { useAssessment } from '../lib/assessmentContext'

const CLASS_LABELS = ['Healthy', 'Mild', 'Moderate', 'Severe']

const HEATMAP_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const HEATMAP_HOURS = ['2A', '4A', '6A', '8A', '10A', '12P', '2P', '4P', '6P', '8P', '10P', '12A']

function heatCellColor(intensity: number) {
  // Low intensity reads as primary (calm); elevated stress events shift toward error red.
  if (intensity >= 0.55) return `rgba(186, 26, 26, ${0.3 + intensity * 0.6})`
  return `rgba(0, 101, 145, ${0.08 + intensity * 0.7})`
}

export function PopulationAnalytics() {
  const [data, setData] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [errorState, setErrorState] = useState<string | null>(null)

  useEffect(() => {
    getAnalytics()
      .then((res) => {
        setData(res)
        setLoading(false)
      })
      .catch((err) => {
        setErrorState(err instanceof Error ? err.message : 'Failed to load analytics data')
        setLoading(false)
      })
  }, [])

  const { prediction, explanation, running, phaseLabel, error, backendReachable } = useAssessment()

  if (loading) {
    return (
      <AppShell>
        <div className="flex h-[400px] items-center justify-center">
          <div className="flex flex-col items-center gap-md">
            <MaterialIcon name="progress_activity" className="animate-spin text-4xl text-primary" />
            <span className="font-label-md text-label-md text-on-surface-variant">Loading analytics data...</span>
          </div>
        </div>
      </AppShell>
    )
  }

  if (errorState || !data) {
    return (
      <AppShell>
        <div className="flex h-[400px] flex-col items-center justify-center gap-md">
          <MaterialIcon name="error" className="text-4xl text-error" />
          <span className="font-label-md text-label-md text-error">{errorState || 'No analytics data available'}</span>
        </div>
      </AppShell>
    )
  }

  // The cohort panels on this page stay sample data. A single assessment is
  // not a population, and deriving a heatmap, a risk-segment breakdown or a
  // correlation matrix from one session would be fabrication -- so the live
  // session gets its own clearly-scoped panel instead of being averaged into
  // charts that imply a cohort behind them.
  const classProbs = prediction?.class_probs ?? []
  const probBars = classProbs.map((p, i) => ({ label: CLASS_LABELS[i] ?? `Class ${i}`, pct: Math.round(p * 100) }))
  const modalityBars = explanation
    ? [
        { label: 'Signals', pct: Math.round(explanation.modality_weights.tabular * 100) },
        { label: 'Face', pct: Math.round(explanation.modality_weights.face * 100) },
        { label: 'Speech', pct: Math.round(explanation.modality_weights.speech * 100) },
      ]
    : []

  return (
    <AppShell
      eyebrow="Cohort"
      title="Executive Analytics"
      subtitle="Aggregate patterns across every stored assessment. Derived from real history — never from placeholder data."
    >
      {(running || error || backendReachable === false) && (
        <div
          role="status"
          className={`mb-lg flex items-center gap-sm rounded-xl border px-lg py-md shadow-level-1 ${
            error || backendReachable === false
              ? 'border-error/40 bg-error/10'
              : 'border-outline-variant bg-surface-container-lowest'
          }`}
        >
          <MaterialIcon
            name={error || backendReachable === false ? 'error' : 'progress_activity'}
            className={`text-[20px] ${error || backendReachable === false ? 'text-error' : 'animate-spin text-primary'}`}
          />
          <span
            className={`font-label-md text-label-md ${
              error || backendReachable === false ? 'text-error' : 'text-on-surface-variant'
            }`}
          >
            {error ?? (backendReachable === false ? 'Cannot reach the CortexAI API.' : phaseLabel)}
          </span>
        </div>
      )}

      {prediction && (
        <section className="mb-lg rounded-xl panel p-lg">
          <div className="mb-md flex flex-wrap items-center justify-between gap-sm">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Current Session (live model output)</h3>
            <span className="font-label-sm text-label-sm text-on-surface-variant">
              Session {prediction.session_id.slice(0, 8)}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-lg md:grid-cols-2">
            <div>
              <h4 className="mb-sm font-label-md text-label-md text-on-surface-variant">
                Class probability (MC-dropout mean)
              </h4>
              <div className="flex flex-col gap-sm">
                {probBars.map((b) => (
                  <div key={b.label}>
                    <div className="flex items-center justify-between">
                      <span className="font-body-sm text-body-sm text-on-surface-variant">{b.label}</span>
                      <span className="font-label-md text-label-md text-on-surface">{b.pct}%</span>
                    </div>
                    <div className="mt-xs h-2 w-full overflow-hidden rounded-full bg-surface-container">
                      <div className="h-full bg-primary" style={{ width: `${b.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="mb-sm font-label-md text-label-md text-on-surface-variant">
                Modality contribution (fusion gate)
              </h4>
              <div className="flex flex-col gap-sm">
                {modalityBars.map((b) => (
                  <div key={b.label}>
                    <div className="flex items-center justify-between">
                      <span className="font-body-sm text-body-sm text-on-surface-variant">{b.label}</span>
                      <span className="font-label-md text-label-md text-on-surface">{b.pct}%</span>
                    </div>
                    <div className="mt-xs h-2 w-full overflow-hidden rounded-full bg-surface-container">
                      <div className="h-full bg-secondary" style={{ width: `${b.pct}%` }} />
                    </div>
                  </div>
                ))}
                {!explanation && (
                  <span className="font-label-sm text-label-sm text-on-surface-variant">
                    Explanations unavailable for this session.
                  </span>
                )}
              </div>
            </div>
          </div>
          <p className="mt-md border-t border-outline-variant pt-sm font-label-sm text-label-sm italic text-on-surface-variant">
            The cohort panels below remain sample data — a single assessment is not a population.
          </p>
        </section>
      )}
      {/* Filters & Controls Bar */}
      <div className="mb-lg flex flex-col items-start justify-between gap-md rounded-xl panel p-md sm:flex-row sm:items-center">
        <div className="flex flex-wrap items-center gap-md">
          <div className="flex items-center gap-2">
            <label className="font-label-sm text-label-sm text-on-surface-variant">Time Range:</label>
            <select className="rounded-md border border-outline-variant bg-surface-container-low px-3 py-1.5 font-label-sm text-label-sm text-on-surface outline-none focus:border-primary focus:ring-1 focus:ring-primary">
              <option>Last 30 Days</option>
              <option>Last Quarter</option>
              <option>YTD</option>
              <option>All Time</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="font-label-sm text-label-sm text-on-surface-variant">Demographic:</label>
            <select className="rounded-md border border-outline-variant bg-surface-container-low px-3 py-1.5 font-label-sm text-label-sm text-on-surface outline-none focus:border-primary focus:ring-1 focus:ring-primary">
              <option>All Patients</option>
              <option>Adults (18-64)</option>
              <option>Seniors (65+)</option>
              <option>Adolescents (12-17)</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="font-label-sm text-label-sm text-on-surface-variant">Condition:</label>
            <select className="rounded-md border border-outline-variant bg-surface-container-low px-3 py-1.5 font-label-sm text-label-sm text-on-surface outline-none focus:border-primary focus:ring-1 focus:ring-primary">
              <option>All Monitored</option>
              <option>Anxiety Disorders</option>
              <option>Depressive Disorders</option>
              <option>PTSD</option>
            </select>
          </div>
        </div>
        <div className="flex items-center gap-sm">
          <SampleDataBadge />
          <button
            type="button"
            className="flex items-center gap-2 rounded-md border border-secondary-container/50 bg-secondary-container/20 px-md py-1.5 font-label-sm text-label-sm text-on-secondary-container transition-colors hover:bg-secondary-container/30"
          >
            <MaterialIcon name="download" className="text-sm" />
            Export Report
          </button>
        </div>
      </div>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-12 gap-lg">
        {/* KPI cards */}
        {data.kpis.map((kpi) => (
          <div
            key={kpi.label}
            className={`relative col-span-12 flex flex-col gap-sm overflow-hidden rounded-xl panel p-lg sm:col-span-6 lg:col-span-3 ${
              kpi.isAi ? 'ai-glow' : ''
            }`}
          >
            <div className="flex items-start justify-between">
              <span className={`font-label-md text-label-md ${kpi.isAi ? 'text-tertiary' : 'text-on-surface-variant'}`}>
                {kpi.label}
              </span>
            </div>
            <div className="flex items-baseline gap-sm">
              <h3 className="font-display-lg text-display-lg text-on-surface">{kpi.value}</h3>
              {kpi.delta && (
                <span
                  className={`flex items-center gap-1 font-label-sm text-label-sm ${
                    kpi.deltaDirection === 'up' && kpi.label === 'Critical Alerts' ? 'text-error' : 'text-secondary'
                  }`}
                >
                  <MaterialIcon name="trending_up" className="text-[14px]" /> {kpi.delta}
                </span>
              )}
              {kpi.isAi && (
                <span className="flex items-center gap-1 font-label-sm text-label-sm text-secondary">
                  <MaterialIcon name="check_circle" className="text-[14px]" /> Optimal
                </span>
              )}
            </div>
            {kpi.sub && <p className="mt-auto font-body-sm text-body-sm text-on-surface-variant">{kpi.sub}</p>}
            {kpi.isAi && <div className="absolute bottom-0 left-0 h-1 w-full bg-gradient-to-r from-tertiary to-secondary" />}
          </div>
        ))}

        {/* Stress Severity Heatmap */}
        <div className="col-span-12 flex min-h-[360px] flex-col rounded-xl panel p-lg lg:col-span-7">
          <div className="mb-md flex items-center justify-between">
            <div>
              <h3 className="font-headline-sm text-headline-sm text-on-surface">Stress Severity Heatmap</h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant">
                Temporal distribution of elevated stress events across population.
              </p>
            </div>
          </div>
          <div className="flex flex-1 items-center justify-center overflow-x-auto rounded-lg border border-outline-variant/50 bg-surface-container-low p-md">
            <div className="flex min-w-[480px] gap-2">
              <div className="flex w-8 flex-col justify-between border-r border-outline-variant/30 py-4 pr-2 text-right font-label-sm text-[10px] text-outline">
                {HEATMAP_DAYS.map((d) => (
                  <span key={d}>{d}</span>
                ))}
              </div>
              <div className="flex-1">
                <div className="grid grid-cols-12 gap-1">
                  {data.heatmap.map((cell) => (
                    <div
                      key={`${cell.day}-${cell.hour}`}
                      className="aspect-square rounded-sm"
                      style={{ background: heatCellColor(cell.intensity) }}
                      title={`${cell.day} ${cell.hour}: ${(cell.intensity * 100).toFixed(0)}%`}
                    />
                  ))}
                </div>
                <div className="mt-1 flex border-t border-outline-variant/30 pt-1 font-label-sm text-[10px] text-outline">
                  {HEATMAP_HOURS.map((h) => (
                    <div key={h} className="flex-1 text-center">
                      {h}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="mt-sm flex items-center justify-end gap-2 font-label-sm text-label-sm text-on-surface-variant">
            <span>Low</span>
            <div className="h-2 w-24 rounded-full bg-gradient-to-r from-primary/10 via-primary to-error" />
            <span>High</span>
          </div>
        </div>

        {/* Emotion Frequency (radar) */}
        <div className="col-span-12 flex min-h-[360px] flex-col rounded-xl panel p-lg lg:col-span-5">
          <div className="mb-md flex items-center justify-between">
            <div>
              <h3 className="font-headline-sm text-headline-sm text-on-surface">Emotion Frequency</h3>
              <p className="font-body-sm text-body-sm text-on-surface-variant">NLP derived emotional states.</p>
            </div>
            <SampleDataBadge />
          </div>
          <div className="flex-1 p-md">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={data.emotionFrequency} outerRadius="75%">
                <PolarGrid stroke={chartColors.outlineVariant} />
                <PolarAngleAxis dataKey="emotion" tick={{ fill: chartColors.onSurfaceVariant, fontSize: 12 }} />
                <Radar
                  dataKey="value"
                  stroke={chartColors.tertiary}
                  fill={chartColors.tertiary}
                  fillOpacity={0.25}
                  strokeWidth={2}
                />
                <Tooltip formatter={(v) => [`${Number(v)}`, 'Frequency']} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Population Risk Breakdown */}
        <div className="col-span-12 flex min-h-[300px] flex-col rounded-xl panel p-lg lg:col-span-6">
          <div className="mb-md">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Population Risk Breakdown</h3>
            <p className="font-body-sm text-body-sm text-on-surface-variant">Stratification by demographic segments.</p>
          </div>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.riskBreakdown} layout="vertical" margin={{ left: 12 }}>
                <XAxis type="number" hide domain={[0, 100]} />
                <YAxis
                  type="category"
                  dataKey="segment"
                  width={80}
                  tick={{ fill: chartColors.onSurfaceVariant, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip formatter={(v, name) => [`${Number(v)}%`, String(name)]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="low" stackId="risk" name="Low" fill={chartColors.secondary} radius={[4, 0, 0, 4]} barSize={24} />
                <Bar dataKey="medium" stackId="risk" name="Medium" fill={chartColors.primary} barSize={24} />
                <Bar dataKey="high" stackId="risk" name="High" fill={chartColors.error} radius={[0, 4, 4, 0]} barSize={24} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Physiological Correlation matrix */}
        <div className="col-span-12 flex min-h-[300px] flex-col rounded-xl panel p-lg lg:col-span-6">
          <div className="mb-md">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Physiological Correlation</h3>
            <p className="font-body-sm text-body-sm text-on-surface-variant">Signal relationship to stress events.</p>
          </div>
          <div className="flex flex-1 items-center justify-center p-md">
            <div className="w-full max-w-sm">
              <div className="grid grid-cols-4 gap-1">
                <div />
                {['HRV', 'EDA', 'Temp'].map((c) => (
                  <div key={c} className="origin-bottom-left ml-4 rotate-45 text-center font-label-sm text-[10px] text-on-surface-variant">
                    {c}
                  </div>
                ))}
                {['Stress', 'Sleep', 'Activity'].flatMap((rowLabel) => [
                  <div key={`${rowLabel}-label`} className="self-center pr-2 text-right font-label-sm text-[10px] text-on-surface-variant">
                    {rowLabel}
                  </div>,
                  ...['HRV', 'EDA', 'Temp'].map((colLabel) => {
                    const cell = data.correlation.find((c) => c.rowLabel === rowLabel && c.colLabel === colLabel)
                    const v = cell?.value ?? 0
                    const magnitude = Math.abs(v)
                    const bg = v >= 0 ? `rgba(186, 26, 26, ${magnitude})` : `rgba(0, 107, 95, ${magnitude})`
                    return (
                      <div
                        key={`${rowLabel}-${colLabel}`}
                        className="flex aspect-square items-center justify-center rounded-sm font-bold text-[10px] text-white"
                        style={{ background: bg, color: magnitude > 0.5 ? '#fff' : '#0b1c30' }}
                      >
                        {v.toFixed(2)}
                      </div>
                    )
                  }),
                ])}
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
