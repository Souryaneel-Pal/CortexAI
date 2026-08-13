import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AppShell } from '../components/layout/AppShell'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import { SampleDataBadge } from '../components/ui/SampleDataBadge'
import { SeverityDot, StatusChip } from '../components/ui/SeverityChip'
import { getDashboard, type DashboardData } from '../lib/api'
import { chartColors, severityColor } from '../lib/chartColors'
import { useAssessment } from '../lib/assessmentContext'
import type { Severity } from '../types'

/** Backend class name -> the severity token the design system already uses. */
const CLASS_TO_SEVERITY: Record<string, Severity> = {
  Healthy: 'Healthy',
  Mild_Stress: 'Mild',
  Moderate_Stress: 'Moderate',
  Severe_Stress: 'Severe',
}

export function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [errorState, setErrorState] = useState<string | null>(null)

  useEffect(() => {
    getDashboard()
      .then((res) => {
        setData(res)
        setLoading(false)
      })
      .catch((err) => {
        setErrorState(err instanceof Error ? err.message : 'Failed to load dashboard data')
        setLoading(false)
      })
  }, [])

  const { prediction, explanation, completedAt, running, phaseLabel, error, backendReachable, hasLiveResult } =
    useAssessment()

  if (loading) {
    return (
      <AppShell>
        <div className="flex h-[400px] items-center justify-center">
          <div className="flex flex-col items-center gap-md">
            <MaterialIcon name="progress_activity" className="animate-spin text-4xl text-primary" />
            <span className="font-label-md text-label-md text-on-surface-variant">Loading dashboard data...</span>
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
          <span className="font-label-md text-label-md text-error">{errorState || 'No dashboard data available'}</span>
        </div>
      </AppShell>
    )
  }

  const donutData = data.stressDistribution.map((s) => ({ name: s.label, value: s.pct, severity: s.severity }))
  const liveSeverity: Severity | null = prediction ? CLASS_TO_SEVERITY[prediction.predicted_class] ?? null : null

  return (
    <AppShell showSearch>
      {/* In-flight / failure banner — the dashboard is where a user lands
          after navigating away mid-run, so the state has to be visible here. */}
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
            {error ??
              (backendReachable === false
                ? 'Cannot reach the CortexAI API. Start it with `uvicorn src.api.main:app`.'
                : phaseLabel)}
          </span>
        </div>
      )}

      {/* Latest live assessment — real backend values, shown only when a
          session exists so it can never be confused with the sample cohort. */}
      {prediction && (
        <section className="mb-xl rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
          <div className="mb-md flex flex-wrap items-center justify-between gap-sm">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Latest Assessment</h3>
            <div className="flex items-center gap-sm">
              {prediction.deferred_to_human && (
                <span className="rounded-full bg-error/10 px-md py-xs font-label-sm text-label-sm text-error">
                  Deferred for human review
                </span>
              )}
              <span className="font-label-sm text-label-sm text-on-surface-variant">
                {completedAt ? new Date(completedAt).toLocaleString() : ''}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-md sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-outline-variant bg-surface p-md">
              <span className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">
                Predicted Status
              </span>
              <div className="mt-xs flex items-center gap-xs">
                {liveSeverity && <SeverityDot severity={liveSeverity} />}
              </div>
              <span className="mt-xs block font-label-sm text-label-sm text-on-surface-variant">
                {Math.round(prediction.confidence * 100)}% confidence
              </span>
            </div>
            {(
              [
                ['Depression', prediction.scores.Depression_Score, 34],
                ['Anxiety', prediction.scores.Anxiety_Score, 24],
                ['Stress', prediction.scores.Stress_Score, 39],
              ] as const
            ).map(([label, score, max]) => (
              <div key={label} className="rounded-lg border border-outline-variant bg-surface p-md">
                <span className="font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">
                  {label}
                </span>
                <div className="mt-xs font-headline-md text-headline-md text-on-surface">
                  {score.toFixed(1)}
                  <span className="font-label-sm text-label-sm text-on-surface-variant"> / {max}</span>
                </div>
                <div className="mt-xs h-2 w-full overflow-hidden rounded-full bg-surface-container">
                  <div className="h-full bg-primary" style={{ width: `${(score / max) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>

          {explanation?.masked_distress_index && !explanation.masked_distress_index.unavailable_reason && (
            <div className="mt-md flex flex-wrap items-center gap-md border-t border-outline-variant pt-md">
              <span className="font-label-sm text-label-sm text-on-surface-variant">
                Masked-Distress Index:{' '}
                <span className={explanation.masked_distress_index.flag ? 'text-error' : 'text-on-surface'}>
                  {explanation.masked_distress_index.mdi.toFixed(2)}
                </span>
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant">
                Modality weighting: {Math.round(explanation.modality_weights.tabular * 100)}% signals ·{' '}
                {Math.round(explanation.modality_weights.face * 100)}% face ·{' '}
                {Math.round(explanation.modality_weights.speech * 100)}% speech
              </span>
              <Link to="/results" className="ml-auto font-label-sm text-label-sm text-primary hover:underline">
                View full explainability →
              </Link>
            </div>
          )}
          {prediction.is_demo_untrained_model && (
            <p className="mt-md font-label-sm text-label-sm text-on-surface-variant">
              Model checkpoints are not loaded — these values are structurally valid but not a real signal.
            </p>
          )}
        </section>
      )}

      {/* Hero */}
      <section className="relative mb-xl flex flex-col gap-lg overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1 md:flex-row md:items-end md:justify-between">
        <div className="pointer-events-none absolute right-0 top-0 h-64 w-64 -translate-y-1/2 translate-x-1/4 rounded-full bg-gradient-to-br from-tertiary/10 to-transparent blur-3xl" />
        <div className="relative z-10 max-w-3xl">
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface mb-sm">
            Clinical Distress Screening Panel
          </h2>
          <p className="font-body-lg text-body-lg text-on-surface-variant">
            Decision-support system integrating behavioral, physiological, and audio-visual metrics to screen for mental distress levels.
          </p>
        </div>
        <div className="relative z-10 shrink-0">
          <Link
            to="/assessment/new"
            className="flex items-center gap-2 rounded-lg bg-primary px-6 py-3 font-label-md text-label-md text-on-primary shadow-level-1 transition-colors hover:bg-on-primary-fixed-variant"
          >
            <MaterialIcon name="add" className="text-[20px]" />
            Start New Assessment
          </Link>
        </div>
      </section>

      {/* KPI Cards Row */}
      <section className="mb-xl grid grid-cols-2 gap-md md:grid-cols-5">
        {data.heroStats.map((stat) => (
          <div
            key={stat.label}
            className={`flex flex-col rounded-lg border border-outline-variant bg-surface-container-lowest p-md shadow-level-1 ${
              stat.accentClassName ? `border-l-4 ${stat.accentClassName}` : ''
            }`}
          >
            <span className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-on-surface-variant">
              {stat.label}
            </span>
            <span className="font-headline-md text-headline-md text-on-surface">{stat.value}</span>
            <div className="mt-2 flex items-center gap-1 text-sm text-outline">
              {stat.label === 'Total Assessments' && <MaterialIcon name="trending_up" className="text-[16px] text-secondary" />}
              <span className="font-label-sm text-label-sm">{stat.meta}</span>
            </div>
          </div>
        ))}
      </section>

      {/* Main Data Grid */}
      <section className="mb-xl grid grid-cols-1 gap-lg lg:grid-cols-3">
        {/* Stress Distribution Donut */}
        <div className="col-span-1 flex flex-col rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
          <div className="mb-md flex items-center justify-between">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Distress Classification Distribution</h3>
            <SampleDataBadge />
          </div>
          <div className="relative flex min-h-[200px] flex-1 items-center justify-center">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={donutData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius="66%"
                  outerRadius="100%"
                  paddingAngle={1}
                  startAngle={90}
                  endAngle={-270}
                >
                  {donutData.map((d) => (
                    <Cell key={d.severity} fill={severityColor[d.severity as Severity]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip formatter={(value, name) => [`${Number(value)}%`, String(name)]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute flex flex-col items-center text-center">
              <span className="font-headline-md text-headline-md text-on-surface">
                {(data.totalAssessments / 1000).toFixed(1)}k
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant">Total</span>
            </div>
          </div>
          <div className="mt-md grid grid-cols-2 gap-sm">
            {data.stressDistribution.map((s) => (
              <div key={s.severity} className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full" style={{ background: severityColor[s.severity as Severity] }} />
                <span className="font-label-sm text-label-sm text-on-surface-variant">
                  {s.label} ({s.pct}%)
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Monthly Assessment Trend */}
        <div className="col-span-1 flex flex-col rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1 lg:col-span-2">
          <div className="mb-md flex items-center justify-between">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Monthly Assessment Volumetrics</h3>
            <div className="flex items-center gap-sm">
              <SampleDataBadge />
              <select className="cursor-pointer rounded-md border-none bg-surface-container-low py-1 pr-8 font-label-sm text-label-sm text-on-surface focus:ring-1 focus:ring-primary">
                <option>Last 6 Months</option>
                <option>This Year</option>
              </select>
            </div>
          </div>
          <div className="min-h-[200px] flex-1">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={data.trend} margin={{ left: 4, right: 12, top: 8, bottom: 0 }}>
                <defs>
                  <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={chartColors.primary} stopOpacity={0.2} />
                    <stop offset="100%" stopColor={chartColors.primary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={chartColors.outlineVariant} strokeDasharray="4 4" vertical={false} />
                <XAxis
                  dataKey="month"
                  tick={{ fill: chartColors.onSurfaceVariant, fontSize: 12 }}
                  axisLine={{ stroke: chartColors.outlineVariant }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: chartColors.onSurfaceVariant, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  width={36}
                />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="assessments"
                  stroke={chartColors.primary}
                  strokeWidth={2}
                  fill="url(#trendFill)"
                  dot={{ r: 3, fill: chartColors.primaryContainer, stroke: '#fff', strokeWidth: 1.5 }}
                  activeDot={{ r: 5 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      {/* Recent Assessments Table */}
      <section className="mb-xl overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest shadow-level-1">
        <div className="flex items-center justify-between border-b border-outline-variant bg-surface-container-low p-lg">
          <h3 className="font-headline-sm text-headline-sm text-on-surface">Recent Patient Screenings</h3>
          <button type="button" className="font-label-sm text-label-sm text-primary hover:underline">
            View All
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-outline-variant bg-surface-bright">
                <th className="px-6 py-3 font-label-sm text-label-sm font-medium uppercase tracking-wider text-on-surface-variant">
                  Patient ID
                </th>
                <th className="px-6 py-3 font-label-sm text-label-sm font-medium uppercase tracking-wider text-on-surface-variant">
                  Date &amp; Time
                </th>
                <th className="px-6 py-3 font-label-sm text-label-sm font-medium uppercase tracking-wider text-on-surface-variant">
                  Risk Level
                </th>
                <th className="px-6 py-3 font-label-sm text-label-sm font-medium uppercase tracking-wider text-on-surface-variant">
                  Status
                </th>
                <th className="px-6 py-3 text-right font-label-sm text-label-sm font-medium uppercase tracking-wider text-on-surface-variant">
                  Action
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant font-body-sm text-body-sm text-on-surface">
              {hasLiveResult && prediction && liveSeverity && (
                <tr className="bg-primary/5 transition-colors hover:bg-surface-container-low">
                  <td className="px-6 py-4 font-label-sm text-label-sm font-medium">
                    {prediction.session_id.slice(0, 8)}
                  </td>
                  <td className="px-6 py-4 text-on-surface-variant">
                    {completedAt ? new Date(completedAt).toLocaleString() : ''}
                  </td>
                  <td className="px-6 py-4">
                    <SeverityDot severity={liveSeverity} />
                  </td>
                  <td className="px-6 py-4">
                    <StatusChip status={prediction.deferred_to_human ? 'AI Reviewing' : 'Completed'} />
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link to="/results" className="text-primary transition-colors hover:text-primary-container">
                      <MaterialIcon name="chevron_right" className="text-[20px]" />
                    </Link>
                  </td>
                </tr>
              )}
              {data.recentAssessments.map((row) => (
                <tr key={row.patientId} className="transition-colors hover:bg-surface-container-low">
                  <td className="px-6 py-4 font-label-sm text-label-sm font-medium">{row.patientId}</td>
                  <td className="px-6 py-4 text-on-surface-variant">{row.dateTime}</td>
                  <td className="px-6 py-4">
                    <SeverityDot severity={row.riskLevel as Severity} />
                  </td>
                  <td className="px-6 py-4">
                    <StatusChip status={row.status} />
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link to="/results" className="text-primary transition-colors hover:text-primary-container">
                      <MaterialIcon name="chevron_right" className="text-[20px]" />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </AppShell>
  )
}
