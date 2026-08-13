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
import { getDashboardData } from '../lib/mockData'
import { chartColors, severityColor } from '../lib/chartColors'

export function Dashboard() {
  const data = getDashboardData()
  const donutData = data.stressDistribution.map((s) => ({ name: s.label, value: s.pct, severity: s.severity }))

  return (
    <AppShell showSearch>
      {/* Hero */}
      <section className="relative mb-xl flex flex-col gap-lg overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1 md:flex-row md:items-end md:justify-between">
        <div className="pointer-events-none absolute right-0 top-0 h-64 w-64 -translate-y-1/2 translate-x-1/4 rounded-full bg-gradient-to-br from-tertiary/10 to-transparent blur-3xl" />
        <div className="relative z-10 max-w-3xl">
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface mb-sm">
            AI-Powered Multimodal Psychiatric Assessment
          </h2>
          <p className="font-body-lg text-body-lg text-on-surface-variant">
            Integrating facial expressions, speech emotions, behavioral patterns, and physiological indicators for
            explainable mental health evaluation.
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
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Stress Distribution</h3>
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
                    <Cell key={d.severity} fill={severityColor[d.severity]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number, name: string) => [`${value}%`, name]} />
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
                <div className="h-3 w-3 rounded-full" style={{ background: severityColor[s.severity] }} />
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
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Monthly Assessment Trend</h3>
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
          <h3 className="font-headline-sm text-headline-sm text-on-surface">Recent Assessments</h3>
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
              {data.recentAssessments.map((row) => (
                <tr key={row.patientId} className="transition-colors hover:bg-surface-container-low">
                  <td className="px-6 py-4 font-label-sm text-label-sm font-medium">{row.patientId}</td>
                  <td className="px-6 py-4 text-on-surface-variant">{row.dateTime}</td>
                  <td className="px-6 py-4">
                    <SeverityDot severity={row.riskLevel} />
                  </td>
                  <td className="px-6 py-4">
                    <StatusChip status={row.status} />
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link to="/insights" className="text-primary transition-colors hover:text-primary-container">
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
