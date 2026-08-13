import { useState } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { AppShell } from '../components/layout/AppShell'
import { InsightBanner } from '../components/ui/InsightBanner'
import { SampleDataBadge } from '../components/ui/SampleDataBadge'
import { CrisisResourcesCard } from '../components/ui/CrisisResourcesCard'
import { getExplainabilityData } from '../lib/mockData'
import { chartColors } from '../lib/chartColors'

const MODALITY_COLOR: Record<string, string> = {
  Physiological: chartColors.primaryContainer,
  Facial: chartColors.secondary,
  Behavioral: chartColors.tertiary,
  Speech: chartColors.outlineVariant,
}

const ACOUSTIC_COLOR: Record<'primary' | 'secondary' | 'tertiary', string> = {
  primary: chartColors.primary,
  secondary: chartColors.secondary,
  tertiary: chartColors.tertiary,
}

export function ExplainableInsights() {
  const [expanded, setExpanded] = useState(false)
  const data = getExplainabilityData(expanded)
  const isSevereFlag = data.summary.headline.includes('high likelihood')

  return (
    <AppShell title="Explainable AI (XAI) Flagship">
      <div className="mb-xl flex flex-col gap-sm md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface mb-xs">
            Patient Assessment Analysis
          </h2>
          <p className="font-body-lg text-body-lg text-on-surface-variant">
            Patient ID: <span className="font-label-md text-label-md text-primary">{data.patientId}</span> · Assessment
            Date: {data.assessmentDate}
          </p>
        </div>
        <SampleDataBadge />
      </div>

      <div className="grid grid-cols-1 gap-lg lg:grid-cols-3">
        {/* AI Explanation Summary */}
        <InsightBanner title="AI Explanation Summary" className="lg:col-span-3">
          <p>
            The model predicts a <strong className="text-error">{data.summary.headline}</strong>. This conclusion is
            primarily driven by{' '}
            {data.summary.driverHighlights.map((d, i) => (
              <span key={d.text}>
                <span
                  className={`rounded px-1 font-medium ${
                    d.colorToken === 'primary' ? 'bg-primary/10 text-primary' : 'bg-secondary/10 text-secondary'
                  }`}
                >
                  {d.text}
                </span>
                {i < data.summary.driverHighlights.length - 1 ? ' and ' : ' '}
              </span>
            ))}
            during stress-inducing question segments. {data.summary.body}
          </p>
        </InsightBanner>

        {/* Feature Importance (SHAP) */}
        <div className="rounded-xl border border-outline-variant bg-surface p-lg shadow-level-1 lg:col-span-2">
          <div className="mb-md flex items-center justify-between">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Feature Importance (SHAP Values)</h3>
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="font-label-sm text-label-sm text-primary hover:underline"
            >
              {expanded ? 'Hide Detailed Metrics' : 'View Detailed Metrics'}
            </button>
          </div>
          <div style={{ height: data.shap.length * 40 + 20 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.shap} layout="vertical" margin={{ left: 24, right: 24 }}>
                <XAxis type="number" domain={[-1, 1]} hide />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={140}
                  tick={{ fill: chartColors.onSurfaceVariant, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip formatter={(v: number) => v.toFixed(2)} />
                <Bar dataKey="value" radius={4} barSize={20}>
                  {data.shap.map((d) => (
                    <Cell key={d.feature} fill={d.value >= 0 ? chartColors.primary : chartColors.secondary} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-md border-t border-outline-variant pt-sm text-center">
            <span className="font-label-sm text-label-sm italic text-on-surface-variant">
              Features ranked by impact on final clinical prediction. Primary = raises predicted severity, Secondary =
              lowers it.
            </span>
          </div>
        </div>

        {/* Modality Contribution Donut */}
        <div className="flex flex-col rounded-xl border border-outline-variant bg-surface p-lg shadow-level-1 lg:col-span-1">
          <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Modality Contribution</h3>
          <div className="relative flex flex-1 items-center justify-center py-md">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={data.modalityContribution}
                  dataKey="pct"
                  nameKey="modality"
                  innerRadius="70%"
                  outerRadius="100%"
                  startAngle={90}
                  endAngle={-270}
                >
                  {data.modalityContribution.map((m) => (
                    <Cell key={m.modality} fill={MODALITY_COLOR[m.modality]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number, name: string) => [`${value}%`, name]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="font-headline-md text-headline-md font-bold text-on-surface">
                {data.modalityContribution.length}
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant">Modalities</span>
            </div>
          </div>
          <div className="mt-auto grid grid-cols-2 gap-sm pt-sm">
            {data.modalityContribution.map((m) => (
              <div key={m.modality} className="flex items-center gap-xs">
                <div className="h-3 w-3 rounded" style={{ background: MODALITY_COLOR[m.modality] }} />
                <span className="font-label-sm text-label-sm text-on-surface-variant">
                  {m.modality} ({m.pct}%)
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Detailed Breakdown Table */}
        <div className="rounded-xl border border-outline-variant bg-surface p-lg shadow-level-1 lg:col-span-3">
          <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Detailed Modality Breakdown</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-outline-variant">
                  <th className="px-md py-sm font-label-md text-label-md font-medium text-on-surface-variant">Modality</th>
                  <th className="px-md py-sm font-label-md text-label-md font-medium text-on-surface-variant">Key Indicator</th>
                  <th className="px-md py-sm font-label-md text-label-md font-medium text-on-surface-variant">Deviation from Baseline</th>
                  <th className="px-md py-sm font-label-md text-label-md font-medium text-on-surface-variant">Clinical Relevance</th>
                </tr>
              </thead>
              <tbody>
                {data.breakdown.map((row) => (
                  <tr key={row.modality} className="border-b border-outline-variant transition-colors last:border-b-0 hover:bg-surface-container-low">
                    <td className="px-md py-md font-body-sm text-body-sm text-on-surface">{row.modality}</td>
                    <td className="px-md py-md font-label-sm text-label-sm" style={{ color: MODALITY_COLOR[row.modality] }}>
                      {row.keyIndicator}
                    </td>
                    <td className="px-md py-md font-body-sm text-body-sm text-on-surface">{row.deviation}</td>
                    <td className="px-md py-md font-body-sm text-body-sm text-on-surface-variant">{row.clinicalRelevance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {expanded && (
          <>
            {/* Acoustic & Prosodic Analysis */}
            <div className="rounded-xl border border-outline-variant bg-surface p-lg shadow-level-1 lg:col-span-1">
              <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Acoustic &amp; Prosodic Analysis</h3>
              <div className="flex flex-col gap-md">
                {data.acoustic.map((m) => (
                  <div key={m.label}>
                    <div className="flex items-center justify-between">
                      <span className="font-body-sm text-on-surface-variant">{m.label}</span>
                      <span className="font-label-md text-label-md" style={{ color: ACOUSTIC_COLOR[m.colorToken] }}>
                        {m.value}
                      </span>
                    </div>
                    <div className="mt-xs h-2 w-full overflow-hidden rounded-full bg-surface-container">
                      <div className="h-full" style={{ width: `${m.pct}%`, background: ACOUSTIC_COLOR[m.colorToken] }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Facial Micro-expression Timeline */}
            <div className="rounded-xl border border-outline-variant bg-surface p-lg shadow-level-1 lg:col-span-2">
              <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Facial Micro-expression Timeline</h3>
              <div style={{ height: 140 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.microExpressionTimeline} margin={{ top: 8 }}>
                    <XAxis dataKey="minute" hide />
                    <YAxis hide domain={[0, 100]} />
                    <Tooltip formatter={(v: number) => [`${v}%`, 'Intensity']} labelFormatter={(l) => `Minute ${l}`} />
                    <Bar dataKey="intensityPct" radius={[4, 4, 0, 0]} fill={chartColors.secondary} fillOpacity={0.6} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-sm flex justify-between font-label-sm text-label-sm text-on-surface-variant">
                <span>Start</span>
                <span>Assessment Duration (10m)</span>
                <span>End</span>
              </div>
            </div>

            {/* Longitudinal Comparison */}
            <div className="rounded-xl border border-outline-variant bg-surface p-lg shadow-level-1 lg:col-span-3">
              <div className="mb-md flex items-center justify-between">
                <h3 className="font-headline-sm text-headline-sm text-on-surface">Longitudinal Comparison (vs. 3 Months Ago)</h3>
                <div className="flex gap-md">
                  <div className="flex items-center gap-xs">
                    <div className="h-3 w-3 rounded-full bg-primary" />
                    <span className="font-label-sm text-label-sm text-on-surface-variant">Current</span>
                  </div>
                  <div className="flex items-center gap-xs">
                    <div className="h-3 w-3 rounded-full bg-outline-variant" />
                    <span className="font-label-sm text-label-sm text-on-surface-variant">Baseline</span>
                  </div>
                </div>
              </div>
              <div className="flex flex-col gap-md">
                {data.longitudinal.map((m) => (
                  <div key={m.label} className="grid grid-cols-12 items-center gap-md">
                    <div className="col-span-3 font-label-sm text-label-sm text-on-surface-variant">{m.label}</div>
                    <div className="col-span-9 flex flex-col gap-1">
                      <div className="h-3 w-full overflow-hidden rounded-full bg-surface-container">
                        <div className="h-full bg-primary" style={{ width: `${m.current}%` }} />
                      </div>
                      <div className="h-3 w-full overflow-hidden rounded-full bg-surface-container">
                        <div className="h-full bg-outline-variant" style={{ width: `${m.baseline}%` }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {isSevereFlag && (
          <div className="lg:col-span-3">
            <CrisisResourcesCard />
          </div>
        )}
      </div>
    </AppShell>
  )
}
