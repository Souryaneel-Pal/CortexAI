import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
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
import { useAssessment } from '../lib/assessmentContext'
import { MaterialIcon } from '../components/ui/MaterialIcon'

const MODALITY_COLOR: Record<string, string> = {
  Physiological: chartColors.primaryContainer,
  Facial: chartColors.secondary,
  Behavioral: chartColors.tertiary,
  Speech: chartColors.outlineVariant,
  'Behavioral & Physiological': chartColors.primaryContainer,
}

const ACOUSTIC_COLOR: Record<'primary' | 'secondary' | 'tertiary', string> = {
  primary: chartColors.primary,
  secondary: chartColors.secondary,
  tertiary: chartColors.tertiary,
}

const SEVERITY_LABEL: Record<string, string> = {
  Healthy: 'Healthy',
  Mild_Stress: 'Mild Stress',
  Moderate_Stress: 'Moderate Stress',
  Severe_Stress: 'Severe Stress',
}

function pct(value: number): number {
  return Math.round(value * 100)
}

export function Results() {
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState(false)
  const data = getExplainabilityData(expanded)

  const { prediction, explanation, faceImageDataUrl, completedAt, generateReport, phase, running } = useAssessment()

  async function handleGenerateReport() {
    if (!prediction) return
    const res = await generateReport(prediction.session_id)
    if (res) {
      navigate('/reports')
    }
  }

  const shapChart = explanation?.signed_shap?.length
    ? [...explanation.signed_shap]
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
        .slice(0, 6)
        .map((d) => ({ feature: d.feature.replace(/_/g, ' '), value: d.shap }))
    : data.shap

  const modalityContribution = explanation
    ? [
        { modality: 'Behavioral & Physiological', pct: pct(explanation.modality_weights.tabular) },
        { modality: 'Facial', pct: pct(explanation.modality_weights.face) },
        { modality: 'Speech', pct: pct(explanation.modality_weights.speech) },
      ]
    : data.modalityContribution

  const mdi = explanation?.masked_distress_index ?? null
  const isSevereFlag = prediction
    ? prediction.predicted_class === 'Severe_Stress' || Boolean(mdi?.flag)
    : data.summary.headline.includes('high likelihood')

  const shapDomain: [number, number] = explanation?.signed_shap?.length
    ? (() => {
        const peak = Math.max(...shapChart.map((d) => Math.abs(d.value)), 1e-6)
        return [-peak, peak]
      })()
    : [-1, 1]

  return (
    <AppShell
      eyebrow="Step 2 of 2"
      title="Diagnostic Result Details"
      subtitle="Prediction, severity scores, and the evidence behind them — attribution per modality, per feature, and per frame."
    >
      <div className="mb-xl flex flex-col gap-md md:flex-row md:items-center md:justify-between border-b border-outline-variant pb-md">
        <div>
          <p className="font-body-md text-body-md text-on-surface-variant">
            {prediction ? (
              <>
                Session: <span className="font-label-md text-label-md text-primary">{prediction.session_id.slice(0, 8)}</span> ·
                Assessment Date: {completedAt ? new Date(completedAt).toLocaleDateString() : '—'}
              </>
            ) : (
              <>
                Patient ID: <span className="font-label-md text-label-md text-primary">{data.patientId}</span> · Assessment
                Date: {data.assessmentDate}
              </>
            )}
          </p>
        </div>

        {/* Generate Report Action */}
        <div className="flex items-center gap-md">
          {prediction && (
            <button
              type="button"
              onClick={handleGenerateReport}
              disabled={running || phase === 'reporting'}
              className="ai-glow flex items-center justify-center gap-sm rounded-lg bg-gradient-to-r from-tertiary to-primary-container px-xl py-sm font-label-md text-label-md text-on-primary transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {phase === 'reporting' ? (
                <>
                  <MaterialIcon name="progress_activity" className="animate-spin text-[18px]" />
                  Generating Clinical Report...
                </>
              ) : (
                <>
                  <MaterialIcon name="clinical_notes" className="text-[18px]" />
                  Generate Clinical Report
                </>
              )}
            </button>
          )}

          {prediction ? (
            prediction.is_demo_untrained_model && (
              <span className="flex items-center gap-xs rounded-full border border-outline-variant bg-surface-container px-md py-xs font-label-sm text-label-sm text-on-surface-variant">
                Untrained model
              </span>
            )
          ) : (
            <SampleDataBadge />
          )}
        </div>
      </div>

      {prediction && (
        <div className="mb-lg grid grid-cols-1 gap-lg sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl panel p-lg">
            <span className="font-label-sm text-label-sm text-on-surface-variant">Predicted Status</span>
            <p className="mt-xs font-headline-sm text-headline-sm text-on-surface">
              {SEVERITY_LABEL[prediction.predicted_class] ?? prediction.predicted_class}
            </p>
            <span className="font-label-sm text-label-sm text-on-surface-variant">
              {pct(prediction.confidence)}% confidence (MC-dropout)
            </span>
          </div>
          {(
            [
              ['Depression', prediction.scores.Depression_Score, 34],
              ['Anxiety', prediction.scores.Anxiety_Score, 24],
              ['Stress', prediction.scores.Stress_Score, 39],
            ] as const
          ).map(([label, score, max]) => (
            <div key={label} className="rounded-xl panel p-lg">
              <span className="font-label-sm text-label-sm text-on-surface-variant">{label} Score</span>
              <p className="mt-xs font-headline-sm text-headline-sm text-on-surface">
                {score.toFixed(1)}
                <span className="font-label-sm text-label-sm text-on-surface-variant"> / {max}</span>
              </p>
              <div className="mt-xs h-2 w-full overflow-hidden rounded-full bg-surface-container">
                <div className="h-full bg-primary" style={{ width: `${(score / max) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 gap-lg lg:grid-cols-3">
        {/* AI Explanation Summary */}
        <InsightBanner title="AI Explanation Summary" className="lg:col-span-3">
          {prediction && explanation ? (
            <p>
              The model predicts{' '}
              <strong className={prediction.predicted_class === 'Severe_Stress' ? 'text-error' : 'text-primary'}>
                {SEVERITY_LABEL[prediction.predicted_class] ?? prediction.predicted_class}
              </strong>{' '}
              at {pct(prediction.confidence)}% confidence. The fusion gate weighted this call{' '}
              {modalityContribution.map((m, i) => (
                <span key={m.modality}>
                  <span className="rounded px-1 font-medium bg-primary/10 text-primary">
                    {m.pct}% {m.modality.toLowerCase()}
                  </span>
                  {i < modalityContribution.length - 1 ? ', ' : '. '}
                </span>
              ))}
              {explanation.top_shap_features[0] && (
                <>
                  The strongest single feature was{' '}
                  <span className="rounded px-1 font-medium bg-secondary/10 text-secondary">
                    {explanation.top_shap_features[0].feature.replace(/_/g, ' ')}
                  </span>
                  .{' '}
                </>
              )}
              {prediction.deferred_to_human
                ? 'Confidence is below the deferral threshold — this case is flagged for human review rather than reported as a result.'
                : 'Confidence is above the deferral threshold.'}
            </p>
          ) : (
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
          )}
        </InsightBanner>

        {/* Grad-CAM */}
        {explanation?.gradcam && (
          <div className="rounded-xl panel p-lg lg:col-span-1">
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Facial Attention Mapping (Grad-CAM)</h3>
            <div className="flex items-center justify-center gap-md">
              {faceImageDataUrl && (
                <figure className="flex flex-col items-center gap-xs">
                  <img
                    src={faceImageDataUrl}
                    alt="Submitted face"
                    className="h-32 w-32 rounded-lg border border-outline-variant object-cover"
                  />
                  <figcaption className="font-label-sm text-label-sm text-on-surface-variant">Input</figcaption>
                </figure>
              )}
              <figure className="flex flex-col items-center gap-xs">
                <img
                  src={`data:image/png;base64,${explanation.gradcam.overlay_png_base64}`}
                  alt="Grad-CAM attribution heatmap"
                  className="h-32 w-32 rounded-lg border border-outline-variant object-cover"
                />
                <figcaption className="font-label-sm text-label-sm text-on-surface-variant">Attribution</figcaption>
              </figure>
            </div>
            <p className="mt-md border-t border-outline-variant pt-sm font-label-sm text-label-sm text-on-surface-variant">
              Brighter regions drove the <span className="text-primary">{explanation.gradcam.predicted_emotion}</span>{' '}
              reading. Hooked at <code>{explanation.gradcam.target_layer}</code>.
            </p>
          </div>
        )}

        {/* MDI */}
        {mdi && !mdi.unavailable_reason && (
          <div className="rounded-xl panel p-lg lg:col-span-2">
            <div className="mb-md flex items-center justify-between">
              <h3 className="font-headline-sm text-headline-sm text-on-surface">Masked-Distress Index (MDI)</h3>
              <span
                className={`rounded-full px-md py-xs font-label-sm text-label-sm ${
                  mdi.flag ? 'bg-error/10 text-error' : 'bg-secondary/10 text-secondary'
                }`}
              >
                {mdi.flag ? 'Contradiction flagged' : 'No contradiction'}
              </span>
            </div>
            <div className="flex items-baseline gap-sm">
              <span className="font-headline-lg text-headline-lg text-on-surface">{mdi.mdi.toFixed(2)}</span>
              <span className="font-label-sm text-label-sm text-on-surface-variant">/ 1.00</span>
            </div>
            <div className="mt-sm h-2 w-full overflow-hidden rounded-full bg-surface-container">
              <div
                className={`h-full ${mdi.flag ? 'bg-error' : 'bg-secondary'}`}
                style={{ width: `${pct(mdi.mdi)}%` }}
              />
            </div>
            <div className="mt-md grid grid-cols-1 gap-sm sm:grid-cols-3">
              {(
                [
                  ['Face reads calm', mdi.face_calm],
                  ['Voice high-arousal', mdi.voice_high_arousal],
                  ['Physiology high-arousal', mdi.physio_high_arousal],
                ] as const
              ).map(([label, value]) => (
                <div key={label}>
                  <div className="flex items-center justify-between">
                    <span className="font-body-sm text-body-sm text-on-surface-variant">{label}</span>
                    <span className="font-label-md text-label-md text-on-surface">{pct(value ?? 0)}%</span>
                  </div>
                  <div className="mt-xs h-2 w-full overflow-hidden rounded-full bg-surface-container">
                    <div className="h-full bg-primary" style={{ width: `${pct(value ?? 0)}%` }} />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-md border-t border-outline-variant pt-sm font-label-sm text-label-sm italic text-on-surface-variant">
              MDI is high only when the face reads calm while voice or physiology read high-arousal.
              {mdi.dominant_contradiction && ` Driven here by ${mdi.dominant_contradiction}.`}
            </p>
          </div>
        )}

        {/* Feature Importance (SHAP) */}
        <div className="rounded-xl panel p-lg lg:col-span-2">
          <div className="mb-md flex items-center justify-between">
            <h3 className="font-headline-sm text-headline-sm text-on-surface">Attribution Analysis (SHAP Values)</h3>
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="font-label-sm text-label-sm text-primary hover:underline"
            >
              {expanded ? 'Hide Detailed Metrics' : 'View Detailed Metrics'}
            </button>
          </div>
          <div style={{ height: shapChart.length * 40 + 20 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shapChart} layout="vertical" margin={{ left: 24, right: 24 }}>
                <XAxis type="number" domain={shapDomain} hide />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={140}
                  tick={{ fill: chartColors.onSurfaceVariant, fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                <Bar dataKey="value" radius={4} barSize={20}>
                  {shapChart.map((d) => (
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
        <div className="flex flex-col rounded-xl panel p-lg lg:col-span-1">
          <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Contribution Weights by Modality</h3>
          <div className="relative flex flex-1 items-center justify-center py-md">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={modalityContribution}
                  dataKey="pct"
                  nameKey="modality"
                  innerRadius="70%"
                  outerRadius="100%"
                  startAngle={90}
                  endAngle={-270}
                >
                  {modalityContribution.map((m) => (
                    <Cell key={m.modality} fill={MODALITY_COLOR[m.modality]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip formatter={(value, name) => [`${Number(value)}%`, String(name)]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span className="font-headline-md text-headline-md font-bold text-on-surface">
                {modalityContribution.length}
              </span>
              <span className="font-label-sm text-label-sm text-on-surface-variant">Modalities</span>
            </div>
          </div>
          <div className="mt-auto grid grid-cols-2 gap-sm pt-sm border-t border-outline-variant/30">
            {modalityContribution.map((m) => (
              <div key={m.modality} className="flex items-center gap-xs">
                <div className="h-3 w-3 rounded" style={{ background: MODALITY_COLOR[m.modality] }} />
                <span className="font-label-sm text-label-sm text-on-surface-variant">
                  {m.modality} ({m.pct}%)
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Speech Integrated Gradients */}
        {explanation?.audio_integrated_gradients && (
          <div className="rounded-xl panel p-lg lg:col-span-3">
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">
              Acoustic Attribution Timeline (Integrated Gradients)
            </h3>
            <div className="flex h-24 items-end gap-[2px]">
              {explanation.audio_integrated_gradients.frame_importance.map((v, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t bg-secondary"
                  style={{ height: `${Math.max(2, v * 100)}%`, opacity: 0.35 + v * 0.65 }}
                  title={`${(i * explanation.audio_integrated_gradients!.frame_ms).toFixed(0)} ms — ${(v * 100).toFixed(0)}%`}
                />
              ))}
            </div>
            <div className="mt-sm flex justify-between font-label-sm text-label-sm text-on-surface-variant">
              <span>0 ms</span>
              <span>
                Frames driving the{' '}
                <span className="text-primary">{explanation.audio_integrated_gradients.predicted_emotion}</span> reading
              </span>
              <span>
                {(
                  explanation.audio_integrated_gradients.frame_importance.length *
                  explanation.audio_integrated_gradients.frame_ms
                ).toFixed(0)}{' '}
                ms
              </span>
            </div>
          </div>
        )}

        {/* Detailed Breakdown Table */}
        <div className="rounded-xl panel p-lg lg:col-span-3">
          <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Categorical Signal Diagnostics</h3>
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
                  <tr key={row.modality} className="border-b border-outline-variant last:border-b-0 hover:bg-surface-container-low transition-colors">
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
            <div className="rounded-xl panel p-lg lg:col-span-1">
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
            <div className="rounded-xl panel p-lg lg:col-span-2">
              <h3 className="font-headline-sm text-headline-sm text-on-surface mb-md">Facial Micro-expression Timeline</h3>
              <div style={{ height: 140 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.microExpressionTimeline} margin={{ top: 8 }}>
                    <XAxis dataKey="minute" hide />
                    <YAxis hide domain={[0, 100]} />
                    <Tooltip formatter={(v) => [`${Number(v)}%`, 'Intensity']} labelFormatter={(l) => `Minute ${l}`} />
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
            <div className="rounded-xl panel p-lg lg:col-span-3">
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
