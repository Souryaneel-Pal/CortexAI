import { useState } from 'react'
import { Link } from 'react-router-dom'
import { AppShell } from '../components/layout/AppShell'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import { SampleDataBadge } from '../components/ui/SampleDataBadge'
import { CrisisResourcesCard } from '../components/ui/CrisisResourcesCard'
import { getClinicalReportData } from '../lib/mockData'
import { DECISION_SUPPORT_REPORT_FOOTER } from '../lib/responsibleAI'
import { useReportForCurrentSession } from '../lib/assessmentContext'

const TREND_ICON: Record<'up' | 'down' | 'flat', string> = {
  up: 'trending_up',
  down: 'trending_down',
  flat: 'trending_flat',
}

export function ClinicalReport() {
  const report = getClinicalReportData()
  const [notes, setNotes] = useState(report.clinicianNotes)
  // The RAG-grounded narrative for the current session. Every sentence of
  // clinical advice in it traces to a retrieved source, and the citations are
  // rendered below it so a clinician can check them.
  const ragReport = useReportForCurrentSession()

  return (
    <AppShell title="Reports" showSearch searchPlaceholder="Search reports..." bareMain>
      <div className="flex-1 overflow-y-auto bg-surface-container-lowest p-margin-mobile md:p-margin-desktop">
        {/* Action Bar */}
        <div className="mx-auto mb-lg flex max-w-[900px] items-center justify-between">
          <Link to="/" className="flex items-center gap-sm font-label-md text-label-md text-on-surface-variant transition-colors hover:text-primary">
            <MaterialIcon name="arrow_back" />
            Back to Patient Profile
          </Link>
          <div className="flex items-center gap-sm">
            <SampleDataBadge />
            <button type="button" className="shadow-level-1 flex items-center gap-xs rounded-lg border border-outline-variant bg-surface px-4 py-2 font-label-md text-label-md text-on-surface transition-colors hover:bg-surface-container">
              <MaterialIcon name="share" className="text-[18px]" /> Share
            </button>
            <button type="button" className="shadow-level-1 flex items-center gap-xs rounded-lg border border-outline-variant bg-surface px-4 py-2 font-label-md text-label-md text-on-surface transition-colors hover:bg-surface-container">
              <MaterialIcon name="print" className="text-[18px]" /> Print
            </button>
            <button type="button" className="shadow-level-1 flex items-center gap-xs rounded-lg bg-primary-container px-4 py-2 font-label-md text-label-md text-on-primary transition-opacity hover:opacity-90">
              <MaterialIcon name="picture_as_pdf" className="text-[18px]" /> Export PDF
            </button>
          </div>
        </div>

        {/* The Report Document */}
        <div className="shadow-level-2 mx-auto mb-2xl max-w-[900px] rounded-xl border border-outline-variant bg-surface-container-lowest p-8 md:p-12">
          {/* Report Header */}
          <div className="mb-xl flex items-start justify-between border-b border-outline-variant pb-xl">
            <div>
              <div className="mb-2 font-label-sm text-label-sm uppercase tracking-wider text-primary">
                Comprehensive Psychiatric Evaluation
              </div>
              <h1 className="font-headline-lg text-headline-lg mb-xs font-bold text-on-surface">{report.patient.name}</h1>
              <div className="flex gap-md font-body-sm text-body-sm text-on-surface-variant">
                <span>ID: #{report.patient.id}</span>
                <span>
                  DOB: {report.patient.dob} ({report.patient.age}y)
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="mb-1 font-label-sm text-label-sm text-on-surface-variant">Assessment Date</div>
              <div className="mb-3 font-label-md text-label-md font-bold text-on-surface">{report.assessmentDate}</div>
              <div className="mb-1 font-label-sm text-label-sm text-on-surface-variant">Clinician</div>
              <div className="font-label-md text-label-md text-on-surface">{report.clinician}</div>
            </div>
          </div>

          {/* Modalities & Context */}
          <div className="mb-xl grid grid-cols-1 gap-md md:grid-cols-3">
            <div className="rounded-lg border border-outline-variant bg-surface-container p-4">
              <div className="mb-1 font-label-sm text-label-sm text-on-surface-variant">Session Type</div>
              <div className="flex items-center gap-xs font-body-md text-body-md font-medium text-on-surface">
                <MaterialIcon name="videocam" className="text-[18px] text-primary" />
                {report.sessionType}
              </div>
            </div>
            <div className="rounded-lg border border-outline-variant bg-surface-container p-4">
              <div className="mb-1 font-label-sm text-label-sm text-on-surface-variant">Duration</div>
              <div className="flex items-center gap-xs font-body-md text-body-md font-medium text-on-surface">
                <MaterialIcon name="schedule" className="text-[18px] text-primary" />
                {report.duration}
              </div>
            </div>
            <div className="rounded-lg border border-outline-variant bg-surface-container p-4">
              <div className="mb-1 font-label-sm text-label-sm text-on-surface-variant">Assessment Tools</div>
              <div className="flex items-center gap-xs font-body-md text-body-md font-medium text-on-surface">
                <MaterialIcon name="quiz" className="text-[18px] text-primary" />
                {report.assessmentTools}
              </div>
            </div>
          </div>

          {/* Clinical Summary */}
          <div className="mb-xl">
            <h2 className="mb-md flex items-center gap-sm font-headline-sm text-headline-sm font-bold text-on-surface">
              <MaterialIcon name="clinical_notes" className="text-primary" />
              Clinical Summary
            </h2>
            <p className="font-body-md text-body-md leading-relaxed text-on-surface-variant">{report.clinicalSummary}</p>
          </div>

          {/* Severity Scores */}
          <div className="mb-xl">
            <h2 className="mb-md flex items-center gap-sm font-headline-sm text-headline-sm font-bold text-on-surface">
              <MaterialIcon name="bar_chart" className="text-primary" />
              Severity Scores
            </h2>
            <div className="grid grid-cols-1 gap-md md:grid-cols-2">
              {report.severityScores.map((s) => (
                <div key={s.instrument} className="flex flex-col justify-between rounded-xl border border-outline-variant bg-surface p-md">
                  <div className="mb-4 flex items-start justify-between">
                    <div>
                      <div className="mb-1 font-label-sm text-label-sm text-on-surface-variant">{s.label}</div>
                      <div className="font-headline-md text-headline-md font-bold text-on-surface">{s.instrument}</div>
                    </div>
                    <div
                      className={`rounded-full px-3 py-1 font-bold text-headline-sm ${
                        s.trendDirection === 'up' ? 'bg-error-container text-error' : 'bg-surface-dim text-on-surface'
                      }`}
                    >
                      {s.score}
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 flex justify-between font-label-sm text-label-sm">
                      <span className="text-on-surface-variant">{s.band}</span>
                      <span className="text-on-surface-variant">{s.bandRangeLabel}</span>
                    </div>
                    <div className="mb-2 h-2 w-full rounded-full bg-surface-container-highest">
                      <div
                        className={`h-2 rounded-full ${s.trendDirection === 'up' ? 'bg-error' : 'bg-surface-tint'}`}
                        style={{ width: `${s.pctOfMax}%` }}
                      />
                    </div>
                    <div className="flex items-center gap-1 text-body-sm text-on-surface-variant">
                      <MaterialIcon
                        name={TREND_ICON[s.trendDirection]}
                        className={`text-[16px] ${s.trendDirection === 'up' ? 'text-error' : 'text-primary'}`}
                      />
                      <span>{s.trendLabel}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Insights */}
          <div className="ai-glow mb-xl rounded-xl border border-tertiary-container bg-gradient-to-br from-tertiary-container/30 to-surface-bright p-lg">
            <div className="mb-md flex items-center gap-sm">
              <div className="rounded-lg bg-gradient-to-r from-tertiary to-secondary-container p-2 text-white">
                <MaterialIcon name="psychology" filled />
              </div>
              <h2 className="font-headline-sm text-headline-sm font-bold text-on-surface">CortexAI Insights</h2>
              {ragReport ? (
                <span className="ml-auto flex items-center gap-xs rounded-full border border-outline-variant bg-surface px-md py-xs font-label-sm text-label-sm text-on-surface-variant">
                  {ragReport.cached ? 'Templated summary' : `Generated by ${ragReport.generator}`}
                </span>
              ) : (
                <SampleDataBadge className="ml-auto" />
              )}
            </div>
            {ragReport ? (
              <div className="flex flex-col gap-md">
                <p className="whitespace-pre-line font-body-sm text-body-sm text-on-surface-variant">
                  {ragReport.narrative}
                </p>
                {ragReport.citations.length > 0 && (
                  <div className="border-t border-outline-variant pt-md">
                    <div className="mb-xs font-label-md text-label-md font-bold text-on-surface">Sources</div>
                    <ul className="flex flex-col gap-xs">
                      {ragReport.citations.map((citation) => (
                        <li key={citation} className="flex items-start gap-xs">
                          <MaterialIcon name="link" className="mt-[2px] text-[16px] text-tertiary" />
                          <span className="font-body-sm text-body-sm text-on-surface-variant">{citation}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {ragReport.fallback_reason && (
                  <p className="rounded-lg border border-outline-variant bg-surface-container-low px-md py-sm font-label-sm text-label-sm text-on-surface-variant">
                    <strong>Narrative not model-generated.</strong> {ragReport.fallback_reason}
                  </p>
                )}
                <p className="border-t border-outline-variant pt-sm font-label-sm text-label-sm italic text-on-surface-variant">
                  {ragReport.disclaimer}
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-md">
                {report.aiInsights.map((insight, i) => (
                  <div
                    key={insight.title}
                    className={`flex items-start gap-md ${i > 0 ? 'border-t border-outline-variant pt-md' : ''}`}
                  >
                    <MaterialIcon name={insight.icon} className="mt-1 text-tertiary" />
                    <div>
                      <div className="mb-1 font-label-md text-label-md font-bold text-on-surface">{insight.title}</div>
                      <p className="font-body-sm text-body-sm text-on-surface-variant">{insight.body}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mb-xl">
            <CrisisResourcesCard />
          </div>

          {/* Clinician Notes */}
          <div className="mb-lg">
            <h2 className="mb-md flex items-center gap-sm font-headline-sm text-headline-sm font-bold text-on-surface">
              <MaterialIcon name="edit_note" className="text-primary" />
              Clinician Notes &amp; Plan
            </h2>
            <div className="overflow-hidden rounded-lg border border-outline-variant bg-surface transition-all focus-within:border-primary focus-within:ring-1 focus-within:ring-primary">
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add diagnosis codes, treatment plan, and follow-up notes here..."
                className="min-h-[150px] w-full resize-y border-none bg-transparent p-4 font-body-md text-body-md text-on-surface focus:ring-0"
              />
            </div>
          </div>

          {/* Responsible-AI footer */}
          <div className="mb-xl flex items-start gap-sm rounded-lg bg-surface-container-low p-md">
            <MaterialIcon name="verified_user" className="mt-0.5 text-secondary" />
            <p className="font-body-sm text-body-sm text-on-surface-variant">{DECISION_SUPPORT_REPORT_FOOTER}</p>
          </div>

          {/* Signature Block */}
          <div className="mt-2xl flex justify-end">
            <div className="w-64">
              <div className="mb-2 flex h-16 items-end justify-center border-b-2 border-outline-variant">
                <span className="font-label-md text-label-md italic text-primary">Electronically Signed</span>
              </div>
              <div className="text-center font-label-sm text-label-sm text-on-surface-variant">
                {report.signedBy}
                <br />
                License #{report.licenseNumber}
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
