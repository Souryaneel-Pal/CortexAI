/**
 * Domain types for the CortexAI frontend.
 *
 * These are shaped to match what the real `/predict`, `/explain` and `/report`
 * FastAPI endpoints (src/api/, built in parallel — see PROJECT_PLAN.md P5) are
 * expected to return, so that swapping the mock data layer (src/lib/mockData.ts)
 * for real API calls later is a data-source change, not a component rewrite.
 */

export type Severity = 'Healthy' | 'Mild' | 'Moderate' | 'Severe'

export interface KpiStat {
  label: string
  value: string
  meta: string
  accentClassName?: string
}

export interface StressDistributionSlice {
  label: string
  severity: Severity
  pct: number
}

export interface TrendPoint {
  month: string
  assessments: number
}

export interface RecentAssessmentRow {
  patientId: string
  dateTime: string
  riskLevel: Severity
  status: 'AI Reviewing' | 'Completed'
}

export interface DashboardData {
  heroStats: KpiStat[]
  stressDistribution: StressDistributionSlice[]
  totalAssessments: number
  trend: TrendPoint[]
  recentAssessments: RecentAssessmentRow[]
}

export interface ModalityContribution {
  modality: 'Physiological' | 'Facial' | 'Behavioral' | 'Speech'
  pct: number
  note?: string
}

export interface ShapFeature {
  feature: string
  value: number // signed contribution, e.g. +0.85 / -0.25
}

export interface ModalityBreakdownRow {
  modality: string
  keyIndicator: string
  deviation: string
  clinicalRelevance: string
}

export interface AcousticMetric {
  label: string
  value: string
  pct: number
  colorToken: 'primary' | 'secondary' | 'tertiary'
}

export interface MicroExpressionPoint {
  minute: number
  intensityPct: number
}

export interface LongitudinalMetric {
  label: string
  current: number
  baseline: number
}

export interface ExplainabilityData {
  patientId: string
  assessmentDate: string
  summary: {
    headline: string
    driverHighlights: { text: string; colorToken: 'primary' | 'secondary' }[]
    body: string
  }
  shap: ShapFeature[]
  modalityContribution: ModalityContribution[]
  breakdown: ModalityBreakdownRow[]
  acoustic: AcousticMetric[]
  microExpressionTimeline: MicroExpressionPoint[]
  longitudinal: LongitudinalMetric[]
}

export interface SeverityScore {
  instrument: 'PHQ-9' | 'GAD-7'
  label: string
  score: number
  maxScore: number
  band: string
  bandRangeLabel: string
  pctOfMax: number
  trendLabel: string
  trendDirection: 'up' | 'down' | 'flat'
}

export interface AiReportInsight {
  icon: string
  title: string
  body: string
}

export interface ClinicalReportData {
  patient: {
    name: string
    id: string
    dob: string
    age: number
  }
  assessmentDate: string
  clinician: string
  sessionType: string
  duration: string
  assessmentTools: string
  clinicalSummary: string
  severityScores: SeverityScore[]
  aiInsights: AiReportInsight[]
  clinicianNotes: string
  signedBy: string
  licenseNumber: string
}

export interface PopulationKpi {
  label: string
  value: string
  delta?: string
  deltaDirection?: 'up' | 'down'
  sub?: string
  isAi?: boolean
}

export interface HeatmapCell {
  day: string
  hour: string
  intensity: number // 0..1
}

export interface EmotionFrequencyPoint {
  emotion: string
  value: number // 0..100
}

export interface RiskBreakdownRow {
  segment: string
  low: number
  medium: number
  high: number
}

export interface CorrelationCell {
  rowLabel: string
  colLabel: string
  value: number // -1..1
}

export interface PopulationAnalyticsData {
  kpis: PopulationKpi[]
  heatmap: HeatmapCell[]
  emotionFrequency: EmotionFrequencyPoint[]
  riskBreakdown: RiskBreakdownRow[]
  correlation: CorrelationCell[]
}

export interface AssessmentDraft {
  Sleep_Quality: number | ''
  Social_Engagement: number | ''
  Heart_Rate_BPM: number | ''
  GSR_Level: number | ''
  Eye_Blink_Rate: number | ''
}
