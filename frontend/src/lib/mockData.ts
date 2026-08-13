/**
 * Mock / sample data layer.
 *
 * There is no backend yet (src/api/ is being built in parallel — see
 * PROJECT_PLAN.md P5). Every page reads from here instead of hardcoding
 * numbers inline, so wiring the real `/predict`, `/explain` and `/report`
 * endpoints later is a matter of swapping these functions for `fetch` calls,
 * not restructuring components.
 *
 * IMPORTANT — responsible-AI framing: nothing here is a real computed result.
 * Values are illustrative sample data carried over from the approved static
 * designs (docs/frontend_design/*\/code.html) for visual fidelity. Every page
 * that reads from this module surfaces a `SampleDataBadge` so numbers are
 * never presented as if they were a real diagnosis or a real model output.
 */
import type {
  AcousticMetric,
  ClinicalReportData,
  CorrelationCell,
  DashboardData,
  EmotionFrequencyPoint,
  ExplainabilityData,
  HeatmapCell,
  LongitudinalMetric,
  MicroExpressionPoint,
  PopulationAnalyticsData,
  RiskBreakdownRow,
} from '../types'

export const SAMPLE_DATA_NOTE =
  'Sample data — illustrative only, not a real patient result.'

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
export function getDashboardData(): DashboardData {
  return {
    totalAssessments: 1248,
    heroStats: [
      { label: 'Total Assessments', value: '1,248', meta: '+12% this month' },
      { label: 'Healthy Cases', value: '856', meta: '68.5% of total', accentClassName: 'border-l-secondary' },
      { label: 'Mild Stress', value: '214', meta: '17.1% of total', accentClassName: 'border-l-primary-container' },
      { label: 'Moderate Stress', value: '132', meta: '10.5% of total', accentClassName: 'border-l-tertiary' },
      { label: 'Severe Stress', value: '46', meta: '3.6% of total', accentClassName: 'border-l-error' },
    ],
    stressDistribution: [
      { label: 'Healthy', severity: 'Healthy', pct: 68 },
      { label: 'Mild', severity: 'Mild', pct: 17 },
      { label: 'Mod.', severity: 'Moderate', pct: 11 },
      { label: 'Severe', severity: 'Severe', pct: 4 },
    ],
    trend: [
      { month: 'Jul', assessments: 60 },
      { month: 'Aug', assessments: 120 },
      { month: 'Sep', assessments: 75 },
      { month: 'Oct', assessments: 210 },
      { month: 'Nov', assessments: 165 },
      { month: 'Dec', assessments: 240 },
    ],
    recentAssessments: [
      { patientId: 'PT-8842', dateTime: 'Oct 24, 2023 · 09:15 AM', riskLevel: 'Severe', status: 'AI Reviewing' },
      { patientId: 'PT-9105', dateTime: 'Oct 24, 2023 · 08:30 AM', riskLevel: 'Moderate', status: 'Completed' },
      { patientId: 'PT-7721', dateTime: 'Oct 23, 2023 · 04:45 PM', riskLevel: 'Mild', status: 'Completed' },
      { patientId: 'PT-6540', dateTime: 'Oct 23, 2023 · 02:10 PM', riskLevel: 'Healthy', status: 'Completed' },
    ],
  }
}

// ---------------------------------------------------------------------------
// Explainable AI Insights (base + expanded)
// ---------------------------------------------------------------------------
const baseBreakdown = [
  {
    modality: 'Physiological',
    baseKeyIndicator: 'Heart Rate Var.',
    expandedKeyIndicator: 'HRV HF/LF Ratio: 1.8',
    deviation: '+15% above norm',
    clinicalRelevance: 'Strong indicator of acute stress response during Q4-Q7.',
  },
  {
    modality: 'Facial',
    baseKeyIndicator: 'Micro-tremors (AU12)',
    expandedKeyIndicator: 'Micro-tremors (AU12)',
    deviation: 'High frequency',
    clinicalRelevance: 'Correlates with suppressed anxiety markers.',
  },
  {
    modality: 'Behavioral',
    baseKeyIndicator: 'Sleep Quality',
    expandedKeyIndicator: 'EDA Peaks: 12/min',
    deviation: '-2 hrs avg/week',
    clinicalRelevance: 'Self-reported disruption aligning with physiological stress.',
  },
  {
    modality: 'Speech',
    baseKeyIndicator: 'Fluency',
    expandedKeyIndicator: 'Latency: 850ms',
    deviation: 'Normal range',
    clinicalRelevance: 'No significant indicators detected in vocal patterns.',
  },
]

export function getExplainabilityData(expanded: boolean): ExplainabilityData {
  return {
    patientId: 'CX-992-A',
    assessmentDate: 'Oct 24, 2023',
    summary: {
      headline: 'high likelihood of generalized anxiety indicators',
      driverHighlights: [
        { text: 'elevated resting heart rate anomalies (34% impact)', colorToken: 'primary' },
        { text: 'consistent micro-tremors detected in facial video analysis (28% impact)', colorToken: 'secondary' },
      ],
      body: 'Speech patterns showed minor, but notable, hesitation metrics.',
    },
    shap: [
      { feature: 'Heart Rate Variance', value: 0.85 },
      { feature: 'Facial Micro-tremors', value: 0.65 },
      { feature: 'Sleep Quality (Self-Rep)', value: 0.45 },
      { feature: 'Speech Fluency', value: -0.25 },
      { feature: 'Galvanic Skin Resp.', value: 0.2 },
    ],
    modalityContribution: [
      { modality: 'Physiological', pct: 40 },
      { modality: 'Facial', pct: 30 },
      { modality: 'Behavioral', pct: 20 },
      { modality: 'Speech', pct: 10 },
    ],
    breakdown: baseBreakdown.map((row) => ({
      modality: row.modality,
      keyIndicator: expanded ? row.expandedKeyIndicator : row.baseKeyIndicator,
      deviation: row.deviation,
      clinicalRelevance: row.clinicalRelevance,
    })),
    acoustic: ACOUSTIC_METRICS,
    microExpressionTimeline: MICRO_EXPRESSION_TIMELINE,
    longitudinal: LONGITUDINAL_METRICS,
  }
}

const ACOUSTIC_METRICS: AcousticMetric[] = [
  { label: 'Jitter (Local)', value: '1.24%', pct: 65, colorToken: 'primary' },
  { label: 'Shimmer (APQ3)', value: '0.38 dB', pct: 42, colorToken: 'secondary' },
  { label: 'Pitch Variability', value: '42.5 Hz', pct: 58, colorToken: 'tertiary' },
]

const MICRO_EXPRESSION_TIMELINE: MicroExpressionPoint[] = [20, 45, 80, 30, 95, 60, 40, 25].map(
  (intensityPct, i) => ({ minute: i + 1, intensityPct }),
)

const LONGITUDINAL_METRICS: LongitudinalMetric[] = [
  { label: 'Anxiety Index', current: 72, baseline: 45 },
  { label: 'Stress Resilience', current: 38, baseline: 60 },
]

// ---------------------------------------------------------------------------
// Clinical Report
// ---------------------------------------------------------------------------
export function getClinicalReportData(): ClinicalReportData {
  return {
    patient: { name: 'Eleanor Vance', id: 'CX-88924', dob: '04/12/1985', age: 38 },
    assessmentDate: 'Oct 24, 2023',
    clinician: 'Dr. Julian Vance',
    sessionType: 'Telehealth (Video)',
    duration: '45 Minutes',
    assessmentTools: 'PHQ-9, GAD-7',
    clinicalSummary:
      'Patient presents with persistent low mood, anhedonia, and difficulty concentrating, ongoing for approximately 3 months. She reports poor sleep quality (early morning awakening) and decreased appetite resulting in a 5lb weight loss over the past month. Denies acute suicidal ideation or intent. Anxiety symptoms are secondary, primarily related to occupational stress.',
    severityScores: [
      {
        instrument: 'PHQ-9',
        label: 'Depression Severity',
        score: 18,
        maxScore: 27,
        band: 'Moderately Severe',
        bandRangeLabel: 'Range: 15-19',
        pctOfMax: 70,
        trendLabel: '+3 points since last assessment (Sep 15)',
        trendDirection: 'up',
      },
      {
        instrument: 'GAD-7',
        label: 'Anxiety Severity',
        score: 11,
        maxScore: 21,
        band: 'Moderate',
        bandRangeLabel: 'Range: 10-14',
        pctOfMax: 50,
        trendLabel: 'Unchanged since last assessment',
        trendDirection: 'flat',
      },
    ],
    aiInsights: [
      {
        icon: 'lightbulb',
        title: 'Symptom Cluster Analysis',
        body: 'Linguistic analysis of the session transcript indicates a high frequency of words associated with lethargy and cognitive fog, correlating strongly with the somatic symptoms reported in the PHQ-9. The AI model flags a 78% confidence interval that somatic symptoms are driving the overall severity increase.',
      },
      {
        icon: 'warning',
        title: 'Risk Stratification',
        body: 'While explicit SI is denied, the trajectory of anhedonia combined with recent sleep disturbances suggests escalating risk. Recommendation: Shortened follow-up interval (2 weeks vs standard 4 weeks).',
      },
    ],
    clinicianNotes: `1. Diagnosis: Major Depressive Disorder, recurrent, moderate to severe (F33.1).
2. Plan: Initiate Escitalopram 10mg daily. Discussed potential side effects (GI upset, sleep changes) and black box warning.
3. Recommend behavioral activation strategies; provided handout on sleep hygiene.
4. Follow-up: Scheduled in 2 weeks to monitor medication tolerability and symptom response per AI insight recommendation.`,
    signedBy: 'Dr. Julian Vance, MD',
    licenseNumber: 'ME-48291',
  }
}

// ---------------------------------------------------------------------------
// Population Analytics
// ---------------------------------------------------------------------------
const HEATMAP_ROWS: number[][] = [
  [0.1, 0.2, 0.4, 0.6, 0.3, 0.6, 0.8, 0.5, 0.3, 0.1, 0.1, 0.05],
  [0.05, 0.1, 0.3, 0.5, 0.4, 0.7, 0.9, 0.4, 0.4, 0.2, 0.1, 0.05],
  [0.1, 0.2, 0.4, 0.6, 0.2, 0.5, 0.7, 0.6, 0.3, 0.1, 0.05, 0.05],
  [0.2, 0.3, 0.5, 0.8, 0.5, 0.8, 0.9, 0.5, 0.5, 0.2, 0.1, 0.05],
  [0.1, 0.1, 0.2, 0.4, 0.7, 0.4, 0.6, 0.5, 0.2, 0.1, 0.05, 0.05],
  [0.05, 0.05, 0.1, 0.2, 0.4, 0.5, 0.4, 0.2, 0.1, 0.05, 0.05, 0.05],
  [0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05],
]
const HEATMAP_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const HEATMAP_HOURS = ['2A', '4A', '6A', '8A', '10A', '12P', '2P', '4P', '6P', '8P', '10P', '12A']

function buildHeatmap(): HeatmapCell[] {
  const cells: HeatmapCell[] = []
  HEATMAP_ROWS.forEach((row, dayIdx) => {
    row.forEach((intensity, hourIdx) => {
      cells.push({ day: HEATMAP_DAYS[dayIdx], hour: HEATMAP_HOURS[hourIdx], intensity })
    })
  })
  return cells
}

const EMOTION_FREQUENCY: EmotionFrequencyPoint[] = [
  { emotion: 'Anxiety', value: 88 },
  { emotion: 'Agitation', value: 62 },
  { emotion: 'Fatigue', value: 74 },
  { emotion: 'Apathy', value: 50 },
  { emotion: 'Sadness', value: 58 },
  { emotion: 'Anger', value: 40 },
]

const RISK_BREAKDOWN: RiskBreakdownRow[] = [
  { segment: 'Adults', low: 60, medium: 25, high: 15 },
  { segment: 'Seniors', low: 45, medium: 35, high: 20 },
  { segment: 'Adolescents', low: 70, medium: 20, high: 10 },
]

const CORRELATION: CorrelationCell[] = [
  { rowLabel: 'Stress', colLabel: 'HRV', value: 0.89 },
  { rowLabel: 'Stress', colLabel: 'EDA', value: 0.72 },
  { rowLabel: 'Stress', colLabel: 'Temp', value: 0.15 },
  { rowLabel: 'Sleep', colLabel: 'HRV', value: -0.65 },
  { rowLabel: 'Sleep', colLabel: 'EDA', value: -0.32 },
  { rowLabel: 'Sleep', colLabel: 'Temp', value: 0.05 },
  { rowLabel: 'Activity', colLabel: 'HRV', value: 0.42 },
  { rowLabel: 'Activity', colLabel: 'EDA', value: 0.58 },
  { rowLabel: 'Activity', colLabel: 'Temp', value: 0.28 },
]

export function getPopulationAnalyticsData(): PopulationAnalyticsData {
  return {
    kpis: [
      { label: 'Active Patients', value: '1,248', delta: '+12%', deltaDirection: 'up' },
      { label: 'Critical Alerts', value: '24', delta: '+3', deltaDirection: 'up', sub: 'Requiring immediate review' },
      { label: 'AI Efficacy Score', value: '94%', sub: 'Prediction accuracy this month', isAi: true },
      { label: 'Avg. Stress Index', value: '6.8', sub: '/ 10' },
    ],
    heatmap: buildHeatmap(),
    emotionFrequency: EMOTION_FREQUENCY,
    riskBreakdown: RISK_BREAKDOWN,
    correlation: CORRELATION,
  }
}
