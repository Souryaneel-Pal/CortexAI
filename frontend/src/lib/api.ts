/**
 * Typed client for the CortexAI FastAPI backend (src/api/main.py).
 *
 * The backend is the source of truth for every number rendered as a real
 * result. Where a page has no live session it keeps rendering `mockData.ts`
 * behind a `SampleDataBadge` — the two are never blended, so a value on
 * screen is either clearly labelled sample data or a real model output.
 */

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:8000'

// The 18 features in docs/Dataset_Description.docx order.
export const TABULAR_FEATURE_COLUMNS = [
  'Sleep_Quality',
  'Social_Engagement',
  'Daily_App_Usage_Min',
  'Typing_Speed_WPM',
  'Session_Frequency',
  'Idle_Time_Min',
  'Facial_Emotion_Variance',
  'Eye_Blink_Rate',
  'Smile_Intensity',
  'Head_Motion_Index',
  'MFCC_Mean',
  'MFCC_Variance',
  'Pitch_Mean',
  'Speech_Rate',
  'Heart_Rate_BPM',
  'HRV_Index',
  'Skin_Temperature',
  'GSR_Level',
] as const

export type TabularFeatureName = (typeof TABULAR_FEATURE_COLUMNS)[number]
export type TabularFeatures = Record<TabularFeatureName, number>

/**
 * Per-column medians of the 4,000-row training table. Used only to fill
 * features the assessment form does not collect, so a partially-filled form
 * still produces a complete, in-distribution 18-feature vector rather than
 * zeros (which sit far outside every column's observed range and would make
 * the prediction meaningless).
 */
export const FEATURE_MEDIANS: TabularFeatures = {
  Sleep_Quality: 3,
  Social_Engagement: 3,
  Daily_App_Usage_Min: 248,
  Typing_Speed_WPM: 53,
  Session_Frequency: 10,
  Idle_Time_Min: 91,
  Facial_Emotion_Variance: 0.553,
  Eye_Blink_Rate: 22,
  Smile_Intensity: 0.501,
  Head_Motion_Index: 0.497,
  MFCC_Mean: -0.878,
  MFCC_Variance: 15.407,
  Pitch_Mean: 191.75,
  Speech_Rate: 4.02,
  Heart_Rate_BPM: 87,
  HRV_Index: 55.07,
  Skin_Temperature: 34.5,
  GSR_Level: 2.51,
}

export interface ModalityWeights {
  face: number
  speech: number
  tabular: number
}

export interface ScoreBreakdown {
  Depression_Score: number
  Anxiety_Score: number
  Stress_Score: number
}

export interface UncertaintyInfo {
  defer: boolean
  reason: string | null
  mc_dropout_confidence: number
}

export interface PredictionResponse {
  session_id: string
  predicted_class: string
  confidence: number
  class_probs: number[]
  scores: ScoreBreakdown
  modality_weights: ModalityWeights
  face_emotion_probs: Record<string, number>
  speech_emotion_probs: Record<string, number>
  uncertainty: UncertaintyInfo
  deferred_to_human: boolean
  is_demo_untrained_model: boolean
  disclaimer: string
}

export interface GradCAMPayload {
  overlay_png_base64: string
  heatmap: number[][]
  target_layer: string
  predicted_emotion: string
}

export interface AudioIGPayload {
  frame_importance: number[]
  frame_ms: number
  predicted_emotion: string
}

export interface MaskedDistressIndex {
  mdi: number
  flag: boolean
  face_calm?: number
  voice_high_arousal?: number
  physio_high_arousal?: number
  dominant_contradiction?: 'voice' | 'physiology' | null
  unavailable_reason?: string
}

export interface ExplanationResponse {
  session_id: string
  top_shap_features: { feature: string; mean_abs_shap: number }[]
  signed_shap: { feature: string; shap: number }[]
  modality_weights: ModalityWeights
  masked_distress_index: MaskedDistressIndex | null
  gradcam: GradCAMPayload | null
  audio_integrated_gradients: AudioIGPayload | null
  is_demo_untrained_model: boolean
  disclaimer: string
}

export interface ReportResponse {
  session_id: string
  narrative: string
  citations: string[]
  cached: boolean
  /** 'ollama:llama3.1' | 'anthropic:<model>' | 'template' */
  generator: string
  /** Set when the local LLM was unavailable and a templated summary was served. */
  fallback_reason: string | null
  disclaimer: string
}

export interface FollowUpResponse {
  session_id: string
  answer: string
  citations: string[]
  cached: boolean
  generator: string
  fallback_reason: string | null
  disclaimer: string
}

/** State of the local Ollama reasoning stack, reported by /health. */
export interface ReasoningStatus {
  ollama_reachable: boolean
  base_url: string
  llm_model: string
  llm_available: boolean
  embedding_model: string
  embedding_available: boolean
  retrieval_backend: string | null
  vector_index: string | null
  detail: string | null
}

export interface HealthResponse {
  status: string
  is_demo_untrained_model: boolean | null
  reasoning?: ReasoningStatus
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = sessionStorage.getItem('auth_token')
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...headers,
      ...init?.headers,
    },
  })
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`${init?.method ?? 'GET'} ${path} failed (${response.status}): ${body}`)
  }
  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function postAssessment(payload: {
  tabular_features: TabularFeatures
  face_image_base64?: string | null
  speech_audio_base64?: string | null
  patient_id?: string | null
  demographic?: string | null
}): Promise<PredictionResponse> {
  // The backend registers this handler at both /predict and /assess.
  return request<PredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getExplanation(sessionId: string): Promise<ExplanationResponse> {
  return request<ExplanationResponse>(`/explain/${sessionId}`)
}

export function getReport(sessionId: string): Promise<ReportResponse> {
  return request<ReportResponse>(`/report/${sessionId}`)
}

export function postFollowUp(sessionId: string, question: string): Promise<FollowUpResponse> {
  return request<FollowUpResponse>('/follow-up', {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, question }),
  })
}

/** Strips the `data:<mime>;base64,` prefix a FileReader data URL carries. */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('Could not read file'))
    reader.onload = () => {
      const result = String(reader.result)
      resolve(result.slice(result.indexOf(',') + 1))
    }
    reader.readAsDataURL(file)
  })
}

export interface UserInfo {
  email: string
  name: string
  role: 'Admin' | 'Clinician'
}

export interface LoginResponse {
  token: string
  user: UserInfo
}

export interface SettingsState {
  uncertainty_threshold: number
  mdi_threshold: number
  ignore_face: boolean
  ignore_speech: boolean
  ignore_tabular: boolean
}

export interface DashboardData {
  totalAssessments: number
  heroStats: { label: string; value: string; meta: string; accentClassName?: string }[]
  stressDistribution: { label: string; severity: string; pct: number }[]
  trend: { month: string; assessments: number }[]
  recentAssessments: { patientId: string; dateTime: string; riskLevel: string; status: string }[]
}

export interface AnalyticsData {
  kpis: { label: string; value: string; delta?: string; deltaDirection?: string; sub?: string; isAi?: boolean }[]
  heatmap: { day: string; hour: string; intensity: number }[]
  emotionFrequency: { emotion: string; value: number }[]
  riskBreakdown: { segment: string; low: number; medium: number; high: number }[]
  correlation: { rowLabel: string; colLabel: string; value: number }[]
}

export function loginUser(payload: { email: string; password: string }): Promise<LoginResponse> {
  return request<LoginResponse>('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getSettings(): Promise<SettingsState> {
  return request<SettingsState>('/api/settings')
}

export function putSettings(payload: SettingsState): Promise<{ status: string }> {
  return request<{ status: string }>('/api/settings', {
    method: 'PUT',
    body: JSON.stringify(payload),
  })
}

export function getDashboard(): Promise<DashboardData> {
  return request<DashboardData>('/api/dashboard')
}

export function getAnalytics(): Promise<AnalyticsData> {
  return request<AnalyticsData>('/api/analytics')
}
