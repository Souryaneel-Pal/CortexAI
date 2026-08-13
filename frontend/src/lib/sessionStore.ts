/**
 * Holds the most recent live assessment so the Explainable-Insights page can
 * render it after `NewAssessment` navigates away.
 *
 * Backed by sessionStorage rather than a router param or a context provider:
 * it survives the page reload a dev server does on hot-reload, is scoped to
 * the tab (so a shared machine doesn't leak one clinician's session into
 * another's), and is cleared on tab close. Nothing here is persisted to disk
 * or sent anywhere — the face image and audio clip themselves are never
 * stored, only the model's outputs.
 */
import type { ExplanationResponse, PredictionResponse } from './api'

const STORAGE_KEY = 'cortexai.lastAssessment'

export interface StoredAssessment {
  prediction: PredictionResponse
  explanation: ExplanationResponse | null
  /** Data URL of the submitted face, kept in-tab only so the Grad-CAM panel
   *  can show the original beside the heatmap overlay. */
  faceImageDataUrl: string | null
  completedAt: string
}

export function saveAssessment(assessment: StoredAssessment): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(assessment))
  } catch {
    // Quota exceeded (a large face image) is not worth failing the flow over —
    // the assessment already succeeded; only the insights hand-off is lost.
  }
}

export function loadAssessment(): StoredAssessment | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as StoredAssessment) : null
  } catch {
    return null
  }
}

export function clearAssessment(): void {
  sessionStorage.removeItem(STORAGE_KEY)
}
