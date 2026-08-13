/**
 * Global assessment state.
 *
 * One React Context owns the whole live-result lifecycle — running an
 * assessment, the in-flight flag, the error, and the resulting
 * prediction / explanation / report — so every page reads the same source of
 * truth instead of each re-reading sessionStorage on mount.
 *
 * Why Context rather than a store library: the app has exactly one piece of
 * cross-page state (the current assessment) and five routes. Context costs no
 * dependency and no bundle weight, and the provider still persists to
 * sessionStorage so a hard refresh or a dev-server hot reload doesn't lose the
 * result mid-demo.
 *
 * Responsible-AI note: pages must render *either* live values *or* the clearly
 * labelled sample data, never a mix. `hasLiveResult` is the single flag every
 * page branches on, so that rule is enforced in one place.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  fileToBase64,
  getExplanation,
  getHealth,
  getReport,
  postAssessment,
  type ExplanationResponse,
  type HealthResponse,
  type PredictionResponse,
  type ReportResponse,
  type TabularFeatures,
} from './api'
import { clearAssessment, loadAssessment, saveAssessment, type StoredAssessment } from './sessionStore'

export interface RunAssessmentInput {
  tabularFeatures: TabularFeatures
  faceFile: File | null
  speechFile: File | null
  patientId?: string
  demographic?: string
}

/** Which stage the pipeline is on, so the UI can narrate progress honestly. */
export type AssessmentPhase = 'idle' | 'uploading' | 'predicting' | 'explaining' | 'reporting' | 'done' | 'error'

const PHASE_LABEL: Record<AssessmentPhase, string> = {
  idle: '',
  uploading: 'Encoding media...',
  predicting: 'Running multimodal inference...',
  explaining: 'Computing SHAP, Grad-CAM and MDI...',
  reporting: 'Generating the grounded report...',
  done: 'Assessment complete',
  error: 'Assessment failed',
}

interface AssessmentContextValue {
  prediction: PredictionResponse | null
  explanation: ExplanationResponse | null
  report: ReportResponse | null
  faceImageDataUrl: string | null
  completedAt: string | null
  /** True only when a real backend result is loaded — the flag pages branch on. */
  hasLiveResult: boolean

  running: boolean
  phase: AssessmentPhase
  phaseLabel: string
  error: string | null

  health: HealthResponse | null
  backendReachable: boolean | null

  runAssessment: (input: RunAssessmentInput) => Promise<PredictionResponse | null>
  generateReport: (sessionId: string) => Promise<ReportResponse | null>
  clear: () => void
  dismissError: () => void
}

const AssessmentContext = createContext<AssessmentContextValue | null>(null)

function toMessage(error: unknown): string {
  if (error instanceof TypeError) {
    // fetch() rejects with TypeError when it cannot reach the host at all.
    return 'Cannot reach the CortexAI API. Start it with `uvicorn src.api.main:app` and retry.'
  }
  return error instanceof Error ? error.message : String(error)
}

export function AssessmentProvider({ children }: { children: ReactNode }) {
  const [stored, setStored] = useState<StoredAssessment | null>(null)
  const [report, setReport] = useState<ReportResponse | null>(null)
  const [phase, setPhase] = useState<AssessmentPhase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null)

  // Rehydrate any assessment from earlier in this tab, and probe the backend
  // once so pages can show connection state before anything is submitted.
  useEffect(() => {
    setStored(loadAssessment())
    let cancelled = false
    getHealth()
      .then((h) => {
        if (cancelled) return
        setHealth(h)
        setBackendReachable(true)
      })
      .catch(() => {
        if (!cancelled) setBackendReachable(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const runAssessment = useCallback(async (input: RunAssessmentInput) => {
    setPhase('uploading')
    setError(null)
    setReport(null)
    try {
      const [faceBase64, speechBase64] = await Promise.all([
        input.faceFile ? fileToBase64(input.faceFile) : Promise.resolve(null),
        input.speechFile ? fileToBase64(input.speechFile) : Promise.resolve(null),
      ])

      setPhase('predicting')
      const prediction = await postAssessment({
        tabular_features: input.tabularFeatures,
        face_image_base64: faceBase64,
        speech_audio_base64: speechBase64,
        patient_id: input.patientId,
        demographic: input.demographic,
      })

      // Explanations and the report are fetched separately and are allowed to
      // fail independently: a SHAP or LLM problem must not discard a
      // prediction the user already waited for.
      setPhase('explaining')
      let explanation: ExplanationResponse | null = null
      try {
        explanation = await getExplanation(prediction.session_id)
      } catch (explainError) {
        console.warn('Explanation unavailable for this session', explainError)
      }

      const next: StoredAssessment = {
        prediction,
        explanation,
        faceImageDataUrl: faceBase64
          ? `data:${input.faceFile?.type || 'image/png'};base64,${faceBase64}`
          : null,
        completedAt: new Date().toISOString(),
      }
      setStored(next)
      setReport(null)
      saveAssessment(next)
      setPhase('done')
      return prediction
    } catch (assessError) {
      setError(toMessage(assessError))
      setPhase('error')
      return null
    }
  }, [])

  const generateReport = useCallback(async (sessionId: string) => {
    setPhase('reporting')
    setError(null)
    try {
      const reportResult = await getReport(sessionId)
      setReport(reportResult)
      setPhase('done')
      return reportResult
    } catch (reportError) {
      setError(toMessage(reportError))
      setPhase('error')
      return null
    }
  }, [])

  const clear = useCallback(() => {
    clearAssessment()
    setStored(null)
    setReport(null)
    setPhase('idle')
    setError(null)
  }, [])

  const value = useMemo<AssessmentContextValue>(
    () => ({
      prediction: stored?.prediction ?? null,
      explanation: stored?.explanation ?? null,
      report,
      faceImageDataUrl: stored?.faceImageDataUrl ?? null,
      completedAt: stored?.completedAt ?? null,
      hasLiveResult: stored?.prediction != null,
      running: phase === 'uploading' || phase === 'predicting' || phase === 'explaining' || phase === 'reporting',
      phase,
      phaseLabel: PHASE_LABEL[phase],
      error,
      health,
      backendReachable,
      runAssessment,
      generateReport,
      clear,
      dismissError: () => setError(null),
    }),
    [stored, report, phase, error, health, backendReachable, runAssessment, generateReport, clear],
  )

  return <AssessmentContext.Provider value={value}>{children}</AssessmentContext.Provider>
}

export function useAssessment(): AssessmentContextValue {
  const context = useContext(AssessmentContext)
  if (!context) {
    throw new Error('useAssessment must be used inside an <AssessmentProvider>')
  }
  return context
}

/**
 * Lazily fetch the grounded report for the current session if the provider
 * doesn't already hold one (e.g. the user landed on Reports after a refresh).
 */
export function useReportForCurrentSession(): ReportResponse | null {
  const { prediction, report } = useAssessment()
  const [lazy, setLazy] = useState<ReportResponse | null>(null)

  useEffect(() => {
    if (report || !prediction) return
    let cancelled = false
    getReport(prediction.session_id)
      .then((r) => {
        if (!cancelled) setLazy(r)
      })
      .catch(() => {
        /* Sample narrative stays on screen; not worth an error toast. */
      })
    return () => {
      cancelled = true
    }
  }, [prediction, report])

  return report ?? lazy
}
