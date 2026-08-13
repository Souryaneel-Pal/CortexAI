import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppShell } from '../components/layout/AppShell'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import { useAssessment } from '../lib/assessmentContext'
import { FEATURE_MEDIANS } from '../lib/api'
import type { AssessmentDraft } from '../types'

const INITIAL_DRAFT: AssessmentDraft = {
  Sleep_Quality: '',
  Social_Engagement: '',
  Heart_Rate_BPM: '',
  GSR_Level: '',
  Eye_Blink_Rate: '',
}

export function NewAssessment() {
  const navigate = useNavigate()
  const [draft, setDraft] = useState<AssessmentDraft>(INITIAL_DRAFT)
  const [faceFile, setFaceFile] = useState<File | null>(null)
  const [speechFile, setSpeechFile] = useState<File | null>(null)
  const [patientId, setPatientId] = useState('')
  const [demographic, setDemographic] = useState('Adults')

  const { runAssessment, running, phaseLabel, error, backendReachable, health, dismissError } = useAssessment()

  const faceInputRef = useRef<HTMLInputElement>(null)
  const speechInputRef = useRef<HTMLInputElement>(null)

  const faceCaptured = faceFile !== null
  const speechCaptured = speechFile !== null
  const modelsReady = backendReachable === true
  const reasoning = health?.reasoning

  function update<K extends keyof AssessmentDraft>(key: K, value: number | '') {
    setDraft((d) => ({ ...d, [key]: value }))
  }

  async function handleRunAssessment() {
    // Overwrite the 5 user-provided features onto the pre-computed medians
    const tabularFeatures = {
      ...FEATURE_MEDIANS,
      Sleep_Quality: draft.Sleep_Quality === '' ? FEATURE_MEDIANS.Sleep_Quality : Number(draft.Sleep_Quality),
      Social_Engagement: draft.Social_Engagement === '' ? FEATURE_MEDIANS.Social_Engagement : Number(draft.Social_Engagement),
      Heart_Rate_BPM: draft.Heart_Rate_BPM === '' ? FEATURE_MEDIANS.Heart_Rate_BPM : Number(draft.Heart_Rate_BPM),
      GSR_Level: draft.GSR_Level === '' ? FEATURE_MEDIANS.GSR_Level : Number(draft.GSR_Level),
      Eye_Blink_Rate: draft.Eye_Blink_Rate === '' ? FEATURE_MEDIANS.Eye_Blink_Rate : Number(draft.Eye_Blink_Rate),
    }

    const prediction = await runAssessment({
      tabularFeatures,
      faceFile,
      speechFile,
      patientId: patientId.trim() || undefined,
      demographic: demographic || undefined,
    })
    if (prediction) navigate('/results')
  }

  return (
    <AppShell>
      <div className="flex flex-col gap-xs mb-xl">
        <h1 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface">
          New Screening Session
        </h1>
        <p className="max-w-3xl font-body-lg text-body-lg leading-relaxed text-on-surface-variant">
          Input patient demographic, behavioral, physiological, and audiovisual metrics to compute the cross-modal distress assessment index.
        </p>
      </div>

      <div className="mt-xl grid grid-cols-1 gap-lg lg:grid-cols-3">
        {/* Step 1 & 2: Media Upload Column */}
        <div className="flex flex-col gap-lg lg:col-span-1">
          {/* Patient Details */}
          <section className="flex flex-col gap-md rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                P
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Demographics & Identifiers</h2>
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Provide the unique clinical identifier and demographic segment for archiving.
            </p>
            <div className="flex flex-col gap-md">
              <label className="flex flex-col gap-xs">
                <span className="font-label-sm text-label-sm text-on-surface-variant">Patient ID</span>
                <input
                  type="text"
                  placeholder="e.g., PT-8842"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                />
              </label>
              <label className="flex flex-col gap-xs">
                <span className="font-label-sm text-label-sm text-on-surface-variant">Demographic Segment</span>
                <select
                  value={demographic}
                  onChange={(e) => setDemographic(e.target.value)}
                  className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="Adults">Adults</option>
                  <option value="Seniors">Seniors</option>
                  <option value="Adolescents">Adolescents</option>
                </select>
              </label>
            </div>
          </section>

          {/* Step 1: Facial Analysis Upload */}
          <section className="group relative flex flex-col gap-md overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                1
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Facial Feature Upload</h2>
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Upload a clear portrait photo for visual expression and eye-activity profiling.
            </p>
            <input
              ref={faceInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => setFaceFile(e.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              onClick={() => faceInputRef.current?.click()}
              className="flex min-h-[160px] flex-col items-center justify-center gap-md rounded-lg border-2 border-dashed border-outline-variant bg-surface-container-low p-xl text-center transition-colors hover:bg-surface-container group-hover:border-primary-fixed-dim"
            >
              <MaterialIcon name="image" className="text-4xl text-outline" />
              <div>
                <p className="font-body-sm text-body-sm text-on-surface max-w-[200px] truncate">
                  {faceCaptured ? faceFile?.name : 'Upload JPEG or PNG file'}
                </p>
                <p className="mt-xs font-label-sm text-label-sm text-on-surface-variant">or drag it here</p>
              </div>
              <span className="rounded-md border border-outline-variant bg-surface-container-highest px-md py-sm font-label-md text-label-md text-on-surface transition-all hover:bg-primary hover:text-on-primary">
                {faceCaptured ? 'Replace Image' : 'Browse Image'}
              </span>
            </button>
          </section>

          {/* Step 2: Speech Analysis Upload */}
          <section className="group relative flex flex-col gap-md overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                2
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Voice Recording Upload</h2>
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant">
              Upload a short voice sample (.wav format, 16kHz preferred) for prosodic acoustic analysis.
            </p>
            <input
              ref={speechInputRef}
              type="file"
              accept="audio/*,.wav"
              className="hidden"
              onChange={(e) => setSpeechFile(e.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              onClick={() => speechInputRef.current?.click()}
              className="flex min-h-[160px] flex-col items-center justify-center gap-md rounded-lg border-2 border-dashed border-outline-variant bg-surface-container-low p-xl text-center transition-colors hover:bg-surface-container group-hover:border-primary-fixed-dim"
            >
              <MaterialIcon name="mic" className="text-4xl text-outline" />
              <div>
                <p className="font-body-sm text-body-sm text-on-surface max-w-[200px] truncate">
                  {speechCaptured ? speechFile?.name : 'Upload WAV audio file'}
                </p>
                <p className="mt-xs font-label-sm text-label-sm text-on-surface-variant">or drag it here</p>
              </div>
              <span className="rounded-md border border-outline-variant bg-surface-container-highest px-md py-sm font-label-md text-label-md text-on-surface transition-all hover:bg-primary hover:text-on-primary">
                {speechCaptured ? 'Replace Audio' : 'Browse Audio'}
              </span>
            </button>
          </section>
        </div>

        {/* Step 3: 5 Key Clinical Features Form */}
        <div className="lg:col-span-2">
          <section className="flex flex-col gap-md rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="mb-sm flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                3
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Core Diagnostic Indicators</h2>
            </div>
            <p className="font-body-sm text-body-sm text-on-surface-variant mb-md">
              Input parameters for the 5 core observations. Secondary features will be automatically imputed using training data distribution medians.
            </p>
            
            <form className="grid grid-cols-1 gap-lg md:grid-cols-2" onSubmit={(e) => e.preventDefault()}>
              
              {/* Behavioral Metrics */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md font-bold text-primary">
                  <MaterialIcon name="directions_run" className="text-[18px]" /> Behavioral
                </legend>
                
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Sleep Quality (1.0 - 5.0)</span>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    step={0.1}
                    placeholder="e.g., 3.5"
                    value={draft.Sleep_Quality}
                    onChange={(e) => update('Sleep_Quality', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                  />
                </label>

                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Social Engagement (1.0 - 5.0)</span>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    step={0.1}
                    placeholder="e.g., 4.0"
                    value={draft.Social_Engagement}
                    onChange={(e) => update('Social_Engagement', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                  />
                </label>
              </fieldset>

              {/* Physiological Metrics */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md font-bold text-primary">
                  <MaterialIcon name="favorite" className="text-[18px]" /> Physiological
                </legend>

                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Heart Rate (bpm)</span>
                  <input
                    type="number"
                    min={0}
                    placeholder="e.g., 72"
                    value={draft.Heart_Rate_BPM}
                    onChange={(e) => update('Heart_Rate_BPM', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                  />
                </label>

                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">GSR Level (µS)</span>
                  <input
                    type="number"
                    min={0}
                    step={0.01}
                    placeholder="e.g., 2.3"
                    value={draft.GSR_Level}
                    onChange={(e) => update('GSR_Level', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                  />
                </label>
              </fieldset>

              {/* Facial Observables */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md md:col-span-2">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md font-bold text-primary">
                  <MaterialIcon name="visibility" className="text-[18px]" /> Observables
                </legend>

                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Eye Blink Rate (blinks/min)</span>
                  <input
                    type="number"
                    min={0}
                    placeholder="e.g., 18"
                    value={draft.Eye_Blink_Rate}
                    onChange={(e) => update('Eye_Blink_Rate', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                  />
                </label>
              </fieldset>

            </form>
          </section>
        </div>
      </div>

      {/* Global Action Footer */}
      <div className="mt-xl flex flex-col items-start gap-lg border-t border-outline-variant pt-lg md:flex-row md:items-center md:justify-between">
        <div className="mr-auto flex max-w-md flex-1 flex-col gap-xs">
          <div className="flex items-center gap-sm">
            <MaterialIcon
              name="progress_activity"
              className={`text-body-md text-primary ${modelsReady && !running ? '' : 'animate-spin'}`}
              style={{ fontSize: 18 }}
            />
            <span className="font-label-md text-label-md text-on-surface-variant">
              {running
                ? phaseLabel
                : backendReachable === false
                  ? 'Backend unreachable'
                  : modelsReady
                    ? 'AI Models Ready'
                    : 'AI Models Initializing...'}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-variant">
            <div
              className={`h-full rounded-full bg-primary transition-all duration-1000 ${modelsReady && !running ? 'w-full' : 'w-1/3'}`}
            />
          </div>

          {/* Local LLM Offline/Online indicator */}
          {reasoning && !reasoning.ollama_reachable && (
            <span className="font-label-sm text-label-sm text-on-surface-variant">
              Local LLM offline — report generation will use templated summary.
            </span>
          )}

          {error && (
            <div
              role="alert"
              className="flex items-start gap-xs rounded-lg border border-error/40 bg-error/10 px-md py-sm"
            >
              <MaterialIcon name="error" className="mt-[2px] text-[16px] text-error" />
              <span className="flex-1 font-label-sm text-label-sm text-error">{error}</span>
              <button
                type="button"
                onClick={dismissError}
                className="font-label-sm text-label-sm text-error underline"
              >
                Dismiss
              </button>
            </div>
          )}
        </div>
        
        <button
          type="button"
          onClick={handleRunAssessment}
          disabled={running || !modelsReady}
          className="ai-glow flex items-center gap-sm rounded-lg bg-gradient-to-r from-tertiary to-primary-container px-3xl py-md font-headline-sm text-headline-sm text-on-primary transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <MaterialIcon name="psychology" />
          {running ? 'Analyzing...' : 'Run AI Assessment'}
        </button>
      </div>
    </AppShell>
  )
}
