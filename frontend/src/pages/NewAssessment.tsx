import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppShell } from '../components/layout/AppShell'
import { MaterialIcon } from '../components/ui/MaterialIcon'
import type { AssessmentDraft } from '../types'

const INITIAL_DRAFT: AssessmentDraft = {
  sleepQuality: '',
  socialEngagement: '',
  heartRate: '',
  gsrBaseline: '',
  blinkRate: '',
  eyeContactQuality: 'Normal',
  pitchVariability: 'Normal',
  speechRate: 'Normal',
}

export function NewAssessment() {
  const navigate = useNavigate()
  const [draft, setDraft] = useState<AssessmentDraft>(INITIAL_DRAFT)
  const [faceCaptured, setFaceCaptured] = useState(false)
  const [speechCaptured, setSpeechCaptured] = useState(false)
  const [modelsReady, setModelsReady] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setModelsReady(true), 1400)
    return () => clearTimeout(t)
  }, [])

  function update<K extends keyof AssessmentDraft>(key: K, value: AssessmentDraft[K]) {
    setDraft((d) => ({ ...d, [key]: value }))
  }

  function runAssessment() {
    navigate('/insights')
  }

  return (
    <AppShell>
      <div className="flex flex-col gap-xs">
        <h1 className="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-on-surface">
          New Multimodal Assessment
        </h1>
        <p className="max-w-2xl font-body-md text-body-md text-on-surface-variant">
          Follow the three steps below to gather patient data. The AI will analyze facial expressions, vocal
          tonality, and self-reported metrics to generate a comprehensive psychiatric insight report.
        </p>
      </div>

      <div className="mt-xl grid grid-cols-1 gap-lg lg:grid-cols-3">
        {/* Step 1 & 2 Column */}
        <div className="flex flex-col gap-lg lg:col-span-1">
          {/* Step 1: Facial Analysis */}
          <section className="group relative flex flex-col gap-md overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                1
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Facial Analysis</h2>
            </div>
            <button
              type="button"
              onClick={() => setFaceCaptured(true)}
              className="flex min-h-[200px] flex-col items-center justify-center gap-md rounded-lg border-2 border-dashed border-outline-variant bg-surface-container-low p-xl text-center transition-colors hover:bg-surface-container group-hover:border-primary-fixed-dim"
            >
              <MaterialIcon name="videocam" className="text-4xl text-outline" />
              <div>
                <p className="font-body-sm text-body-sm text-on-surface">Drag &amp; drop video file or click to browse</p>
                <p className="mt-xs font-label-sm text-label-sm text-on-surface-variant">or</p>
              </div>
              <span className="rounded-md border border-outline-variant bg-surface-container-highest px-md py-sm font-label-md text-label-md text-on-surface transition-all hover:bg-primary hover:text-on-primary">
                Start Webcam
              </span>
            </button>
            <div className={`mt-sm flex flex-col gap-xs transition-opacity ${faceCaptured ? 'opacity-100' : 'opacity-50'}`}>
              <span className="font-label-sm text-label-sm text-on-surface-variant">
                {faceCaptured ? 'Detected Signals' : 'Detected Signals (Awaiting Media)'}
              </span>
              <div className="flex h-2 w-full gap-xs overflow-hidden rounded-full bg-surface-variant">
                <div className="w-[10%] bg-secondary-container" />
                <div className="w-[20%] bg-primary-container" />
                <div className="w-[70%] bg-tertiary-container" />
              </div>
              <div className="flex justify-between font-label-sm text-[10px] text-label-sm text-outline">
                <span>Happy</span>
                <span>Sad</span>
                <span>Neutral</span>
              </div>
            </div>
          </section>

          {/* Step 2: Speech Analysis */}
          <section className="group relative flex flex-col gap-md overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                2
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Speech Analysis</h2>
            </div>
            <button
              type="button"
              onClick={() => setSpeechCaptured(true)}
              className="flex min-h-[160px] flex-col items-center justify-center gap-md rounded-lg border-2 border-dashed border-outline-variant bg-surface-container-low p-xl text-center transition-colors hover:bg-surface-container group-hover:border-primary-fixed-dim"
            >
              <MaterialIcon name="mic" className="text-4xl text-outline" />
              <div>
                <p className="font-body-sm text-body-sm text-on-surface">Upload audio file</p>
                <p className="mt-xs font-label-sm text-label-sm text-on-surface-variant">or</p>
              </div>
              <span className="rounded-md border border-outline-variant bg-surface-container-highest px-md py-sm font-label-md text-label-md text-on-surface transition-all hover:bg-primary hover:text-on-primary">
                Record Audio
              </span>
            </button>
            <div className={`mt-sm flex h-12 w-full items-center justify-center rounded bg-surface-variant transition-opacity ${speechCaptured ? 'opacity-100' : 'opacity-30'}`}>
              <span className="font-label-sm text-label-sm text-outline">
                {speechCaptured ? 'Waveform Captured' : 'Waveform Ready'}
              </span>
            </div>
          </section>
        </div>

        {/* Step 3 Column */}
        <div className="lg:col-span-2">
          <section className="flex h-full flex-col gap-md rounded-xl border border-outline-variant bg-surface-container-lowest p-lg shadow-level-1">
            <div className="mb-sm flex items-center gap-sm">
              <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-container font-label-sm text-label-sm text-on-primary">
                3
              </div>
              <h2 className="font-headline-sm text-headline-sm text-on-surface">Clinical Metrics Input</h2>
            </div>
            <form
              className="grid flex-1 grid-cols-1 gap-lg md:grid-cols-2"
              onSubmit={(e) => e.preventDefault()}
            >
              {/* Behavioral */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md text-primary">
                  <MaterialIcon name="directions_run" /> Behavioral
                </legend>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Sleep Quality (1-10)</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={draft.sleepQuality}
                    onChange={(e) => update('sleepQuality', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  />
                </label>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Social Engagement Level</span>
                  <select
                    value={draft.socialEngagement}
                    onChange={(e) => update('socialEngagement', e.target.value)}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  >
                    <option value="">Select level...</option>
                    <option>High</option>
                    <option>Moderate</option>
                    <option>Low</option>
                    <option>Isolated</option>
                  </select>
                </label>
              </fieldset>

              {/* Physiological */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md text-primary">
                  <MaterialIcon name="favorite" /> Physiological
                </legend>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Heart Rate (bpm)</span>
                  <input
                    type="number"
                    placeholder="e.g., 72"
                    value={draft.heartRate}
                    onChange={(e) => update('heartRate', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  />
                </label>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">GSR Baseline (µS)</span>
                  <input
                    type="text"
                    placeholder="e.g., 2.4"
                    value={draft.gsrBaseline}
                    onChange={(e) => update('gsrBaseline', e.target.value)}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  />
                </label>
              </fieldset>

              {/* Facial Observables */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md text-primary">
                  <MaterialIcon name="visibility" /> Facial Observables
                </legend>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Blink Rate (per min)</span>
                  <input
                    type="number"
                    value={draft.blinkRate}
                    onChange={(e) => update('blinkRate', e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  />
                </label>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Eye Contact Quality</span>
                  <select
                    value={draft.eyeContactQuality}
                    onChange={(e) => update('eyeContactQuality', e.target.value)}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  >
                    <option>Normal</option>
                    <option>Avoidant</option>
                    <option>Staring</option>
                  </select>
                </label>
              </fieldset>

              {/* Speech Characteristics */}
              <fieldset className="flex flex-col gap-md rounded-lg border border-surface-variant bg-surface-container-low p-md">
                <legend className="flex items-center gap-xs border-b border-outline-variant pb-xs font-label-md text-label-md text-primary">
                  <MaterialIcon name="graphic_eq" /> Speech Characteristics
                </legend>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Pitch Variability</span>
                  <select
                    value={draft.pitchVariability}
                    onChange={(e) => update('pitchVariability', e.target.value)}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  >
                    <option>Monotone</option>
                    <option>Normal</option>
                    <option>Highly Variable</option>
                  </select>
                </label>
                <label className="flex flex-col gap-xs">
                  <span className="font-label-sm text-label-sm text-on-surface-variant">Speech Rate</span>
                  <select
                    value={draft.speechRate}
                    onChange={(e) => update('speechRate', e.target.value)}
                    className="w-full rounded border-outline-variant bg-surface px-sm py-xs font-body-sm text-body-sm text-on-surface focus:border-primary-container focus:ring-1 focus:ring-primary-container"
                  >
                    <option>Slow</option>
                    <option>Normal</option>
                    <option>Pressured/Fast</option>
                  </select>
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
              className={`text-body-md text-primary ${modelsReady ? '' : 'animate-spin'}`}
              style={{ fontSize: 18 }}
            />
            <span className="font-label-md text-label-md text-on-surface-variant">
              {modelsReady ? 'AI Models Ready' : 'AI Models Initializing...'}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-variant">
            <div
              className={`h-full rounded-full bg-primary transition-all duration-1000 ${modelsReady ? 'w-full' : 'w-1/3 animate-pulse'}`}
            />
          </div>
        </div>
        <button
          type="button"
          onClick={runAssessment}
          className="ai-glow flex items-center gap-sm rounded-lg bg-gradient-to-r from-tertiary to-primary-container px-3xl py-md font-headline-sm text-headline-sm text-on-primary transition-opacity hover:opacity-90"
        >
          <MaterialIcon name="psychology" />
          Run AI Assessment
        </button>
      </div>
    </AppShell>
  )
}
