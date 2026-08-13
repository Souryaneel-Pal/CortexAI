/**
 * Maps the clinician-facing assessment form onto the model's 18-feature
 * input vector (docs/Dataset_Description.docx).
 *
 * The form deliberately asks for a small number of things a clinician can
 * actually observe or read off a device in a session; the model expects all
 * 18 columns. Every field the form does not collect is filled with that
 * column's median from the 4,000-row training table (`FEATURE_MEDIANS`) —
 * an in-distribution "no information" value — rather than zero, which for
 * columns like `Pitch_Mean` (observed range 80–300 Hz) or `Skin_Temperature`
 * (32–37 °C) would be far outside anything the encoder ever saw.
 *
 * The categorical → numeric conversions below are a **UI convenience for the
 * demo form, not a validated clinical instrument**. They are stated
 * explicitly here rather than buried in a component so they can be reviewed,
 * changed, or replaced by real device capture (the wearable/WebRTC path in
 * docs/Proposal.pdf Sec. 08) without hunting through JSX.
 */
import { FEATURE_MEDIANS, type TabularFeatures } from './api'

export interface AssessmentFormValues {
  sleepQuality: number | ''
  socialEngagement: string
  heartRate: number | ''
  gsrBaseline: string
  blinkRate: number | ''
  eyeContactQuality: string
  pitchVariability: string
  speechRate: string
}

/** The form's 1–10 sleep scale onto the dataset's documented 1–5 scale. */
function sleepQualityToScale(value: number): number {
  return Math.min(5, Math.max(1, Math.round(value / 2)))
}

const SOCIAL_ENGAGEMENT_SCALE: Record<string, number> = {
  High: 5,
  Moderate: 3,
  Low: 2,
  Isolated: 1,
}

/** Head_Motion_Index proxy: gaze aversion presents as more head movement. */
const EYE_CONTACT_TO_HEAD_MOTION: Record<string, number> = {
  Normal: FEATURE_MEDIANS.Head_Motion_Index,
  Avoidant: 0.75,
  Staring: 0.25,
}

/** MFCC_Variance proxy: prosodic flattening shows up as low spectral variance. */
const PITCH_VARIABILITY_TO_MFCC_VARIANCE: Record<string, number> = {
  Monotone: 5.0,
  Normal: FEATURE_MEDIANS.MFCC_Variance,
  'Highly Variable': 25.0,
}

/** Speech_Rate in words/second (dataset range 2.0–6.0). */
const SPEECH_RATE_SCALE: Record<string, number> = {
  Slow: 2.5,
  Normal: FEATURE_MEDIANS.Speech_Rate,
  'Pressured/Fast': 5.5,
}

function numeric(value: number | '', fallback: number): number {
  return value === '' || Number.isNaN(Number(value)) ? fallback : Number(value)
}

function parseFloatOr(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function buildTabularFeatures(form: AssessmentFormValues): TabularFeatures {
  return {
    ...FEATURE_MEDIANS,

    // Behavioural
    Sleep_Quality:
      form.sleepQuality === ''
        ? FEATURE_MEDIANS.Sleep_Quality
        : sleepQualityToScale(Number(form.sleepQuality)),
    Social_Engagement: SOCIAL_ENGAGEMENT_SCALE[form.socialEngagement] ?? FEATURE_MEDIANS.Social_Engagement,

    // Facial observables
    Eye_Blink_Rate: numeric(form.blinkRate, FEATURE_MEDIANS.Eye_Blink_Rate),
    Head_Motion_Index: EYE_CONTACT_TO_HEAD_MOTION[form.eyeContactQuality] ?? FEATURE_MEDIANS.Head_Motion_Index,

    // Acoustic
    MFCC_Variance:
      PITCH_VARIABILITY_TO_MFCC_VARIANCE[form.pitchVariability] ?? FEATURE_MEDIANS.MFCC_Variance,
    Speech_Rate: SPEECH_RATE_SCALE[form.speechRate] ?? FEATURE_MEDIANS.Speech_Rate,

    // Physiological
    Heart_Rate_BPM: numeric(form.heartRate, FEATURE_MEDIANS.Heart_Rate_BPM),
    GSR_Level: parseFloatOr(form.gsrBaseline, FEATURE_MEDIANS.GSR_Level),
  }
}

/** Which of the 18 features the form actually supplied, for UI disclosure. */
export function collectedFeatureCount(form: AssessmentFormValues): number {
  return [
    form.sleepQuality !== '',
    Boolean(SOCIAL_ENGAGEMENT_SCALE[form.socialEngagement]),
    form.heartRate !== '',
    Number.isFinite(Number.parseFloat(form.gsrBaseline)),
    form.blinkRate !== '',
    true, // eye contact -> Head_Motion_Index (always has a value)
    true, // pitch variability -> MFCC_Variance
    true, // speech rate -> Speech_Rate
  ].filter(Boolean).length
}
