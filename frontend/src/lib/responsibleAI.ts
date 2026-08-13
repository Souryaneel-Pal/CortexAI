/**
 * Responsible-AI copy, centralized so it is worded consistently everywhere
 * it appears (per PROJECT_PLAN.md: "decision-support ... framing wired into
 * every report/API response/UI copy string — non-negotiable").
 *
 * The six approved Stitch pages (docs/frontend_design/*\/code.html) don't
 * themselves carry this copy — they're single-screen mockups. The project's
 * secondary reference (docs/MindScope_UI_Template.jsx) does, and P4/P6 of
 * PROJECT_PLAN.md require it be present across the UI, not added later. The
 * three strings below are carried over near-verbatim from that reference.
 */

export const DECISION_SUPPORT_SIDEBAR =
  'Assists professionals. Not a diagnosis. Always route people to qualified human care.'

export const DECISION_SUPPORT_REPORT_FOOTER =
  'This narrative is generated decision-support, not a diagnosis. Interpretation and any clinical action remain the responsibility of a qualified professional.'

export const CRISIS_SURFACING_NOTE =
  "When severe-distress indicators co-occur, this interface surfaces support resources and prompts human review."

export const CRISIS_RESOURCES = [
  {
    name: '988 Suicide & Crisis Lifeline (US)',
    detail: 'Call or text 988 — free, confidential, 24/7.',
  },
  {
    name: 'International Association for Suicide Prevention',
    detail: 'Crisis center directory for locations outside the US: iasp.info/resources/Crisis_Centres',
  },
]
