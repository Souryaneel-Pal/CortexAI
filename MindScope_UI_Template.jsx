import React, { useState, useEffect } from "react";
import {
  Brain, LayoutDashboard, ScanFace, Mic, Activity, HeartPulse,
  Gauge, LineChart as LineIcon, FileText, ShieldCheck, Sparkles,
  ChevronRight, Upload, Play, Camera, Waves, TrendingUp, AlertTriangle,
  CircleDot, Quote, Download, ArrowRight, Info
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Line, LineChart
} from "recharts";

/* ------------------------------------------------------------------ *
 *  MindScope — multimodal mental-health decision-support console
 *  A multipage UI/UX template for the Hack4Health project.
 *  Modality color language:  face = indigo · speech = sky · behaviour/physio = teal
 * ------------------------------------------------------------------ */

const C = {
  ink: "#0f172a", inkSoft: "#334155", muted: "#64748b", line: "#e6e8ef",
  bg: "#f5f6fb", surface: "#ffffff",
  face: "#6366f1", speech: "#0ea5e9", body: "#14b8a6", fusion: "#f59e0b",
  xai: "#8b5cf6", rag: "#ef4444",
  healthy: "#10b981", mild: "#f59e0b", moderate: "#f97316", severe: "#ef4444",
};

const SEVERITY = {
  Healthy:  { c: C.healthy,  i: 0 },
  "Mild":   { c: C.mild,     i: 1 },
  "Moderate": { c: C.moderate, i: 2 },
  "Severe": { c: C.severe,   i: 3 },
};

/* ------------------------------- mock data ------------------------------- */
const RESULT = {
  cls: "Moderate",
  confidence: 0.78,
  scores: [
    { key: "Depression", val: 19, max: 34, band: "Moderate", c: C.moderate },
    { key: "Anxiety",    val: 14, max: 24, band: "Moderate", c: C.moderate },
    { key: "Stress",     val: 27, max: 39, band: "Severe",   c: C.severe },
  ],
  contrib: [
    { key: "Physiological", pct: 46, c: C.body,   note: "Low HRV · elevated GSR" },
    { key: "Facial",        pct: 32, c: C.face,   note: "Brow tension · low smile" },
    { key: "Speech",        pct: 22, c: C.speech, note: "Flat pitch · fast rate" },
  ],
  shap: [
    { f: "HRV Index",            v: -0.34 },
    { f: "Sleep Quality",        v: -0.28 },
    { f: "GSR Level",            v:  0.22 },
    { f: "Facial Emotion Var.",  v:  0.18 },
    { f: "Speech Rate",          v:  0.14 },
    { f: "Social Engagement",    v: -0.11 },
  ],
  counterfactual: {
    lever: "Sleep Quality", from: 2, to: 4, result: "Mild",
  },
};

const TRAJECTORY = [
  { s: "May", Depression: 24, Anxiety: 18, Stress: 31 },
  { s: "Jun", Depression: 22, Anxiety: 17, Stress: 30 },
  { s: "Jul", Depression: 21, Anxiety: 16, Stress: 29 },
  { s: "Aug", Depression: 19, Anxiety: 14, Stress: 27 },
];

const RECENT = [
  { id: "PT-2041", cls: "Moderate", conf: 78, when: "12 min ago" },
  { id: "PT-2040", cls: "Healthy",  conf: 91, when: "1 hr ago" },
  { id: "PT-2039", cls: "Mild",     conf: 83, when: "2 hr ago" },
  { id: "PT-2038", cls: "Severe",   conf: 74, when: "Today, 09:14" },
  { id: "PT-2037", cls: "Healthy",  conf: 88, when: "Yesterday" },
];

const NUMERIC_GROUPS = [
  { g: "Behavioural", c: C.body, fields: ["Sleep Quality", "Social Engagement", "Daily App Usage", "Typing Speed", "Session Frequency", "Idle Time"] },
  { g: "Facial (video)", c: C.face, fields: ["Emotion Variance", "Eye-Blink Rate", "Smile Intensity", "Head-Motion Index"] },
  { g: "Acoustic", c: C.speech, fields: ["MFCC Mean", "MFCC Variance", "Pitch Mean", "Speech Rate"] },
  { g: "Physiological", c: C.body, fields: ["Heart Rate", "HRV Index", "Skin Temperature", "GSR Level"] },
];

/* ------------------------------- primitives ------------------------------- */
const Card = ({ children, className = "", style = {} }) => (
  <div className={"rounded-2xl border " + className}
       style={{ background: C.surface, borderColor: C.line, boxShadow: "0 1px 2px rgba(15,23,42,.04), 0 8px 24px -18px rgba(15,23,42,.25)", ...style }}>
    {children}
  </div>
);

const Eyebrow = ({ children, color = C.muted }) => (
  <div className="text-[11px] font-semibold uppercase" style={{ letterSpacing: ".14em", color }}>{children}</div>
);

const SeverityPill = ({ cls, small }) => {
  const s = SEVERITY[cls] || SEVERITY.Healthy;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full font-semibold"
      style={{ background: s.c + "1a", color: s.c, padding: small ? "2px 9px" : "4px 12px", fontSize: small ? 11 : 12.5 }}>
      <CircleDot size={small ? 11 : 13} /> {cls}{!small && " Stress"}
    </span>
  );
};

/* confidence ring (SVG) */
const Ring = ({ pct, color = C.fusion, size = 132 }) => {
  const r = size / 2 - 10, circ = 2 * Math.PI * r;
  const [dash, setDash] = useState(circ);
  useEffect(() => { const t = setTimeout(() => setDash(circ * (1 - pct)), 120); return () => clearTimeout(t); }, [pct, circ]);
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size/2} cy={size/2} r={r} stroke={C.line} strokeWidth="10" fill="none" />
      <circle cx={size/2} cy={size/2} r={r} stroke={color} strokeWidth="10" fill="none"
        strokeLinecap="round" strokeDasharray={circ} strokeDashoffset={dash}
        style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(.22,1,.36,1)" }} />
    </svg>
  );
};

/* ------------------------------- shell ------------------------------- */
const NAV = [
  { id: "overview",   label: "Overview",       icon: LayoutDashboard },
  { id: "assess",     label: "New assessment", icon: ScanFace },
  { id: "results",    label: "Results",        icon: Gauge },
  { id: "trajectory", label: "Trajectory",     icon: LineIcon },
  { id: "report",     label: "Report",         icon: FileText },
];

function Sidebar({ page, setPage }) {
  return (
    <aside className="shrink-0 flex flex-col justify-between py-6 px-4"
      style={{ width: 246, background: C.ink, minHeight: "100%" }}>
      <div>
        <div className="flex items-center gap-2.5 px-2 mb-8">
          <div className="grid place-items-center rounded-xl" style={{ width: 38, height: 38, background: "linear-gradient(135deg,#6366f1,#14b8a6)" }}>
            <Brain size={21} color="#fff" />
          </div>
          <div>
            <div className="text-white font-semibold" style={{ letterSpacing: "-.02em", fontSize: 17 }}>MindScope</div>
            <div style={{ color: "#94a3b8", fontSize: 10.5, letterSpacing: ".08em" }}>PSYCHIATRIC AI</div>
          </div>
        </div>
        <nav className="flex flex-col gap-1">
          {NAV.map(n => {
            const on = page === n.id; const Icon = n.icon;
            return (
              <button key={n.id} onClick={() => setPage(n.id)}
                className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors"
                style={{ background: on ? "rgba(255,255,255,.08)" : "transparent", color: on ? "#fff" : "#94a3b8" }}>
                <Icon size={18} style={{ color: on ? "#a5b4fc" : "#64748b" }} />
                <span style={{ fontSize: 13.5, fontWeight: on ? 600 : 500 }}>{n.label}</span>
                {on && <ChevronRight size={15} className="ml-auto" style={{ color: "#a5b4fc" }} />}
              </button>
            );
          })}
        </nav>
      </div>
      <div className="rounded-xl px-3 py-3" style={{ background: "rgba(255,255,255,.05)" }}>
        <div className="flex items-center gap-2 mb-1">
          <ShieldCheck size={15} style={{ color: "#5eead4" }} />
          <span className="text-white" style={{ fontSize: 12, fontWeight: 600 }}>Decision support</span>
        </div>
        <p style={{ color: "#94a3b8", fontSize: 10.5, lineHeight: 1.5 }}>
          Assists professionals. Not a diagnosis. Always route people to qualified human care.
        </p>
      </div>
    </aside>
  );
}

function Topbar({ page }) {
  const titles = {
    overview: ["Clinical overview", "Snapshot of today's multimodal assessments"],
    assess: ["New assessment", "Capture face, voice and behavioural signals"],
    results: ["Assessment results", "Prediction, severity and explanation for PT-2041"],
    trajectory: ["Trajectory", "How this person's scores change over time"],
    report: ["Clinical report", "Grounded, cited narrative for the record"],
  };
  const [t, s] = titles[page];
  return (
    <div className="flex items-center justify-between px-8 py-5 border-b" style={{ borderColor: C.line, background: C.surface }}>
      <div>
        <h1 style={{ color: C.ink, fontSize: 21, fontWeight: 700, letterSpacing: "-.02em" }}>{t}</h1>
        <p style={{ color: C.muted, fontSize: 12.5, marginTop: 2 }}>{s}</p>
      </div>
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-full px-3 py-1.5" style={{ background: C.bg, border: `1px solid ${C.line}` }}>
          <span className="rounded-full" style={{ width: 8, height: 8, background: C.healthy }} />
          <span style={{ fontSize: 12, color: C.inkSoft, fontWeight: 500 }}>Model online · v0.9</span>
        </div>
        <div className="grid place-items-center rounded-full font-semibold text-white"
          style={{ width: 34, height: 34, background: "linear-gradient(135deg,#6366f1,#0ea5e9)", fontSize: 13 }}>DR</div>
      </div>
    </div>
  );
}

/* ------------------------------- Overview ------------------------------- */
function Overview({ setPage }) {
  const kpis = [
    { label: "Assessments today", val: "24", sub: "+6 vs yesterday", c: C.face, icon: Activity },
    { label: "Mean confidence", val: "82%", sub: "across all runs", c: C.fusion, icon: Gauge },
    { label: "Flagged (Severe)", val: "3", sub: "resource layer shown", c: C.severe, icon: AlertTriangle },
    { label: "Modalities complete", val: "91%", sub: "graceful fallback on 9%", c: C.body, icon: Waves },
  ];
  return (
    <div className="p-8 grid gap-6">
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(4,1fr)" }}>
        {kpis.map(k => (
          <Card key={k.label} className="p-5">
            <div className="flex items-center justify-between">
              <Eyebrow>{k.label}</Eyebrow>
              <div className="grid place-items-center rounded-lg" style={{ width: 30, height: 30, background: k.c + "18" }}>
                <k.icon size={16} style={{ color: k.c }} />
              </div>
            </div>
            <div style={{ fontSize: 30, fontWeight: 700, color: C.ink, marginTop: 8, letterSpacing: "-.02em" }}>{k.val}</div>
            <div style={{ fontSize: 11.5, color: C.muted }}>{k.sub}</div>
          </Card>
        ))}
      </div>

      <div className="grid gap-6" style={{ gridTemplateColumns: "1.55fr 1fr" }}>
        <Card className="p-6">
          <div className="flex items-center justify-between mb-1">
            <div>
              <Eyebrow color={C.face}>Population trend</Eyebrow>
              <div style={{ fontSize: 16, fontWeight: 600, color: C.ink, marginTop: 3 }}>Mean stress score · last 4 cycles</div>
            </div>
            <TrendingUp size={18} style={{ color: C.body }} />
          </div>
          <div style={{ height: 220, marginTop: 10 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={TRAJECTORY} margin={{ left: -18, right: 6, top: 8 }}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={C.face} stopOpacity={0.28} />
                    <stop offset="100%" stopColor={C.face} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.line} vertical={false} />
                <XAxis dataKey="s" tick={{ fill: C.muted, fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: C.muted, fontSize: 12 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: `1px solid ${C.line}`, fontSize: 12 }} />
                <Area type="monotone" dataKey="Stress" stroke={C.face} strokeWidth={2.5} fill="url(#g1)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="p-6">
          <div className="flex items-center justify-between mb-3">
            <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Recent assessments</div>
            <button onClick={() => setPage("results")} className="flex items-center gap-1" style={{ color: C.face, fontSize: 12.5, fontWeight: 600 }}>
              View <ArrowRight size={14} />
            </button>
          </div>
          <div className="flex flex-col">
            {RECENT.map((r, i) => (
              <button key={r.id} onClick={() => setPage("results")}
                className="flex items-center justify-between py-2.5 text-left"
                style={{ borderTop: i ? `1px solid ${C.line}` : "none" }}>
                <div>
                  <div style={{ fontSize: 13.5, fontWeight: 600, color: C.ink }}>{r.id}</div>
                  <div style={{ fontSize: 11, color: C.muted }}>{r.when}</div>
                </div>
                <div className="flex items-center gap-3">
                  <span style={{ fontSize: 12, color: C.muted, fontVariantNumeric: "tabular-nums" }}>{r.conf}%</span>
                  <SeverityPill cls={r.cls} small />
                </div>
              </button>
            ))}
          </div>
        </Card>
      </div>

      <Card className="p-6 flex items-center justify-between"
        style={{ background: "linear-gradient(100deg,#eef2ff,#ecfeff)" }}>
        <div className="flex items-center gap-4">
          <div className="grid place-items-center rounded-xl" style={{ width: 46, height: 46, background: "linear-gradient(135deg,#6366f1,#14b8a6)" }}>
            <Sparkles size={22} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 15.5, fontWeight: 700, color: C.ink }}>Run a new multimodal assessment</div>
            <div style={{ fontSize: 12.5, color: C.inkSoft }}>Face · voice · behavioural signals — with live capture and missing-modality fallback.</div>
          </div>
        </div>
        <button onClick={() => setPage("assess")} className="rounded-xl px-5 py-3 text-white font-semibold flex items-center gap-2"
          style={{ background: C.ink, fontSize: 13.5 }}>
          Start assessment <ArrowRight size={16} />
        </button>
      </Card>
    </div>
  );
}

/* ------------------------------- New assessment ------------------------------- */
function Assess({ setPage }) {
  const [running, setRunning] = useState(false);
  const inputs = [
    { key: "face", label: "Facial expression", icon: ScanFace, c: C.face, action: "Drop image or capture live", sub: "48×48 grayscale · FER pipeline" },
    { key: "speech", label: "Speech emotion", icon: Mic, c: C.speech, action: "Record 3s or upload .wav", sub: "Wav2Vec2 · log-Mel spectrogram" },
  ];
  const run = () => { setRunning(true); setTimeout(() => setPage("results"), 1400); };
  return (
    <div className="p-8 grid gap-6">
      <div className="grid gap-6" style={{ gridTemplateColumns: "1fr 1fr" }}>
        {inputs.map(m => (
          <Card key={m.key} className="p-6">
            <div className="flex items-center gap-2.5 mb-4">
              <div className="grid place-items-center rounded-lg" style={{ width: 32, height: 32, background: m.c + "18" }}>
                <m.icon size={17} style={{ color: m.c }} />
              </div>
              <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>{m.label}</div>
            </div>
            <div className="rounded-xl grid place-items-center text-center"
              style={{ border: `1.5px dashed ${m.c}66`, background: m.c + "0a", height: 148 }}>
              <div>
                <div className="grid place-items-center mx-auto rounded-full mb-2" style={{ width: 40, height: 40, background: m.c + "1a" }}>
                  {m.key === "face" ? <Camera size={19} style={{ color: m.c }} /> : <Play size={19} style={{ color: m.c }} />}
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>{m.action}</div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>{m.sub}</div>
              </div>
            </div>
            <div className="flex gap-2 mt-3">
              <button className="flex-1 rounded-lg py-2 flex items-center justify-center gap-1.5 font-medium"
                style={{ background: C.bg, color: C.inkSoft, fontSize: 12.5, border: `1px solid ${C.line}` }}>
                <Upload size={14} /> Upload
              </button>
              <button className="flex-1 rounded-lg py-2 flex items-center justify-center gap-1.5 font-medium text-white"
                style={{ background: m.c, fontSize: 12.5 }}>
                <Camera size={14} /> Live capture
              </button>
            </div>
          </Card>
        ))}
      </div>

      <Card className="p-6">
        <div className="flex items-center gap-2.5 mb-4">
          <div className="grid place-items-center rounded-lg" style={{ width: 32, height: 32, background: C.body + "18" }}>
            <HeartPulse size={17} style={{ color: C.body }} />
          </div>
          <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Behavioural, acoustic &amp; physiological signals</div>
          <span className="ml-2 rounded-full px-2 py-0.5" style={{ background: C.body + "14", color: C.body, fontSize: 10.5, fontWeight: 600 }}>18 features</span>
        </div>
        <div className="grid gap-5" style={{ gridTemplateColumns: "repeat(4,1fr)" }}>
          {NUMERIC_GROUPS.map(grp => (
            <div key={grp.g}>
              <div className="flex items-center gap-1.5 mb-2">
                <span className="rounded-full" style={{ width: 7, height: 7, background: grp.c }} />
                <span style={{ fontSize: 11.5, fontWeight: 700, color: C.inkSoft, letterSpacing: ".02em" }}>{grp.g}</span>
              </div>
              <div className="flex flex-col gap-2">
                {grp.fields.map(f => (
                  <div key={f}>
                    <div style={{ fontSize: 10.5, color: C.muted, marginBottom: 2 }}>{f}</div>
                    <input placeholder="—" className="w-full rounded-lg px-2.5 py-1.5"
                      style={{ border: `1px solid ${C.line}`, fontSize: 12.5, color: C.ink, background: C.bg }} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2" style={{ color: C.muted, fontSize: 12 }}>
          <Info size={14} /> Missing a modality? The model runs with graceful fallback and lowers its confidence.
        </div>
        <button onClick={run} disabled={running}
          className="rounded-xl px-6 py-3 text-white font-semibold flex items-center gap-2"
          style={{ background: running ? C.muted : C.ink, fontSize: 14 }}>
          {running ? <><Waves size={16} className="animate-pulse" /> Fusing modalities…</> : <><Gauge size={16} /> Run assessment</>}
        </button>
      </div>
    </div>
  );
}

/* ------------------------------- Results ------------------------------- */
function ContribMeter() {
  const [w, setW] = useState(false);
  useEffect(() => { const t = setTimeout(() => setW(true), 150); return () => clearTimeout(t); }, []);
  return (
    <div>
      <div className="flex rounded-full overflow-hidden" style={{ height: 16 }}>
        {RESULT.contrib.map(m => (
          <div key={m.key} style={{ width: (w ? m.pct : 0) + "%", background: m.c, transition: "width 1s cubic-bezier(.22,1,.36,1)" }} />
        ))}
      </div>
      <div className="flex flex-col gap-2.5 mt-4">
        {RESULT.contrib.map(m => (
          <div key={m.key} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="rounded" style={{ width: 10, height: 10, background: m.c }} />
              <span style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>{m.key}</span>
              <span style={{ fontSize: 11.5, color: C.muted }}>· {m.note}</span>
            </div>
            <span style={{ fontSize: 13, fontWeight: 700, color: m.c, fontVariantNumeric: "tabular-nums" }}>{m.pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Results({ setPage }) {
  const s = SEVERITY[RESULT.cls];
  return (
    <div className="p-8 grid gap-6">
      <div className="grid gap-6" style={{ gridTemplateColumns: "1fr 1.35fr" }}>
        {/* headline */}
        <Card className="p-6 flex flex-col items-center text-center">
          <Eyebrow color={C.muted}>Mental-health status</Eyebrow>
          <div className="relative my-3 grid place-items-center">
            <Ring pct={RESULT.confidence} color={s.c} />
            <div className="absolute grid place-items-center">
              <div style={{ fontSize: 30, fontWeight: 700, color: C.ink, lineHeight: 1 }}>{Math.round(RESULT.confidence*100)}%</div>
              <div style={{ fontSize: 10.5, color: C.muted, letterSpacing: ".06em" }}>CONFIDENCE</div>
            </div>
          </div>
          <SeverityPill cls={RESULT.cls} />
          <div className="mt-4 w-full rounded-xl px-3 py-2.5 flex items-center gap-2 text-left"
            style={{ background: C.fusion + "12" }}>
            <Info size={15} style={{ color: C.fusion, flexShrink: 0 }} />
            <span style={{ fontSize: 11.5, color: C.inkSoft }}>Confidence below 80% — corroborate with clinical judgement.</span>
          </div>
        </Card>

        {/* scores */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Symptom severity scores</div>
            <span style={{ fontSize: 11.5, color: C.muted }}>DASS-aligned bands</span>
          </div>
          <div className="flex flex-col gap-5">
            {RESULT.scores.map(sc => (
              <div key={sc.key}>
                <div className="flex items-center justify-between mb-1.5">
                  <span style={{ fontSize: 13.5, fontWeight: 600, color: C.ink }}>{sc.key}</span>
                  <div className="flex items-center gap-2">
                    <span style={{ fontSize: 13, color: C.inkSoft, fontVariantNumeric: "tabular-nums" }}>{sc.val}<span style={{ color: C.muted }}>/{sc.max}</span></span>
                    <SeverityPill cls={sc.band} small />
                  </div>
                </div>
                <ScoreBar val={sc.val} max={sc.max} color={sc.c} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* explainability */}
      <div className="grid gap-6" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-1">
            <Sparkles size={16} style={{ color: C.xai }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Modality contribution</div>
          </div>
          <p style={{ fontSize: 12, color: C.muted, marginBottom: 14 }}>Which signals drove this prediction, from the fusion attention.</p>
          <ContribMeter />
        </Card>

        <Card className="p-6">
          <div className="flex items-center gap-2 mb-1">
            <Activity size={16} style={{ color: C.xai }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Top feature attributions</div>
          </div>
          <p style={{ fontSize: 12, color: C.muted, marginBottom: 12 }}>SHAP values — red raises stress, green lowers it.</p>
          <div className="flex flex-col gap-2">
            {RESULT.shap.map(x => <ShapRow key={x.f} f={x.f} v={x.v} />)}
          </div>
        </Card>
      </div>

      {/* face heatmap + counterfactual */}
      <div className="grid gap-6" style={{ gridTemplateColumns: "1fr 1.4fr" }}>
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-3">
            <ScanFace size={16} style={{ color: C.face }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Grad-CAM · facial focus</div>
          </div>
          <div className="rounded-xl grid place-items-center relative overflow-hidden" style={{ height: 150, background: "#111827" }}>
            <div className="absolute rounded-full" style={{ width: 74, height: 74, top: 34, background: "radial-gradient(circle,#ef4444cc,transparent 70%)", filter: "blur(2px)" }} />
            <div className="absolute rounded-full" style={{ width: 46, height: 30, top: 92, background: "radial-gradient(circle,#f59e0baa,transparent 70%)", filter: "blur(2px)" }} />
            <ScanFace size={80} color="#334155" />
          </div>
          <p style={{ fontSize: 11.5, color: C.muted, marginTop: 8 }}>Highest activation around the brow and mouth — consistent with tension.</p>
        </Card>

        <Card className="p-6" style={{ background: "linear-gradient(100deg,#ecfdf5,#f0fdfa)" }}>
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp size={16} style={{ color: C.healthy }} />
            <div style={{ fontSize: 15, fontWeight: 600, color: C.ink }}>Counterfactual · what would help</div>
          </div>
          <p style={{ fontSize: 13, color: C.inkSoft, lineHeight: 1.6 }}>
            Raising <b style={{ color: C.ink }}>{RESULT.counterfactual.lever}</b> from
            {" "}<b>{RESULT.counterfactual.from}</b> to <b>{RESULT.counterfactual.to}</b> is predicted to move status from
            {" "}<SeverityPill cls="Moderate" small /> to <SeverityPill cls={RESULT.counterfactual.result} small />.
          </p>
          <div className="flex items-center gap-6 mt-4">
            {[["Now","Moderate",C.moderate],["Target","Mild",C.mild]].map(([k,v,col]) => (
              <div key={k}>
                <div style={{ fontSize: 10.5, color: C.muted, letterSpacing: ".06em" }}>{k.toUpperCase()}</div>
                <div className="flex items-center gap-1.5 mt-1">
                  <span className="rounded-full" style={{ width: 9, height: 9, background: col }} />
                  <span style={{ fontSize: 14, fontWeight: 700, color: C.ink }}>{v}</span>
                </div>
              </div>
            ))}
            <ArrowRight size={18} style={{ color: C.muted }} className="mt-3" />
            <button onClick={() => setPage("report")} className="ml-auto rounded-xl px-4 py-2.5 text-white font-semibold flex items-center gap-2"
              style={{ background: C.ink, fontSize: 13 }}>
              <FileText size={15} /> Generate report
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}

const ScoreBar = ({ val, max, color }) => {
  const [w, setW] = useState(0);
  useEffect(() => { const t = setTimeout(() => setW((val/max)*100), 140); return () => clearTimeout(t); }, [val, max]);
  return (
    <div className="rounded-full" style={{ height: 9, background: C.line }}>
      <div className="rounded-full" style={{ height: 9, width: w + "%", background: color, transition: "width 1s cubic-bezier(.22,1,.36,1)" }} />
    </div>
  );
};

const ShapRow = ({ f, v }) => {
  const pos = v > 0, col = pos ? C.severe : C.healthy, mag = Math.min(Math.abs(v) / 0.4, 1) * 50;
  const [w, setW] = useState(0);
  useEffect(() => { const t = setTimeout(() => setW(mag), 140); return () => clearTimeout(t); }, [mag]);
  return (
    <div className="flex items-center" style={{ fontSize: 12 }}>
      <div className="text-right pr-3" style={{ width: 130, color: C.inkSoft }}>{f}</div>
      <div className="flex-1 flex items-center" style={{ height: 20 }}>
        <div className="flex justify-end" style={{ width: "50%" }}>
          {!pos && <div className="rounded-l" style={{ width: w + "%", height: 14, background: col, transition: "width .9s ease" }} />}
        </div>
        <div style={{ width: 1, height: 20, background: C.line }} />
        <div style={{ width: "50%" }}>
          {pos && <div className="rounded-r" style={{ width: w + "%", height: 14, background: col, transition: "width .9s ease" }} />}
        </div>
      </div>
      <div className="pl-3 text-right" style={{ width: 46, color: col, fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{v > 0 ? "+" : ""}{v}</div>
    </div>
  );
};

/* ------------------------------- Trajectory ------------------------------- */
function Trajectory() {
  const series = [
    { k: "Depression", c: C.face }, { k: "Anxiety", c: C.speech }, { k: "Stress", c: C.body },
  ];
  return (
    <div className="p-8 grid gap-6">
      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(3,1fr)" }}>
        {series.map(s => {
          const first = TRAJECTORY[0][s.k], last = TRAJECTORY[TRAJECTORY.length-1][s.k], d = last - first;
          return (
            <Card key={s.k} className="p-5">
              <Eyebrow color={s.c}>{s.k}</Eyebrow>
              <div className="flex items-end gap-2 mt-2">
                <div style={{ fontSize: 28, fontWeight: 700, color: C.ink }}>{last}</div>
                <div className="flex items-center gap-1 mb-1.5" style={{ color: d < 0 ? C.healthy : C.severe, fontSize: 12.5, fontWeight: 600 }}>
                  <TrendingUp size={14} style={{ transform: d < 0 ? "rotate(180deg)" : "none" }} /> {d > 0 ? "+" : ""}{d}
                </div>
              </div>
              <div style={{ fontSize: 11.5, color: C.muted }}>since first session · improving</div>
            </Card>
          );
        })}
      </div>
      <Card className="p-6">
        <div style={{ fontSize: 15, fontWeight: 600, color: C.ink, marginBottom: 4 }}>Longitudinal score trajectory</div>
        <p style={{ fontSize: 12, color: C.muted, marginBottom: 8 }}>Four assessment cycles for PT-2041 — the trend clinicians actually treat.</p>
        <div style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={TRAJECTORY} margin={{ left: -18, right: 8, top: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.line} vertical={false} />
              <XAxis dataKey="s" tick={{ fill: C.muted, fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: C.muted, fontSize: 12 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ borderRadius: 12, border: `1px solid ${C.line}`, fontSize: 12 }} />
              {series.map(s => (
                <Line key={s.k} type="monotone" dataKey={s.k} stroke={s.c} strokeWidth={2.5}
                  dot={{ r: 3, fill: s.c }} activeDot={{ r: 5 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center gap-5 mt-3">
          {series.map(s => (
            <div key={s.k} className="flex items-center gap-1.5">
              <span className="rounded-full" style={{ width: 9, height: 9, background: s.c }} />
              <span style={{ fontSize: 12, color: C.inkSoft }}>{s.k}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------- Report ------------------------------- */
function Report() {
  const cites = [
    { n: 1, src: "DSM-5-TR", note: "Stress-related disorders, diagnostic criteria" },
    { n: 2, src: "DASS-21 manual", note: "Severity bands for Depression / Anxiety / Stress" },
    { n: 3, src: "WHO mhGAP", note: "Guidance on escalation and referral" },
  ];
  return (
    <div className="p-8 grid gap-6" style={{ gridTemplateColumns: "1.7fr 1fr" }}>
      <Card className="p-8">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2.5">
            <div className="grid place-items-center rounded-lg" style={{ width: 32, height: 32, background: C.rag + "18" }}>
              <FileText size={17} style={{ color: C.rag }} />
            </div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Grounded clinical narrative</div>
              <div style={{ fontSize: 11.5, color: C.muted }}>Every claim is retrieval-grounded and cited.</div>
            </div>
          </div>
          <button className="rounded-lg px-3.5 py-2 flex items-center gap-1.5 text-white font-medium" style={{ background: C.ink, fontSize: 12.5 }}>
            <Download size={14} /> Export PDF
          </button>
        </div>

        <div className="flex items-center gap-3 mb-5 pb-5" style={{ borderBottom: `1px solid ${C.line}` }}>
          <span style={{ fontSize: 12.5, color: C.muted }}>Subject <b style={{ color: C.ink }}>PT-2041</b></span>
          <span style={{ color: C.line }}>|</span>
          <SeverityPill cls="Moderate" small />
          <span style={{ color: C.line }}>|</span>
          <span style={{ fontSize: 12.5, color: C.muted }}>78% confidence</span>
        </div>

        {[
          { h: "Summary", t: <>The integrated multimodal assessment indicates <b>Moderate Stress</b>. Physiological signals — reduced heart-rate variability and elevated galvanic skin response — were the dominant contributors, corroborated by facial tension and a flattened speech profile.<Cite n={2}/></> },
          { h: "Severity", t: <>Depression and Anxiety fall in the moderate band, while Stress reaches the severe band on a DASS-aligned scale. The pattern is consistent with an acute stress presentation rather than a persistent mood disorder.<Cite n={1}/><Cite n={2}/></> },
          { h: "Recommendation", t: <>Corroborate with clinical interview given sub-80% confidence. A brief sleep-focused intervention is a high-value lever: the model estimates that improving sleep quality would shift status toward Mild. Escalate for in-person review if distress persists.<Cite n={3}/></> },
        ].map(sec => (
          <div key={sec.h} className="mb-5">
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: ".1em", color: C.rag, textTransform: "uppercase", marginBottom: 4 }}>{sec.h}</div>
            <p style={{ fontSize: 13.5, color: C.inkSoft, lineHeight: 1.7 }}>{sec.t}</p>
          </div>
        ))}

        <div className="rounded-xl px-4 py-3 flex items-start gap-2.5" style={{ background: C.bg }}>
          <ShieldCheck size={16} style={{ color: C.body, flexShrink: 0, marginTop: 1 }} />
          <p style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.6 }}>
            This narrative is generated decision-support, not a diagnosis. Interpretation and any clinical action remain the responsibility of a qualified professional.
          </p>
        </div>
      </Card>

      <div className="grid gap-6" style={{ alignContent: "start" }}>
        <Card className="p-6">
          <div className="flex items-center gap-2 mb-3">
            <Quote size={15} style={{ color: C.rag }} />
            <div style={{ fontSize: 14.5, fontWeight: 600, color: C.ink }}>Sources retrieved</div>
          </div>
          <div className="flex flex-col gap-3">
            {cites.map(c => (
              <div key={c.n} className="flex gap-3">
                <div className="grid place-items-center rounded-md shrink-0 text-white font-bold"
                  style={{ width: 22, height: 22, background: C.rag, fontSize: 11 }}>{c.n}</div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: C.ink }}>{c.src}</div>
                  <div style={{ fontSize: 11.5, color: C.muted }}>{c.note}</div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6" style={{ background: C.ink }}>
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} style={{ color: "#fca5a5" }} />
            <div style={{ fontSize: 14.5, fontWeight: 600, color: "#fff" }}>Safety layer</div>
          </div>
          <p style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6, marginBottom: 12 }}>
            When severe-distress indicators co-occur, the interface surfaces support resources and prompts human review.
          </p>
          <button className="w-full rounded-lg py-2.5 font-semibold" style={{ background: "#fff", color: C.ink, fontSize: 12.5 }}>
            View support resources
          </button>
        </Card>
      </div>
    </div>
  );
}

const Cite = ({ n }) => (
  <sup><span className="inline-grid place-items-center rounded align-middle text-white font-bold ml-0.5"
    style={{ width: 14, height: 14, background: C.rag, fontSize: 9 }}>{n}</span></sup>
);

/* ------------------------------- App ------------------------------- */
export default function App() {
  const [page, setPage] = useState("overview");
  const PAGES = { overview: Overview, assess: Assess, results: Results, trajectory: Trajectory, report: Report };
  const Page = PAGES[page];
  return (
    <div className="flex" style={{ minHeight: 720, background: C.bg, fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", color: C.ink }}>
      <Sidebar page={page} setPage={setPage} />
      <div className="flex-1 flex flex-col" style={{ minWidth: 0 }}>
        <Topbar page={page} />
        <div className="flex-1 overflow-auto">
          <Page setPage={setPage} />
        </div>
      </div>
    </div>
  );
}
