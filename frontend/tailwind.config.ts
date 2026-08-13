import type { Config } from 'tailwindcss'

/**
 * CortexAI design tokens — transcribed exactly from
 * docs/frontend_design/mindscope_ai/DESIGN.md (source of truth for P5 frontend).
 *
 * Sky-blue (primary), teal (secondary) and violet (tertiary) are the only three
 * palette hues. Violet is reserved EXCLUSIVELY for AI-generated content/insights.
 */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surfaces
        surface: '#f8f9ff',
        'surface-dim': '#cbdbf5',
        'surface-bright': '#f8f9ff',
        'surface-container-lowest': '#ffffff',
        'surface-container-low': '#eff4ff',
        'surface-container': '#e5eeff',
        'surface-container-high': '#dce9ff',
        'surface-container-highest': '#d3e4fe',
        'surface-variant': '#d3e4fe',
        'surface-tint': '#006591',
        'on-surface': '#0b1c30',
        'on-surface-variant': '#3e4850',
        'inverse-surface': '#213145',
        'inverse-on-surface': '#eaf1ff',
        outline: '#6e7881',
        'outline-variant': '#bec8d2',
        background: '#f8f9ff',
        'on-background': '#0b1c30',

        // Primary — Sky Blue
        primary: '#006591',
        'on-primary': '#ffffff',
        'primary-container': '#0ea5e9',
        'on-primary-container': '#003751',
        'inverse-primary': '#89ceff',
        'primary-fixed': '#c9e6ff',
        'primary-fixed-dim': '#89ceff',
        'on-primary-fixed': '#001e2f',
        'on-primary-fixed-variant': '#004c6e',

        // Secondary — Teal
        secondary: '#006b5f',
        'on-secondary': '#ffffff',
        'secondary-container': '#6df5e1',
        'on-secondary-container': '#006f64',
        'secondary-fixed': '#71f8e4',
        'secondary-fixed-dim': '#4fdbc8',
        'on-secondary-fixed': '#00201c',
        'on-secondary-fixed-variant': '#005048',

        // Tertiary — Violet (AI-generated content ONLY, per DESIGN.md)
        tertiary: '#6d3bd7',
        'on-tertiary': '#ffffff',
        'tertiary-container': '#a986ff',
        'on-tertiary-container': '#3e0097',
        'tertiary-fixed': '#e9ddff',
        'tertiary-fixed-dim': '#d0bcff',
        'on-tertiary-fixed': '#23005c',
        'on-tertiary-fixed-variant': '#5516be',

        // Error
        error: '#ba1a1a',
        'on-error': '#ffffff',
        'error-container': '#ffdad6',
        'on-error-container': '#93000a',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        // Named type-scale families (DESIGN.md typography tokens)
        'display-lg': ['Inter', 'sans-serif'],
        'headline-lg': ['Inter', 'sans-serif'],
        'headline-lg-mobile': ['Inter', 'sans-serif'],
        'headline-md': ['Inter', 'sans-serif'],
        'headline-sm': ['Inter', 'sans-serif'],
        'body-lg': ['Inter', 'sans-serif'],
        'body-md': ['Inter', 'sans-serif'],
        'body-sm': ['Inter', 'sans-serif'],
        'label-md': ['JetBrains Mono', 'monospace'],
        'label-sm': ['JetBrains Mono', 'monospace'],
      },
      fontSize: {
        'display-lg': ['48px', { lineHeight: '56px', letterSpacing: '-0.02em', fontWeight: '700' }],
        'headline-lg': ['32px', { lineHeight: '40px', letterSpacing: '-0.01em', fontWeight: '600' }],
        'headline-lg-mobile': ['28px', { lineHeight: '36px', fontWeight: '600' }],
        'headline-md': ['24px', { lineHeight: '32px', fontWeight: '600' }],
        'headline-sm': ['20px', { lineHeight: '28px', fontWeight: '600' }],
        'body-lg': ['18px', { lineHeight: '28px', fontWeight: '400' }],
        'body-md': ['16px', { lineHeight: '24px', fontWeight: '400' }],
        'body-sm': ['14px', { lineHeight: '20px', fontWeight: '400' }],
        'label-md': ['14px', { lineHeight: '20px', letterSpacing: '0.02em', fontWeight: '500' }],
        'label-sm': ['12px', { lineHeight: '16px', letterSpacing: '0.04em', fontWeight: '500' }],
      },
      spacing: {
        base: '4px',
        xs: '4px',
        sm: '8px',
        md: '16px',
        lg: '24px',
        xl: '32px',
        '2xl': '48px',
        '3xl': '64px',
        gutter: '24px',
        'margin-mobile': '16px',
        'margin-desktop': '40px',
      },
      borderRadius: {
        // DESIGN.md "Shapes" scale, transcribed exactly.
        sm: '0.25rem',
        DEFAULT: '0.5rem',
        md: '0.75rem',
        lg: '1rem',
        xl: '1.5rem',
        full: '9999px',
      },
      boxShadow: {
        // DESIGN.md "Elevation & Depth"
        'level-1': '0 2px 4px 0 rgba(11, 28, 48, 0.05)',
        'level-2': '0 10px 20px 0 rgba(11, 28, 48, 0.1)',
        // DESIGN.md "AI Elements": Tertiary glow, #8B5CF6 @ 15% opacity
        'ai-glow': '0 0 15px rgba(139, 92, 246, 0.15)',
      },
      maxWidth: {
        canvas: '1440px',
      },
      backdropBlur: {
        nav: '12px',
      },
    },
  },
  plugins: [],
} satisfies Config
