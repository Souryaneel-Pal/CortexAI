---
name: MindScope AI
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#3e4850'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#6e7881'
  outline-variant: '#bec8d2'
  surface-tint: '#006591'
  primary: '#006591'
  on-primary: '#ffffff'
  primary-container: '#0ea5e9'
  on-primary-container: '#003751'
  inverse-primary: '#89ceff'
  secondary: '#006b5f'
  on-secondary: '#ffffff'
  secondary-container: '#6df5e1'
  on-secondary-container: '#006f64'
  tertiary: '#6d3bd7'
  on-tertiary: '#ffffff'
  tertiary-container: '#a986ff'
  on-tertiary-container: '#3e0097'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#c9e6ff'
  primary-fixed-dim: '#89ceff'
  on-primary-fixed: '#001e2f'
  on-primary-fixed-variant: '#004c6e'
  secondary-fixed: '#71f8e4'
  secondary-fixed-dim: '#4fdbc8'
  on-secondary-fixed: '#00201c'
  on-secondary-fixed-variant: '#005048'
  tertiary-fixed: '#e9ddff'
  tertiary-fixed-dim: '#d0bcff'
  on-tertiary-fixed: '#23005c'
  on-tertiary-fixed-variant: '#5516be'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.04em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  2xl: 48px
  3xl: 64px
  gutter: 24px
  margin-mobile: 16px
  margin-desktop: 40px
---

## Brand & Style

The design system is engineered for a premium healthcare SaaS environment, prioritizing clarity, clinical precision, and technological sophistication. The aesthetic sits at the intersection of **Corporate Modern** and **Minimalism**, drawing heavy influence from high-performance developer tools to instill a sense of reliability and speed.

The brand personality is trustworthy and data-driven. It avoids clinical coldness by utilizing soft gradients and subtle glassmorphism to signify the presence of AI. The interface should feel "quiet" to allow complex medical data to remain the focal point, using generous whitespace to reduce cognitive load for healthcare professionals.

## Colors

The palette is anchored by **Sky Blue** (Primary) to evoke trust and **Teal** (Secondary) to maintain a modern medical feel. **Violet** (Tertiary) is reserved exclusively for AI-generated insights, predictions, and automated features, creating a distinct visual mental model for the user.

**Implementation Details:**
- **Light Mode:** Use white (#FFFFFF) for the primary canvas and Slate-50 (#F8FAFC) for background layering and sidebars.
- **Dark Mode:** Use Slate-950 (#0F172A) for the base and Slate-900 (#1E293B) for cards and elevated surfaces.
- **Status Colors:** Use standard semantic reds and ambers for alerts, but desaturate them slightly to fit the premium aesthetic.
- **Gradients:** Use subtle linear gradients (Primary to Secondary) for high-impact call-to-actions or data visualizations.

## Typography

The design system utilizes **Inter** for all primary communication due to its exceptional legibility in data-dense environments and neutral, professional tone. To emphasize the "AI/Technical" nature of the product, **JetBrains Mono** is used for small labels, metadata, and status indicators.

- **Weight Usage:** Use Semibold (600) for headlines to maintain a strong hierarchy without appearing overly aggressive.
- **Letter Spacing:** Apply slight negative tracking on large headlines to tighten the visual lockup.
- **Readability:** Ensure body text never drops below 14px for accessibility in clinical settings.

## Layout & Spacing

This design system employs a **Fixed-Fluid Hybrid Grid**. Content is constrained to a maximum width of 1440px for readability, while background elements and sidebars extend to the screen edge.

- **Grid Model:** A 12-column grid for desktop (24px gutter) and a 4-column grid for mobile (16px gutter).
- **Rhythm:** An 8px linear scale drives all padding and margins. 
- **Density:** Provide two density modes: "Standard" for general patient overviews and "Compact" for intensive data tables and diagnostic logs.

## Elevation & Depth

Hierarchy is established using **Tonal Layering** and **Ambient Shadows**. 

- **Level 0 (Base):** Background color (Slate-50 in Light, Slate-950 in Dark).
- **Level 1 (Cards):** Surface color with a 1px border (Slate-200) and a very soft, diffused shadow (Y: 2px, Blur: 4px, Opacity: 5%).
- **Level 2 (Popovers/Modals):** Higher elevation with a larger shadow spread (Y: 10px, Blur: 20px, Opacity: 10%).
- **AI Elements:** Elements using the Tertiary color should utilize a subtle glow (Box-shadow with #8B5CF6 at 15% opacity) to signify an active intelligence layer.
- **Glassmorphism:** Use `backdrop-filter: blur(12px)` for navigation bars and sticky headers to maintain context while scrolling.

## Shapes

The shape language is "Soft-Modern." All primary containers, including cards and input fields, use a 0.5rem (8px) radius. Larger layout containers or dashboard widgets should scale up to 1rem (16px) or 1.5rem (24px) for a more approachable, premium feel. 

Buttons and interactive chips should maintain a consistent 8px radius to feel surgical and precise, avoiding fully rounded pill shapes except for status tags.

## Components

### Buttons
- **Primary:** Solid Sky-500 (#0EA5E9) with white text. 
- **Secondary:** Slate-100 background with Slate-900 text.
- **AI Action:** Gradient background (Violet to Teal) with white text.

### Cards
- Cards must use a 1px border (Slate-200) even when shadowed to ensure definition on white backgrounds.
- Padding inside cards should default to `lg` (24px).

### Input Fields
- Use a 1px Slate-300 border that transitions to Sky-500 on focus.
- Labels use `label-sm` (JetBrains Mono) for a technical appearance.

### Data Visualization
- Charts should use the Primary, Secondary, and Tertiary palette exclusively.
- Use rounded corners on bar charts (radius: 4px) to match the UI shape language.

### Specialized Components
- **Insight Banner:** A specific component for AI feedback using a light Violet tint background and a subtle 1px border.
- **Patient Status Chip:** Small, semi-transparent chips with a dot indicator for "Stable," "Critical," or "Pending."