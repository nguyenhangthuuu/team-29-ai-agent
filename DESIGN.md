# DESIGN.md — Belfius ReBel Neo-Terminal Design System
### Impeccable Design Specification | QTEM Data Challenge (QDC) 2026–2027

---

## 1. Core Vision & Surface Mode
- **Surface Mode:** `Operate` (High-density institutional financial terminal; clarity and scanability over decorative fluff).
- **Inspiration:** Bloomberg Terminal modular information architecture, modernized with a bright, minimalist, elegant European private banking aesthetic.
- **Tone:** Sharp, elegant, minimalist, and authoritative.

---

## 2. Color Palette & Brand Tokens

### Primary Brand
- **Brand Crimson Accent:** `#B40036` (Official Belfius Deep Crimson / Magenta Red).
  - Used surgically: active navigation states, primary action CTA buttons, execution triggers, and focus rings.
- **Brand Glow Aura:** `rgba(180, 0, 54, 0.18)` (Subtle, restrained accent glow; zero neon oversaturation).
- **Soft Lavender / Lilac Accent:** `#E9D5FF` / `#F3E8FF` (ReBel secondary token for thematic category tags and informative pill chips).

### Surfaces & Canvas (Modern Bright Neo-Terminal)
- **Canvas Base:** `#0D111A` (Deep Obsidian Slate; high-contrast, clean, and modern).
- **Surface Cards:** `#131926` (Subtle elevation with flat 1px micro-borders).
- **Micro-Border:** `#1F293D` (Crisp structural separator; zero blurred heavy shadows).
- **Card Hover:** `#182030` (Subtle interactive feedback).

### Typography & Contrast Hierarchy
- **Text Primary:** `#F8FAFC` (Crisp White for key financial figures and active titles).
- **Text Secondary:** `#94A3B8` (Refined Slate Gray for metric labels, captions, and table headers).
- **Text Muted:** `#64748B` (Subtle metadata, timestamps, and ISIN codes).

### Financial Semantic Accents
- **Alpha / Positive Gain:** `#10B981` (Emerald Green).
- **Drawdown / Loss:** `#F43F5E` (Rose Red).
- **Warning / Constraint Clamp:** `#F59E0B` (Amber).

---

## 3. Typography & Typesetting Rules
- **Font Family (UI Copy):** `'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
  - Style: Slim, elegant, minimal lettering.
  - Letter Spacing: `-0.015em` to `0.005em` (sharp, refined geometric tracking).
  - Font Weights: Light (`300`) for secondary descriptions, Regular (`400`) for body prose, Medium (`500`) for table headers, Semibold (`600`) for primary KPI figures.
- **Financial Numerals:** `'JetBrains Mono', 'Fira Code', 'SF Mono', monospace`
  - Rule: Mandatory tabular numbers (`font-variant-numeric: tabular-nums; font-feature-settings: 'tnum'`).
  - Guarantee: Decimals and commas align vertically across rows without jitter.

---

## 4. Layout Boundaries & Anti-Slop Enforcement
- **Strict Boundary Constraint:** Every text block, table cell, and badge MUST stay within container boundaries.
  - Mandatory `white-space: nowrap` + `text-overflow: ellipsis` on ticker/name columns.
  - Zero horizontal page overflow; all grid layouts must compute with `minmax(0, 1fr)`.
- **Card Clutter Elimination:** No cards-inside-cards with nested drop shadows. Clean 1px micro-borders only.
- **Zero Emojis:** Strictly professional financial iconography (Lucide / Feather SVG vectors) or pure typography.
