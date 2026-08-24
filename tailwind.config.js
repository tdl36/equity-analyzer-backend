/** @type {import('tailwindcss').Config} */

// Every colour below resolves through a CSS custom property defined per theme
// in src/theme.css. The `<alpha-value>` placeholder is what keeps Tailwind's
// opacity modifiers working — `bg-white/10`, `bg-amber-500/20` and friends all
// still compose correctly, they just read a themed triple instead of a literal.
// See the header comment in src/theme.css for what each ramp rung means.
const rgb = (v) => `rgb(var(${v}) / <alpha-value>)`;

const slate = {
  50: rgb('--c-slate-50'),
  100: rgb('--c-slate-100'),
  200: rgb('--c-slate-200'),
  300: rgb('--c-slate-300'),
  400: rgb('--c-slate-400'),
  500: rgb('--c-slate-500'),
  600: rgb('--c-slate-600'),
  700: rgb('--c-slate-700'),
  800: rgb('--c-slate-800'),
  900: rgb('--c-slate-900'),
  950: rgb('--c-slate-900'),
};

// One accent per theme (anti-slop rule 7). Charlie reaches for amber, yellow
// and brand interchangeably for the same brand moments, so all three alias to
// the single accent ramp rather than introducing a second brand colour.
const accent = {
  50: rgb('--c-accent-300'),
  100: rgb('--c-accent-300'),
  200: rgb('--c-accent-300'),
  300: rgb('--c-accent-300'),
  400: rgb('--c-accent-400'),
  500: rgb('--c-accent-500'),
  600: rgb('--c-accent-600'),
  700: rgb('--c-accent-700'),
  800: rgb('--c-accent-800'),
  900: rgb('--c-accent-900'),
  950: rgb('--c-accent-900'),
};

const ramp = (prefix) => ({
  50: rgb(`--c-${prefix}-300`),
  100: rgb(`--c-${prefix}-300`),
  200: rgb(`--c-${prefix}-300`),
  300: rgb(`--c-${prefix}-300`),
  400: rgb(`--c-${prefix}-400`),
  500: rgb(`--c-${prefix}-500`),
  600: rgb(`--c-${prefix}-600`),
  700: rgb(`--c-${prefix}-700`),
  800: rgb(`--c-${prefix}-700`),
  900: rgb(`--c-${prefix}-700`),
  950: rgb(`--c-${prefix}-700`),
});

const pos = ramp('pos');
const neg = ramp('neg');
const info = ramp('info');

module.exports = {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Overlay / primary-contrast colour. Inverts per theme so the ~1,900
        // `bg-white/x` and `border-white/x` glass utilities read as a light film
        // on Dusk and a dark film on Ink/Oak/Bloc. `black` is intentionally left
        // alone so `bg-black/30` modal scrims still work on light grounds.
        white: rgb('--c-white'),

        slate,
        gray: slate,
        zinc: slate,
        neutral: slate,

        amber: accent,
        yellow: accent,
        brand: accent,
        orange: accent,

        green: pos,
        emerald: pos,
        teal: pos,
        red: neg,
        rose: neg,
        blue: info,
        sky: info,
        indigo: info,

        accent: {
          DEFAULT: 'var(--accent)',
          ink: 'var(--accent-ink)',
          on: 'var(--on-accent)',
        },

        // Direct handles for the non-ramp tokens, for markup that wants the
        // semantic name rather than a ramp rung.
        ground: 'var(--bg)',
        surface: 'var(--surface)',
        rule: 'var(--line)',
        'rule-strong': 'var(--line-strong)',
      },
      fontFamily: {
        // Preflight sets `font-sans` on <html>, so remapping it themes the whole
        // app's body copy without a single class change in app.jsx.
        sans: ['var(--font-body)'],
        serif: ['var(--font-display)'],
        mono: ['var(--font-mono)'],
        display: ['var(--font-display)'],
      },
      borderRadius: {
        DEFAULT: 'var(--radius)',
        sm: 'var(--radius)',
        md: 'var(--radius)',
        lg: 'var(--radius-lg)',
        xl: 'var(--radius-xl)',
        '2xl': 'var(--radius-2xl)',
        '3xl': 'var(--radius-3xl)',
        // `rounded-full` stays a true pill: it carries avatars, the Charlie
        // logo and status dots, which should not square off even in Bloc.
        full: '9999px',
      },
      letterSpacing: {
        label: 'var(--label-tracking)',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
