/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#f5faff',
        ink: '#1a1a2e',
        muted: '#64748b',
        border: '#e2e8f0',
        accent: '#1e40af',
        'accent-hover': '#1e3a8a',
        'accent-light': '#eff6ff',
        surface: '#ffffff',
        'surface-raised': '#f7fbff',
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        display: ['Playfair Display', 'Georgia', 'serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        'soft': '0 2px 8px -2px rgba(0,0,0,0.08), 0 4px 16px -4px rgba(0,0,0,0.06)',
        'card': '0 4px 24px -4px rgba(37,99,235,0.08), 0 2px 8px -2px rgba(0,0,0,0.04)',
        'glow': '0 0 24px -4px rgba(37,99,235,0.25)',
      },
    },
  },
  plugins: [],
}
