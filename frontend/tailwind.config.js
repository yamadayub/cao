/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        // Premium color palette
        primary: {
          50: '#faf7f8',
          100: '#f5eff1',
          200: '#eddfe3',
          300: '#dfc5cc',
          400: '#cba3ae',
          500: '#b78291',
          600: '#9d6b7a',
          700: '#8b6f7a', // Main brand color
          800: '#6d4a56',
          900: '#5c3f49',
          950: '#352127',
        },
        accent: {
          DEFAULT: '#9d8189',
          light: '#b9a4ab',
          dark: '#7a636a',
        },
        neutral: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
        },
      },
      fontFamily: {
        serif: ['Cormorant Garamond', 'serif'],
        sans: ['Noto Sans JP', 'sans-serif'],
      },
      fontSize: {
        'display-1': ['5rem', { lineHeight: '1.1', letterSpacing: '-0.02em' }],
        'display-2': ['4rem', { lineHeight: '1.15', letterSpacing: '-0.02em' }],
        'display-3': ['3rem', { lineHeight: '1.2', letterSpacing: '-0.01em' }],
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
        '30': '7.5rem',
      },
      boxShadow: {
        'elegant': '0 4px 24px rgba(0, 0, 0, 0.08)',
        'elegant-lg': '0 8px 40px rgba(0, 0, 0, 0.12)',
        'elegant-xl': '0 12px 60px rgba(0, 0, 0, 0.15)',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.8s ease-out forwards',
        'fade-in': 'fadeIn 0.6s ease-out forwards',
        'bounce-slow': 'bounce 2s infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      transitionDuration: {
        '400': '400ms',
      },
    },
  },
  plugins: [],
};
