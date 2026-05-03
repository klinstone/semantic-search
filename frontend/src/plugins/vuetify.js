import { createVuetify } from 'vuetify'
import { ru, en } from 'vuetify/locale'

export default createVuetify({
  theme: {
    defaultTheme: 'light',
    themes: {
      light: {
        dark: false,
        colors: {
          primary: '#1F3A8A',     // deep indigo, distinctive but academic
          secondary: '#475569',
          surface: '#FFFFFF',
          background: '#F6F7FA',
          error: '#B91C1C',
          info: '#1D4ED8',
          success: '#15803D',
          warning: '#B45309'
        }
      }
    }
  },
  locale: {
    locale: 'ru',
    fallback: 'en',
    messages: { ru, en }
  },
  defaults: {
    VBtn: { variant: 'flat' },
    VTextField: { variant: 'outlined', density: 'comfortable' },
    VSelect: { variant: 'outlined', density: 'comfortable' },
    VCard: { rounded: 'lg' },
    VChip: { rounded: 'sm' }
  }
})
