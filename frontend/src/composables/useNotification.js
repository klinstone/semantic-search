import { reactive } from 'vue'

// Single shared snackbar state for the whole app.
const state = reactive({
  visible: false,
  message: '',
  color: 'info',
  timeout: 4000
})

function show(message, color = 'info', timeout = 4000) {
  state.message = message
  state.color = color
  state.timeout = timeout
  state.visible = true
}

export function useNotification() {
  return {
    state,
    show,
    success: (m, t) => show(m, 'success', t ?? 4000),
    error: (m, t) => show(m, 'error', t ?? 6000),
    info: (m, t) => show(m, 'info', t ?? 4000),
    warning: (m, t) => show(m, 'warning', t ?? 5000)
  }
}
