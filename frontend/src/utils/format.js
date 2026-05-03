/**
 * Format a byte count into a short human-readable string ("3.4 MB").
 */
export function formatBytes(bytes) {
  if (bytes == null || bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  const value = bytes / Math.pow(1024, i)
  return `${i === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[i]}`
}

/** ISO 8601 (UTC) -> локальная дата-время. */
export function formatDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  return d.toLocaleString('ru-RU', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

export const STATUS_LABELS = {
  pending: 'В очереди',
  processing: 'Обрабатывается',
  indexed: 'Готов',
  failed: 'Ошибка'
}

export const STATUS_COLORS = {
  pending: 'grey',
  processing: 'info',
  indexed: 'success',
  failed: 'error'
}

export const TERMINAL_STATUSES = new Set(['indexed', 'failed'])

export function isTerminalStatus(status) {
  return TERMINAL_STATUSES.has(status)
}

export function mimeIcon(mime) {
  switch (mime) {
    case 'application/pdf':
      return 'mdi-file-pdf-box'
    case 'text/plain':
      return 'mdi-file-document-outline'
    case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
      return 'mdi-file-word-box'
    default:
      return 'mdi-file-outline'
  }
}
