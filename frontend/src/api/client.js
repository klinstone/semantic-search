/**
 * Lightweight wrapper over fetch + XMLHttpRequest (for upload progress).
 *
 * Knows about the unified backend error format:
 *   { "error": { "code": "...", "message": "...", "details": {...} } }
 *
 * The base URL defaults to a relative `/api/v1` so the same code works both
 * with the Vite dev proxy and the production nginx proxy.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

export class ApiError extends Error {
  constructor(status, code, message, details = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

async function parseError(response) {
  let body = null
  try {
    body = await response.json()
  } catch {
    /* not JSON */
  }
  const err = body?.error
  if (err && typeof err === 'object') {
    return new ApiError(
      response.status,
      err.code || 'UNKNOWN',
      err.message || response.statusText || `HTTP ${response.status}`,
      err.details ?? null
    )
  }
  return new ApiError(
    response.status,
    'UNKNOWN',
    body?.message || response.statusText || `HTTP ${response.status}`,
    null
  )
}

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`
  let response
  try {
    response = await fetch(url, options)
  } catch (e) {
    throw new ApiError(0, 'NETWORK_ERROR', 'Сервер недоступен', { cause: String(e) })
  }
  if (!response.ok) {
    throw await parseError(response)
  }
  if (response.status === 204) return null
  const ct = response.headers.get('content-type') || ''
  if (ct.includes('application/json')) {
    return await response.json()
  }
  return await response.text()
}

/** Upload a file with progress events. fetch() doesn't support upload progress. */
function uploadWithProgress(path, file, { onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const fd = new FormData()
    fd.append('file', file)

    xhr.open('POST', `${BASE_URL}${path}`)

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    })

    xhr.addEventListener('load', () => {
      let body = null
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body)
      } else {
        const err = body?.error
        reject(
          new ApiError(
            xhr.status,
            err?.code || 'UNKNOWN',
            err?.message || xhr.statusText || `HTTP ${xhr.status}`,
            err?.details ?? null
          )
        )
      }
    })

    xhr.addEventListener('error', () => {
      reject(new ApiError(0, 'NETWORK_ERROR', 'Сервер недоступен'))
    })

    xhr.addEventListener('abort', () => {
      reject(new ApiError(0, 'ABORTED', 'Загрузка прервана'))
    })

    xhr.send(fd)
  })
}

export const api = {
  health() {
    return request('/health')
  },

  listDocuments({ limit = 20, offset = 0, status = null } = {}) {
    const params = new URLSearchParams()
    params.set('limit', String(limit))
    params.set('offset', String(offset))
    if (status) params.set('status', status)
    return request(`/documents?${params.toString()}`)
  },

  getDocument(id) {
    return request(`/documents/${encodeURIComponent(id)}`)
  },

  uploadDocument(file, opts = {}) {
    return uploadWithProgress('/documents', file, opts)
  },

  deleteDocument(id) {
    return request(`/documents/${encodeURIComponent(id)}`, { method: 'DELETE' })
  },

  search(query, { limit = 10, documentIds = null } = {}) {
    return request('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        limit,
        document_ids: documentIds
      })
    })
  }
}
