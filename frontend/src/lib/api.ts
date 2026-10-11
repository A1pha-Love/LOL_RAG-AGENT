/** Typed client for the local FastAPI backend. */

import { readSse } from './sse'
import type {
  HealthReport,
  SessionDetail,
  SessionSummary,
  StreamEvent,
} from '@/types'

/** Vite proxies /api in development; in a build the backend serves the bundle. */
const BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const payload = await response.json()
      if (payload?.detail) detail = typeof payload.detail === 'string' ? payload.detail : detail
    } catch {
      /* the body was not JSON; keep the status line */
    }
    throw new Error(detail)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  health: () => request<HealthReport>('/api/health'),

  listSessions: () => request<{ sessions: SessionSummary[] }>('/api/sessions'),

  createSession: (title = '') =>
    request<{ session_id: string; title: string }>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),

  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),

  deleteSession: (id: string) =>
    request<void>(`/api/sessions/${id}`, { method: 'DELETE' }),
}

/**
 * Stream one conversational turn.
 *
 * Returns an async iterator of parsed events so the caller can render each one as
 * it arrives. Passing a signal lets the UI stop generation, which also aborts the
 * request so the backend stops spending tokens.
 */
export async function* streamTurn(
  question: string,
  sessionId: string | null,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${BASE}/api/sessions/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`请求失败：${response.status} ${response.statusText}`)
  }
  if (!response.body) {
    throw new Error('响应没有可读流，无法流式接收')
  }

  for await (const frame of readSse(response.body)) {
    if (!frame.data) continue
    try {
      yield JSON.parse(frame.data) as StreamEvent
    } catch {
      // A malformed frame must not kill the whole stream.
      yield { type: 'error', message: `无法解析事件：${frame.data.slice(0, 120)}` }
    }
  }
}
