/** Typed client for the local FastAPI backend. */

// ============================================================================
// 中文说明
//
// 这一层是前端访问本地 FastAPI 后端的唯一出口：普通请求走 request()（fetch +
// JSON），问答这一轮因为要流式返回，所以单独用 streamTurn()（异步生成器）。
//
// 之所以自己封装而不引入 axios 之类：只需要 fetch，且错误语义很单一——
// 后端用 FastAPI 的 HTTPException 返回 {"detail": "..."}，前端把它取出来当成
// Error.message，好让界面直接展示一句人能看懂的中文而不是「500 Internal
// Server Error」。
//
// 被谁使用：composables/useAgent.ts（健康检查、会话增删改查、发起一轮问答）。
// ============================================================================

import { readSse } from './sse'
import type {
  HealthReport,
  SessionDetail,
  SessionSummary,
  StreamEvent,
} from '@/types'

/**
 * Vite proxies /api in development; in a build the backend serves the bundle.
 *
 * 中文说明：后端地址前缀。
 *
 * 开发时 vite.config.ts 把 /api 代理到本地后端，因此默认空串即可；
 * 构建产物由后端自己托管时同样是同源请求。只有在前后端分开部署时，
 * 才用环境变量 VITE_API_BASE 指定绝对地址（例如 http://127.0.0.1:8000）。
 */
const BASE = import.meta.env.VITE_API_BASE ?? ''

/**
 * 发送一个请求并解析 JSON 响应。
 *
 * 参数:
 *     path: 以 /api 开头的接口路径。
 *     init: 可选 fetch 配置（method/body 等）；headers 可被它覆盖。
 *
 * 返回:
 *     解析后的响应体。204 无内容时返回 undefined。
 *
 * 异常:
 *     响应非 2xx 时抛出 Error，消息优先取后端的 detail 字段。
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    // 默认用状态行兜底，保证任何失败都至少有一个可读信息。
    let detail = `${response.status} ${response.statusText}`
    try {
      const payload = await response.json()
      // detail 只在是字符串时采用：FastAPI 校验失败时它可能是数组或对象，
      // 直接塞进 Error 会显示成 [object Object]。
      if (payload?.detail) detail = typeof payload.detail === 'string' ? payload.detail : detail
    } catch {
      /* the body was not JSON; keep the status line */
    }
    throw new Error(detail)
  }
  // DELETE 返回 204，没有响应体，此时 response.json() 会抛解析错误。
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  /** 后端健康检查：顶栏据此显示「索引就绪 / 过期 / 缺失」。 */
  health: () => request<HealthReport>('/api/health'),

  /** 拉取会话列表（已由后端排好序，前端不再重排）。 */
  listSessions: () => request<{ sessions: SessionSummary[] }>('/api/sessions'),

  /** 新建空会话；title 留空则由后端生成默认标题。 */
  createSession: (title = '') =>
    request<{ session_id: string; title: string }>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify({ title }),
    }),

  /** 读取某个会话的全部历史消息。 */
  getSession: (id: string) => request<SessionDetail>(`/api/sessions/${id}`),

  /** 删除会话（连同其消息）。 */
  deleteSession: (id: string) =>
    request<void>(`/api/sessions/${id}`, { method: 'DELETE' }),

  /** Pin or unpin a conversation; pinned ones sort above the rest. */
  pinSession: (id: string, pinned: boolean) =>
    request<{ session_id: string; pinned: boolean }>(`/api/sessions/${id}/pin`, {
      method: 'PUT',
      body: JSON.stringify({ pinned }),
    }),
}

/**
 * Stream one conversational turn.
 *
 * Returns an async iterator of parsed events so the caller can render each one as
 * it arrives. Passing a signal lets the UI stop generation, which also aborts the
 * request so the backend stops spending tokens.
 */
// 中文说明：发起一轮问答并流式产出事件。
//
// 这里用 fetch + 手写 SSE 解析，而不是浏览器的 EventSource，原因见 lib/sse.ts
// 顶部注释：EventSource 只能发 GET、无法携带请求体，而问答必须 POST JSON。
//
// 参数:
//     question: 用户问题（调用方已 trim）。
//     sessionId: 归属会话；为 null 表示这是一个新会话，后端会在此轮创建它。
//     signal: 用于「停止生成」的 AbortSignal，abort 会同时中断请求，
//         让后端也停止消耗 token。
//
// 产出:
//     逐个解析好的 StreamEvent。注意解析失败不会中断整个流，而是产出一个
//     error 事件（见下方 catch），这样后面的 token 仍能正常显示。
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

  // 注意：这里不经过 request()，因为要在拿到响应头之后立刻开始读流，
  // 不能先 await response.json()。
  if (!response.ok) {
    throw new Error(`请求失败：${response.status} ${response.statusText}`)
  }
  if (!response.body) {
    throw new Error('响应没有可读流，无法流式接收')
  }

  for await (const frame of readSse(response.body)) {
    // 只有注释/心跳的帧没有 data，跳过即可（sse-starlette 会周期性发心跳）。
    if (!frame.data) continue
    try {
      yield JSON.parse(frame.data) as StreamEvent
    } catch {
      // A malformed frame must not kill the whole stream.
      yield { type: 'error', message: `无法解析事件：${frame.data.slice(0, 120)}` }
    }
  }
}
