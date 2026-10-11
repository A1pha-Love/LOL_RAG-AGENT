/**
 * Store behaviour tests.
 *
 * These cover the awkward cases that empirical testing with a real browser could
 * not reach reliably: a stalled stream, a stream that throws mid-way, and a slow
 * session-list refresh. All three previously left `state.streaming` stuck at
 * true, which disabled the composer, the suggested questions and the sidebar —
 * the whole UI becoming unresponsive rather than merely slow.
 *
 *     npm test
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

const apiStub = {
  health: vi.fn(async () => ({
    status: 'ok',
    index_exists: true,
    index_stale: false,
    index_reason: 'ok',
    session_db: '',
  })),
  listSessions: vi.fn(async () => ({ sessions: [] })),
  createSession: vi.fn(async () => ({ session_id: 's1', title: '' })),
  getSession: vi.fn(async () => ({
    session_id: 's1',
    title: '',
    created_at: '',
    updated_at: '',
    turns: [],
  })),
  deleteSession: vi.fn(async () => undefined),
}

let streamStub: (...args: unknown[]) => AsyncGenerator<unknown>

vi.mock('@/lib/api', () => ({
  get api() {
    return apiStub
  },
  get streamTurn() {
    return streamStub
  },
}))

/** Yields the given events; `stallAt` makes the generator hang at that index. */
function scripted(events: unknown[], { stallAt = -1 } = {}) {
  return async function* () {
    for (const [index, event] of events.entries()) {
      if (index === stallAt) await new Promise(() => {})
      yield event
    }
  }
}

const doneTurn = (overrides: Record<string, unknown> = {}) => [
  { type: 'rewrite', query: 'q', fallback: null },
  { type: 'delta', text: '答' },
  {
    type: 'done',
    session_id: 's9',
    answer: '答案',
    retrieval_query: 'q',
    tool_calls: [],
    seconds: 1,
    ...overrides,
  },
]

async function loadStore() {
  vi.resetModules()
  return (await import('@/composables/useAgent')).store
}

beforeEach(() => {
  vi.clearAllMocks()
  apiStub.listSessions.mockResolvedValue({ sessions: [] })
  streamStub = scripted(doneTurn())
})

describe('ask', () => {
  it('clears the streaming flag and records the answer', async () => {
    const store = await loadStore()

    await store.ask('问题')

    expect(store.state.streaming).toBe(false)
    expect(store.state.turns).toHaveLength(1)
    expect(store.state.turns[0].answer).toBe('答案')
  })

  it('adopts the session id the server created implicitly', async () => {
    const store = await loadStore()

    await store.ask('问题')

    expect(store.state.activeSessionId).toBe('s9')
  })

  it('frees the UI when the stream throws', async () => {
    streamStub = async function* () {
      throw new Error('connection lost')
    }
    const store = await loadStore()

    await store.ask('问题')

    expect(store.state.streaming).toBe(false)
    expect(store.state.turns[0].error).toMatch(/connection lost/)
  })

  it('resolves ask() even when the session refresh never settles', async () => {
    // Regression: the refresh was awaited inside the finally block, so a refresh
    // that never resolved meant ask() never resolved either. Callers awaiting the
    // turn hung forever, which is how the whole UI (composer, suggestions,
    // sidebar) ended up unresponsive rather than merely slow.
    apiStub.listSessions.mockImplementation(() => new Promise(() => {}))
    const store = await loadStore()

    const outcome = await Promise.race([
      store.ask('问题').then(() => 'resolved'),
      new Promise((resolve) => setTimeout(() => resolve('hung'), 250)),
    ])

    expect(outcome).toBe('resolved')
    expect(store.state.streaming).toBe(false)
    expect(store.state.turns[0].answer).toBe('答案')
  })

  it('still refreshes the session list after a turn', async () => {
    apiStub.listSessions.mockResolvedValue({
      sessions: [
        {
          session_id: 's9',
          title: '问题',
          created_at: '',
          updated_at: '',
          turn_count: 2,
        },
      ],
    })
    const store = await loadStore()

    await store.ask('问题')
    await new Promise((resolve) => setTimeout(resolve, 20))

    expect(apiStub.listSessions).toHaveBeenCalled()
    expect(store.state.sessions).toHaveLength(1)
  })

  it('ignores a second question while one is in flight', async () => {
    let started = 0
    streamStub = async function* () {
      started += 1
      yield { type: 'delta', text: '答' }
      await new Promise(() => {})
    }
    const store = await loadStore()

    void store.ask('第一问')
    await new Promise((resolve) => setTimeout(resolve, 20))
    await store.ask('第二问')

    expect(started).toBe(1)
    store.stop()
  })

  it('treats a deliberate stop as a stop, not an error', async () => {
    streamStub = async function* (_q: unknown, _s: unknown, signal: AbortSignal) {
      yield { type: 'delta', text: '部分' }
      await new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () =>
          reject(Object.assign(new Error('aborted'), { name: 'AbortError' })),
        )
      })
    }
    const store = await loadStore()

    const pending = store.ask('问题')
    await new Promise((resolve) => setTimeout(resolve, 20))
    store.stop()
    await pending

    expect(store.state.streaming).toBe(false)
    expect(store.state.turns[0].error).toBeNull()
  })

  it('collects retrieval details for the inspector', async () => {
    streamStub = scripted([
      { type: 'rewrite', query: '改写后的查询', fallback: null },
      { type: 'tools', names: ['search_project_knowledge'] },
      {
        type: 'retrieval',
        retrieval_query: '改写后的查询',
        results: [{ source: 'kbase/a.md', heading_path: 'A > B', rerank_score: 0.9 }],
      },
      { type: 'done', session_id: 's1', answer: '答', retrieval_query: '改写后的查询', tool_calls: ['search_project_knowledge'], seconds: 2 },
    ])
    const store = await loadStore()

    await store.ask('那他的胜率呢？')

    const turn = store.state.turns[0]
    expect(turn.toolCalls).toEqual(['search_project_knowledge'])
    expect(turn.chunks).toHaveLength(1)
    expect(store.retrievalView.value?.rewritten).toBe(true)
    expect(store.retrievalView.value?.chunks[0].rerank_score).toBe(0.9)
  })
})

describe('selectSession', () => {
  it('rebuilds the transcript from stored turns', async () => {
    apiStub.getSession.mockResolvedValue({
      session_id: 's2',
      title: '历史会话',
      created_at: '',
      updated_at: '',
      turns: [
        { role: 'user', content: '旧问题', created_at: '', retrieval_query: null, tool_calls: [] },
        {
          role: 'assistant',
          content: '旧答案',
          created_at: '',
          retrieval_query: '改写后的查询',
          tool_calls: ['get_ban_stats'],
        },
      ],
    })
    const store = await loadStore()

    await store.selectSession('s2')

    expect(store.state.activeSessionId).toBe('s2')
    expect(store.state.turns).toHaveLength(1)
    expect(store.state.turns[0].question).toBe('旧问题')
    expect(store.state.turns[0].answer).toBe('旧答案')
    expect(store.state.turns[0].retrievalQuery).toBe('改写后的查询')
    expect(store.state.streaming).toBe(false)
  })

  it('propagates a load failure without wedging the UI', async () => {
    apiStub.getSession.mockRejectedValue(new Error('404 not found'))
    const store = await loadStore()

    await expect(store.selectSession('missing')).rejects.toThrow('404 not found')
    expect(store.state.streaming).toBe(false)
  })
})

describe('startNewSession', () => {
  it('clears the active session and transcript', async () => {
    const store = await loadStore()
    await store.ask('问题')
    expect(store.state.turns).toHaveLength(1)

    store.startNewSession()

    expect(store.state.activeSessionId).toBeNull()
    expect(store.state.turns).toHaveLength(0)
    expect(store.state.selectedTurnId).toBeNull()
  })
})
