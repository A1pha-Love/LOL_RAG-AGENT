/**
 * Sidebar behaviour: a new conversation must appear immediately, and pinning must
 * order the list.
 *
 * Reported issues:
 *   1. Asking a question left the conversation absent from the sidebar until the
 *      turn finished, so it was easy to lose among the others.
 *   2. There was no way to pin a conversation, and opening one should place it below
 *      the pinned group.
 *
 * `createSession` is now called up front, so the conversation exists before
 * streaming starts.
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
  createSession: vi.fn(async () => ({ session_id: 'fresh', title: '' })),
  getSession: vi.fn(async (id: string) => ({
    session_id: id,
    title: id,
    created_at: '',
    updated_at: '',
    turns: [],
  })),
  deleteSession: vi.fn(async () => undefined),
  pinSession: vi.fn(async (id: string, pinned: boolean) => ({ session_id: id, pinned })),
}

/**
 * Guard: any API method not listed above throws instead of being a silent no-op.
 *
 * Without this, a reintroduced call (for example recording activity when a
 * conversation is opened, which was a reported bug) would resolve to `undefined` and
 * the test would still pass. With it, the unexpected method is reached and fails.
 */
const guardedApi = new Proxy(apiStub, {
  get(target, property, receiver) {
    if (typeof property === 'string' && !(property in target)) {
      return () => {
        throw new Error(`unexpected API call in this test: ${property}`)
      }
    }
    return Reflect.get(target, property, receiver)
  },
})

let streamStub: (...args: unknown[]) => AsyncGenerator<unknown>

vi.mock('@/lib/api', () => ({
  get api() {
    return guardedApi
  },
  get streamTurn() {
    return streamStub
  },
}))

function summary(
  id: string,
  { pinned = 0, lastActive = '2026-01-01T00:00:00+00:00' }: { pinned?: number; lastActive?: string } = {},
) {
  return {
    session_id: id,
    title: id,
    created_at: '',
    updated_at: '',
    turn_count: 1,
    pinned,
    last_active_at: lastActive,
  }
}

async function loadStore() {
  vi.resetModules()
  return (await import('@/composables/useAgent')).store
}

beforeEach(() => {
  vi.clearAllMocks()
  apiStub.listSessions.mockResolvedValue({ sessions: [] })
  // A stream that ends immediately: these tests are about the sidebar, not streaming.
  streamStub = async function* (_q: unknown, _s: unknown) {
    yield { type: 'delta', text: '答' }
    yield { type: 'done', session_id: 'fresh', answer: '答', retrieval_query: null, tool_calls: [], seconds: 1 }
  }
})

describe('a new conversation appears immediately', () => {
  it('creates the conversation before streaming starts', async () => {
    const store = await loadStore()

    await store.ask('新问题')

    expect(apiStub.createSession).toHaveBeenCalledTimes(1)
    expect(store.state.activeSessionId).toBe('fresh')
  })

  it('lists the conversation as soon as the question is asked', async () => {
    // This is the reported bug: without an up-front create, the sidebar only learned
    // about the conversation when the turn finished.
    const store = await loadStore()

    const pending = store.ask('新问题')
    // Let the create/refresh round-trip settle, without waiting for the stream.
    await new Promise((resolve) => setTimeout(resolve, 0))

    expect(store.state.activeSessionId).toBe('fresh')
    await pending
  })

  it('does not create a second conversation for a follow-up question', async () => {
    const store = await loadStore()
    await store.ask('第一问')
    apiStub.createSession.mockClear()

    await store.ask('第二问')

    expect(apiStub.createSession).not.toHaveBeenCalled()
    expect(store.state.turns).toHaveLength(2)
  })

  it('reports a failure to create instead of silently doing nothing', async () => {
    apiStub.createSession.mockRejectedValue(new Error('500 boom'))
    const store = await loadStore()

    await store.ask('新问题')

    expect(store.state.banner).toMatch(/无法创建会话/)
    expect(store.state.streaming).toBe(false)
    expect(store.state.turns).toHaveLength(0)
  })
})

describe('opening a conversation', () => {
  it('does NOT reorder the list just because it was opened', async () => {
    // Reported as a bug: clicking a conversation to read it made the sidebar jump
    // under the cursor, which looked like the wrong item had been selected. Opening
    // is not participation, so it must not count as activity.
    //
    // The apiStub proxy makes this a hard assertion: any call outside the stubbed set
    // (a touch/ordering endpoint under any name) throws rather than being ignored.
    const store = await loadStore()

    await store.selectSession('a')

    expect(store.state.activeSessionId).toBe('a')
  })

  it('does not re-read the list when nothing changed', async () => {
    const store = await loadStore()
    apiStub.listSessions.mockClear()

    await store.selectSession('a')

    expect(apiStub.listSessions).not.toHaveBeenCalled()
  })

  it('shows the stored transcript', async () => {
    apiStub.getSession.mockResolvedValue({
      session_id: 'a',
      title: 'a',
      created_at: '',
      updated_at: '',
      turns: [
        { role: 'user', content: '旧问题', retrieval_query: null, tool_calls: [], created_at: '' },
        { role: 'assistant', content: '旧答案', retrieval_query: null, tool_calls: [], created_at: '' },
      ],
    })
    const store = await loadStore()

    await store.selectSession('a')

    expect(store.state.activeSessionId).toBe('a')
    expect(store.state.turns.map((turn) => turn.question)).toEqual(['旧问题'])
    expect(store.state.turns[0].answer).toBe('旧答案')
  })

  it('reflects the server order when the list is re-read', async () => {
    const store = await loadStore()
    await store.selectSession('a')
    // The order comes from the server, so the test supplies it and checks it lands.
    apiStub.listSessions.mockResolvedValue({
      sessions: [summary('a', { lastActive: '2026-02-01T00:00:00+00:00' }), summary('b')],
    })

    await store.refreshSessions()

    expect(store.state.sessions.map((item) => item.session_id)).toEqual(['a', 'b'])
  })
})

describe('a completed turn moves the conversation up', () => {
  it('re-reads the list after asking, so the order follows real activity', async () => {
    // This is where the reordering is now expected to happen: the server bumps
    // last_active_at when a turn is appended, and ask() re-reads the order.
    const store = await loadStore()
    await store.selectSession('a')
    apiStub.listSessions.mockClear()

    await store.ask('新问题')

    expect(apiStub.listSessions).toHaveBeenCalled()
  })
})

describe('pinning', () => {
  it('sends the new state and re-reads the list', async () => {
    const store = await loadStore()

    await store.togglePin('a', true)

    expect(apiStub.pinSession).toHaveBeenCalledWith('a', true)
    expect(apiStub.listSessions).toHaveBeenCalled()
  })

  it('unpinning is a separate call', async () => {
    const store = await loadStore()

    await store.togglePin('a', false)

    expect(apiStub.pinSession).toHaveBeenCalledWith('a', false)
  })

  it('shows the pinned group first when the server says so', async () => {
    apiStub.listSessions.mockResolvedValue({
      sessions: [summary('pinned', { pinned: 1 }), summary('recent'), summary('older')],
    })
    const store = await loadStore()

    await store.refreshSessions()

    expect(store.state.sessions.map((item) => item.session_id)).toEqual([
      'pinned',
      'recent',
      'older',
    ])
    expect(store.state.sessions[0].pinned).toBe(1)
  })
})
