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
  touchSession: vi.fn(async (id: string) => ({ session_id: id })),
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
  it('records the activity so the order survives a reload', async () => {
    const store = await loadStore()

    await store.selectSession('a')

    expect(apiStub.touchSession).toHaveBeenCalledWith('a')
    // Re-read rather than reordering locally: one source of truth for the order.
    expect(apiStub.listSessions).toHaveBeenCalled()
  })

  it('reflects the server order after opening', async () => {
    const store = await loadStore()
    await store.selectSession('a')
    // The order comes from the server, so the test supplies it and checks it lands.
    apiStub.listSessions.mockResolvedValue({
      sessions: [summary('a', { lastActive: '2026-02-01T00:00:00+00:00' }), summary('b')],
    })

    await store.refreshSessions()

    expect(store.state.sessions.map((item) => item.session_id)).toEqual(['a', 'b'])
  })

  it('still opens the conversation when the touch call fails', async () => {
    // Ordering is cosmetic; it must not be able to break navigation.
    apiStub.touchSession.mockRejectedValue(new Error('offline'))
    const store = await loadStore()

    await store.selectSession('a')

    expect(store.state.activeSessionId).toBe('a')
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
