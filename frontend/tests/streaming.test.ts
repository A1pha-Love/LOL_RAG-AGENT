/**
 * Reactivity tests: the transcript must update *while* a turn is streaming.
 *
 * Reported bug: the answer only appeared after switching to another conversation
 * and back. That is the signature of updates landing on an object the template is
 * not tracking, so these tests assert on the store's public state via a Vue
 * effect — the same mechanism the template uses — rather than only after a turn
 * completes.
 *
 *     npm test
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { effect, nextTick } from 'vue'

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
  pinSession: vi.fn(async (id: string, pinned: boolean) => ({ session_id: id, pinned })),
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

/** A stream the test drives step by step, so mid-turn state can be inspected. */
function controllableStream() {
  let release: (() => void) | null = null
  const queue: unknown[] = []
  let finished = false

  async function* generator() {
    while (!finished) {
      if (!queue.length) {
        await new Promise<void>((resolve) => {
          release = resolve
        })
      }
      if (finished) return
      while (queue.length) yield queue.shift()
    }
  }

  return {
    generator,
    async send(event: unknown) {
      queue.push(event)
      release?.()
      release = null
      await new Promise((resolve) => setTimeout(resolve, 0))
    },
    async end() {
      finished = true
      release?.()
      await new Promise((resolve) => setTimeout(resolve, 0))
    },
  }
}

async function loadStore() {
  vi.resetModules()
  return (await import('@/composables/useAgent')).store
}

/**
 * Poll until `check()` passes.
 *
 * Driven streams cross several await boundaries (the test's send, the generator's
 * yield, the store's pump), so asserting immediately after `send()` would test
 * microtask ordering rather than the behaviour under test. Polling also flushes
 * Vue's scheduler, which is what the template waits on.
 */
async function waitFor(check: () => boolean, label: string, timeoutMs = 2_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    await nextTick()
    if (check()) return
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error(`timed out waiting for: ${label}`)
}

beforeEach(() => {
  vi.clearAllMocks()
  apiStub.listSessions.mockResolvedValue({ sessions: [] })
})

describe('live streaming updates', () => {
  it('publishes each delta to a reactive reader while the turn is still open', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()

    // Exactly what the template does: read `turns` inside a reactive effect.
    const seen: string[] = []
    const stop = effect(() => {
      const turn = store.state.turns[0]
      if (turn) seen.push(turn.answer)
    })

    const pending = store.ask('问题')
    await nextTick()

    await stream.send({ type: 'delta', text: '第一段' })
    await waitFor(() => store.state.turns[0]?.answer === '第一段', 'first delta to be visible')
    expect(seen.at(-1)).toBe('第一段')

    await stream.send({ type: 'delta', text: '第二段' })
    await waitFor(() => store.state.turns[0]?.answer === '第一段第二段', 'second delta to be appended')
    expect(seen.at(-1)).toBe('第一段第二段')

    await stream.send({
      type: 'done',
      session_id: 's1',
      answer: '第一段第二段',
      retrieval_query: null,
      tool_calls: [],
      seconds: 1,
    })
    await stream.end()
    await pending
    await nextTick()

    stop()
    expect(seen.at(-1)).toBe('第一段第二段')
  })

  it('shows the turn immediately, before any event arrives', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()

    const observed: number[] = []
    const stop = effect(() => observed.push(store.state.turns.length))

    const pending = store.ask('问题')
    // The conversation is created before streaming, so the turn appears one tick
    // after that round-trip rather than synchronously.
    await waitFor(() => store.state.turns.length === 1, 'the turn to appear')

    expect(store.state.turns).toHaveLength(1)
    expect(store.state.turns[0].question).toBe('问题')
    expect(store.state.turns[0].streaming).toBe(true)
    expect(observed).toContain(1)

    await stream.send({ type: 'done', session_id: 's1', answer: '答', retrieval_query: null, tool_calls: [], seconds: 1 })
    await stream.end()
    await pending
    stop()
  })

  it('publishes the retrieval details mid-turn for the inspector', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()

    const chunksSeen: number[] = []
    const stop = effect(() => chunksSeen.push(store.state.turns[0]?.chunks.length ?? -1))

    const pending = store.ask('问题')
    await nextTick()

    await stream.send({ type: 'tools', names: ['search_project_knowledge'] })
    await waitFor(
      () => store.state.turns[0]?.toolCalls.length === 1,
      'tool call to be visible',
    )
    await stream.send({
      type: 'retrieval',
      retrieval_query: 'q',
      results: [{ source: 'kbase/a.md', heading_path: 'A > B', rerank_score: 0.9 }],
    })
    await waitFor(() => store.state.turns[0]?.chunks.length === 1, 'chunks to be visible')

    expect(store.state.turns[0].chunks).toHaveLength(1)
    expect(chunksSeen.at(-1)).toBe(1)

    await stream.send({ type: 'done', session_id: 's1', answer: '答', retrieval_query: 'q', tool_calls: [], seconds: 1 })
    await stream.end()
    await pending
    stop()
  })
})
