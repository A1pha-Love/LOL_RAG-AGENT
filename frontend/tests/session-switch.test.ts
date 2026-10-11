/**
 * Switching conversations must stay possible while a turn is streaming.
 *
 * Reported bug: with the model running, the sidebar's conversation buttons were
 * disabled, so the user was locked in place until the answer finished. The cause was
 * that the in-flight turn lived only in `state.turns`, which a session switch
 * replaces — so switching had to be blocked to avoid losing it.
 *
 * These tests pin the intended behaviour: the stream keeps running in the
 * background, its live turn survives switching away and back, and neither the
 * conversation list nor the transcript is corrupted.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

const apiStub = {
  health: vi.fn(async () => ({
    status: 'ok',
    index_exists: true,
    index_stale: false,
    index_reason: 'ok',
    session_db: '',
  })),
  listSessions: vi.fn(async () => ({ sessions: [] })),
  createSession: vi.fn(async () => ({ session_id: 'new', title: '' })),
  getSession: vi.fn(async (id: string) => ({
    session_id: id,
    title: id,
    created_at: '',
    updated_at: '',
    turns: [
      { role: 'user', content: `${id} 的旧问题`, created_at: '', retrieval_query: null, tool_calls: [] },
      { role: 'assistant', content: `${id} 的旧答案`, created_at: '', retrieval_query: null, tool_calls: [] },
    ],
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

/** A stream the test drives, with an abort hook so stalls can be ended. */
function controllableStream() {
  let release: (() => void) | null = null
  let aborted = false
  const queue: unknown[] = []
  let finished = false

  async function* generator(_q: unknown, _s: unknown, signal: AbortSignal) {
    signal.addEventListener('abort', () => {
      aborted = true
      release?.()
    })
    while (!finished && !aborted) {
      if (!queue.length) {
        await new Promise<void>((resolve) => {
          release = resolve
        })
      }
      if (finished || aborted) return
      while (queue.length) yield queue.shift()
    }
  }

  return {
    generator,
    get aborted() {
      return aborted
    },
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

async function waitFor(check: () => boolean, label: string, timeoutMs = 2_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    await nextTick()
    if (check()) return
    await new Promise((resolve) => setTimeout(resolve, 5))
  }
  throw new Error(`timed out waiting for: ${label}`)
}

async function loadStore() {
  vi.resetModules()
  return (await import('@/composables/useAgent')).store
}

beforeEach(() => {
  vi.clearAllMocks()
  apiStub.listSessions.mockResolvedValue({ sessions: [] })
})

describe('switching sessions while streaming', () => {
  it('does not report the app as busy for a different session', async () => {
    // `streaming` is what the sidebar uses to disable its buttons, so it must
    // describe the visible conversation, not the app as a whole.
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const pending = store.ask('问题')
    await waitFor(() => store.state.streaming, 'streaming to start')

    expect(store.state.streaming).toBe(true)
    expect(store.state.streamingSessionId).toBe('a')

    stream.end()
    await pending
  })

  it('keeps the live turn when switching away and back', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const pending = store.ask('第一问')
    await stream.send({ type: 'delta', text: '正在生成' })
    await waitFor(
      () => store.state.turns.at(-1)?.answer === '正在生成',
      'the streamed delta to appear',
    )

    // Switch to another conversation and keep streaming there without finishing.
    await store.selectSession('b')
    expect(store.state.turns.at(-1)?.question).toBe('b 的旧问题')

    // Returning restores the still-running turn rather than the stored snapshot.
    await store.selectSession('a')
    const restored = store.state.turns.find((turn) => turn.question === '第一问')
    expect(restored, 'the in-flight turn must survive a switch').toBeDefined()
    expect(restored?.answer).toBe('正在生成')

    stream.end()
    await pending
  })

  it('keeps applying deltas while the user is in another conversation', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const pending = store.ask('问题')
    await store.selectSession('b')

    await stream.send({ type: 'delta', text: '后台生成中' })
    await stream.send({
      type: 'done',
      session_id: 'a',
      answer: '后台生成中',
      retrieval_query: null,
      tool_calls: [],
      seconds: 1,
    })
    await stream.end()
    await pending

    // The visible conversation is untouched...
    expect(store.state.turns.at(-1)?.question).toBe('b 的旧问题')
    // ...and the background turn finished cleanly.
    expect(store.state.streaming).toBe(false)
    expect(store.state.streamingSessionId).toBeNull()
  })

  it('lets a second conversation be asked while the first still streams', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const first = store.ask('第一问')
    await waitFor(() => store.state.streaming, 'first turn to start')

    await store.selectSession('b')
    // A different conversation must be usable: the composer is only blocked by the
    // conversation it belongs to.
    expect(store.state.streaming).toBe(false)

    stream.end()
    await first
  })

  it('refuses a second turn in the same conversation', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const first = store.ask('第一问')
    await waitFor(() => store.state.streaming, 'first turn to start')

    await store.ask('第二问')
    expect(store.state.turns.filter((turn) => turn.streaming)).toHaveLength(1)

    stream.end()
    await first
  })

  it('keeps stop() targeting the streaming conversation', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const pending = store.ask('问题')
    await waitFor(() => store.state.streaming, 'streaming to start')
    await store.selectSession('b')

    store.stop()
    await pending

    expect(stream.aborted).toBe(true)
    expect(store.state.streaming).toBe(false)
  })

  it('reports streaming per session in the list', async () => {
    const stream = controllableStream()
    streamStub = stream.generator as unknown as typeof streamStub
    const store = await loadStore()
    await store.selectSession('a')

    const pending = store.ask('问题')
    await waitFor(() => store.state.streaming, 'streaming to start')

    expect(store.isSessionStreaming('a')).toBe(true)
    expect(store.isSessionStreaming('b')).toBe(false)

    stream.end()
    await pending
    expect(store.isSessionStreaming('a')).toBe(false)
  })
})
