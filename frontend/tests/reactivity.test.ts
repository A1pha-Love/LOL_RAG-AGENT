/**
 * Pinpoint test: does mutating a turn obtained from the store reach a reactive
 * reader?
 *
 * If this passes, the store's reactive plumbing is sound and the problem is in
 * how the turn object is created/handed to `applyEvent`. If it fails, `reactive`
 * is not tracking the objects pushed into the array.
 */

import { describe, expect, it, vi } from 'vitest'
import { effect, nextTick, reactive } from 'vue'

describe('reactive array of objects (Vue semantics)', () => {
  it('only tracks writes made through a reactive reference', async () => {
    // The trap behind the reported bug, isolated: the object stored in a reactive
    // array is proxied, but the plain reference kept by the caller is not, so
    // writes through it are invisible. Wrapping the reference fixes it — which is
    // why `newTurn` returns `reactive(...)`.
    const state = reactive({ turns: [] as Array<{ answer: string }> })
    const raw = { answer: '' }
    state.turns.push(raw)

    const seen: string[] = []
    const stop = effect(() => seen.push(state.turns[0].answer))

    raw.answer = 'through the raw reference'
    await nextTick()
    expect(seen.at(-1)).toBe('', 'writes through the raw reference are not tracked')

    const wrapped = reactive(raw)
    wrapped.answer = 'through a reactive reference'
    await nextTick()

    stop()
    expect(seen.at(-1)).toBe('through a reactive reference')
  })

  it('DOES track a reactive-wrapped turn', async () => {
    // This is the shape useAgent uses: the reference handed to applyEvent is
    // itself reactive, so writes notify the template immediately.
    const state = reactive({ turns: [] as Array<{ answer: string }> })
    const turn = reactive({ answer: '' })
    state.turns.push(turn)

    const seen: string[] = []
    const stop = effect(() => seen.push(state.turns[0].answer))

    turn.answer = 'x'
    await nextTick()

    stop()
    expect(seen.at(-1)).toBe('x')
  })

  it('tracks mutations made through a reactive-wrapped turn', async () => {
    // This is the shape useAgent must use: the reference handed to applyEvent
    // must itself be reactive, otherwise writes to it bypass the proxy stored in
    // the array and the template never sees them.
    const state = reactive({ turns: [] as Array<{ answer: string }> })
    const turn = reactive({ answer: '' })
    state.turns.push(turn)

    const seen: string[] = []
    const stop = effect(() => seen.push(state.turns[0].answer))

    turn.answer = 'x'
    await nextTick()

    stop()
    expect(seen.at(-1)).toBe('x')
  })

  it('tracks mutations made through the proxy read back from the array', async () => {
    const state = reactive({ turns: [] as Array<{ answer: string }> })
    const turn = { answer: '' }
    state.turns.push(turn)

    const seen: string[] = []
    const stop = effect(() => seen.push(state.turns[0].answer))

    // Reading the element out of a reactive array returns the proxy.
    state.turns[0].answer = 'via proxy'
    await nextTick()

    stop()
    expect(seen.at(-1)).toBe('via proxy')
  })
})

describe('store turn identity', () => {
  it('the object pushed is the object the store exposes', async () => {
    vi.resetModules()
    const apiStub = {
      health: vi.fn(),
      listSessions: vi.fn(async () => ({ sessions: [] })),
      createSession: vi.fn(),
      getSession: vi.fn(),
      deleteSession: vi.fn(),
    }
    vi.doMock('@/lib/api', () => ({
      api: apiStub,
      streamTurn: async function* () {
        await new Promise(() => {})
      },
    }))

    const { store } = await import('@/composables/useAgent')
    void store.ask('问题')
    await nextTick()

    const fromArray = store.state.turns[0]
    expect(fromArray).toBeDefined()
    expect(fromArray.question).toBe('问题')

    // Mutating through the array reference must notify readers.
    const seen: string[] = []
    const stop = effect(() => seen.push(store.state.turns[0].answer))
    fromArray.answer = '直接写入'
    await nextTick()
    stop()

    expect(seen.at(-1)).toBe('直接写入')
  })
})
