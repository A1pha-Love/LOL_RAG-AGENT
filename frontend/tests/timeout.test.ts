/**
 * Tests for the stream watchdog.
 *
 * A stalled connection is the failure the user actually experiences: the UI sits
 * on "generating…" forever. These cover both shapes of stall — a stream that never
 * delivers its first event, and one that goes quiet mid-turn.
 *
 * Fake timers keep the suite fast: the timeouts are awaited for real, so
 * advancing the clock is what triggers them.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StreamTimeoutError, withIdleTimeout } from '@/lib/timeout'

/**
 * Abortable stand-in for a stalled connection.
 *
 * A bare `new Promise(() => {})` would deadlock the *test*: nothing can reject
 * it, so the iteration never settles. Real streams unblock when the request is
 * aborted, which is exactly what `withIdleTimeout` does on timeout, so the stub
 * mirrors that.
 */
function stallUntilAborted() {
  let abort = () => {}
  const aborted = new Promise<never>((_resolve, reject) => {
    abort = () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' }))
  })
  return { aborted, abort: () => abort() }
}

async function* neverYields(stall: ReturnType<typeof stallUntilAborted>) {
  await stall.aborted
}

async function* yieldsThenStalls(stall: ReturnType<typeof stallUntilAborted>) {
  yield 'first'
  await stall.aborted
}

async function* yieldsAll(values: string[]) {
  for (const value of values) yield value
}

/** Runs the guarded stream to completion, reporting how it ended. */
async function settle(iterable: AsyncIterable<unknown>) {
  const seen: unknown[] = []
  try {
    for await (const value of iterable) seen.push(value)
    return { status: 'completed' as const, seen }
  } catch (error) {
    return { status: 'threw' as const, error, seen }
  }
}

/** Lets pending microtasks run, then fires the pending timer. */
async function advance(ms: number) {
  await vi.advanceTimersByTimeAsync(ms)
}

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('withIdleTimeout', () => {
  it('passes a healthy stream through untouched', async () => {
    const outcome = await settle(withIdleTimeout(yieldsAll(['a', 'b', 'c']), () => {}))
    expect(outcome.status).toBe('completed')
    expect(outcome.seen).toEqual(['a', 'b', 'c'])
  })

  it('times out a stream that never produces a first event', async () => {
    const stall = stallUntilAborted()
    const running = settle(withIdleTimeout(neverYields(stall), stall.abort))
    await advance(60_000)

    const outcome = await running
    expect(outcome.status).toBe('threw')
    expect((outcome as { error: unknown }).error).toBeInstanceOf(StreamTimeoutError)
  })

  it('times out a stream that goes quiet mid-turn', async () => {
    const stall = stallUntilAborted()
    const running = settle(withIdleTimeout(yieldsThenStalls(stall), stall.abort))
    await advance(45_000)

    const outcome = await running
    expect(outcome.status).toBe('threw')
    expect((outcome as { error: unknown }).error).toMatchObject({ phase: 'idle' })
  })

  it('does not fire while events keep arriving', async () => {
    // Each event resets the watchdog, so a long turn that keeps streaming must
    // survive far beyond a single idle window.
    async function* slowButAlive() {
      for (let index = 0; index < 5; index += 1) {
        yield index
        await new Promise((resolve) => setTimeout(resolve, 30_000))
      }
    }

    const running = settle(withIdleTimeout(slowButAlive(), () => {}))
    await advance(30_000 * 5)

    const outcome = await running
    expect(outcome.status).toBe('completed')
    expect(outcome.seen).toHaveLength(5)
  })

  it('explains the failure in terms the user can act on', async () => {
    const stall = stallUntilAborted()
    const running = settle(withIdleTimeout(neverYields(stall), stall.abort))
    await advance(60_000)

    const outcome = await running
    expect((outcome as { error: Error }).error.message).toMatch(/中止/)
  })
})
