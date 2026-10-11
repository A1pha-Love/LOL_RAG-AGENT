/**
 * Guard against a stream that stalls.
 *
 * A turn legitimately has quiet stretches: the reasoning model "thinks" before
 * its first token, and tool calls run between tokens. But if nothing at all
 * arrives for long enough, the connection is dead while the UI still shows
 * "generating…" — indistinguishable from a hang, and the user waits forever.
 *
 * Two thresholds, because the two phases differ: waiting for the first token is
 * slower than waiting between tokens.
 */

/** Longest tolerated wait for the first byte of a turn. */
export const FIRST_EVENT_TIMEOUT_MS = 60_000

/** Longest tolerated gap once the turn has started producing events. */
export const IDLE_TIMEOUT_MS = 45_000

/**
 * Absolute ceiling for one turn.
 *
 * The idle watchdog covers a stalled read, but it cannot cover a `fetch` that
 * hangs before any response is returned (connection setup, or a proxy that reads
 * the request and never answers). This bound guarantees the UI always recovers.
 */
export const TOTAL_TIMEOUT_MS = 150_000

export class StreamTimeoutError extends Error {
  readonly seconds: number
  readonly phase: 'first' | 'idle'

  constructor(seconds: number, phase: 'first' | 'idle') {
    super(
      phase === 'first'
        ? `等待首个响应超过 ${seconds} 秒，已中止。可能是模型繁忙或网络中断。`
        : `响应中断超过 ${seconds} 秒没有新内容，已中止。可以重新提问或稍后重试。`,
    )
    this.name = 'StreamTimeoutError'
    this.seconds = seconds
    this.phase = phase
  }
}

/**
 * Wrap an async iterable so each `next()` races the watchdog.
 *
 * Racing the iterator rather than running a timer beside it means a stalled read
 * is detected even when the underlying socket never errors — a half-open
 * connection can otherwise leave `read()` pending indefinitely.
 */
export async function* withIdleTimeout<T>(
  source: AsyncGenerator<T>,
  onTimeout: () => void,
  {
    firstMs = FIRST_EVENT_TIMEOUT_MS,
    idleMs = IDLE_TIMEOUT_MS,
  }: { firstMs?: number; idleMs?: number } = {},
): AsyncGenerator<T> {
  const iterator = source[Symbol.asyncIterator]()
  let started = false

  try {
    while (true) {
      const limit = started ? idleMs : firstMs
      let timer: ReturnType<typeof setTimeout> | undefined

      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new StreamTimeoutError(limit / 1000, started ? 'idle' : 'first')), limit)
      })

      let result: IteratorResult<T>
      try {
        result = await Promise.race([iterator.next(), timeout])
      } catch (error) {
        // Abort the underlying request so the server stops working on it too.
        onTimeout()
        throw error
      } finally {
        if (timer) clearTimeout(timer)
      }

      if (result.done) return
      started = true
      yield result.value
    }
  } finally {
    // Releasing the generator must not wait for a pending read: on a stalled
    // connection `return()` would block on the same read that never resolves, so
    // the failure would hang instead of surfacing. `onTimeout()` has already
    // aborted the request, which is what actually unblocks the reader.
    void iterator.return?.(undefined as never)?.catch(() => undefined)
  }
}
