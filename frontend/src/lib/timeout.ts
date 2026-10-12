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

// ============================================================================
// 中文说明
//
// 这一层解决的是「流卡住了但界面还在转圈」的问题：连接可能半开（TCP 层不报错、
// 也永远不返回数据），此时 UI 一直显示「生成中…」，用户只能干等。
//
// 三档阈值，各自防的**不是**同一件事：
//
//   1) 首事件超时 FIRST_EVENT_TIMEOUT_MS = 60s
//      防「一直没有第一个响应」。这一轮还没开始产出任何东西：推理模型要先
//      「想」一遍才吐第一个 token，所以第一段静默天然比后面长，阈值给得最宽。
//   2) 空闲超时 IDLE_TIMEOUT_MS = 45s
//      防「中途断流」。已经产出过事件，之后长时间没有任何新内容——通常是
//      连接掉了，或者服务端进程死了、socket 却没有报错。因为已经有证据说明
//      链路是通的，所以容忍度比首事件小。
//   3) 总时长上限 TOTAL_TIMEOUT_MS = 150s（在 composables/useAgent.ts 里配合
//      AbortController 使用）
//      防「前两档都覆盖不到的情况」：空闲看门狗只在迭代器被 await 时才会推进，
//      如果 fetch 本身卡在建立连接上没有返回 response，迭代器根本还没被创建，
//      看门狗就无从计时。总时长基于墙上时钟，无论卡在哪一步都会触发，并 abort
//      掉请求把 fetch 释放出来。
//
// 为什么是「迭代器每一步与定时器赛跑」（withIdleTimeout）而不是在旁边跑一个
// 定时器：半开连接下 reader.read() 会永远 pending 且不抛错，只有让 next() 与
// 超时竞争，才能把「没有任何事件到达」这件事变成可捕获的异常。
//
// 被谁使用：composables/useAgent.ts 的 ask()。
// 单测：tests/timeout.test.ts。
// ============================================================================

/** Longest tolerated wait for the first byte of a turn. */
// 中文说明：等待本轮第一个事件的最长时间（60 秒）。
export const FIRST_EVENT_TIMEOUT_MS = 60_000

/** Longest tolerated gap once the turn has started producing events. */
// 中文说明：本轮已开始产出后，两次事件之间允许的最长间隔（45 秒）。
export const IDLE_TIMEOUT_MS = 45_000

/**
 * Absolute ceiling for one turn.
 *
 * The idle watchdog covers a stalled read, but it cannot cover a `fetch` that
 * hangs before any response is returned (connection setup, or a proxy that reads
 * the request and never answers). This bound guarantees the UI always recovers.
 */
// 中文说明：一轮问答的绝对墙上时钟上限（150 秒），作为最后的兜底。
export const TOTAL_TIMEOUT_MS = 150_000

/**
 * 中文说明：超时错误，携带两个信息供界面使用。
 *
 * seconds: 触发时用的阈值（秒），用于拼出「超过 60 秒」这样的文案。
 * phase:   'first' 表示等首个响应就超时；'idle' 表示中途断流。
 *          注意 useAgent 触发总时长兜底时传的是 'idle'，因为那种情况通常是
 *          请求卡在建立连接上，用「响应中断」的文案比「等待首个响应」更贴切。
 */
export class StreamTimeoutError extends Error {
  readonly seconds: number
  readonly phase: 'first' | 'idle'

  constructor(seconds: number, phase: 'first' | 'idle') {
    // 两种阶段给不同的提示语：前者怀疑模型繁忙/网络不通，后者建议重试。
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
// 中文说明：给异步生成器套一层「每一步都限时」的包装。
//
// 参数:
//     source: 上游生成器（这里是 streamTurn 产出的事件流）。
//     onTimeout: 超时回调。这一层只负责「通知超时」，具体动作（abort 请求）
//         交给调用方，因此这里不需要知道 AbortController 的存在。
//     firstMs / idleMs: 覆盖默认阈值，便于测试注入极小值。
//
// 产出:
//     与 source 同类型的事件，只多了一层超时保护。
//
// 异常:
//     任一步超过阈值即抛 StreamTimeoutError（抛出前先调用 onTimeout 中断请求）。
export async function* withIdleTimeout<T>(
  source: AsyncGenerator<T>,
  onTimeout: () => void,
  {
    firstMs = FIRST_EVENT_TIMEOUT_MS,
    idleMs = IDLE_TIMEOUT_MS,
  }: { firstMs?: number; idleMs?: number } = {},
): AsyncGenerator<T> {
  const iterator = source[Symbol.asyncIterator]()
  /** 是否已产出过至少一个事件：决定下一步用 firstMs 还是 idleMs。 */
  let started = false

  try {
    while (true) {
      const limit = started ? idleMs : firstMs
      let timer: ReturnType<typeof setTimeout> | undefined

      // 每次迭代新建定时器：这是「每步重置」的空闲计时器，而不是整轮只跑一次的
      // 总计时器（总计时器在 composables/useAgent.ts 里另有一套）。
      const timeout = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new StreamTimeoutError(limit / 1000, started ? 'idle' : 'first')), limit)
      })

      let result: IteratorResult<T>
      try {
        // 谁先完成谁说了算：正常拿到下一个事件，或超时抛错。
        result = await Promise.race([iterator.next(), timeout])
      } catch (error) {
        // Abort the underlying request so the server stops working on it too.
        onTimeout()
        throw error
      } finally {
        // 必须清理定时器：否则生成器结束后定时器仍会触发，抛出没有接收者的
        // unhandled rejection。
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
    // 中文说明：这里刻意 **不 await** 上游的 return()。卡住的连接上 return()
    // 会去等那个永远不 resolve 的 read；若 await 它，超时异常会被这个等待吞掉，
    // 表现为「超时了但界面一直不恢复」。真正解开阻塞的是 onTimeout() 里的
    // abort。catch 掉 rejection 是因为此时已无人关心这个结果。
    void iterator.return?.(undefined as never)?.catch(() => undefined)
  }
}
