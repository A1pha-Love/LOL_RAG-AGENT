/**
 * Server-Sent Events reader.
 *
 * The browser's EventSource cannot be used here: it only issues GET requests and
 * cannot carry a body, while this API needs POST with a JSON payload. So the SSE
 * framing is parsed by hand from a streamed fetch response.
 *
 * Framing follows the SSE specification, where a line ends with CRLF, LF or a lone
 * CR, and a frame ends at a blank line. That matters in practice: the backend emits
 * CRLF (sse-starlette does), so a parser that looks for "\n\n" finds no boundary at
 * all, accumulates the entire stream, and eventually fails with several JSON
 * objects concatenated into one "event".
 *
 * Multi-line `data:` fields are joined with a newline. Comment lines (a leading
 * ":", used for keepalive pings) are ignored.
 */

// ============================================================================
// 中文说明
//
// 为什么不能用浏览器的 EventSource？
//   1. EventSource 只能发起 GET 请求，不能带请求体；本项目的问答必须把
//      {question, session_id} 作为 JSON POST 上去。
//   2. EventSource 无法读取自定义状态码、无法自定义请求头，也无法在拿到
//      响应前主动中止。
//   因此这里改用 fetch 拿到 ReadableStream<Uint8Array>，再手写 SSE 分帧。
//
// 分帧为什么必须按规范处理 CR / LF / CRLF 三种行结束符？
//   SSE 规范（WHATWG HTML "Server-sent events"）规定：一行以 CRLF、单独的 LF
//   或单独的 CR 结束；一个「帧」由一个空行结束。后端用的是 sse-starlette，
//   它发的是 CRLF（`\r\n`）。项目早期实现只查找 `"\n\n"` 作为帧边界，于是
//   CRLF 流里根本找不到边界：解析器把整段流一直攒着，直到最后把多个 JSON
//   对象拼在一起当成一个事件去 JSON.parse，报「无法解析事件」，而且所有回答
//   是等生成结束才一次性出现的。修复办法就是下面这样严格按规范逐行判定。
//
// 整体结构（三段）：
//   readLine()    —— 从缓冲区当前位置取出一整行，行尾不完整时返回 null 等待。
//   parseFrame()  —— 把一帧内的若干行按 `字段: 值` 解析成 {event, data}。
//   readSse()     —— 把字节流解码成文本，按空行切帧并逐个 yield 出去。
//
// 被谁使用：lib/api.ts 的 streamTurn()。
// ============================================================================

/** 回车符 '\r' 的字符码。 */
const CR = 13
/** 换行符 '\n' 的字符码。 */
const LF = 10

interface Line {
  text: string
  /** Index just past this line's terminator. */
  end: number
}

/**
 * The next line in `buffer` from `start`, or null when it is still incomplete.
 */
// 中文说明：从 `start` 起取出下一整行。
//
// 返回:
//     `{ text, end }`：text 不含行结束符，end 指向行结束符**之后**的位置
//     （即下一行的起点）。若从 start 到缓冲区末尾都还没有行结束符，
//     返回 null，表示这一行被网络分片截断了，需要等下一个 chunk 到齐再判。
//
// 关键点：遇到 CR 时不能立即定案，因为后面可能跟着 LF（CRLF），也可能没有
// （单独的 CR）。如果 CR 正好是缓冲区最后一个字符，就无法分辨——此时返回
// null 让调用方等待，否则会把 CRLF 拆成「一行 + 一个空行」，凭空多出一个帧
// 边界，把一条合法事件截断。
function readLine(buffer: string, start: number): Line | null {
  for (let index = start; index < buffer.length; index += 1) {
    const code = buffer.charCodeAt(index)
    if (code === LF) {
      // 单独的 LF：行内容为 [start, index)，下一行从 index+1 开始。
      return { text: buffer.slice(start, index), end: index + 1 }
    }
    if (code === CR) {
      // A following LF belongs to the same terminator, so the line only ends here
      // once that decision is possible.
      if (index + 1 >= buffer.length) return null
      const next = buffer.charCodeAt(index + 1)
      // CRLF 时行结束符是 2 个字符，单独的 CR 是 1 个字符。
      return { text: buffer.slice(start, index), end: index + (next === LF ? 2 : 1) }
    }
  }
  // 没找到任何行结束符：这一行还没收完。
  return null
}

/** Parse one frame's field lines into an event. */
// 中文说明：解析一帧（不含结尾空行的那段文本）。
//
// 返回:
//     `{ event, data }`；若这一帧没有任何 data 行（例如纯心跳/注释帧）则返回
//     null，调用方会跳过它。
//
// 注意 data 的多行是用 `\n` 拼接的，这是 SSE 规范要求的：同一帧内多个
// `data:` 行之间以换行相连再交给消费者。
function parseFrame(frame: string): { event: string; data: string } | null {
  // 规范默认事件名为 "message"；后端显式发了 `event: ...` 时会被覆盖。
  // 本项目 api.ts 并不区分事件名，所有帧都当 JSON 解析。
  let event = 'message'
  const dataLines: string[] = []

  for (const line of splitLines(frame)) {
    applyField(line, (name, value) => {
      if (name === 'event') event = value
      else if (name === 'data') dataLines.push(value)
      // 其余字段（id、retry）本项目用不到，静默忽略。
    })
  }

  if (!dataLines.length) return null
  return { event, data: dataLines.join('\n') }
}

/** Every complete line in `text`, ignoring a trailing partial line. */
// 中文说明：把一整帧文本按 CR / LF / CRLF 切成行。
//
// 与 readLine 不同，这里的文本已经确定是一个完整帧（以空行结尾），所以不需要
// 「等待更多数据」的逻辑；但仍要正确识别三种行结束符，且 CRLF 只能算一个
// 换行（因此遇到 CR 后若下一个是 LF 就多跳一格）。
function splitLines(text: string): string[] {
  const lines: string[] = []
  let start = 0

  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index)
    if (code === LF) {
      lines.push(text.slice(start, index))
      start = index + 1
    } else if (code === CR) {
      lines.push(text.slice(start, index))
      // CRLF：把 LF 一并吃掉，避免下一轮再切出一个空行。
      if (text.charCodeAt(index + 1) === LF) index += 1
      start = index + 1
    }
  }

  // 末尾没有换行符的残余内容也算一行。
  if (start < text.length) lines.push(text.slice(start))
  return lines
}

/**
 * 中文说明：解析单行 `字段: 值` 并回调。
 *
 * 规范细节：
 *   - 空行是帧结束标志，不属于字段；以 `:` 开头的是注释（后端用它做心跳），
 *     都要跳过。
 *   - 没有冒号的行，整行作为字段名、值为空串。
 *   - 冒号后**最多一个**前导空格属于格式，不是值的一部分（`data: 1` → `1`）。
 */
function applyField(line: string, apply: (name: string, value: string) => void): void {
  if (!line || line.startsWith(':')) return // blank or keepalive comment
  const colon = line.indexOf(':')
  if (colon === -1) {
    apply(line, '')
    return
  }
  // A single leading space after the colon is part of the framing, not the value.
  const value = line.slice(colon + 1).replace(/^ /, '')
  apply(line.slice(0, colon), value)
}

/**
 * 中文说明：把一个字节流按 SSE 规范解码成一个个事件。
 *
 * 参数:
 *     body: fetch 响应的 ReadableStream<Uint8Array>。
 *
 * 产出:
 *     `{ event, data }`。data 是原始字符串，调用方负责 JSON.parse。
 *
 * 两个必须小心的边界（下面的代码就是围绕它们写的）：
 *   1. **一次 read() 拿到的不是一帧**：数据可能把一行、甚至一个多字节 UTF-8
 *      汉字切成两半。所以用 TextDecoder 的 `{ stream: true }` 保留半个字符的
 *      状态，并且只消费「已经见到行结束符」的部分，残余留在 buffer 里。
 *   2. **流可能没有末尾空行**：模型结束、连接关闭时最后一帧可能不含收尾空行，
 *      因此循环结束后要再解析一次 buffer 的残余。
 */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<{ event: string; data: string }> {
  const reader = body.getReader()
  // 默认按 UTF-8 解码；stream: true 允许跨 chunk 的半个字符。
  const decoder = new TextDecoder()
  /** 尚未消费的文本；始终以「未完整的行」开始和结束。 */
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      // stream: true 表示「后面还有数据」，让解码器缓存不完整的多字节序列。
      buffer += decoder.decode(value, { stream: true })

      // frameStart 指向当前正在累积的这一帧的起点，cursor 是逐行扫描的游标，
      // 两者不同：cursor 会走过帧内的每一行，frameStart 只在遇到空行时才前移。
      let frameStart = 0
      let cursor = 0
      while (true) {
        const line = readLine(buffer, cursor)
        // 行还没收全（或被分片截断），跳出等下一个 chunk。
        if (!line) break
        if (line.text === '') {
          // A blank line dispatches the accumulated frame.
          // 空行 = 帧边界：把它之前的内容（frameStart..cursor）交给 parseFrame。
          const parsed = parseFrame(buffer.slice(frameStart, cursor))
          if (parsed) yield parsed
          // 下一帧从空行结束符之后开始。
          frameStart = line.end
        }
        cursor = line.end
      }
      // Keep only the unconsumed remainder: complete frames must not be re-parsed
      // when the next chunk arrives.
      // 丢弃已消费部分，只留残余——否则下一个 chunk 到达后会重复解析已经
      // yield 过的帧。
      if (frameStart > 0) buffer = buffer.slice(frameStart)
    }

    // The stream can end without a final blank line.
    // 连接结束但服务端没发收尾空行时，残余里的最后一帧也要交出去。
    const tail = parseFrame(buffer)
    if (tail) yield tail
  } finally {
    // Release the lock so the body is not left locked if the consumer stops early
    // (for example after an abort). `cancel()` is deliberately not called: the
    // caller owns the request and aborts it through its own AbortController.
    // 中文说明：只释放锁，不 cancel 流——请求的生命周期由调用方的
    // AbortController 掌管（见 useAgent.ask），这里 cancel 会与它重复且可能
    // 抛出「已中止」的次级错误。releaseLock 放在 try 里是因为生成器可能已经
    // 被 return() 提前终结，此时锁可能已不存在。
    try {
      reader.releaseLock()
    } catch {
      /* already released */
    }
  }
}
