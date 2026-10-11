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

const CR = 13
const LF = 10

interface Line {
  text: string
  /** Index just past this line's terminator. */
  end: number
}

/** The next line in `buffer` from `start`, or null when it is still incomplete. */
function readLine(buffer: string, start: number): Line | null {
  for (let index = start; index < buffer.length; index += 1) {
    const code = buffer.charCodeAt(index)
    if (code === LF) {
      return { text: buffer.slice(start, index), end: index + 1 }
    }
    if (code === CR) {
      // A following LF belongs to the same terminator, so the line only ends here
      // once that decision is possible.
      if (index + 1 >= buffer.length) return null
      const next = buffer.charCodeAt(index + 1)
      return { text: buffer.slice(start, index), end: index + (next === LF ? 2 : 1) }
    }
  }
  return null
}

/** Parse one frame's field lines into an event. */
function parseFrame(frame: string): { event: string; data: string } | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const line of splitLines(frame)) {
    applyField(line, (name, value) => {
      if (name === 'event') event = value
      else if (name === 'data') dataLines.push(value)
    })
  }

  if (!dataLines.length) return null
  return { event, data: dataLines.join('\n') }
}

/** Every complete line in `text`, ignoring a trailing partial line. */
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
      if (text.charCodeAt(index + 1) === LF) index += 1
      start = index + 1
    }
  }

  if (start < text.length) lines.push(text.slice(start))
  return lines
}

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

export async function* readSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<{ event: string; data: string }> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      let frameStart = 0
      let cursor = 0
      while (true) {
        const line = readLine(buffer, cursor)
        if (!line) break
        if (line.text === '') {
          // A blank line dispatches the accumulated frame.
          const parsed = parseFrame(buffer.slice(frameStart, cursor))
          if (parsed) yield parsed
          frameStart = line.end
        }
        cursor = line.end
      }
      // Keep only the unconsumed remainder: complete frames must not be re-parsed
      // when the next chunk arrives.
      if (frameStart > 0) buffer = buffer.slice(frameStart)
    }

    // The stream can end without a final blank line.
    const tail = parseFrame(buffer)
    if (tail) yield tail
  } finally {
    // Release the lock so the body is not left locked if the consumer stops early
    // (for example after an abort). `cancel()` is deliberately not called: the
    // caller owns the request and aborts it through its own AbortController.
    try {
      reader.releaseLock()
    } catch {
      /* already released */
    }
  }
}
