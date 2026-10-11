/**
 * Minimal Server-Sent Events reader.
 *
 * The browser's EventSource cannot be used here: it only issues GET requests and
 * cannot carry a body, while this API needs POST with a JSON payload. So the SSE
 * framing is parsed by hand from a streamed fetch response — about forty lines,
 * and it keeps the endpoint a normal POST.
 *
 * Framing handled: fields are `event:` / `data:` lines terminated by a blank
 * line, with multi-line `data:` concatenated by newline (SSE spec). Comment
 * lines (leading `:`, which is how sse-starlette sends keepalive pings) are
 * ignored.
 */
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

      let separator = buffer.indexOf('\n\n')
      while (separator !== -1) {
        const frame = buffer.slice(0, separator)
        buffer = buffer.slice(separator + 2)
        const parsed = parseFrame(frame)
        if (parsed) yield parsed
        separator = buffer.indexOf('\n\n')
      }
    }
    // A final frame may arrive without its trailing blank line.
    const tail = parseFrame(buffer)
    if (tail) yield tail
  } finally {
    reader.releaseLock()
  }
}

function parseFrame(frame: string): { event: string; data: string } | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const rawLine of frame.split('\n')) {
    const line = rawLine.replace(/\r$/, '')
    if (!line || line.startsWith(':')) continue // blank or keepalive comment
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    const value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '')
    if (field === 'event') event = value
    else if (field === 'data') dataLines.push(value)
  }

  if (!dataLines.length) return null
  return { event, data: dataLines.join('\n') }
}
