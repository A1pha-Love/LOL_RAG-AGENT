/**
 * SSE framing tests.
 *
 * The reported failure: several JSON events arrived concatenated inside one
 * "event", producing `无法解析事件：{...} {...}`. The cause was line endings — the
 * backend emits CRLF, and a parser that splits on "\n\n" finds no boundary at all,
 * so it buffered the whole stream and treated it as a single frame.
 *
 * These tests therefore feed the parser exactly what the backend sends, including
 * chunk boundaries in awkward places.
 */

import { describe, expect, it } from 'vitest'
import { readSse } from '@/lib/sse'

/** Build a ReadableStream that emits the given byte chunks verbatim. */
function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const events: Array<{ event: string; data: string }> = []
  for await (const event of readSse(stream)) events.push(event)
  return events
}

/** What sse-starlette actually writes: CRLF, with a blank line between events. */
const CRLF = '\r\n'

describe('readSse framing', () => {
  it('splits CRLF-separated events (the reported bug)', async () => {
    const payload =
      'event: status' + CRLF + 'data: {"type":"status","phase":"rewriting"}' + CRLF + CRLF +
      'event: rewrite' + CRLF + 'data: {"type":"rewrite","query":"q"}' + CRLF + CRLF

    const events = await collect(streamOf(payload))

    expect(events).toHaveLength(2)
    expect(events[0]).toEqual({ event: 'status', data: '{"type":"status","phase":"rewriting"}' })
    expect(events[1]).toEqual({ event: 'rewrite', data: '{"type":"rewrite","query":"q"}' })
  })

  it('splits LF-separated events too', async () => {
    const payload = 'event: a\ndata: 1\n\nevent: b\ndata: 2\n\n'
    const events = await collect(streamOf(payload))
    expect(events.map((event) => event.data)).toEqual(['1', '2'])
  })

  it('splits lone-CR-separated events', async () => {
    const payload = 'event: a\rdata: 1\r\revent: b\rdata: 2\r\r'
    const events = await collect(streamOf(payload))
    expect(events.map((event) => event.data)).toEqual(['1', '2'])
  })

  it('is not confused by a chunk boundary inside the CRLFCRLF separator', async () => {
    // The realistic worst case: the network splits exactly between the CR and LF.
    const events = await collect(
      streamOf('event: a\r\ndata: 1\r', '\n\r\nevent: b\r\ndata: 2\r\n\r\n'),
    )
    expect(events.map((event) => event.data)).toEqual(['1', '2'])
  })

  it('is not confused by a chunk boundary inside a JSON payload', async () => {
    const events = await collect(
      streamOf('event: a\r\ndata: {"type":"de', 'lta","text":"你好"}\r\n\r\n'),
    )
    expect(events).toHaveLength(1)
    expect(JSON.parse(events[0].data)).toEqual({ type: 'delta', text: '你好' })
  })

  it('ignores keepalive comment lines', async () => {
    const payload =
      ': ping - 2026-01-01' + CRLF + CRLF +
      'event: a' + CRLF + 'data: 1' + CRLF + CRLF

    const events = await collect(streamOf(payload))
    expect(events).toEqual([{ event: 'a', data: '1' }])
  })

  it('joins multi-line data fields with a newline', async () => {
    const payload = 'event: a\r\ndata: line1\r\ndata: line2\r\n\r\n'
    const events = await collect(streamOf(payload))
    expect(events[0].data).toBe('line1\nline2')
  })

  it('strips exactly one leading space after the colon', async () => {
    // Per the SSE spec only the first space is framing; the rest is the value.
    const single = await collect(streamOf('event: a\r\ndata: spaced\r\n\r\n'))
    expect(single[0].data).toBe('spaced')

    const double = await collect(streamOf('event: a\r\ndata:  spaced\r\n\r\n'))
    expect(double[0].data).toBe(' spaced')
  })

  it('emits a final frame that lacks a trailing blank line', async () => {
    const payload = 'event: a\r\ndata: 1\r\n\r\nevent: b\r\ndata: 2'
    const events = await collect(streamOf(payload))
    expect(events.map((event) => event.data)).toEqual(['1', '2'])
  })

  it('defaults the event name to "message"', async () => {
    const events = await collect(streamOf('data: 1\n\n'))
    expect(events[0].event).toBe('message')
  })

  it('skips frames that carry no data field', async () => {
    const events = await collect(streamOf('event: a\r\n\r\nevent: b\r\ndata: 2\r\n\r\n'))
    expect(events).toEqual([{ event: 'b', data: '2' }])
  })
})
