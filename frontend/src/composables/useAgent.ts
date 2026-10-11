/**
 * Application state and turn orchestration.
 *
 * Implemented as a module-level singleton rather than a store library: the state
 * is small (session list, one transcript, one in-flight stream) and the only
 * asynchronous logic is a single streamed turn, so Pinia or Vuex would add a
 * dependency without removing complexity.
 *
 * The whole state object is `reactive`, so templates bind to `store.turns`
 * directly with no `.value` unwrapping.
 */

import { computed, reactive } from 'vue'
import { api, streamTurn } from '@/lib/api'
import { StreamTimeoutError, TOTAL_TIMEOUT_MS, withIdleTimeout } from '@/lib/timeout'
import type {
  HealthReport,
  RetrievedChunk,
  SessionSummary,
  StreamEvent,
} from '@/types'

/** One displayed turn: a question and its answer or in-flight progress. */
export interface UiTurn {
  id: number
  question: string
  answer: string
  streaming: boolean
  error: string | null
  /** Set when the stream stalled; the UI offers a one-click retry. */
  timedOut: boolean
  status: string | null
  retrievalQuery: string | null
  rewriteFallback: string | null
  toolCalls: string[]
  chunks: RetrievedChunk[]
  seconds: number | null
}

/** What the retrieval inspector shows for the selected turn. */
export interface RetrievalView {
  query: string | null
  rewritten: boolean
  fallback: string | null
  tools: string[]
  chunks: RetrievedChunk[]
}

let nextTurnId = 1
let controller: AbortController | null = null

const state = reactive({
  sessions: [] as SessionSummary[],
  activeSessionId: null as string | null,
  turns: [] as UiTurn[],
  streaming: false,
  selectedTurnId: null as number | null,
  health: null as HealthReport | null,
  banner: null as string | null,
})

const selectedTurn = computed<UiTurn | null>(
  () => state.turns.find((turn) => turn.id === state.selectedTurnId) ?? state.turns.at(-1) ?? null,
)

const retrievalView = computed<RetrievalView | null>(() => {
  const turn = selectedTurn.value
  if (!turn) return null
  return {
    query: turn.retrievalQuery,
    rewritten: Boolean(turn.retrievalQuery && turn.retrievalQuery !== turn.question),
    fallback: turn.rewriteFallback,
    tools: turn.toolCalls,
    chunks: turn.chunks,
  }
})

async function refreshHealth(): Promise<void> {
  try {
    state.health = await api.health()
    state.banner = null
  } catch (error) {
    state.health = null
    state.banner = `无法连接后端：${(error as Error).message}`
  }
}

async function refreshSessions(): Promise<void> {
  const { sessions } = await api.listSessions()
  state.sessions = sessions
}

async function selectSession(id: string): Promise<void> {
  const detail = await api.getSession(id)
  state.activeSessionId = id
  state.turns = []
  for (const stored of detail.turns) {
    if (stored.role === 'user') {
      state.turns.push(newTurn(stored.content, false))
    } else {
      const previous = state.turns.at(-1)
      if (previous) {
        previous.answer = stored.content
        previous.retrievalQuery = stored.retrieval_query
        previous.toolCalls = [...stored.tool_calls]
      }
    }
  }
  state.selectedTurnId = state.turns.at(-1)?.id ?? null
}

function startNewSession(): void {
  state.activeSessionId = null
  state.turns = []
  state.selectedTurnId = null
}

async function removeSession(id: string): Promise<void> {
  await api.deleteSession(id)
  if (id === state.activeSessionId) startNewSession()
  await refreshSessions()
}

function stop(): void {
  controller?.abort()
}

async function ask(question: string): Promise<void> {
  const trimmed = question.trim()
  if (!trimmed || state.streaming) return

  state.banner = null
  const turn = newTurn(trimmed, true)
  state.turns.push(turn)
  state.selectedTurnId = turn.id
  state.streaming = true
  controller = new AbortController()

  try {
    const events = streamTurn(trimmed, state.activeSessionId, controller.signal)
    const guarded = withIdleTimeout(events, () => controller?.abort())

    // Operation-level ceiling. The idle watchdog only advances when the iterator
    // is awaited, so it cannot rescue a fetch that never returns a response at
    // all; this watchdog fires on wall-clock time regardless of where the turn is
    // stuck, and aborting the controller releases the fetch.
    let watchdog: ReturnType<typeof setTimeout> | undefined
    const deadline = new Promise<never>((_, reject) => {
      watchdog = setTimeout(() => {
        controller?.abort()
        reject(new StreamTimeoutError(TOTAL_TIMEOUT_MS / 1000, 'idle'))
      }, TOTAL_TIMEOUT_MS)
    })

    try {
      const pump = async () => {
        for await (const event of guarded) applyEvent(turn, event)
      }
      await Promise.race([pump(), deadline])
    } finally {
      if (watchdog) clearTimeout(watchdog)
    }
  } catch (error) {
    if (error instanceof StreamTimeoutError) {
      // Keep the question on screen and offer a retry: dropping the turn would
      // make the user retype it, and a silent disappearance looks like a bug.
      turn.timedOut = true
      turn.error = error.message
      state.banner = null
    } else if ((error as Error).name === 'AbortError') {
      // A deliberate stop, not a failure worth reporting.
      turn.error = null
    } else {
      turn.error = (error as Error).message
    }
  } finally {
    turn.streaming = false
    turn.status = null
    state.streaming = false
    controller = null
  }

  // Deliberately outside the try/finally and not awaited: refreshing the sidebar
  // is cosmetic, and awaiting it here meant a refresh that never settled left
  // ask() unresolved — with `busy` still true the composer, the suggested
  // questions and the session list all stopped responding.
  void refreshSessions().catch(() => undefined)
}

function applyEvent(turn: UiTurn, event: StreamEvent): void {
  switch (event.type) {
    case 'status':
      turn.status = event.message
      break
    case 'rewrite':
      turn.retrievalQuery = event.query
      turn.rewriteFallback = event.fallback
      break
    case 'tools':
      turn.toolCalls = [...event.names]
      break
    case 'retrieval':
      turn.chunks = event.results ?? []
      break
    case 'delta':
      turn.answer += event.text
      break
    case 'done':
      turn.answer = event.answer
      turn.seconds = event.seconds
      turn.toolCalls = [...event.tool_calls]
      if (event.retrieval_query) turn.retrievalQuery = event.retrieval_query
      // The first question of a new conversation creates the session
      // server-side; adopting its id makes the next question continue it.
      if (!state.activeSessionId) state.activeSessionId = event.session_id
      break
    case 'error':
      turn.error = event.message
      break
  }
}

/** Re-ask a question that failed, replacing the failed turn. */
async function retry(turnId: number): Promise<void> {
  const index = state.turns.findIndex((turn) => turn.id === turnId)
  if (index === -1) return
  const question = state.turns[index].question
  state.turns.splice(index, 1)
  await ask(question)
}

function newTurn(question: string, streaming: boolean): UiTurn {
  return {
    id: nextTurnId++,
    question,
    answer: '',
    streaming,
    error: null,
    timedOut: false,
    status: null,
    retrievalQuery: null,
    rewriteFallback: null,
    toolCalls: [],
    chunks: [],
    seconds: null,
  }
}

export const store = {
  state,
  selectedTurn,
  retrievalView,
  refreshHealth,
  refreshSessions,
  selectSession,
  startNewSession,
  removeSession,
  ask,
  retry,
  stop,
}

export type AgentStore = typeof store
