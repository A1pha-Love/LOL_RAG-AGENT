/** Shared types mirroring the backend payloads (see apps/api.py). */

export interface StoredTurn {
  role: 'user' | 'assistant'
  content: string
  created_at: string
  retrieval_query: string | null
  tool_calls: string[]
}

export interface SessionSummary {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  turn_count: number
}

export interface SessionDetail {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  turns: StoredTurn[]
}

/**
 * A retrieved chunk as returned by search_project_knowledge.
 *
 * `rerank_score` is only meaningful within one request — the API documents it as
 * a relative score — which is why it is displayed per query rather than compared
 * across queries.
 */
export interface RetrievedChunk {
  chunk_id?: string
  source?: string
  heading?: string
  heading_path?: string
  category?: string
  version?: string
  content?: string
  /** First ~180 characters of the chunk, sent by the backend for the inspector. */
  preview?: string
  similarity?: number
  vector_score?: number
  rerank_score?: number | null
  rerank_rank?: number | null
  rerank_applied?: boolean
}

/** The "rewrite" event: what retrieval will actually search for. */
export interface RewriteEvent {
  type: 'rewrite'
  query: string | null
  fallback: string | null
}

export interface StatusEvent {
  type: 'status'
  phase: string
  message: string
}

export interface ToolsEvent {
  type: 'tools'
  names: string[]
}

/** The chunks matched by the knowledge retrieval, for the retrieval panel. */
export interface RetrievalEvent {
  type: 'retrieval'
  retrieval_query: string | null
  results: RetrievedChunk[]
  error?: string | null
}

export interface DeltaEvent {
  type: 'delta'
  text: string
}

export interface DoneEvent {
  type: 'done'
  session_id: string
  answer: string
  retrieval_query: string | null
  tool_calls: string[]
  seconds: number
}

export interface ErrorEvent {
  type: 'error'
  message: string
}

export type StreamEvent =
  | StatusEvent
  | RewriteEvent
  | ToolsEvent
  | RetrievalEvent
  | DeltaEvent
  | DoneEvent
  | ErrorEvent

export interface HealthReport {
  status: string
  index_exists: boolean
  index_stale: boolean
  index_reason: string
  session_db: string
}
