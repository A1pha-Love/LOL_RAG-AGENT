/** Shared types mirroring the backend payloads (see apps/api.py). */

// ============================================================================
// 中文说明
//
// 本文件是前后端之间的“类型契约”，一一对应后端 FastAPI 的返回体与 SSE 事件体
// （后端定义见 apps/api.py）。它本身不含任何运行时代码，编译后会被完全擦除——
// 存在的意义是让前端在编译期就知道后端会给出哪些字段，改后端时类型检查会立刻报错。
//
// 命名约定：后端用的是 Python/JSON 的 snake_case，前端刻意不做大小写转换，
// 保持与后端字段完全一致，避免“前端 camelCase ↔ 后端 snake_case”的映射层
// 出现偏差而难以排查。
//
// 数据流向大致是：
//   api.listSessions()  → SessionSummary[]     （左栏会话列表）
//   api.getSession(id)  → SessionDetail        （打开会话时载入历史回合）
//   streamTurn()        → StreamEvent 流       （一轮问答的实时进度）
//   api.health()        → HealthReport         （顶栏索引状态徽标）
// ============================================================================

/** 会话详情里的一条历史消息，对应数据库中的一行。 */
export interface StoredTurn {
  /** 消息角色：用户提问或助手回答。 */
  role: 'user' | 'assistant'
  content: string
  /** ISO 时间字符串，由后端生成。 */
  created_at: string
  /** 该轮实际用于检索的查询（可能是改写后的）；未检索时为 null。 */
  retrieval_query: string | null
  /** 该轮调用过的工具名列表。 */
  tool_calls: string[]
}

/** 左侧会话列表中的一行摘要（不含消息正文）。 */
export interface SessionSummary {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  turn_count: number
  /** 1 when pinned; pinned conversations sort above the rest. */
  pinned: number
  /** Last activity (or open) time, used to order the non-pinned group. */
  last_active_at: string | null
}

/** 打开某个会话时后端返回的完整对话内容。 */
export interface SessionDetail {
  session_id: string
  title: string
  created_at: string
  updated_at: string
  /** 按时间正序排列的全部消息。 */
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
  /** 分块在索引中的唯一标识。 */
  chunk_id?: string
  /** 来源文件路径（如 `kbase/xxx.md`）。 */
  source?: string
  /** 该分块所属的最末级标题。 */
  heading?: string
  /** 完整标题路径（如 `A > B > C`），用于展示分块在文档中的位置。 */
  heading_path?: string
  /** 分块类别标签（project/domain/conflict/meta/offtopic/noise）。 */
  category?: string
  /** 文档版本号，可能为 `n/a`。 */
  version?: string
  /** 分块全文；列表接口通常不返回，只返回 preview。 */
  content?: string
  /** First ~180 characters of the chunk, sent by the backend for the inspector. */
  preview?: string
  /**
   * 一阶段召回得分（0~1 区间）：后端按 `VECTOR_WEIGHT * 向量余弦 + 词面重合`
   * 加权得到，因此它**不是**纯向量相似度，而是用于挑选候选池的融合分。
   */
  similarity?: number
  /** 未经加权的原始向量余弦得分，仅用于对照，界面上不展示。 */
  vector_score?: number
  /** rerank 之后的得分，仅在启用 rerank 时非空。 */
  rerank_score?: number | null
  /** rerank 之后的名次（1 为最相关）。 */
  rerank_rank?: number | null
  /** 该分块是否真的经过了 rerank 模型打分。 */
  rerank_applied?: boolean
}

/** The "rewrite" event: what retrieval will actually search for. */
export interface RewriteEvent {
  type: 'rewrite'
  /** 改写后的检索查询；为 null 表示本轮不做知识检索。 */
  query: string | null
  /** 改写失败时回退使用的查询（通常是用户原问题）。 */
  fallback: string | null
}

/** 进度提示事件：模型思考、检索中、生成中等阶段性文案。 */
export interface StatusEvent {
  type: 'status'
  /** 机器可读的阶段名。 */
  phase: string
  /** 直接展示给用户的中文文案。 */
  message: string
}

/** 工具调用事件：模型这一轮选用了哪些工具。 */
export interface ToolsEvent {
  type: 'tools'
  names: string[]
}

/** The chunks matched by the knowledge retrieval, for the retrieval panel. */
export interface RetrievalEvent {
  type: 'retrieval'
  retrieval_query: string | null
  results: RetrievedChunk[]
  /** 检索过程出错时的说明；出错不影响回答继续生成。 */
  error?: string | null
}

/** 增量文本事件：一个 token 片段，前端追加到答案末尾。 */
export interface DeltaEvent {
  type: 'delta'
  text: string
}

/** 回合结束事件：给出最终答案与统计信息（是权威结果，会覆盖流式拼接的文本）。 */
export interface DoneEvent {
  type: 'done'
  /** 本轮最终归属的会话 id（新会话在此刻才由后端创建）。 */
  session_id: string
  /** 完整答案，用于纠正流式拼接可能存在的误差。 */
  answer: string
  retrieval_query: string | null
  tool_calls: string[]
  /** 后端计时的耗时（秒），前端不自行估算。 */
  seconds: number
}

/** 错误事件：本轮失败的原因，前端展示在回答区域。 */
export interface ErrorEvent {
  type: 'error'
  message: string
}

/**
 * 流中可能出现的全部事件，用 `type` 做判别联合（discriminated union）。
 *
 * 好处是 `applyEvent` 的 switch 能获得穷尽性检查：后端新增一种事件类型时，
 * TypeScript 会在缺少分支的地方报错。
 */
export type StreamEvent =
  | StatusEvent
  | RewriteEvent
  | ToolsEvent
  | RetrievalEvent
  | DeltaEvent
  | DoneEvent
  | ErrorEvent

/** 后端健康检查结果，用于顶栏的索引状态徽标。 */
export interface HealthReport {
  status: string
  /** 知识库索引文件是否存在；不存在则检索必然失败。 */
  index_exists: boolean
  /** 索引是否落后于知识库源文件。 */
  index_stale: boolean
  /** 判定过期/正常的原因说明，作为徽标的 tooltip。 */
  index_reason: string
  /** 会话数据库路径，仅用于诊断展示。 */
  session_db: string
}
