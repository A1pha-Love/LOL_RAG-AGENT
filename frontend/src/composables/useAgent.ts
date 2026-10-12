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

// ============================================================================
// 中文说明
//
// 这是整个前端的「大脑」：全局状态 + 一轮问答的编排逻辑，全部集中在这一个模块里。
//
// 为什么是模块级单例，而不是 Pinia / Vuex：
//   状态很小（会话列表、当前对话的回合数组、一个在飞的流），唯一的异步逻辑就是
//   「流式跑完一轮问答」，引入状态库只会多一个依赖而不会减少复杂度。导出的
//   `store` 对象本身就是 API，组件直接 `store.ask(...)`。
//
// 为什么整个 state 用 reactive 而不是 ref：
//   模板里就能直接写 `state.turns` / `store.turns`，不必到处 `.value` 解包。
//
// 三个必须理解的点（都是本项目实际踩过的坑）：
//   1. newTurn() 返回的对象必须用 reactive() 包一层——否则 applyEvent 持有的是
//      原始对象引用，写入绕过 Vue 代理，界面不更新（详见 newTurn 注释）。
//   2. ask() 里有两处 refreshSessions() 刻意 **不 await**——await 一个永不返回的
//      刷新会让整个界面（输入框、推荐问题、会话列表）失去响应（详见各处注释）。
//   3. liveTurn 让正在生成的回合「存活」于会话切换——否则流式期间必须锁死侧栏
//      （详见 liveTurn 的注释）。
//
// 状态与界面的对应关系：
//   state.sessions        → 左栏 SessionList
//   state.turns           → 中栏 ChatStream + 右栏 RetrievalPanel（选中的回合）
//   state.streaming /
//   state.streamingSessionId → 输入框禁用 / 侧栏「生成中…」标记
//   state.health          → 顶栏索引状态徽标
//   state.banner          → 顶部告警条
//
// 单测：tests/store.test.ts、tests/streaming.test.ts、tests/session-switch.test.ts、
//       tests/sidebar.test.ts、tests/reactivity.test.ts。
// ============================================================================

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
// 中文说明：界面上「一问一答」的完整视图模型，比后端存的 StoredTurn 丰富得多：
// 除了问题与答案，还包括流式过程中的临时状态（status）、检索详情（chunks）、
// 以及失败/超时的标记。它是一次性的展示结构，不落库。
export interface UiTurn {
  /** 前端自增的回合标识，用作 v-for 的 key 与推荐牌堆的索引。 */
  id: number
  question: string
  /** 答案正文；流式期间不断追加 delta。 */
  answer: string
  /** 该回合是否正在生成。 */
  streaming: boolean
  /** 失败原因；null 表示没有错误。 */
  error: string | null
  /** Set when the stream stalled; the UI offers a one-click retry. */
  /** 中文说明：是否是超时导致的失败。只有它才显示「重试」按钮。 */
  timedOut: boolean
  /** 当前进度文案（来自 status 事件），如「正在检索知识库…」。 */
  status: string | null
  /** 本轮实际使用的检索查询（可能是改写后的）。 */
  retrievalQuery: string | null
  /** 查询改写失败时的回退值。 */
  rewriteFallback: string | null
  toolCalls: string[]
  /** 检索命中的分块，供右侧检索面板展示。 */
  chunks: RetrievedChunk[]
  /** 后端计时得到的耗时（秒）；未结束时为 null。 */
  seconds: number | null
}

/** What the retrieval inspector shows for the selected turn. */
// 中文说明：右侧「检索过程」面板的展示数据，由 selectedTurn 派生（见 retrievalView）。
export interface RetrievalView {
  query: string | null
  /** 检索查询是否与用户原问题不同（即发生过改写）。 */
  rewritten: boolean
  fallback: string | null
  tools: string[]
  chunks: RetrievedChunk[]
}

/** 回合 id 自增计数器；模块级，所以整页生命周期内不会重复。 */
let nextTurnId = 1
/**
 * 当前这一轮流式请求的中止句柄。
 *
 * 只有「同时只允许一个流」这一条约束，所以用单个变量即可（ask 开头会挡掉
 * 可见会话正在流式时的重复提交）。stop() 通过 abort() 同时中断前端读取与后端生成。
 */
let controller: AbortController | null = null

/**
 * The turn currently streaming, keyed by BOTH the id of the turn and the session it
 * belongs to.
 *
 * It cannot live only in `state.turns`: switching conversations replaces that array,
 * so the in-flight turn would be lost and switching had to be blocked while the
 * model ran — which locked the user in place for the whole answer. Keeping the live
 * turn here lets the stream continue in the background and be restored on return.
 */
// 中文说明：正在流式生成的那一轮，同时记录「回合对象」与「它属于哪个会话」。
//
// 为什么不能只放在 state.turns 里：
//   state.turns 的语义是「当前可见会话的回合列表」，切换会话时会被整体替换
//   （selectSession 里 `state.turns = []` 然后重建）。如果正在生成的回合只存在于
//   这个数组里，一切换就被丢掉，于是早期实现只能「生成期间禁止切换会话」——
//   一轮回答要几十秒，用户就被锁在原地动弹不得。
//
//   把它挪到模块级变量后：切走时流继续在后台跑（applyEvent 仍然写这个对象），
//   切回来时再把它塞回 state.turns（见 selectSession），于是侧栏可以随时切换，
//   并且用 streamingSessionId 标出「哪个会话还在生成」。
//
// sessionId 可能为 null：新会话在第一轮 done 事件到达前还没有服务端 id。
let liveTurn: { sessionId: string | null; turn: UiTurn } | null = null

/**
 * 中文说明：全局响应式状态。
 *
 * 注意 `state.turns` 与 `state.streaming` 描述的都是**当前可见**的会话；
 * 后台仍在生成的会话由 streamingSessionId 表示，两者可能不是同一个。
 */
const state = reactive({
  sessions: [] as SessionSummary[],
  /** 当前打开的会话 id；null 表示「新会话」尚未落库的状态。 */
  activeSessionId: null as string | null,
  turns: [] as UiTurn[],
  /** True when the *visible* conversation has a turn in flight. */
  /** 中文说明：可见会话是否有回合在生成（决定输入框/发送按钮是否禁用）。 */
  streaming: false,
  /** Which conversation the in-flight turn belongs to, wherever the user is now. */
  /** 中文说明：在飞的回合属于哪个会话——用户此刻可能已经在看别的会话了。 */
  streamingSessionId: null as string | null,
  /** 右侧检索面板当前展示的回合；null 时回落到最后一个回合。 */
  selectedTurnId: null as number | null,
  health: null as HealthReport | null,
  /** 顶部告警条文案；null 表示无告警。 */
  banner: null as string | null,
})

/**
 * 中文说明：右侧面板要展示的回合——优先用户点选的那一轮，否则取最后一轮。
 *
 * 这两个字段是 computed（而不是把结果存进 state），这样流式追加文字、切换选中
 * 回合都会自动重新计算，不需要手动同步。
 */
const selectedTurn = computed<UiTurn | null>(
  () => state.turns.find((turn) => turn.id === state.selectedTurnId) ?? state.turns.at(-1) ?? null,
)

/**
 * 中文说明：把选中回合整理成检索面板需要的视图。
 *
 * `rewritten` 的判定方式是「检索查询 ≠ 用户原问题」：这是一个启发式的近似，
 * 后端并没有单独发一个「是否改写」的布尔值。
 */
const retrievalView = computed<RetrievalView | null>(() => {
  const turn = selectedTurn.value
  // 没有任何回合时不展示面板（组件会显示空状态提示）。
  if (!turn) return null
  return {
    query: turn.retrievalQuery,
    rewritten: Boolean(turn.retrievalQuery && turn.retrievalQuery !== turn.question),
    fallback: turn.rewriteFallback,
    tools: turn.toolCalls,
    chunks: turn.chunks,
  }
})

/**
 * 中文说明：刷新后端健康状态。
 *
 * 这里刻意吞掉异常并转成 banner 文案：后端没起来时界面仍要能打开并给出提示，
 * 而不是白屏或抛未捕获异常。成功时顺手清掉 banner（后端恢复后提示要消失）。
 */
async function refreshHealth(): Promise<void> {
  try {
    state.health = await api.health()
    state.banner = null
  } catch (error) {
    state.health = null
    state.banner = `无法连接后端：${(error as Error).message}`
  }
}

/**
 * 中文说明：重新拉取会话列表。
 *
 * 排序完全由后端负责（置顶在前、其余按最后活动时间倒序），前端只做整体替换，
 * 不做本地重排——这样「顺序」只有一个真相来源。
 */
async function refreshSessions(): Promise<void> {
  const { sessions } = await api.listSessions()
  state.sessions = sessions
}

/** Whether a given conversation has a turn in flight, wherever the user is. */
// 中文说明：某个会话是否有回合正在生成——用户当前可能并不在这个会话里。
function isSessionStreaming(id: string): boolean {
  return state.streamingSessionId === id
}

/**
 * 中文说明：打开一个会话，把服务端存的历史消息还原成 UiTurn 列表。
 *
 * 还原规则：user 消息开一个新回合，紧跟其后的 assistant 消息填进该回合的答案
 * 与检索信息。这是把「扁平的数据库行」折回「一问一答」的过程。
 *
 * 注意下面三件刻意为之的事：
 *   1. 这里 **不调用任何排序/「标记活跃」接口**（见函数末尾的注释）。
 *   2. 如果这个会话有在飞的回合，用 liveTurn 覆盖掉服务端的旧快照。
 *   3. state.streaming 只反映当前可见会话，因此切换后输入框会立刻恢复可用。
 */
async function selectSession(id: string): Promise<void> {
  const detail = await api.getSession(id)
  state.activeSessionId = id
  state.turns = []
  for (const stored of detail.turns) {
    if (stored.role === 'user') {
      state.turns.push(newTurn(stored.content, false))
    } else {
      // assistant 消息并入上一条 user 回合。历史回合一律 streaming=false：
      // 它们是已经落库的完成态。
      const previous = state.turns.at(-1)
      if (previous) {
        previous.answer = stored.content
        previous.retrievalQuery = stored.retrieval_query
        // 展开复制，避免把后端响应的数组直接塞进响应式对象（保持与 applyEvent 一致）。
        previous.toolCalls = [...stored.tool_calls]
      }
    }
  }

  // A turn may still be streaming for this conversation. Its live text is newer than
  // the stored snapshot (the answer is only persisted when the stream ends), so swap
  // the stored placeholder for the live turn instead of showing a stale answer.
  // 中文说明：答案只有在流结束时才落库，所以服务端快照比 liveTurn 旧。这里按
  // 「问题文本」找到对应的占位回合并原地替换（splice 保持位置不变），找不到就追加。
  // 用问题文本匹配是因为 liveTurn 的回合 id 是前端生成的，落库的历史回合拿到的是
  // 新的 id，两者对不上。
  if (liveTurn && liveTurn.sessionId === id) {
    const placeholder = state.turns.find((turn) => turn.question === liveTurn?.turn.question)
    if (placeholder) {
      state.turns.splice(state.turns.indexOf(placeholder), 1, liveTurn.turn)
    } else {
      state.turns.push(liveTurn.turn)
    }
  }

  // `streaming` describes the visible conversation: switching to a different one
  // must not leave the composer and sidebar disabled.
  // 中文说明：只根据「这个会话是否就是正在生成的那个」来设置，因此切到别的会话
  // 后输入框立刻可用（后台的流不受影响）。
  state.streaming = liveTurn !== null && liveTurn.sessionId === id
  state.selectedTurnId = state.turns.at(-1)?.id ?? null

  // Deliberately NOT recorded as activity. Opening a conversation to read it is not
  // participation, and reordering the sidebar on click made the list jump under the
  // cursor — it looked like the wrong conversation had been selected. The order now
  // changes when the conversation actually gains a turn: the server updates
  // `last_active_at` when a turn is appended, and ask() re-reads the list afterwards.
  //
  // 中文说明（会话顺序语义，务必理解）：
  //   打开会话**不**算活跃——用户只是来读旧内容，而且过去在点击时重排会让列表
  //   在光标底下跳动，看起来像选错了会话。因此这里**只**读详情，绝不调用任何
  //   「更新活跃时间」或排序相关的接口。
  //   顺序真正变化的时机是「该会话产生了新的一轮对话」：后端在 append_turn 时
  //   更新 last_active_at（列表排序键之一，见 conversation.py），前端则由
  //   ask() 在整轮结束后重读列表来体现。
}

/** Pin or unpin a conversation. */
// 中文说明：切换置顶。置顶会影响分组与排序，所以必须等后端改完再重读列表
// （与 ask() 里那两处「不 await」不同：这里是用户直接操作，等一下是合理的，
// 而且调用方 App.vue 会用 loading 状态禁用按钮，不会造成界面卡死）。
async function togglePin(id: string, pinned: boolean): Promise<void> {
  await api.pinSession(id, pinned)
  await refreshSessions()
}

/**
 * 中文说明：开始一个新会话（清空中栏，进入空状态）。
 *
 * 注意它只是把 activeSessionId 置空、清空回合数组，并**不立即**在后端建库——
 * 真正的会话由第一轮的 ask() 创建（也可以由用户点「＋ 新建」显式创建）。
 * 同时把 state.streaming 置 false：新会话是空的，不可能有回合在生成。
 */
function startNewSession(): void {
  state.activeSessionId = null
  state.turns = []
  state.selectedTurnId = null
  state.streaming = false
}

/**
 * 中文说明：删除会话；如果删的正是当前打开的会话，则退回空状态。
 *
 * 顺序很关键：先删后端，再重读列表。这样即使删除请求失败，界面也不会先
 * 自己把项去掉。
 */
async function removeSession(id: string): Promise<void> {
  await api.deleteSession(id)
  if (id === state.activeSessionId) startNewSession()
  await refreshSessions()
}

/** 中文说明：用户点「停止」——abort 掉请求（服务端也会因此停止生成）。 */
function stop(): void {
  controller?.abort()
}

/**
 * 中文说明：发起一轮问答——本项目最核心、注释最多的函数。
 *
 * 整体流程：
 *   1. 校验问题非空、且可见会话没有正在生成的回合（后台别的会话在生成不妨碍）。
 *   2. 若是新会话，先在后端创建（拿到 id 才能让侧栏立刻出现这一项）。
 *   3. 乐观地往 state.turns 推入一个 streaming 回合，并登记 liveTurn。
 *   4. 用两套看门狗包住流：每步空闲超时（withIdleTimeout）+ 整轮墙上时钟上限。
 *   5. 逐事件 applyEvent 渲染；结束/失败后统一收尾（清 streaming 标记等）。
 *   6. 最后**不 await** 地重读会话列表，让顺序反映真实活跃度。
 */
async function ask(question: string): Promise<void> {
  const trimmed = question.trim()
  // Only the visible conversation is blocked: a turn already running in a different
  // conversation must not stop the user from asking here.
  // 中文说明：只拦「当前可见会话正在生成」这一种情况。当用户在别处生成、
  // 切到这个空闲会话时，应该能继续提问。
  if (!trimmed || state.streaming) return

  // Create the conversation up front rather than letting the first turn create it
  // server-side. The id is then known before streaming starts, so the conversation
  // appears in the sidebar immediately and cannot be lost among the others.
  // 中文说明：先建会话再提问。好处是流开始之前 id 就已确定，侧栏能立刻出现这一项；
  // 否则要等 done 事件才知道 id，中间这段时间用户切走就可能「丢」掉这次对话。
  if (!state.activeSessionId) {
    try {
      const created = await api.createSession()
      state.activeSessionId = created.session_id
      // Added optimistically so the conversation is visible at once. The refresh is
      // deliberately NOT awaited: blocking here is the same mistake as the old
      // finally-block refresh, where a slow request left ask() unresolved and the
      // whole UI disabled. The post-turn refresh reconciles the real counters.
      //
      // 中文说明：这里先插一条本地构造的摘要（turn_count 记 0、pinned 记 0、
      // 时间用本地 now），让侧栏「零延迟」出现新会话；随后那次刷新**刻意不 await**。
      // 原因：曾经在 finally 里 await 刷新，只要刷新请求迟迟不返回，ask() 就一直
      // 不 resolve，界面上的 busy 状态无法解除，输入框、推荐问题、会话列表全部
      // 失去响应。真正的计数与排序由本轮结束后的刷新来对齐。
      state.sessions = [
        {
          session_id: created.session_id,
          title: created.title,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          turn_count: 0,
          pinned: 0,
          last_active_at: new Date().toISOString(),
        },
        ...state.sessions,
      ]
      // 后台刷新，失败也吞掉（`void` 明确表示「不等」，catch 避免未处理拒绝）。
      void refreshSessions().catch(() => undefined)
    } catch (error) {
      // 连会话都建不出来就没必要继续了：给出提示并直接返回。
      state.banner = `无法创建会话：${(error as Error).message}`
      return
    }
  }

  // Captured now, because the user may switch conversations while this runs. The
  // backend keys the answer to this id, so the turn belongs to it regardless of what
  // is on screen.
  // 中文说明：把会话 id 用局部变量固化下来。用户随时可能切走，而这一轮的回答
  // 在后端是挂在 sessionId 上的，所以回合的归属只由这个快照决定，与当前屏幕无关。
  const sessionId = state.activeSessionId

  state.banner = null
  const turn = newTurn(trimmed, true)
  state.turns.push(turn)
  state.selectedTurnId = turn.id
  state.streaming = true
  state.streamingSessionId = sessionId
  // 登记 liveTurn 之后，即使切换会话，applyEvent 仍会写这个回合对象。
  liveTurn = { sessionId, turn }
  controller = new AbortController()

  try {
    const events = streamTurn(trimmed, sessionId, controller.signal)
    // 第一层看门狗：每步空闲超时（首事件 60s / 之后 45s）。超时回调里 abort 请求，
    // 让后端也停下，并把阻塞的 read 解开。
    const guarded = withIdleTimeout(events, () => controller?.abort())

    // Operation-level ceiling. The idle watchdog only advances when the iterator
    // is awaited, so it cannot rescue a fetch that never returns a response at
    // all; this watchdog fires on wall-clock time regardless of where the turn is
    // stuck, and aborting the controller releases the fetch.
    //
    // 中文说明：第二层看门狗——整轮绝对上限（TOTAL_TIMEOUT_MS）。它防的是空闲
    // 看门狗覆盖不到的情况：fetch 卡在建立连接上、根本没返回响应，此时迭代器
    // 还没开始被 await，空闲计时器无从推进。这个 Promise 基于墙上时钟，只要
    // 超时就 abort，把卡住的 fetch 释放掉。
    let watchdog: ReturnType<typeof setTimeout> | undefined
    const deadline = new Promise<never>((_, reject) => {
      watchdog = setTimeout(() => {
        controller?.abort()
        reject(new StreamTimeoutError(TOTAL_TIMEOUT_MS / 1000, 'idle'))
      }, TOTAL_TIMEOUT_MS)
    })

    try {
      // 把整个流泵进 applyEvent；与 deadline 赛跑，谁先结束算谁。
      const pump = async () => {
        for await (const event of guarded) applyEvent(turn, event)
      }
      await Promise.race([pump(), deadline])
    } finally {
      // 无论正常结束还是异常，都要撤掉定时器，否则它会在稍后触发一个
      // 没人接收的 rejection。
      if (watchdog) clearTimeout(watchdog)
    }
  } catch (error) {
    if (error instanceof StreamTimeoutError) {
      // Keep the question on screen and offer a retry: dropping the turn would
      // make the user retype it, and a silent disappearance looks like a bug.
      // 中文说明：超时不删回合、也不清空问题，只标记 timedOut，界面据此显示
      // 「重试」按钮——否则用户得把问题重打一遍，而且问题凭空消失看着像 bug。
      turn.timedOut = true
      turn.error = error.message
      // 超时属于「这一轮的问题」，不该再顶着顶部告警条，清掉它。
      state.banner = null
    } else if ((error as Error).name === 'AbortError') {
      // A deliberate stop, not a failure worth reporting.
      // 中文说明：用户主动点「停止」也会以 AbortError 冒到这里，这不是故障，
      // 所以不显示错误。
      turn.error = null
    } else {
      turn.error = (error as Error).message
    }
  } finally {
    // 收尾无论成败都要执行：结束流式外观、清掉 liveTurn / streamingSessionId。
    turn.streaming = false
    turn.status = null
    liveTurn = null
    // The in-flight turn is over, so no conversation is streaming any more. This
    // must happen unconditionally: clearing it only for the visible conversation
    // would leave the sidebar showing "generating" forever for the one the user
    // switched away from.
    // 中文说明：必须无条件清空。如果只在「当前可见会话 == 本轮会话」时才清，
    // 那么用户切走后侧栏那个「生成中…」标记会永远挂在上面。
    state.streamingSessionId = null
    // Idle again unless the user moved to a different conversation meanwhile.
    //
    // Compared against the id captured at the start, not the current one: a turn
    // that began a new conversation has `sessionId === null` while
    // `activeSessionId` has since been filled in from the done event, so comparing
    // the two would leave `streaming` stuck true and disable the composer.
    //
    // 中文说明：只在「用户没有切走」时解除当前会话的 streaming。比较的是函数
    // 开头固化的 sessionId，而不是现在的 activeSessionId：新会话这一轮的
    // sessionId 是 null，而 activeSessionId 会在 done 事件里被填上真实 id，
    // 拿两者比较会不相等，导致 streaming 卡在 true，输入框永久禁用。
    if (state.activeSessionId === sessionId || sessionId === null) {
      state.streaming = false
    }
    controller = null
  }

  // Deliberately outside the try/finally and not awaited: refreshing the sidebar
  // is cosmetic, and awaiting it here meant a refresh that never settled left
  // ask() unresolved — with `busy` still true the composer, the suggested
  // questions and the session list all stopped responding.
  //
  // 中文说明：本轮结束后的会话列表刷新——刻意放在 try/finally **之外**且**不 await**。
  // 这是修过的真实 bug：刷新只是为了让侧栏显示真实轮数与最新顺序（外观性的），
  // 一旦 await，刷新请求不返回就意味着 ask() 永不 resolve，busy 永远为真，
  // 输入框、推荐问题、会话列表会一起失去响应。
  // （放在 finally 之外还保证了一点：即使用户中断流，这次刷新照样会发出。）
  void refreshSessions().catch(() => undefined)
}

/**
 * 中文说明：把流里的一个事件写进对应的 UI 回合。
 *
 * 每种事件只影响回合的某几个字段，其中 delta 是高频的（每个 token 一次），
 * done 是权威的最终态（会覆盖流式拼接得到的文本，纠正可能的差异）。
 */
function applyEvent(turn: UiTurn, event: StreamEvent): void {
  switch (event.type) {
    case 'status':
      // 进度文案：答案为空时由 MessageBubble 展示成「正在等待模型…」那一行。
      turn.status = event.message
      break
    case 'rewrite':
      turn.retrievalQuery = event.query
      turn.rewriteFallback = event.fallback
      break
    case 'tools':
      // 重新赋一个新数组（而不是 push），保证响应式触发更直观、也便于对比前后值。
      turn.toolCalls = [...event.names]
      break
    case 'retrieval':
      // `?? []` 兜底：后端异常时 results 可能是 null。
      turn.chunks = event.results ?? []
      break
    case 'delta':
      // 流式追加。这条写入之所以能被界面看到，全靠 newTurn() 返回的是 reactive 对象。
      turn.answer += event.text
      break
    case 'done':
      // 用后端给的完整答案覆盖流式拼接结果。
      turn.answer = event.answer
      turn.seconds = event.seconds
      turn.toolCalls = [...event.tool_calls]
      if (event.retrieval_query) turn.retrievalQuery = event.retrieval_query
      // The first question of a new conversation creates the session
      // server-side; adopting its id makes the next question continue it.
      // 中文说明：新会话的第一轮在这里才拿到服务端 id（如果 ask 开头已经建过
      // 会话，activeSessionId 已非空，这里不会覆盖）。采用它，下一轮追问才会
      // 落到同一个会话里。
      if (!state.activeSessionId) state.activeSessionId = event.session_id
      break
    case 'error':
      turn.error = event.message
      break
  }
}

/** Re-ask a question that failed, replacing the failed turn. */
// 中文说明：重试——把失败的回合从列表里移除，再用同一个问题重新走一遍 ask()。
// 先删再问，所以不会出现「失败回合 + 新回合」的重复条目。
async function retry(turnId: number): Promise<void> {
  const index = state.turns.findIndex((turn) => turn.id === turnId)
  // 找不到（例如用户在别处已经删掉）就静默返回。
  if (index === -1) return
  const question = state.turns[index].question
  state.turns.splice(index, 1)
  await ask(question)
}

/**
 * 中文说明：构造一个 UI 回合对象。
 *
 * ⚠ 这里是本项目最关键的一处修复，务必理解：
 *
 *   `state.turns` 是 reactive 数组，所以通过 `state.turns[i]` 读出来的元素是
 *   Vue 代理；但 **向数组里 push 一个普通对象时，数组内部保存的只是这个普通
 *   对象本身**（Vue 的 reactive 是惰性的：读取时才包装）。如果 newTurn 返回普通
 *   对象，ask() 里 `const turn = newTurn(...)` 拿到的就是那个**原始对象引用**，
 *   随后 `applyEvent(turn, event)` 里的 `turn.answer += text` 是直接写原始对象，
 *   **完全绕过代理**，不会触发依赖通知——界面只在切换会话（重新从服务端读数据）
 *   之后才显示答案，看上去像「流式没生效」。
 *
 *   用 reactive(...) 包一层，captured 的引用与数组里存的就是同一个代理对象，
 *   写入才会被追踪。tests/reactivity.test.ts 把这两种写法都验证了一遍。
 *
 * 参数:
 *     question: 用户问题文本。
 *     streaming: 是否处于流式生成中（历史回合为 false）。
 */
function newTurn(question: string, streaming: boolean): UiTurn {
  // Must be reactive in its own right.
  //
  // A plain object pushed into a `reactive` array is only reachable through that
  // array's proxy; the reference kept here would stay raw, so `turn.answer += …`
  // inside applyEvent would bypass the proxy entirely and the template would never
  // re-render. The symptom is an answer that only appears after reloading the
  // conversation. Wrapping here makes the captured reference and the stored proxy
  // the same object.
  return reactive<UiTurn>({
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
  })
}

/**
 * 中文说明：对外暴露的 store。组件只通过这个对象访问状态与方法，
 * 不直接碰模块内部变量（controller / liveTurn / nextTurnId 都是私有的）。
 */
export const store = {
  state,
  selectedTurn,
  retrievalView,
  refreshHealth,
  refreshSessions,
  selectSession,
  isSessionStreaming,
  togglePin,
  startNewSession,
  removeSession,
  ask,
  retry,
  stop,
}

/** 中文说明：store 的类型，供需要显式标注类型的场合使用（例如测试替身）。 */
export type AgentStore = typeof store
