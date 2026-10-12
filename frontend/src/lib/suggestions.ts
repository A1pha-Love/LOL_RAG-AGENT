/**
 * Suggested opening questions for an empty conversation.
 *
 * The pool is curated rather than generated: each question is known to be
 * answerable from the dataset and the knowledge base, and the set deliberately
 * spans the different tools (match count, ban stats, champion stats, first-event
 * metrics, dataset overview, knowledge retrieval) so an empty page advertises what
 * the agent can actually do rather than repeating one kind of question.
 *
 * Kept in its own module so the rotation is testable and the pool can grow without
 * touching the component.
 */

// ============================================================================
// 中文说明
//
// 空会话页面上那三张「试试这些问题」的小卡片，以及每一轮回答下方的「继续问」，
// 都由这里的内容与轮换算法提供。
//
// 为什么问题池是**人工挑选**而不是让模型生成：这些问题都保证能从这份数据集或
// 知识库中答出来，而且刻意覆盖了不同的工具（场次统计、Ban 统计、英雄统计、
// 首个事件指标、数据集概览、知识检索），这样空页面本身就在告诉用户「这个
// Agent 能做什么」，而不是反复推荐同一类问题。
//
// 三层结构：
//   SUGGESTION_POOL  —— 全部候选题（静态常量）。
//   SuggestionDeck   —— 单个「牌堆」：把候选打乱后按页轮换，refresh 就取下一页。
//   SuggestionDecks  —— 每一轮对话一个独立牌堆，避免刷新一行时另一行跳动。
//
// 被谁使用：components/ChatStream.vue（空状态 + 每轮追问行）。
// 单测：tests/suggestions.test.ts。
// ============================================================================

/** How many suggestions are shown at once. */
// 中文说明：一页展示几个推荐问题（同时也是轮换时前进的步长）。
export const SUGGESTIONS_PER_PAGE = 3

// 候选题池。注释按主题分组，便于后续增删时保持覆盖面的多样性。
export const SUGGESTION_POOL = [
  // 数据边界与规模
  '这份数据去重后有多少场比赛？',
  '这份数据覆盖哪个时间段、哪个服务器？',
  '原始行数与去重后的场次差多少？重复是怎么产生的？',
  '数据集里有哪些字段可以用来做统计？',
  // 禁用与出场
  '哪个英雄被禁用次数最多？',
  '禁用率最高的英雄是哪个？',
  'Ban 率按什么口径计算？同一英雄同场多次禁用怎么算？',
  '哪个英雄出场率最高？',
  '英雄登场率与选取率是一回事吗？',
  // 胜率
  '哪个英雄胜率最高？',
  '英雄胜率是怎么算的？分母是什么？',
  // 首个事件
  '一血发生率是多少？',
  '一血队伍的条件胜率是多少？',
  '一血发生率和一血队伍的条件胜率分别是多少？',
  '首个防御塔的归属对胜负有多大影响？',
  // 指标口径与陷阱
  '英雄胜率与英雄出场率的分母有什么区别？',
  '哪些指标容易被误读？',
  '统计里有哪些已知的局限或注意事项？',
] as const

/**
 * Rotating deck of suggestions.
 *
 * Advancing rotates whole pages so a refresh never re-shows a question that is
 * still on screen; with the pool size not a multiple of the page size, successive
 * pages also never repeat immediately.
 */
// 中文说明：单个推荐题牌堆。
//
// 关键设计：`next()` 前进**整页**而不是一条。如果只前进一条，刷新后新一页里
// 会有两条是刚刚看过的，用户会觉得「点了没反应」。整页前进则保证新一页与旧页
// 完全不重叠（前提是页大小不超过池子大小）。
//
// 另外池子长度（19）不是页大小（3）的整数倍，所以页边界会错开，连续两页也不会
// 恰好重合——这属于顺带得到的好处，而非刻意构造。
export class SuggestionDeck {
  /** 打乱（或原样）后的下标顺序，长度为池子大小。 */
  private readonly order: number[]
  /** 当前页在 order 中的起始偏移。 */
  private cursor = 0

  /**
   * 参数:
   *     pool: 候选题；默认用内置池子，单测可注入小池子。
   *     pageSize: 一页几条。
   *     shuffle: 是否打乱顺序；单测传 false 以便断言确定的输出。
   */
  constructor(
    private readonly pool: readonly string[] = SUGGESTION_POOL,
    private readonly pageSize: number = SUGGESTIONS_PER_PAGE,
    shuffle = true,
  ) {
    const indices = pool.map((_, index) => index)
    this.order = shuffle ? shuffled(indices) : indices
  }

  /** The current page of suggestions. */
  current(): string[] {
    return this.sliceAt(this.cursor)
  }

  /** Advance one page and return the new suggestions. */
  next(): string[] {
    // 取模保证游标在池子内循环，不会越界。
    this.cursor = (this.cursor + this.pageSize) % this.order.length
    return this.current()
  }

  /** 按 order 从 start 起取 pageSize 条，环形回绕。 */
  private sliceAt(start: number): string[] {
    const picked: string[] = []
    // 与池子大小取 min：页大小大于池子时不会重复取同一条。
    for (let offset = 0; offset < Math.min(this.pageSize, this.order.length); offset += 1) {
      picked.push(this.pool[this.order[(start + offset) % this.order.length]])
    }
    return picked
  }
}

/** 中文说明：Fisher–Yates 洗牌，返回新数组（不修改入参）。 */
function shuffled(values: number[]): number[] {
  const result = [...values]
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(Math.random() * (index + 1))
    ;[result[index], result[swap]] = [result[swap], result[index]]
  }
  return result
}

/**
 * One independent deck per conversation turn.
 *
 * Each turn's follow-up row refreshes on its own, so a single shared deck would
 * make one row jump when another is refreshed.
 */
// 中文说明：按回合 id 分发的牌堆集合。
//
// 每一轮回答下方的「继续问」是独立刷新的：如果所有回合共用一个牌堆，刷新第 3 轮
// 的推荐时第 1 轮的也会跟着变，看起来像界面错乱。所以按 turnId 各持一个。
// `retain()` 提供「丢掉已不存在回合的牌堆」的能力，否则这个 Map 会随回合数一直增长。
// 注意：截至当前版本，生产代码里还没有调用它（只有 tests/suggestions.test.ts 覆盖），
// 属于预留的清理入口。
export class SuggestionDecks {
  private readonly decks = new Map<number, SuggestionDeck>()

  constructor(private readonly pool: readonly string[] = SUGGESTION_POOL) {}

  /** Current suggestions for a turn, creating its deck on first use. */
  current(turnId: number): string[] {
    return this.deckFor(turnId).current()
  }

  /** Advance one page for a turn and return the new suggestions. */
  rotate(turnId: number): string[] {
    return this.deckFor(turnId).next()
  }

  /** Drop decks for turns that no longer exist, so the map cannot grow forever. */
  retain(turnIds: readonly number[]): void {
    const keep = new Set(turnIds)
    // 先取 key 快照再删除，避免边遍历边改 Map。
    for (const key of [...this.decks.keys()]) {
      if (!keep.has(key)) this.decks.delete(key)
    }
  }

  /** 懒创建：第一次访问某回合时才建牌堆，并沿用同一个 pool 配置。 */
  private deckFor(turnId: number): SuggestionDeck {
    let deck = this.decks.get(turnId)
    if (!deck) {
      deck = new SuggestionDeck(this.pool, SUGGESTIONS_PER_PAGE)
      this.decks.set(turnId, deck)
    }
    return deck
  }
}
