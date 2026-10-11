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

/** How many suggestions are shown at once. */
export const SUGGESTIONS_PER_PAGE = 3

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
export class SuggestionDeck {
  private readonly order: number[]
  private cursor = 0

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
    this.cursor = (this.cursor + this.pageSize) % this.order.length
    return this.current()
  }

  private sliceAt(start: number): string[] {
    const picked: string[] = []
    for (let offset = 0; offset < Math.min(this.pageSize, this.order.length); offset += 1) {
      picked.push(this.pool[this.order[(start + offset) % this.order.length]])
    }
    return picked
  }
}

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
    for (const key of [...this.decks.keys()]) {
      if (!keep.has(key)) this.decks.delete(key)
    }
  }

  private deckFor(turnId: number): SuggestionDeck {
    let deck = this.decks.get(turnId)
    if (!deck) {
      deck = new SuggestionDeck(this.pool, SUGGESTIONS_PER_PAGE)
      this.decks.set(turnId, deck)
    }
    return deck
  }
}
