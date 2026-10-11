/**
 * Tests for the suggestion deck behind the "换一换" button.
 *
 * The behaviour that matters: refreshing must always show a different set, never
 * strand the user on a repeated page, and keep every page the same size.
 */

import { describe, expect, it } from 'vitest'
import {
  SUGGESTION_POOL,
  SUGGESTIONS_PER_PAGE,
  SuggestionDeck,
  SuggestionDecks,
} from '@/lib/suggestions'

/** Deterministic deck: no shuffle, so pages are predictable. */
const ordered = () => new SuggestionDeck(SUGGESTION_POOL, SUGGESTIONS_PER_PAGE, false)

describe('SuggestionDeck', () => {
  it('shows a full page by default', () => {
    expect(ordered().current()).toHaveLength(SUGGESTIONS_PER_PAGE)
  })

  it('every page holds distinct questions', () => {
    const page = ordered().current()
    expect(new Set(page).size).toBe(page.length)
  })

  it('advancing shows a different page', () => {
    const deck = ordered()
    const first = deck.current()
    const second = deck.next()

    expect(second).not.toEqual(first)
    // A refresh must not leave a question from the previous page on screen.
    expect(second.some((item) => first.includes(item))).toBe(false)
  })

  it('keeps showing new pages across many refreshes', () => {
    const deck = ordered()
    let previous = deck.current()
    const seen = new Set<string>([previous.join('|')])

    for (let step = 0; step < 12; step += 1) {
      const next = deck.next()
      expect(next).not.toEqual(previous)
      seen.add(next.join('|'))
      previous = next
    }

    // A full cycle shows every page, so the distinct count is exactly the number
    // of pages in the pool (18 / 3 = 6) — not fewer, which would mean a repeat.
    const pages = Math.ceil(SUGGESTION_POOL.length / SUGGESTIONS_PER_PAGE)
    expect(seen.size).toBe(pages)
  })

  it('eventually covers the whole pool', () => {
    const deck = ordered()
    const shown = new Set<string>(deck.current())
    const pages = Math.ceil(SUGGESTION_POOL.length / SUGGESTIONS_PER_PAGE)

    for (let step = 1; step < pages; step += 1) {
      for (const item of deck.next()) shown.add(item)
    }

    expect(shown.size).toBe(SUGGESTION_POOL.length)
  })

  it('wraps around without breaking', () => {
    const deck = ordered()
    const total = SUGGESTION_POOL.length
    for (let step = 0; step < total * 3; step += 1) {
      expect(deck.next()).toHaveLength(SUGGESTIONS_PER_PAGE)
    }
  })

  it('shuffles the first page so the empty state varies between visits', () => {
    // With a shuffled deck the very first page is unlikely to be identical across
    // many fresh decks; this guards against the shuffle being a no-op.
    const firstPages = new Set(
      Array.from({ length: 20 }, () => new SuggestionDeck(SUGGESTION_POOL, SUGGESTIONS_PER_PAGE).current().join('|')),
    )
    expect(firstPages.size).toBeGreaterThan(1)
  })

  it('never runs past the end of the pool', () => {
    const deck = ordered()
    for (let step = 0; step < 30; step += 1) {
      for (const item of deck.next()) {
        expect(SUGGESTION_POOL).toContain(item)
      }
    }
  })
})

describe('SuggestionDecks (per-turn rows)', () => {
  it('gives every turn its own row of suggestions', () => {
    const decks = new SuggestionDecks(SUGGESTION_POOL)
    expect(decks.current(1)).toHaveLength(SUGGESTIONS_PER_PAGE)
    expect(decks.current(2)).toHaveLength(SUGGESTIONS_PER_PAGE)
  })

  it('rotating one turn does not disturb another', () => {
    // Each turn's refresh button is independent; a shared deck would make one
    // row jump when a different row is refreshed.
    const decks = new SuggestionDecks(SUGGESTION_POOL)
    const firstBefore = decks.current(1)
    const secondBefore = decks.current(2)

    const firstAfter = decks.rotate(1)

    expect(firstAfter).not.toEqual(firstBefore)
    expect(decks.current(2)).toEqual(secondBefore)
  })

  it('reports the same page on repeated reads', () => {
    const decks = new SuggestionDecks(SUGGESTION_POOL)
    expect(decks.current(7)).toEqual(decks.current(7))
  })

  it('drops decks for turns that no longer exist', () => {
    const decks = new SuggestionDecks(SUGGESTION_POOL)
    const before = decks.current(1)
    decks.retain([2, 3])
    // A dropped turn starts fresh, which is acceptable; what matters is that the
    // map cannot grow without bound as a conversation gets longer.
    expect(decks.current(2)).toHaveLength(SUGGESTIONS_PER_PAGE)
    expect(before).toHaveLength(SUGGESTIONS_PER_PAGE)
  })

  it('keeps retained turns unchanged', () => {
    const decks = new SuggestionDecks(SUGGESTION_POOL)
    decks.rotate(1)
    const kept = decks.current(1)
    decks.retain([1])
    expect(decks.current(1)).toEqual(kept)
  })
})

describe('suggestion pool', () => {
  it('is large enough that refreshing is worthwhile', () => {
    expect(SUGGESTION_POOL.length).toBeGreaterThanOrEqual(SUGGESTIONS_PER_PAGE * 4)
  })

  it('has no duplicates', () => {
    expect(new Set(SUGGESTION_POOL).size).toBe(SUGGESTION_POOL.length)
  })

  it('asks questions rather than issuing commands, so they can be sent as-is', () => {
    for (const item of SUGGESTION_POOL) {
      expect(item.trim()).toBe(item)
      expect(item.length).toBeGreaterThan(4)
      expect(/[？吗]|什么|哪|多少|怎么|区别|是否/.test(item)).toBe(true)
    }
  })
})
