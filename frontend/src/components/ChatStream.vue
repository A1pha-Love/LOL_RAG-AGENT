<script setup lang="ts">
/**
 * Transcript plus the composer.
 *
 * Auto-scrolls only when the user is already near the bottom: forcing the view
 * down while someone is reading earlier turns is worse than not scrolling.
 */
import { nextTick, reactive, ref, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'
import RefreshButton from './RefreshButton.vue'
import { SuggestionDeck, SuggestionDecks } from '@/lib/suggestions'
import type { UiTurn } from '@/composables/useAgent'

const props = defineProps<{
  turns: readonly UiTurn[]
  streaming: boolean
  selectedTurnId: number | null
}>()

const emit = defineEmits<{
  (event: 'ask', question: string): void
  (event: 'stop'): void
  (event: 'select-turn', id: number): void
  (event: 'retry', id: number): void
}>()

const draft = ref('')
const scroller = ref<HTMLElement | null>(null)
const pinned = ref(true)

// One deck per visit to the empty state, so the first page is random and every
// refresh shows a different page.
let deck = new SuggestionDeck()
const suggestions = ref<string[]>(deck.current())

// A separate deck per turn for the follow-up rows. The rendered strings live in a
// reactive map so a rotation is a plain assignment the template reacts to.
const followUpDecks = new SuggestionDecks()
const followUps = reactive(new Map<number, string[]>())

function rotateSuggestions(): void {
  suggestions.value = deck.next()
}

function followUpsFor(turnId: number): string[] {
  let current = followUps.get(turnId)
  if (!current) {
    current = followUpDecks.current(turnId)
    followUps.set(turnId, current)
  }
  return current
}

function rotateFollowUps(turnId: number): void {
  followUps.set(turnId, followUpDecks.rotate(turnId))
}

function submit(): void {
  const question = draft.value.trim()
  if (!question || props.streaming) return
  emit('ask', question)
  draft.value = ''
}

function onKeydown(event: KeyboardEvent): void {
  // Enter sends; Shift+Enter inserts a newline.
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

function onScroll(): void {
  const element = scroller.value
  if (!element) return
  pinned.value = element.scrollHeight - element.scrollTop - element.clientHeight < 80
}

watch(
  () => props.turns.map((turn) => turn.answer.length).join(','),
  async () => {
    if (!pinned.value) return
    await nextTick()
    const element = scroller.value
    if (element) element.scrollTop = element.scrollHeight
  },
)
</script>

<template>
  <section class="panel chat">
    <div ref="scroller" class="scroller" @scroll="onScroll">
      <div v-if="!turns.length" class="welcome">
        <h3>英雄联盟历史排位数据分析</h3>
        <p>
          数据为 2017-06-08 至 2017-09-06 的 EUW 排位赛，去重后 51,053 场。
          所有统计数字由本地 Python 计算，回答会标注文档来源。
        </p>
        <div class="suggestions-head">
          <span>试试这些问题</span>
          <RefreshButton @rotate="rotateSuggestions" />
        </div>
        <div class="suggestions">
          <button v-for="item in suggestions" :key="item" class="suggestion" @click="emit('ask', item)">
            {{ item }}
          </button>
        </div>
      </div>

      <MessageBubble
        v-for="turn in turns"
        :key="turn.id"
        :turn="turn"
        :selected="turn.id === selectedTurnId"
        :follow-ups="followUpsFor(turn.id)"
        @select="emit('select-turn', turn.id)"
        @retry="emit('retry', turn.id)"
        @ask="emit('ask', $event)"
        @rotate-follow-ups="rotateFollowUps(turn.id)"
      />
    </div>

    <form class="composer" @submit.prevent="submit">
      <textarea
        v-model="draft"
        rows="1"
        placeholder="输入问题，Enter 发送，Shift+Enter 换行"
        :disabled="streaming"
        @keydown="onKeydown"
      />
      <button v-if="streaming" type="button" class="stop" @click="emit('stop')">停止</button>
      <button v-else type="submit" class="send" :disabled="!draft.trim()">发送</button>
    </form>
  </section>
</template>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.scroller {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.welcome {
  margin: auto;
  max-width: 460px;
  padding: 24px;
  text-align: center;
  color: var(--text-muted);
}

.welcome h3 {
  margin: 0 0 10px;
  font-size: 15px;
  color: var(--text);
}

.welcome p {
  margin: 0 0 18px;
  font-size: 12.5px;
  line-height: 1.8;
}

.suggestions-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 7px;
  padding: 0 2px;
  font-size: 11.5px;
  color: var(--text-faint);
}

/* The refresh button styles live in RefreshButton.vue, which both the empty state
   and the per-turn follow-up rows use. */

.suggestions {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.suggestion {
  padding: 9px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 12.5px;
  text-align: left;
  cursor: pointer;
}

.suggestion:hover {
  border-color: var(--accent-border);
  color: var(--text);
}

.composer {
  display: flex;
  gap: 8px;
  padding: 10px;
  border-top: 1px solid var(--border);
}

.composer textarea {
  flex: 1;
  resize: none;
  max-height: 140px;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 9px;
  background: var(--surface-2);
  color: var(--text);
  font-family: inherit;
  font-size: 13px;
  line-height: 1.6;
}

.composer textarea:focus {
  outline: none;
  border-color: var(--accent-border);
}

.send,
.stop {
  align-self: flex-end;
  padding: 10px 18px;
  border: none;
  border-radius: 9px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.send {
  background: var(--accent);
  color: #fff;
}

.send:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.stop {
  background: var(--surface-3);
  color: var(--danger);
}
</style>
