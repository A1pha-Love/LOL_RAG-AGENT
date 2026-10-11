<script setup lang="ts">
/**
 * One question and its answer.
 *
 * Renders a deliberately small Markdown subset (headings, bold, inline code,
 * lists, tables) instead of pulling in a Markdown library: the answers are
 * structured but predictable, and escaping is handled here so model output can
 * never inject markup.
 */
import { computed } from 'vue'
import type { UiTurn } from '@/composables/useAgent'

const props = defineProps<{
  turn: UiTurn
  selected: boolean
}>()

const emit = defineEmits<{
  (event: 'select'): void
  (event: 'retry'): void
}>()

/** Escape first, then apply inline formatting — never the other way round. */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function inline(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
}

interface Block {
  kind: 'heading' | 'paragraph' | 'list' | 'table' | 'code'
  html?: string
  items?: string[]
  rows?: string[][]
  header?: string[]
}

function parse(source: string): Block[] {
  const lines = source.split('\n')
  const blocks: Block[] = []
  let paragraph: string[] = []
  let list: string[] = []
  let table: string[][] = []
  let code: string[] = []
  let inCode = false

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', html: inline(paragraph.join(' ')) })
      paragraph = []
    }
  }
  const flushList = () => {
    if (list.length) {
      blocks.push({ kind: 'list', items: list.map(inline) })
      list = []
    }
  }
  const flushTable = () => {
    if (!table.length) return
    const header = table[0].map(inline)
    // Row 1 is the |---|---| separator, so body rows start at index 2.
    const body = table.slice(2).map((row) => row.map(inline))
    blocks.push({ kind: 'table', header, rows: body.length ? body : [] })
    table = []
  }

  for (const raw of lines) {
    const line = raw.trimEnd()

    if (line.trim().startsWith('```')) {
      if (inCode) {
        blocks.push({ kind: 'code', html: escapeHtml(code.join('\n')) })
        code = []
        inCode = false
      } else {
        flushParagraph()
        flushList()
        flushTable()
        inCode = true
      }
      continue
    }
    if (inCode) {
      code.push(raw)
      continue
    }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushParagraph()
      flushList()
      table.push(line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim()))
      continue
    }
    flushTable()

    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    if (heading) {
      flushParagraph()
      flushList()
      blocks.push({ kind: 'heading', html: inline(heading[2]) })
      continue
    }

    const bullet = /^\s*[-*]\s+(.*)$/.exec(line)
    const ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (bullet || ordered) {
      flushParagraph()
      list.push((bullet ?? ordered)![1])
      continue
    }

    if (!line.trim()) {
      flushParagraph()
      flushList()
      continue
    }
    paragraph.push(line.trim())
  }

  flushParagraph()
  flushList()
  flushTable()
  if (code.length) blocks.push({ kind: 'code', html: escapeHtml(code.join('\n')) })
  return blocks
}

const blocks = computed(() => parse(props.turn.answer))
</script>

<template>
  <article
    class="turn"
    :class="{ selected, streaming: turn.streaming }"
    @click="emit('select')"
  >
    <div class="question">
      <span class="who">问</span>
      <p>{{ turn.question }}</p>
    </div>

    <div class="answer">
      <span class="who">答</span>
      <div class="body">
        <template v-if="blocks.length">
          <template v-for="(block, index) in blocks" :key="index">
            <h4 v-if="block.kind === 'heading'" v-html="block.html" />
            <p v-else-if="block.kind === 'paragraph'" v-html="block.html" />
            <ul v-else-if="block.kind === 'list'">
              <li v-for="(item, i) in block.items" :key="i" v-html="item" />
            </ul>
            <pre v-else-if="block.kind === 'code'"><code v-html="block.html" /></pre>
            <table v-else-if="block.kind === 'table'">
              <thead>
                <tr><th v-for="(cell, i) in block.header" :key="i" v-html="cell" /></tr>
              </thead>
              <tbody>
                <tr v-for="(row, i) in block.rows" :key="i">
                  <td v-for="(cell, j) in row" :key="j" v-html="cell" />
                </tr>
              </tbody>
            </table>
          </template>
        </template>

        <p v-else-if="turn.status" class="status">
          <span class="dot" />{{ turn.status }}
        </p>
        <p v-else class="status"><span class="dot" />正在等待模型…</p>

        <span v-if="turn.streaming && blocks.length" class="cursor" />
      </div>
    </div>

    <p v-if="turn.error" class="error">
      <span>⚠ {{ turn.error }}</span>
      <button v-if="turn.timedOut" class="retry" @click.stop="emit('retry')">重试</button>
    </p>

    <footer v-if="!turn.streaming" class="foot">
      <span v-if="turn.seconds">{{ turn.seconds }}s</span>
      <span v-if="turn.toolCalls.length">{{ turn.toolCalls.length }} 次工具调用</span>
      <span v-if="turn.chunks.length">{{ turn.chunks.length }} 个命中分块</span>
      <span v-if="turn.retrievalQuery && turn.retrievalQuery !== turn.question" class="rewritten">
        查询已改写
      </span>
    </footer>
  </article>
</template>

<style scoped>
.turn {
  padding: 14px 16px;
  border: 1px solid transparent;
  border-radius: 12px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}

.turn:hover {
  background: var(--surface-2);
}

.turn.selected {
  background: var(--surface-2);
  border-color: var(--accent-border);
}

.question,
.answer {
  display: flex;
  gap: 10px;
}

.question {
  margin-bottom: 10px;
}

.question p {
  margin: 0;
  font-weight: 600;
}

.who {
  flex: none;
  width: 20px;
  height: 20px;
  border-radius: 5px;
  display: grid;
  place-items: center;
  font-size: 11px;
  background: var(--surface-3);
  color: var(--text-muted);
}

.answer .who {
  background: var(--accent-soft);
  color: var(--accent);
}

.body {
  flex: 1;
  min-width: 0;
}

.body :deep(p) {
  margin: 0 0 8px;
  line-height: 1.75;
}

.body :deep(p:last-child) {
  margin-bottom: 0;
}

.body :deep(h4) {
  margin: 12px 0 6px;
  font-size: 13.5px;
  color: var(--accent);
}

.body :deep(ul) {
  margin: 0 0 8px;
  padding-left: 20px;
}

.body :deep(li) {
  line-height: 1.7;
  margin-bottom: 3px;
}

.body :deep(code) {
  font-family: var(--mono);
  font-size: 12px;
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--code-bg);
}

.body :deep(pre) {
  margin: 0 0 8px;
  padding: 10px 12px;
  overflow-x: auto;
  border-radius: 8px;
  background: var(--code-bg);
}

.body :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 0 0 8px;
  font-size: 12.5px;
}

.body :deep(th),
.body :deep(td) {
  border: 1px solid var(--border);
  padding: 5px 8px;
  text-align: left;
}

.body :deep(th) {
  background: var(--surface-3);
}

.status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  color: var(--text-faint);
  font-size: 12.5px;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}

.cursor {
  display: inline-block;
  width: 7px;
  height: 15px;
  vertical-align: text-bottom;
  background: var(--accent);
  animation: blink 1s steps(2, start) infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.25; }
  50% { opacity: 1; }
}

@keyframes blink {
  50% { opacity: 0; }
}

.error {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 8px 0 0 30px;
  font-size: 12.5px;
  color: var(--danger);
}

.retry {
  padding: 3px 10px;
  border: 1px solid color-mix(in srgb, var(--danger) 45%, transparent);
  border-radius: 6px;
  background: transparent;
  color: var(--danger);
  font-size: 11.5px;
  cursor: pointer;
}

.retry:hover {
  background: color-mix(in srgb, var(--danger) 14%, transparent);
}

.foot {
  display: flex;
  gap: 12px;
  margin: 8px 0 0 30px;
  font-size: 11.5px;
  color: var(--text-faint);
}

.rewritten {
  color: var(--accent);
}
</style>
