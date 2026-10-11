<script setup lang="ts">
/**
 * Retrieval inspector: what the turn actually searched for and matched.
 *
 * This panel is the reason the whole turn is streamed rather than just the
 * answer. It shows the rewritten query (the single most useful debugging signal
 * for follow-up questions), the tools the model chose, and each matched chunk
 * with its score and provenance.
 */
import { computed } from 'vue'
import type { RetrievalView } from '@/composables/useAgent'

const props = defineProps<{ view: RetrievalView | null }>()

const expanded = defineModel<number | null>('expanded', { default: null })

const chunks = computed(() => props.view?.chunks ?? [])

function scoreOf(value: number | null | undefined): number {
  return typeof value === 'number' ? value : 0
}

/** Rerank scores are the primary signal when present; otherwise embedding similarity. */
function primaryScore(chunk: { rerank_score?: number | null; similarity?: number }): number {
  return chunk.rerank_score ?? chunk.similarity ?? 0
}

function barWidth(chunk: { rerank_score?: number | null; similarity?: number }): string {
  return `${Math.max(2, Math.min(100, primaryScore(chunk) * 100)).toFixed(1)}%`
}

function sourceLabel(path?: string): string {
  if (!path) return '未知来源'
  return path.replace(/^kbase\//, '')
}

const CATEGORY_LABEL: Record<string, string> = {
  project: '项目',
  domain: '领域',
  conflict: '冲突',
  meta: '元信息',
  offtopic: '干扰',
  noise: '噪声',
}
</script>

<template>
  <aside class="panel inspector">
    <header class="panel-head">
      <h2>检索过程</h2>
      <span v-if="view?.chunks.length" class="count">{{ view.chunks.length }} 块</span>
    </header>

    <p v-if="!view" class="empty">选择左侧任意一轮对话，查看它的检索细节。</p>

    <div v-else class="body">
      <section class="block">
        <h3>检索查询</h3>
        <p class="query">{{ view.query ?? '（本轮未调用知识检索）' }}</p>
        <p v-if="view.rewritten" class="hint ok">已根据对话历史改写</p>
        <p v-else-if="view.query" class="hint">与原问题一致，无需改写</p>
        <p v-if="view.fallback" class="hint warn">改写失败已回退：{{ view.fallback }}</p>
      </section>

      <section v-if="view.tools.length" class="block">
        <h3>工具调用</h3>
        <div class="chips">
          <span v-for="tool in view.tools" :key="tool" class="chip">{{ tool }}</span>
        </div>
      </section>

      <section class="block">
        <h3>命中分块</h3>
        <p v-if="!chunks.length" class="empty-inline">
          本轮没有命中分块（可能是纯统计问题，或检索未通过阈值）。
        </p>
        <ul v-else class="chunk-list">
          <li v-for="(chunk, index) in chunks" :key="index">
            <button class="chunk" @click="expanded = expanded === index ? null : index">
              <div class="chunk-top">
                <span class="rank">{{ chunk.rerank_rank ?? index + 1 }}</span>
                <span class="path">{{ chunk.heading_path || sourceLabel(chunk.source) }}</span>
                <span class="score">{{ primaryScore(chunk).toFixed(3) }}</span>
              </div>
              <div class="bar"><i :style="{ width: barWidth(chunk) }" /></div>
              <div class="chunk-meta">
                <code>{{ sourceLabel(chunk.source) }}</code>
                <span v-if="chunk.category" class="tag">
                  {{ CATEGORY_LABEL[chunk.category] ?? chunk.category }}
                </span>
                <span v-if="chunk.version && chunk.version !== 'n/a'" class="tag">{{ chunk.version }}</span>
                <span v-if="chunk.rerank_score != null" class="tag muted">
                  向量 {{ scoreOf(chunk.similarity).toFixed(3) }}
                </span>
              </div>
            </button>
            <p v-if="expanded === index" class="preview">{{ chunk.preview }}</p>
          </li>
        </ul>
      </section>

      <p v-if="chunks.length" class="foot-note">
        分数为本次请求内的相对值，不能跨问题比较。
      </p>
    </div>
  </aside>
</template>

<style scoped>
.inspector {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.count {
  font-size: 11.5px;
  color: var(--text-faint);
}

.body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.block h3 {
  margin: 0 0 7px;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--text-faint);
}

.query {
  margin: 0;
  padding: 9px 10px;
  border-radius: 8px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  font-size: 12.5px;
  line-height: 1.65;
  word-break: break-word;
}

.hint {
  margin: 6px 0 0;
  font-size: 11.5px;
  color: var(--text-faint);
}

.hint.ok {
  color: var(--accent);
}

.hint.warn {
  color: var(--warn);
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.chip {
  padding: 3px 8px;
  border-radius: 5px;
  background: var(--accent-soft);
  border: 1px solid var(--accent-border);
  color: var(--accent);
  font-family: var(--mono);
  font-size: 11px;
}

.chunk-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.chunk {
  width: 100%;
  padding: 8px 9px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface-2);
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.chunk:hover {
  border-color: var(--accent-border);
}

.chunk-top {
  display: flex;
  align-items: baseline;
  gap: 7px;
}

.rank {
  flex: none;
  width: 16px;
  height: 16px;
  border-radius: 4px;
  display: grid;
  place-items: center;
  background: var(--surface-3);
  font-size: 10.5px;
  color: var(--text-faint);
}

.path {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  line-height: 1.5;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.score {
  flex: none;
  font-family: var(--mono);
  font-size: 11.5px;
  color: var(--accent);
}

.bar {
  margin: 6px 0 5px;
  height: 3px;
  border-radius: 2px;
  background: var(--surface-3);
  overflow: hidden;
}

.bar i {
  display: block;
  height: 100%;
  background: var(--accent);
}

.chunk-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-size: 10.5px;
  color: var(--text-faint);
}

.chunk-meta code {
  font-family: var(--mono);
}

.tag {
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--surface-3);
}

.tag.muted {
  opacity: 0.75;
}

.preview {
  margin: 5px 0 0;
  padding: 8px 9px;
  border-left: 2px solid var(--accent-border);
  background: var(--code-bg);
  border-radius: 0 6px 6px 0;
  font-size: 11.5px;
  line-height: 1.7;
  color: var(--text-muted);
  word-break: break-word;
}

.empty,
.empty-inline {
  color: var(--text-faint);
  font-size: 12px;
  line-height: 1.7;
}

.empty {
  padding: 18px 14px;
}

.empty-inline {
  margin: 0;
}

.foot-note {
  margin: 0;
  font-size: 11px;
  line-height: 1.6;
  color: var(--text-faint);
}
</style>
