<script setup lang="ts">
/**
 * Session list: create, switch, and delete conversations.
 *
 * Switching is deliberately NOT blocked while a turn is streaming. A turn can take
 * tens of seconds, and disabling the sidebar for that whole time locks the user in
 * place; the in-flight turn is kept outside the visible transcript so it survives a
 * switch (see `liveTurn` in useAgent), and a marker shows which conversation is
 * still generating.
 *
 * `busy` therefore only guards the destructive action (delete) and the initial load,
 * never navigation.
 */
import type { SessionSummary } from '@/types'

const props = defineProps<{
  sessions: readonly SessionSummary[]
  activeId: string | null
  /** Suppresses actions during the initial load. */
  loading: boolean
  /** The conversation currently generating, if any. */
  streamingId: string | null
}>()

const emit = defineEmits<{
  (event: 'create'): void
  (event: 'select', id: string): void
  (event: 'remove', id: string): void
  (event: 'pin', id: string, pinned: boolean): void
}>()

/** True for the first row, when at least one conversation is pinned. */
function showPinnedDivider(session: SessionSummary): boolean {
  return props.sessions[0]?.session_id === session.session_id && session.pinned === 1
}

function isStreaming(id: string): boolean {
  return props.streamingId === id
}

function shortId(id: string): string {
  return id.slice(0, 6)
}

function when(iso: string): string {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso.slice(0, 16)
  const now = Date.now()
  const diffMinutes = Math.round((now - date.getTime()) / 60000)
  if (diffMinutes < 1) return '刚刚'
  if (diffMinutes < 60) return `${diffMinutes} 分钟前`
  if (diffMinutes < 60 * 24) return `${Math.round(diffMinutes / 60)} 小时前`
  return date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}
</script>

<template>
  <aside class="panel sessions">
    <header class="panel-head">
      <h2>会话</h2>
      <button class="ghost" :disabled="loading" title="开始一个新会话" @click="emit('create')">
        ＋ 新建
      </button>
    </header>

    <p v-if="!sessions.length" class="empty">还没有会话。<br />在下方输入问题即可开始。</p>

    <ul v-else class="session-list">
      <li v-for="session in sessions" :key="session.session_id">
        <!-- A divider marks where the pinned group ends, so the ordering is
             legible rather than looking arbitrary. -->
        <p v-if="showPinnedDivider(session)" class="group-label">置顶</p>
        <div class="row">
          <button
            class="session-item"
            :class="{ active: session.session_id === activeId, pinned: session.pinned === 1 }"
            @click="emit('select', session.session_id)"
          >
            <span class="title">
              <!-- Never disable navigation: mark the conversation instead, so the user
                   can see where the answer is still being written. -->
              <span v-if="isStreaming(session.session_id)" class="spinner" title="正在生成回答" />
              {{ session.title || '（未命名会话）' }}
            </span>
            <span class="meta">
              <code>{{ shortId(session.session_id) }}</code>
              <span v-if="isStreaming(session.session_id)" class="generating">生成中…</span>
              <span v-else>{{ session.turn_count }} 轮</span>
              <span>{{ when(session.last_active_at || session.updated_at) }}</span>
            </span>
          </button>
          <button
            class="icon pin"
            :class="{ on: session.pinned === 1 }"
            :title="session.pinned === 1 ? '取消置顶' : '置顶该会话'"
            @click.stop="emit('pin', session.session_id, session.pinned !== 1)"
          >
            {{ session.pinned === 1 ? '★' : '☆' }}
          </button>
          <button
            class="icon danger"
            title="删除该会话"
            :disabled="loading || isStreaming(session.session_id)"
            @click.stop="emit('remove', session.session_id)"
          >
            ×
          </button>
        </div>
      </li>
    </ul>
  </aside>
</template>

<style scoped>
.sessions {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.session-list {
  list-style: none;
  margin: 0;
  padding: 6px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.session-list li {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

/* The row holds the conversation button plus its pin and delete actions. */
.session-list li .row {
  display: flex;
  align-items: stretch;
  gap: 2px;
}

.group-label {
  margin: 6px 4px 0;
  font-size: 10.5px;
  letter-spacing: 0.06em;
  color: var(--text-faint);
}

.session-item.pinned .title {
  color: var(--accent);
}

.icon.pin {
  font-size: 13px;
  color: var(--text-faint);
}

.icon.pin:hover {
  color: var(--accent);
}

.icon.pin.on {
  color: var(--accent);
}

.session-item {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 9px 10px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.session-item:hover {
  background: var(--surface-2);
}

.session-item.active {
  background: var(--accent-soft);
  border-color: var(--accent-border);
}

.session-item .title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Marks a conversation whose answer is still being generated. */
.spinner {
  flex: none;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.25; }
  50% { opacity: 1; }
}

.session-item .meta .generating {
  color: var(--accent);
}

.session-item .meta {
  display: flex;
  gap: 8px;
  font-size: 11px;
  color: var(--text-faint);
}

.empty {
  padding: 18px 14px;
  font-size: 12.5px;
  line-height: 1.7;
  color: var(--text-faint);
}
</style>
