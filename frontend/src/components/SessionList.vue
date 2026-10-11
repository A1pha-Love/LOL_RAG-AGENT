<script setup lang="ts">
/**
 * Session list: create, switch, and delete conversations.
 *
 * The active session id is shown because the backend can create a session
 * implicitly on the first question, and seeing the id makes that visible rather
 * than mysterious.
 */
import type { SessionSummary } from '@/types'

defineProps<{
  sessions: readonly SessionSummary[]
  activeId: string | null
  busy: boolean
}>()

const emit = defineEmits<{
  (event: 'create'): void
  (event: 'select', id: string): void
  (event: 'remove', id: string): void
}>()

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
      <button class="ghost" :disabled="busy" title="开始一个新会话" @click="emit('create')">
        ＋ 新建
      </button>
    </header>

    <p v-if="!sessions.length" class="empty">还没有会话。<br />在下方输入问题即可开始。</p>

    <ul v-else class="session-list">
      <li v-for="session in sessions" :key="session.session_id">
        <button
          class="session-item"
          :class="{ active: session.session_id === activeId }"
          :disabled="busy"
          @click="emit('select', session.session_id)"
        >
          <span class="title">{{ session.title || '（未命名会话）' }}</span>
          <span class="meta">
            <code>{{ shortId(session.session_id) }}</code>
            <span>{{ session.turn_count }} 轮</span>
            <span>{{ when(session.updated_at) }}</span>
          </span>
        </button>
        <button
          class="icon danger"
          title="删除该会话"
          :disabled="busy"
          @click.stop="emit('remove', session.session_id)"
        >
          ×
        </button>
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
  align-items: stretch;
  gap: 2px;
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
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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
