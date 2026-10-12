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
// 中文说明
//
// 左栏：会话列表（新建 / 切换 / 置顶 / 删除）。
//
// 核心设计取舍（原注释）：**流式生成期间不禁止切换会话**。一轮回答可能几十秒，
// 这段时间锁死侧栏等于把用户钉在原地。为了让切换安全，正在生成的回合被放在
// store 的 liveTurn 里（见 composables/useAgent.ts），切走时流继续在后台跑，
// 切回来还能接着看；侧栏则用一个「生成中…」标记告诉用户哪个会话在生成。
//
// 因此 loading（App.vue 传入的）只用于两件事：初始加载与删除这类破坏性操作，
// 永远不用来禁用导航。
//
// 分组展示语义（本组件最值得看的部分）：
//   列表顺序完全由后端决定（pinned DESC, pinned_at DESC, last_active_at DESC），
//   前端只负责「在置顶组结束的位置插一条『置顶』分隔标签」，让顺序看起来是有
//   理由的、而不是随机的。见 showPinnedDivider 与模板中的 group-label。
import type { SessionSummary } from '@/types'

const props = defineProps<{
  /** 已由后端排好序的会话摘要列表。 */
  sessions: readonly SessionSummary[]
  /** 当前打开的会话 id；null 表示正处于「新会话」空状态。 */
  activeId: string | null
  /** Suppresses actions during the initial load. */
  /** 中文说明：初始加载中；用于禁用新建/删除，不影响切换。 */
  loading: boolean
  /** The conversation currently generating, if any. */
  /** 中文说明：正在生成回答的会话 id（可能不是当前可见的那个）。 */
  streamingId: string | null
}>()

const emit = defineEmits<{
  (event: 'create'): void
  (event: 'select', id: string): void
  (event: 'remove', id: string): void
  (event: 'pin', id: string, pinned: boolean): void
}>()

/** True for the first row, when at least one conversation is pinned. */
// 中文说明：判断是否要在这一行之前插入「置顶」分隔标签。
//
// 条件是「这是列表第一行 且 这一行本身是置顶的」：由于后端保证置顶项排在最前，
// 第一行是置顶就说明存在置顶组，标签插在它上面正好把「置顶组」与「其余」分开，
// 且只会插一次。若没有任何置顶项，则整组标签都不出现。
function showPinnedDivider(session: SessionSummary): boolean {
  return props.sessions[0]?.session_id === session.session_id && session.pinned === 1
}

/** 中文说明：这个会话是否正在生成——决定标题前的呼吸圆点与「生成中…」文案。 */
function isStreaming(id: string): boolean {
  return props.streamingId === id
}

/** 中文说明：会话 id 太长，只展示前 6 位作为可辨识的短码。 */
function shortId(id: string): string {
  return id.slice(0, 6)
}

/**
 * 中文说明：把时间戳转成「刚刚 / N 分钟前 / N 小时前 / 月-日」。
 *
 * 参数:
 *     iso: 后端给的 ISO 时间字符串（可能为空或是无法解析的格式）。
 *
 * 返回:
 *     相对时间的可读文案；今天以前落到本地化的「月/日」。
 *
 * 两处兜底值得注意：
 *   - 空串直接返回空串（不显示时间）；
 *   - 解析失败（非法日期）时退回截取字符串前 16 个字符（`YYYY-MM-DDTHH:MM`），
 *     这样界面至少能显示原始信息，而不是 "Invalid Date"。
 * 用 Math.round 而不是 Math.floor：59 秒显示「1 分钟前」比「0 分钟前」自然。
 */
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
      <!-- 新建只是切到空状态（见 store.startNewSession），不涉及网络写操作，
           所以只在初始加载时禁用。 -->
      <button class="ghost" :disabled="loading" title="开始一个新会话" @click="emit('create')">
        ＋ 新建
      </button>
    </header>

    <!-- 空列表提示；明确告诉用户「在下方输入问题即可开始」，降低首次使用门槛。 -->
    <p v-if="!sessions.length" class="empty">还没有会话。<br />在下方输入问题即可开始。</p>

    <ul v-else class="session-list">
      <li v-for="session in sessions" :key="session.session_id">
        <!-- A divider marks where the pinned group ends, so the ordering is
             legible rather than looking arbitrary. -->
        <!-- 中文说明：置顶分隔标签。因为置顶项固定排在最前（后端 pinned DESC），
             只要在第一行之前插一个「置顶」，读者就能理解上面的组和下面的组
             是两套排序规则，而不是顺序错乱。 -->
        <p v-if="showPinnedDivider(session)" class="group-label">置顶</p>
        <div class="row">
          <!-- 会话本体按钮：active 高亮、pinned 影响标题颜色。 -->
          <button
            class="session-item"
            :class="{ active: session.session_id === activeId, pinned: session.pinned === 1 }"
            @click="emit('select', session.session_id)"
          >
            <span class="title">
              <!-- Never disable navigation: mark the conversation instead, so the user
                   can see where the answer is still being written. -->
              <!-- 中文说明：正在生成时不禁用这个按钮，只在标题前加一个呼吸圆点 +
                   下方「生成中…」文字。用户既能立刻看出哪个会话在生成，又不会被
                   拦住不让切换。 -->
              <span v-if="isStreaming(session.session_id)" class="spinner" title="正在生成回答" />
              {{ session.title || '（未命名会话）' }}
            </span>
            <span class="meta">
              <code>{{ shortId(session.session_id) }}</code>
              <!-- 生成中时用「生成中…」替换「N 轮」：此时轮数还没更新，显示旧值
                   会让人困惑。 -->
              <span v-if="isStreaming(session.session_id)" class="generating">生成中…</span>
              <span v-else>{{ session.turn_count }} 轮</span>
              <!-- 优先用「最后活动时间」；老数据可能没有该字段，退回 updated_at，
                   保证列表永远有时间可显示。 -->
              <span>{{ when(session.last_active_at || session.updated_at) }}</span>
            </span>
          </button>
          <!-- 置顶星标：★ 表示已置顶，☆ 表示未置顶。@click.stop 防止同时触发
               外层的「选中会话」。发送的是「目标状态」（取反后的值），而不是
               当前状态，避免后端还要自己推断意图。 -->
          <button
            class="icon pin"
            :class="{ on: session.pinned === 1 }"
            :title="session.pinned === 1 ? '取消置顶' : '置顶该会话'"
            @click.stop="emit('pin', session.session_id, session.pinned !== 1)"
          >
            {{ session.pinned === 1 ? '★' : '☆' }}
          </button>
          <!-- 删除：唯一的破坏性操作，所以在 loading 期间禁用；正在生成的会话
               也不允许删除（后端此时可能还在往这个会话写数据）。 -->
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
