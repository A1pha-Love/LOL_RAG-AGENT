<script setup lang="ts">
/**
 * Three-column shell: sessions / conversation / retrieval inspector.
 *
 * The top bar carries backend health, because a missing or stale knowledge index
 * degrades answers silently — the same class of failure as a wrong number, so it
 * belongs on screen rather than in a log.
 */
import { onMounted, ref } from 'vue'
import ChatStream from './components/ChatStream.vue'
import RetrievalPanel from './components/RetrievalPanel.vue'
import SessionList from './components/SessionList.vue'
import { store } from '@/composables/useAgent'

const { state, retrievalView } = store
const expandedChunk = ref<number | null>(null)
const loading = ref(false)

async function withLoading(task: () => Promise<void>): Promise<void> {
  loading.value = true
  try {
    await task()
  } catch (error) {
    state.banner = (error as Error).message
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  await withLoading(async () => {
    await store.refreshHealth()
    await store.refreshSessions()
  })
})

function onAsk(question: string): void {
  expandedChunk.value = null
  void store.ask(question)
}

function onSelectTurn(id: number): void {
  expandedChunk.value = null
  state.selectedTurnId = id
}
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="pulse" />
        <h1>LoL 排位数据分析 Agent</h1>
        <span class="sub">本地运行 · 统计由 Python 计算 · 回答标注文档来源</span>
      </div>

      <div class="status">
        <span v-if="state.streaming" class="badge busy">生成中…</span>
        <span v-if="!state.health" class="badge danger">后端未连接</span>
        <template v-else>
          <span v-if="!state.health.index_exists" class="badge danger">索引缺失</span>
          <span
            v-else-if="state.health.index_stale"
            class="badge warn"
            :title="state.health.index_reason"
          >
            索引过期
          </span>
          <span v-else class="badge ok" :title="state.health.index_reason">索引就绪</span>
        </template>
      </div>
    </header>

    <p v-if="state.banner" class="banner">⚠ {{ state.banner }}</p>

    <main class="columns">
      <SessionList
        :sessions="state.sessions"
        :active-id="state.activeSessionId"
        :busy="loading || state.streaming"
        @create="store.startNewSession()"
        @select="(id: string) => withLoading(() => store.selectSession(id))"
        @remove="(id: string) => withLoading(() => store.removeSession(id))"
      />

      <ChatStream
        :turns="state.turns"
        :streaming="state.streaming"
        :selected-turn-id="state.selectedTurnId"
        @ask="onAsk"
        @stop="store.stop()"
        @select-turn="onSelectTurn"
        @retry="(id: number) => store.retry(id)"
      />

      <RetrievalPanel v-model:expanded="expandedChunk" :view="retrievalView" />
    </main>
  </div>
</template>

<style scoped>
.shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
  min-height: 0;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 0;
}

.brand h1 {
  margin: 0;
  font-size: 14px;
  font-weight: 650;
  white-space: nowrap;
}

.brand .sub {
  font-size: 11.5px;
  color: var(--text-faint);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pulse {
  flex: none;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
}

.status {
  display: flex;
  gap: 6px;
}

.badge {
  padding: 3px 9px;
  border-radius: 20px;
  font-size: 11.5px;
  border: 1px solid transparent;
  white-space: nowrap;
}

.badge.ok {
  color: var(--ok);
  background: color-mix(in srgb, var(--ok) 12%, transparent);
  border-color: color-mix(in srgb, var(--ok) 35%, transparent);
}

.badge.warn {
  color: var(--warn);
  background: color-mix(in srgb, var(--warn) 12%, transparent);
  border-color: color-mix(in srgb, var(--warn) 35%, transparent);
}

.badge.danger {
  color: var(--danger);
  background: color-mix(in srgb, var(--danger) 12%, transparent);
  border-color: color-mix(in srgb, var(--danger) 35%, transparent);
}

.badge.busy {
  color: var(--accent);
  background: var(--accent-soft);
  border-color: var(--accent-border);
}

.banner {
  margin: 0;
  padding: 8px 16px;
  background: color-mix(in srgb, var(--danger) 10%, transparent);
  color: var(--danger);
  font-size: 12.5px;
  border-bottom: 1px solid var(--border);
}

.columns {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr) 340px;
  gap: 1px;
  background: var(--border);
}

.columns > * {
  background: var(--surface);
  min-width: 0;
}

/* Below this width the inspector would squeeze the transcript, so it is dropped
   rather than made unusable. */
@media (max-width: 1180px) {
  .columns {
    grid-template-columns: 200px minmax(0, 1fr);
  }

  .columns > :last-child {
    display: none;
  }
}
</style>
