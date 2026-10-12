<script setup lang="ts">
/**
 * Three-column shell: sessions / conversation / retrieval inspector.
 *
 * The top bar carries backend health, because a missing or stale knowledge index
 * degrades answers silently — the same class of failure as a wrong number, so it
 * belongs on screen rather than in a log.
 */
// 中文说明
//
// 本组件是布局骨架，只做三件事：摆放三栏、顶栏状态、把用户操作转发给 store。
// 它刻意不持有任何业务状态——会话列表、回合内容、检索详情分别由三个子组件
// 通过 props 接收，数据单一来源仍是 composables/useAgent.ts。
//
// 为什么把「后端健康 / 索引是否过期」放在顶栏（原注释）：
//   索引缺失或过期会让回答**静默**变差（检索不到依据、答非所问），这类错误和
//   「算错一个数字」同属一类，必须摆在用户眼前，而不是只写进日志。
//
// 三栏宽度：240px（会话）/ 自适应（对话）/ 340px（检索）；
// 窗口窄于 1180px 时按下面的媒体查询丢掉右栏（参见文件末尾样式）。
import { onMounted, ref } from 'vue'
import ChatStream from './components/ChatStream.vue'
import RetrievalPanel from './components/RetrievalPanel.vue'
import SessionList from './components/SessionList.vue'
import { store } from '@/composables/useAgent'

// 这里解构出的是「稳定引用」：state 是 reactive 对象本身，retrievalView 是
// computed 的 ref（其 .value 会变，但 ref 对象本身不变），所以在 setup 阶段
// 取一次即可，模板中 Vue 会自动解包 computed。
const { state, retrievalView } = store
/** 右侧检索面板当前展开的分块下标；null 表示都收起。切换回合时会被重置。 */
const expandedChunk = ref<number | null>(null)
/** 侧栏是否有一次「破坏性/初始化」操作在进行中（新建、切换、删除、置顶）。 */
const loading = ref(false)

/**
 * 中文说明：统一给异步操作加上 loading 开关与错误兜底。
 *
 * 参数:
 *     task: 要执行的异步操作（通常是 store 的方法）。
 *
 * 行为:
 *     执行期间 loading=true（侧栏会禁用按钮，避免重复点击）；
 *     抛错时把消息写进顶部的 banner 告警条，而不是让异常冒泡成白屏。
 *     用 finally 保证异常路径下也会复位 loading。
 */
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

// 首次进入：并行地拿到健康状态与会话列表。
// 这里刻意 await（而不是像 ask() 里那样 fire-and-forget）：启动阶段还没有任何
// 可交互内容，等一下不会造成「界面不响应」的观感问题。
onMounted(async () => {
  await withLoading(async () => {
    await store.refreshHealth()
    await store.refreshSessions()
  })
})

/** 中文说明：提交一个问题。`void` 表示不等待——ask 自己会驱动界面状态。 */
function onAsk(question: string): void {
  // 新问题会把右侧面板的焦点交给新回合，先收起上一个展开的分块。
  expandedChunk.value = null
  void store.ask(question)
}

/** 中文说明：用户点选某一轮对话，右侧检索面板随之切换。 */
function onSelectTurn(id: number): void {
  // 展开状态是「某个分块下标」，跟回合绑定没有意义，切换时清掉。
  expandedChunk.value = null
  state.selectedTurnId = id
}
</script>

<template>
  <div class="shell">
    <!-- 顶栏：应用标识 + 后端/索引状态。状态区是整页唯一的「系统健康」入口。 -->
    <header class="topbar">
      <div class="brand">
        <span class="pulse" />
        <h1>LoL 排位数据分析 Agent</h1>
        <!-- 副标题同时点明两件事：全部本地运行（数据不出机）、统计由 Python 计算
             （而不是让模型估数），这是本项目可信度的核心主张。 -->
        <span class="sub">本地运行 · 统计由 Python 计算 · 回答标注文档来源</span>
      </div>

      <div class="status">
        <!-- 生成中与后端未连接可以同时成立，所以是两个独立的 v-if。 -->
        <span v-if="state.streaming" class="badge busy">生成中…</span>
        <span v-if="!state.health" class="badge danger">后端未连接</span>
        <template v-else>
          <!-- 索引不存在：知识检索必然失败，属于最严重的情况。 -->
          <span v-if="!state.health.index_exists" class="badge danger">索引缺失</span>
          <!-- 索引过期：能检索但内容可能已落后于源文档，用警告级别。 -->
          <span
            v-else-if="state.health.index_stale"
            class="badge warn"
            :title="state.health.index_reason"
          >
            索引过期
          </span>
          <!-- 正常状态也把判断理由放在 tooltip 里，便于排查「为什么说过期」。 -->
          <span v-else class="badge ok" :title="state.health.index_reason">索引就绪</span>
        </template>
      </div>
    </header>

    <!-- 全局告警条（无法连接后端、操作失败等）。 -->
    <p v-if="state.banner" class="banner">⚠ {{ state.banner }}</p>

    <!-- 三栏布局。左：会话列表；中：对话流与输入框；右：检索过程检视器。 -->
    <main class="columns">
      <!-- 左栏：所有操作都包一层 withLoading，统一 loading 与错误提示。 -->
      <SessionList
        :sessions="state.sessions"
        :active-id="state.activeSessionId"
        :loading="loading"
        :streaming-id="state.streamingSessionId"
        @create="store.startNewSession()"
        @select="(id: string) => withLoading(() => store.selectSession(id))"
        @remove="(id: string) => withLoading(() => store.removeSession(id))"
        @pin="(id: string, pinned: boolean) => withLoading(() => store.togglePin(id, pinned))"
      />

      <!-- 中栏：对话流。ask/stop/retry 直接透传；选回合需要先清掉右栏展开态，
           所以走 onSelectTurn 而不是直接赋值。 -->
      <ChatStream
        :turns="state.turns"
        :streaming="state.streaming"
        :selected-turn-id="state.selectedTurnId"
        @ask="onAsk"
        @stop="store.stop()"
        @select-turn="onSelectTurn"
        @retry="(id: number) => store.retry(id)"
      />

      <!-- 右栏：v-model:expanded 双向绑定展开的分块下标，展开状态由父组件持有，
           这样切换回合时能统一重置。 -->
      <RetrievalPanel v-model:expanded="expandedChunk" :view="retrievalView" />
    </main>
  </div>
</template>

<style scoped>
/* 外层纵向流式布局：顶栏 + 告警条固定高度，三栏区占满剩余空间。
   min-height: 0 是让内部滚动区（.columns 里的各栏）能正确收缩的必要条件，
   否则 flex 子项的最小内容高度会让整页出现滚动条。 */
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

/* 副标题在窄窗口下省略号截断，避免把状态徽标挤出去。 */
.brand .sub {
  font-size: 11.5px;
  color: var(--text-faint);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 品牌色小圆点，纯装饰。 */
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

/* 三种语义色共用一套「淡底 + 同色描边」的写法，靠 color-mix 从单个变量派生，
   这样换主题只需改 --ok / --warn / --danger。 */
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

/* 三栏：固定宽度的侧栏 + 可伸缩的中栏。
   gap:1px + 各自 --border 背景，是常用的「用背景色画出 1px 分隔线」技巧。 */
.columns {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr) 340px;
  gap: 1px;
  background: var(--border);
}

/* minmax(0, 1fr) 与 min-width: 0 都是为了允许中栏被压缩：
   grid/flex 子项默认 min-width:auto，长表格会把中栏撑破而不换行。 */
.columns > * {
  background: var(--surface);
  min-width: 0;
}

/* Below this width the inspector would squeeze the transcript, so it is dropped
   rather than made unusable. */
/* 中文说明：窗口宽度不足 1180px 时，宁可整块隐藏右侧检索面板，也不把它压到
   无法阅读——检视器是辅助功能，对话本身才是主体。 */
@media (max-width: 1180px) {
  .columns {
    grid-template-columns: 200px minmax(0, 1fr);
  }

  .columns > :last-child {
    display: none;
  }
}
</style>
