<script setup lang="ts">
/**
 * Retrieval inspector: what the turn actually searched for and matched.
 *
 * This panel is the reason the whole turn is streamed rather than just the
 * answer. It shows the rewritten query (the single most useful debugging signal
 * for follow-up questions), the tools the model chose, and each matched chunk
 * with its score and provenance.
 */
// 中文说明
//
// 右栏「检索过程」检视器：把这一轮问答的**内部过程**摊开来给用户看。
//
// 为什么值得单独做一栏、也是「整轮都流式推送」的原因（原注释）：
//   只看最终回答，用户无法判断「这个结论到底检索到了什么依据」。这里展示三样：
//     1. 改写后的检索查询——复述类追问（「那它的胜率呢」）最关键的一条线索，
//        能看出模型把上下文补全成了什么查询；
//     2. 模型选择的工具；
//     3. 每个命中分块的来源、类别与分数。
//   后端为此在生成过程中就把 rewrite / tools / retrieval 事件推给前端，
//   这些事件单独看没有意义，只有配合这个面板才有价值。
//
// 数据来源：props.view 由 useAgent 的 retrievalView 派生（它跟随 selectedTurn）。
import { computed } from 'vue'
import type { RetrievalView } from '@/composables/useAgent'

const props = defineProps<{ view: RetrievalView | null }>()

/**
 * 中文说明：展开的分块下标，用 defineModel 做成双向绑定
 * （父组件写 `v-model:expanded="expandedChunk"`）。
 *
 * 为什么要提到父组件：切换回合时父组件需要把它重置为 null，否则「上一个回合
 * 展开了第 2 块」的状态会莫名套用到新回合上。默认 null = 全部收起。
 *
 * 注意类型是**下标**而不是 chunk_id：因为一轮检索结果只是一次性展示数据，
 * 用下标最简单（列表不会在展示期间变化）。
 */
const expanded = defineModel<number | null>('expanded', { default: null })

/** 中文说明：命中的分块列表；view 为 null（还没选回合）时给空数组，模板少写判空。 */
const chunks = computed(() => props.view?.chunks ?? [])

/**
 * 中文说明：把可能缺失的分数归一成数字。
 *
 * 后端字段是可选的，直接 `.toFixed()` 会在 undefined 上抛错，所以统一兜底为 0。
 */
function scoreOf(value: number | null | undefined): number {
  return typeof value === 'number' ? value : 0
}

/** Rerank scores are the primary signal when present; otherwise embedding similarity. */
// 中文说明：主展示分数——有 rerank 分数就用它，否则退回一阶段相似度。
//
// 用 `??` 而不是 `||`：rerank_score 理论上可能为 0，用 || 会把 0 当成缺失而误退回。
function primaryScore(chunk: { rerank_score?: number | null; similarity?: number }): number {
  return chunk.rerank_score ?? chunk.similarity ?? 0
}

/**
 * 中文说明：把分数换算成进度条宽度。
 *
 * 三重约束：
 *   - 乘以 100 转成百分比（分数是 0~1 的量纲）；
 *   - Math.max(2, …) 保证极低分也留一条可见的细线——宽度为 0 会让人以为没渲染出来；
 *   - Math.min(100, …) 防止后端给到 1 以上的值把条撑出容器。
 */
function barWidth(chunk: { rerank_score?: number | null; similarity?: number }): string {
  return `${Math.max(2, Math.min(100, primaryScore(chunk) * 100)).toFixed(1)}%`
}

/**
 * 中文说明：把来源路径处理成适合展示的短标签。
 *
 * 索引里的路径统一带 `kbase/` 前缀（这是索引目录名，对用户没有信息量），
 * 因此展示时剥掉；路径缺失时给出「未知来源」而不是留空。
 */
function sourceLabel(path?: string): string {
  if (!path) return '未知来源'
  return path.replace(/^kbase\//, '')
}

/**
 * 中文说明：分块类别的中文标签。
 *
 * 这些类别是知识库构建阶段给每个分块打的标（见后端索引脚本），其中
 * `offtopic` / `noise` 是刻意放入知识库的**干扰文档**：它们用来验证检索
 * 会不会被无关内容带偏，所以这里也照实展示，不做隐藏。
 * 用 ?? 兜底是因为后端可能新增类别，届时直接显示原值即可。
 */
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
      <!-- 命中块数直接放在标题右侧，一眼可见本轮检索规模。 -->
      <span v-if="view?.chunks.length" class="count">{{ view.chunks.length }} 块</span>
    </header>

    <!-- 还没有选中回合（例如刚打开应用）时的空状态提示。 -->
    <p v-if="!view" class="empty">选择左侧任意一轮对话，查看它的检索细节。</p>

    <div v-else class="body">
      <!-- ① 检索查询：整个面板信息密度最高的一段。 -->
      <section class="block">
        <h3>检索查询</h3>
        <!-- 没有检索查询说明这一轮没调知识检索（纯统计类问题通常如此），
             用括号说明而不是留白，避免用户以为界面坏了。 -->
        <p class="query">{{ view.query ?? '（本轮未调用知识检索）' }}</p>
        <!-- 三种提示互斥/递进：改写成功 / 无需改写 / 改写失败回退。 -->
        <p v-if="view.rewritten" class="hint ok">已根据对话历史改写</p>
        <p v-else-if="view.query" class="hint">与原问题一致，无需改写</p>
        <p v-if="view.fallback" class="hint warn">改写失败已回退：{{ view.fallback }}</p>
      </section>

      <!-- ② 工具调用：模型这一轮选了哪些工具，只有调用了才显示这一节。 -->
      <section v-if="view.tools.length" class="block">
        <h3>工具调用</h3>
        <div class="chips">
          <span v-for="tool in view.tools" :key="tool" class="chip">{{ tool }}</span>
        </div>
      </section>

      <!-- ③ 命中分块：每个分块一张可点开的卡片。 -->
      <section class="block">
        <h3>命中分块</h3>
        <!-- 空结果的两种常见原因都写清楚，用户才知道这不是故障。 -->
        <p v-if="!chunks.length" class="empty-inline">
          本轮没有命中分块（可能是纯统计问题，或检索未通过阈值）。
        </p>
        <ul v-else class="chunk-list">
          <!-- :key 用下标：这份列表是一次性展示数据，不会在展示期间被重排。 -->
          <li v-for="(chunk, index) in chunks" :key="index">
            <!-- 点击在「展开这一块」与「收起」之间切换。 -->
            <button class="chunk" @click="expanded = expanded === index ? null : index">
              <div class="chunk-top">
                <!-- rerank 名次优先；未启用 rerank 时用列表位置（1 起）代替。 -->
                <span class="rank">{{ chunk.rerank_rank ?? index + 1 }}</span>
                <!-- 优先显示标题路径（可读性最好），没有才退回文件路径。 -->
                <span class="path">{{ chunk.heading_path || sourceLabel(chunk.source) }}</span>
                <span class="score">{{ primaryScore(chunk).toFixed(3) }}</span>
              </div>
              <!-- 分数条：宽度由 barWidth 计算，直观比较块之间的相对得分。 -->
              <div class="bar"><i :style="{ width: barWidth(chunk) }" /></div>
              <div class="chunk-meta">
                <code>{{ sourceLabel(chunk.source) }}</code>
                <!-- 类别与版本；version 为 'n/a' 是后端约定的「无版本」，不展示。 -->
                <span v-if="chunk.category" class="tag">
                  {{ CATEGORY_LABEL[chunk.category] ?? chunk.category }}
                </span>
                <span v-if="chunk.version && chunk.version !== 'n/a'" class="tag">{{ chunk.version }}</span>
                <!-- 只有真的经过 rerank 时才显示原始向量分，否则两者相同、展示无意义。 -->
                <span v-if="chunk.rerank_score != null" class="tag muted">
                  向量 {{ scoreOf(chunk.similarity).toFixed(3) }}
                </span>
              </div>
            </button>
            <!-- 展开后显示后端随事件下发的 ~180 字预览（不下发全文，避免响应体过大）。 -->
            <p v-if="expanded === index" class="preview">{{ chunk.preview }}</p>
          </li>
        </ul>
      </section>

      <!-- 全局脚注：这条提醒很重要——分数是「本次请求内的相对值」，
           跨问题比较会得出错误结论（后端 API 文档如此声明）。 -->
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
