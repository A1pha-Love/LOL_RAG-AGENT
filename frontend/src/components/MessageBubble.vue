<script setup lang="ts">
/**
 * One question and its answer.
 *
 * Renders a deliberately small Markdown subset (headings, bold, inline code,
 * lists, tables) instead of pulling in a Markdown library: the answers are
 * structured but predictable, and escaping is handled here so model output can
 * never inject markup.
 */
// 中文说明
//
// 一张「问 + 答」卡片，中栏里每一轮对话就是一个本组件。
//
// 为什么自己写一个极小的 Markdown 子集渲染器（原注释）：
//   1. 模型的回答是自己用提示词约束出来的，格式**可预测**（标题、加粗、行内代码、
//      列表、表格），不需要一个完整 Markdown 解析器；
//   2. 少一个第三方依赖，也少一条「库的 XSS 漏洞」风险链；
//   3. 转义在这里显式完成（见 escapeHtml），模型输出永远无法注入标签。
//
// ⚠ 安全模型（理解这一点比看懂解析逻辑更重要）：
//   先转义、后格式化，且顺序不可颠倒。若先加 `<strong>` 之类标签再转义，转义会把
//   自己生成的标签也变成文本；反之（先转义再插入标记）则模型给出的 `<script>`
//   已经变成 `&lt;script&gt;`，插入的标记只可能是本文件写死的那几种。
//   模板里所有 v-html 绑定的都是 inline()/escapeHtml() 处理过的字符串。
import { computed } from 'vue'
import RefreshButton from './RefreshButton.vue'
import type { UiTurn } from '@/composables/useAgent'

const props = defineProps<{
  turn: UiTurn
  /** 是否是当前选中回合（高亮 + 右侧面板显示它的检索详情）。 */
  selected: boolean
  /** Follow-up questions offered under a finished answer. */
  /** 中文说明：回答完成后展示的追问建议。 */
  followUps?: readonly string[]
}>()

const emit = defineEmits<{
  (event: 'select'): void
  (event: 'retry'): void
  (event: 'ask', question: string): void
  (event: 'rotate-follow-ups'): void
}>()

/** Escape first, then apply inline formatting — never the other way round. */
// 中文说明：HTML 转义。五个字符里处理了 & < > "，& 必须最先替换，
// 否则后面替换出来的 `&lt;` 会被再次转义成 `&amp;lt;`。
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/**
 * 中文说明：行内格式——`代码` 与 **加粗**。
 *
 * 目标串已经转义完毕，所以这里的正则只可能命中「用户写下的反引号/星号」，
 * 不会命中被转义成实体的尖括号。执行顺序是先代码后加粗：代码里的星号应当
 * 原样保留（虽然此处 `([^`]+)` 内部仍会被后一步匹配，属于已知的简化）。
 */
function inline(text: string): string {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
}

/**
 * 中文说明：解析结果的中间表示。解析与渲染分离，好处是模板只需按 kind 分支，
 * 不用在模板里写正则；也便于将来替换渲染方式。
 */
interface Block {
  kind: 'heading' | 'paragraph' | 'list' | 'table' | 'code'
  /** heading/paragraph/code 的已转义 HTML。 */
  html?: string
  /** list 的条目（已转义并应用行内格式）。 */
  items?: string[]
  /** table 的正文行（不含表头与分隔行）。 */
  rows?: string[][]
  /** table 的表头单元格。 */
  header?: string[]
}

/**
 * 中文说明：把回答正文解析成块序列（行级状态机）。
 *
 * 这是本文件的核心函数。它是一个**单遍扫描**的状态机：用几个累加器
 * （paragraph / list / table / code）收集同类行，遇到类型切换时先 flush 再开新的。
 * 之所以不是逐行产出而是「先攒后出」，是因为段落要合并多行、列表要合并多项、
 * 表格要等分隔行才能确定表头。
 *
 * 参数:
 *     source: 模型回答的原始文本（未转义）。
 *
 * 返回:
 *     Block 数组，顺序与原文一致。
 *
 * 注意实现细节：
 *   - 代码围栏 ``` 会先于一切判断，且围栏内的内容**原样保留**（不 trim、不解析），
 *     只做转义。
 *   - 调换 flush 顺序会改变输出，例如表格行之前必须先 flush 段落。
 */
function parse(source: string): Block[] {
  const lines = source.split('\n')
  const blocks: Block[] = []
  // 以下 5 个是可变的累加器（闭包共享），flush 后就地清空。
  let paragraph: string[] = []
  let list: string[] = []
  let table: string[][] = []
  let code: string[] = []
  let inCode = false

  /** 把攒下的段落做成一个 block；段落内多行用空格合并成一段。 */
  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ kind: 'paragraph', html: inline(paragraph.join(' ')) })
      paragraph = []
    }
  }
  /** 把攒下的列表项做成一个 block（有序/无序在此处统一为 <ul>）。 */
  const flushList = () => {
    if (list.length) {
      blocks.push({ kind: 'list', items: list.map(inline) })
      list = []
    }
  }
  /** 把攒下的表格行做成一个 block。 */
  const flushTable = () => {
    if (!table.length) return
    const header = table[0].map(inline)
    // Row 1 is the |---|---| separator, so body rows start at index 2.
    // 中文说明：Markdown 表格的第 2 行是 |---|---| 对齐分隔行，不是数据，
    // 因此正文从下标 2 开始。这里只做了「跳过一行」的最简处理，不解析对齐方式。
    const body = table.slice(2).map((row) => row.map(inline))
    // rows 为空数组时给 []，避免模板 v-for 拿到 undefined。
    blocks.push({ kind: 'table', header, rows: body.length ? body : [] })
    table = []
  }

  for (const raw of lines) {
    // 行尾空白不参与语义；但 raw（未 trim）在代码块里要原样保留缩进。
    const line = raw.trimEnd()

    if (line.trim().startsWith('```')) {
      if (inCode) {
        // 结束围栏：把收集到的代码行整体转义（此前的 flush 已在开围栏时做过）。
        blocks.push({ kind: 'code', html: escapeHtml(code.join('\n')) })
        code = []
        inCode = false
      } else {
        // 开始围栏：先把前面攒的段落/列表/表格收尾，保证块顺序不变。
        flushParagraph()
        flushList()
        flushTable()
        inCode = true
      }
      continue
    }
    if (inCode) {
      // 代码块内不做任何 Markdown 解析，保留原行（含缩进）。
      code.push(raw)
      continue
    }

    // 表格行：形如 | a | b |。判定必须宽松（首尾可有可无空格）。
    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushParagraph()
      flushList()
      // 去掉首尾竖线后按 | 切分，每个单元格 trim。
      table.push(line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim()))
      continue
    }
    // 不是表格行，说明表格（如果有）已经结束。
    flushTable()

    // 标题：# ~ ####（最多 4 级，统一渲染为 <h4>，因为卡片内层级不深）。
    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    if (heading) {
      flushParagraph()
      flushList()
      blocks.push({ kind: 'heading', html: inline(heading[2]) })
      continue
    }

    // 列表项：`- ` / `* ` 为无序，`1. ` / `1) ` 为有序；两种都收进同一个 list 累加器。
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line)
    const ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (bullet || ordered) {
      flushParagraph()
      // 非空断言 `!`：两分支必有一个命中，取到的捕获组即为条目文本。
      list.push((bullet ?? ordered)![1])
      continue
    }

    // 空行：结束当前段落与列表。
    if (!line.trim()) {
      flushParagraph()
      flushList()
      continue
    }
    // 其余非空行都归入段落（连续多行会被合并成一段）。
    paragraph.push(line.trim())
  }

  // 文件/消息结束时的收尾 flush，顺序与前面保持一致。
  flushParagraph()
  flushList()
  flushTable()
  // 未闭合的代码围栏：仍然把已收集的内容作为一个代码块输出，
  // 否则流式过程中（围栏还没闭合）这段代码会整段消失。
  if (code.length) blocks.push({ kind: 'code', html: escapeHtml(code.join('\n')) })
  return blocks
}

/**
 * 中文说明：把答案文本解析成块的派生值。computed 让它只在 turn.answer 变化时
 * 重新解析——流式期间每个 token 都会触发一次，这也是把解析放在 computed 而不是
 * 模板方法里的原因（模板方法每帧都会重跑）。
 */
const blocks = computed(() => parse(props.turn.answer))
</script>

<template>
  <!-- 点击卡片任意位置即可选中该回合（驱动右侧检索面板）；
       class 里的 selected / streaming 控制高亮与光标动画。 -->
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
        <!-- 有内容就渲染解析后的块；块类型分发到对应的 HTML 标签。
             这里所有 v-html 的值都已在 parse/inline 中转义。 -->
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

        <!-- 还没有任何正文：显示后端发来的进度文案（status），
             连进度文案都没有（例如刚提交、首个事件未到）就显示「正在等待模型…」。
             dot 是呼吸动画的小圆点，用来表明还在进行中。 -->
        <p v-else-if="turn.status" class="status">
          <span class="dot" />{{ turn.status }}
        </p>
        <p v-else class="status"><span class="dot" />正在等待模型…</p>

        <!-- 流式光标：只在「正在生成 且 已有正文」时出现，
             避免和上面的等待提示重复。 -->
        <span v-if="turn.streaming && blocks.length" class="cursor" />
      </div>
    </div>

    <!-- 错误行：@click.stop 阻止冒泡，否则点「重试」会顺带切换选中回合。 -->
    <p v-if="turn.error" class="error">
      <span>⚠ {{ turn.error }}</span>
      <button v-if="turn.timedOut" class="retry" @click.stop="emit('retry')">重试</button>
    </p>

    <!-- 页脚统计。这些数字全部来自后端（耗时、工具调用次数、命中块数），
         前端不做任何估算；「查询已改写」用于提示检索用了改写后的查询。 -->
    <footer v-if="!turn.streaming" class="foot">
      <span v-if="turn.seconds">{{ turn.seconds }}s</span>
      <span v-if="turn.toolCalls.length">{{ turn.toolCalls.length }} 次工具调用</span>
      <span v-if="turn.chunks.length">{{ turn.chunks.length }} 个命中分块</span>
      <span v-if="turn.retrievalQuery && turn.retrievalQuery !== turn.question" class="rewritten">
        查询已改写
      </span>
    </footer>

    <!-- Follow-up prompts: only once the answer is complete, so nothing suggests a
         new question while this one is still being written. -->
    <!-- 中文说明：追问建议只在「已结束且无错误」时出现——回答还在写的时候就
         推荐下一个问题会分散注意力，失败时也不该推荐。
         按钮上的 @click.stop 防止点击追问顺带选中本回合。 -->
    <div v-if="!turn.streaming && !turn.error && followUps?.length" class="follow-ups">
      <div class="follow-ups-head">
        <span>继续问</span>
        <RefreshButton @rotate="emit('rotate-follow-ups')" />
      </div>
      <div class="follow-up-list">
        <button
          v-for="item in followUps"
          :key="item"
          type="button"
          class="follow-up"
          @click.stop="emit('ask', item)"
        >
          {{ item }}
        </button>
      </div>
    </div>
  </article>
</template>

<style scoped>
/* 卡片本体。默认透明边框 + 透明背景，只有 hover/选中时才显形，
   这样一屏多轮对话时视觉噪音最小。 */
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

/* 选中态：淡强调底 + 强调色描边，对应右侧检索面板显示的那一轮。 */
.turn.selected {
  background: var(--surface-2);
  border-color: var(--accent-border);
}

/* 问与答都用「左侧徽标 + 右侧内容」的两列结构。 */
.question,
.answer {
  display: flex;
  gap: 10px;
}

.question {
  margin-bottom: 10px;
}

/* 问题加粗：用户扫视时先看到自己问过什么。 */
.question p {
  margin: 0;
  font-weight: 600;
}

/* 「问」「答」小方块徽标。flex: none 保证不被长文本挤压变形。 */
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

/* 答的徽标用强调色，区分角色。 */
.answer .who {
  background: var(--accent-soft);
  color: var(--accent);
}

.body {
  flex: 1;
  min-width: 0;
}

/* ── 以下全部使用 :deep() ──────────────────────────────────────────────
   原因：这些元素（p/h4/ul/code/pre/table）是 v-html 注入的 HTML，
   不带 scoped 属性（Vue 只会给模板里写出来的元素加 data-v-xxxx），
   因此普通选择器匹配不到它们。:deep() 让选择器穿透 scoped 边界。 */
.body :deep(p) {
  margin: 0 0 8px;
  line-height: 1.75;
}

/* 最后一段不留底边距，避免与页脚之间出现多余空隙。 */
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

/* 表格撑满宽度；border-collapse 让相邻单元格共享 1px 边线。 */
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

/* 等待/进度提示行。 */
.status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  color: var(--text-faint);
  font-size: 12.5px;
}

/* 呼吸圆点：「还在进行中」的最小视觉信号。 */
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse 1.2s ease-in-out infinite;
}

/* 流式光标：一个闪烁的细长块，模拟终端光标。 */
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

/* 错误行缩进 30px，与问题/答案正文左对齐（20px 徽标 + 10px 间距）。 */
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

/* 追问区：用虚线上边框与回答正文分隔，暗示「这是另一个问题」。 */
.follow-ups {
  margin: 10px 0 0 30px;
  padding-top: 9px;
  border-top: 1px dashed var(--border);
}

.follow-ups-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 11px;
  color: var(--text-faint);
}

/* 追问按钮用 flex-wrap 自动换行：问题文本长短不一，横向排不下就折行。 */
.follow-up-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.follow-up {
  max-width: 100%;
  padding: 5px 10px;
  border: 1px solid var(--border);
  border-radius: 7px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
  text-align: left;
  cursor: pointer;
}

.follow-up:hover {
  border-color: var(--accent-border);
  color: var(--accent);
}

/* 页脚统计行：同样缩进 30px 与正文对齐。 */
.foot {
  display: flex;
  gap: 12px;
  margin: 8px 0 0 30px;
  font-size: 11.5px;
  color: var(--text-faint);
}

/* 「查询已改写」用强调色，提示检索口径与原问题不同。 */
.rewritten {
  color: var(--accent);
}
</style>
