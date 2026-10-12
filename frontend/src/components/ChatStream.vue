<script setup lang="ts">
/**
 * Transcript plus the composer.
 *
 * Auto-scrolls only when the user is already near the bottom: forcing the view
 * down while someone is reading earlier turns is worse than not scrolling.
 */
// 中文说明
//
// 中栏：对话流（空状态显示欢迎语 + 推荐问题；有内容时逐轮渲染 MessageBubble）
// 加上底部输入框。
//
// 两个值得学习的点：
//   1. 自动滚动是「条件性」的（原注释）：只有当用户本来就在底部附近时才跟着
//      新内容滚。否则用户往上翻看历史时会被不断拽回底部，比不滚动更糟。
//   2. 本组件是**受控展示组件**：回合数据来自 props，提问/停止/重试/选回合都
//      通过 emit 交回 App.vue → store，自身只持有一个输入草稿 `draft`。
import { nextTick, reactive, ref, watch } from 'vue'
import MessageBubble from './MessageBubble.vue'
import RefreshButton from './RefreshButton.vue'
import { SuggestionDeck, SuggestionDecks } from '@/lib/suggestions'
import type { UiTurn } from '@/composables/useAgent'

const props = defineProps<{
  /** 当前可见会话的回合列表（只读，组件不会修改它）。 */
  turns: readonly UiTurn[]
  /** 可见会话是否有回合正在生成——决定输入框是否禁用。 */
  streaming: boolean
  /** 当前选中的回合 id，用于高亮并驱动右侧面板。 */
  selectedTurnId: number | null
}>()

const emit = defineEmits<{
  (event: 'ask', question: string): void
  (event: 'stop'): void
  (event: 'select-turn', id: number): void
  (event: 'retry', id: number): void
}>()

/** 输入框草稿。只在本组件内使用，发送后立刻清空。 */
const draft = ref('')
/** 滚动容器引用，用于读取/设置 scrollTop。 */
const scroller = ref<HTMLElement | null>(null)
/** 用户是否「贴底」；true 时才自动跟随新内容滚动。 */
const pinned = ref(true)

// One deck per visit to the empty state, so the first page is random and every
// refresh shows a different page.
// 中文说明：空状态的推荐题牌堆。模块级 let（非 ref）是因为它只在 JS 逻辑里用，
// 渲染用的是下面 suggestions 这个 ref；牌堆本身不需要响应式。
let deck = new SuggestionDeck()
/** 当前展示的推荐问题（空状态那三张卡片）。 */
const suggestions = ref<string[]>(deck.current())

// A separate deck per turn for the follow-up rows. The rendered strings live in a
// reactive map so a rotation is a plain assignment the template reacts to.
// 中文说明：每轮回答下方的「继续问」各有独立牌堆；渲染内容放在 reactive Map 里，
// 于是「换一换」只是 set 一个新数组，模板会自动更新（Map 的 set 也被 Vue 代理追踪）。
const followUpDecks = new SuggestionDecks()
const followUps = reactive(new Map<number, string[]>())

/** 中文说明：空状态点「换一换」——取牌堆的下一页。 */
function rotateSuggestions(): void {
  suggestions.value = deck.next()
}

/**
 * 中文说明：读取某一轮的追问列表，第一次访问时懒初始化并缓存在 map 里。
 *
 * 模板里每渲染一轮就会调用一次；缓存避免每次渲染都创建牌堆（否则「换一换」
 * 的效果会丢失）。
 */
function followUpsFor(turnId: number): string[] {
  let current = followUps.get(turnId)
  if (!current) {
    current = followUpDecks.current(turnId)
    followUps.set(turnId, current)
  }
  return current
}

/** 中文说明：某一轮点「换一换」——只轮换这一轮的牌堆。 */
function rotateFollowUps(turnId: number): void {
  followUps.set(turnId, followUpDecks.rotate(turnId))
}

/**
 * 中文说明：提交输入框内容。
 *
 * 空内容或正在生成时直接忽略（生成期间输入框本身也是 disabled 的，这里再挡一次
 * 是为了覆盖「点推荐问题/追问」这类不经过输入框的提交路径）。
 */
function submit(): void {
  const question = draft.value.trim()
  if (!question || props.streaming) return
  emit('ask', question)
  // 立即清空，让用户能马上开始写下一个问题（真正的发送由父组件负责）。
  draft.value = ''
}

/**
 * 中文说明：Enter 发送、Shift+Enter 换行。
 *
 * 必须 preventDefault：textarea 里 Enter 默认是插入换行，不阻止就会既发送
 * 又多出一个空行。用输入法（中文拼音）时按 Enter 是「确认候选词」，
 * 这种组合键不会被这里的判断拦下——composition 阶段浏览器不派发 keydown 的
 * Enter 到此处（更准确地说，是输入法在合成期间吞掉了该按键）。
 */
function onKeydown(event: KeyboardEvent): void {
  // Enter sends; Shift+Enter inserts a newline.
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

/**
 * 中文说明：滚动时判断用户是否贴底。
 *
 * 用「距底部小于 80px」而不是「等于 0」，是因为滚动是像素级的：用户滚到底后
 * 一点惯性、一点行高差异都会让 scrollTop 差几个像素，严格判断会导致自动滚动
 * 时灵时不灵。
 */
function onScroll(): void {
  const element = scroller.value
  if (!element) return
  pinned.value = element.scrollHeight - element.scrollTop - element.clientHeight < 80
}

/**
 * 中文说明：内容变化时自动滚到底（仅在用户贴底时）。
 *
 * 监听表达式用的是「各回合答案长度拼成的字符串」而不是 turns 本身：
 *   - 流式期间答案每来一个 token 就变一次，长度串能精确捕捉到这一点；
 *   - 而 turns 数组的引用在追加回合时也会变，但深比较整个回合对象代价高，
 *     且 applyEvent 就地改字段时数组引用不变，单纯的 watch(() => props.turns, ...)
 *     在流式追加文本时**不会**触发。
 *
 * nextTick 是必要的：DOM 还没把新文本渲染出来时读 scrollHeight 会得到旧值。
 */
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
    <!-- 唯一的滚动容器；滚动监听用于决定是否自动跟随。 -->
    <div ref="scroller" class="scroller" @scroll="onScroll">
      <!-- 空状态：欢迎语 + 数据边界说明 + 推荐问题。 -->
      <div v-if="!turns.length" class="welcome">
        <h3>英雄联盟历史排位数据分析</h3>
        <!-- 明确写出数据范围与规模，并强调统计由本地 Python 计算、回答标注来源，
             这是让用户建立可信预期的关键信息。 -->
        <p>
          数据为 2017-06-08 至 2017-09-06 的 EUW 排位赛，去重后 51,053 场。
          所有统计数字由本地 Python 计算，回答会标注文档来源。
        </p>
        <div class="suggestions-head">
          <span>试试这些问题</span>
          <RefreshButton @rotate="rotateSuggestions" />
        </div>
        <div class="suggestions">
          <!-- 直接 emit('ask', item)：点推荐问题等于替用户发送，不经过输入框。 -->
          <button v-for="item in suggestions" :key="item" class="suggestion" @click="emit('ask', item)">
            {{ item }}
          </button>
        </div>
      </div>

      <!-- 每一轮对话一张卡片。:key 用回合 id（而非下标），这样插入/删除回合时
           Vue 会复用正确的 DOM，不会被错位的流式文本串台。 -->
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

    <!-- 输入区。@submit.prevent 同时覆盖了「点发送」和「在输入框按 Enter」两条路径。 -->
    <form class="composer" @submit.prevent="submit">
      <textarea
        v-model="draft"
        rows="1"
        placeholder="输入问题，Enter 发送，Shift+Enter 换行"
        :disabled="streaming"
        @keydown="onKeydown"
      />
      <!-- 生成中把「发送」换成「停止」（同一个位置，避免布局跳动）。 -->
      <button v-if="streaming" type="button" class="stop" @click="emit('stop')">停止</button>
      <button v-else type="submit" class="send" :disabled="!draft.trim()">发送</button>
    </form>
  </section>
</template>

<style scoped>
/* 中栏也是「上下两段」结构：滚动区占满，输入区固定在最底部。 */
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

/* margin:auto 让欢迎卡片在滚动区里水平垂直居中（flex 列方向下的常见写法）。 */
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

/* line-height 1.8：中文段落需要比英文更松的行距。 */
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
/* 中文说明：这里没有「换一换」按钮的样式，因为空状态与每轮追问行共用
   RefreshButton.vue 内的 scoped 样式。 */

.suggestions {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* 推荐问题做成整行按钮（text-align: left），中文长句在窄侧栏里更好读。 */
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

/* 输入区带顶部分隔线，与滚动区视觉分离。 */
.composer {
  display: flex;
  gap: 8px;
  padding: 10px;
  border-top: 1px solid var(--border);
}

/* 输入框用 textarea（而非 input）以便 Shift+Enter 换行；
   resize: none + max-height 让高度由内容决定但不允许手动拖拽破坏布局。 */
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

/* 发送/停止按钮贴底对齐（align-self: flex-end），与输入框最后一行齐平。 */
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

/* 草稿为空时发送按钮变淡；:disabled 只影响外观与点击，不影响布局。 */
.send:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

/* 「停止」用危险色文字 + 中性底，提示这是一个中断动作而不是主操作。 */
.stop {
  background: var(--surface-3);
  color: var(--danger);
}
</style>
