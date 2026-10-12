<script setup lang="ts">
/**
 * Circular "换一换" button: a two-arrow glyph that spins on hover.
 *
 * Extracted so the empty state and the per-turn follow-up rows share one
 * implementation — duplicating it would let the two drift apart.
 *
 * The spin is applied to the glyph, never to the button: the tooltip is a child of
 * the button, so rotating the button also rotated the label.
 */
// 中文说明
//
// 一个圆形「换一换」按钮：两个箭头组成的图标，hover 时转半圈。
//
// 为什么抽成组件（原注释）：空状态的推荐问题区与每轮回答下方的追问区都用它，
// 复刻两份实现早晚会走样（尺寸、动效、无障碍属性不一致）。本组件也是被
// 「依赖最少的叶子组件」——没有 props 依赖，只发一个 rotate 事件。
//
// 关键细节（原注释）：旋转动画加在**图标**上而不是按钮上。因为气泡提示是按钮的
// 子元素，旋转按钮会把「换一换」这几个字也一起转过去。

/** 中文说明：按钮文案（同时用作 title 与 aria-label）；不传则用「换一换」。 */
defineProps<{ label?: string }>()
/** 中文说明：点击时通知父组件轮换到下一页建议（具体轮换逻辑在 lib/suggestions.ts）。 */
const emit = defineEmits<{ (event: 'rotate'): void }>()
</script>

<template>
  <!-- type="button" 很关键：本组件也会出现在表单（输入框）附近，
       不写就会被当作 submit 按钮，点击时意外提交表单。 -->
  <button
    type="button"
    class="refresh"
    :aria-label="label ?? '换一换'"
    :title="label ?? '换一换'"
    @click="emit('rotate')"
  >
    <!-- 只有 glyph 这一层参与旋转（见上方注释与样式）。 -->
    <span class="glyph">
      <!-- aria-hidden：图标对读屏软件无意义，语义由 aria-label 承担。 -->
      <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
        <!-- 弧线：一个接近整圈的圆弧（8 单位半径），右侧留缺口。 -->
        <path
          d="M20 11a8 8 0 1 0-2.3 6.3"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        />
        <!-- 箭头：与弧线起点相连的折线，构成「循环/刷新」的双箭头形状。 -->
        <path
          d="M20 4.5V11h-6.2"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </span>
    <span class="tip">{{ label ?? '换一换' }}</span>
  </button>
</template>

<style scoped>
/* position: relative 是为了让 .tip 以按钮为定位基准。 */
.refresh {
  position: relative;
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 50%;
  background: var(--surface-2);
  color: var(--text-muted);
  cursor: pointer;
  /* flex: none 防止在 flex 行里被长文本压扁成椭圆。 */
  flex: none;
}

.refresh:hover {
  color: var(--accent);
  border-color: var(--accent-border);
}

/*
 * The rotation belongs to the ICON, never the button.
 *
 * The tooltip is a child of the button, so rotating the button also rotated the
 * "换一换" label. Only the glyph may spin.
 */
/* 中文说明：下面两条是同一个 bug 的修复记录——旋转必须落在图标上。
   若把它写在 .refresh:hover 上，作为子元素的气泡文字也会跟着转 180°。 */
.glyph {
  display: grid;
  place-items: center;
  transition: transform 0.4s ease;
}

.refresh:hover .glyph {
  /* A half turn reads as "refreshing" without an animation library. */
  /* 中文说明：半圈（180°）在视觉上就足以表达「换一批」，
     不需要引入动画库，也不需要真的转满一圈。 */
  transform: rotate(180deg);
}

/* 按下时轻微缩小，给一个「已响应」的触感反馈。 */
.refresh:active {
  transform: scale(0.92);
}

/* :focus-visible 只在键盘导航时出现焦点环，鼠标点击不出现。 */
.refresh:focus-visible {
  outline: 2px solid var(--accent-border);
  outline-offset: 2px;
}

/* Tooltip: shown on hover and on keyboard focus. */
/* 中文说明：气泡提示绝对定位在按钮上方，默认透明。
   用 opacity 而不是 display 切换，才能有过渡动画；pointer-events: none
   避免这个透明层挡住下面元素的鼠标事件。 */
.tip {
  position: absolute;
  bottom: calc(100% + 7px);
  left: 50%;
  transform: translateX(-50%) translateY(2px);
  padding: 3px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-3);
  color: var(--text);
  font-size: 11px;
  line-height: 1.4;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s, transform 0.15s;
}

/* hover 与键盘焦点都会显示气泡（无障碍：键盘用户也能看到提示）。
   translateY 从 2px 到 0，形成轻微上浮的效果。 */
.refresh:hover .tip,
.refresh:focus-visible .tip {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}
</style>
