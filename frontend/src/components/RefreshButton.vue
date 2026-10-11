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

defineProps<{ label?: string }>()
const emit = defineEmits<{ (event: 'rotate'): void }>()
</script>

<template>
  <button
    type="button"
    class="refresh"
    :aria-label="label ?? '换一换'"
    :title="label ?? '换一换'"
    @click="emit('rotate')"
  >
    <span class="glyph">
      <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
        <path
          d="M20 11a8 8 0 1 0-2.3 6.3"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        />
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
.glyph {
  display: grid;
  place-items: center;
  transition: transform 0.4s ease;
}

.refresh:hover .glyph {
  /* A half turn reads as "refreshing" without an animation library. */
  transform: rotate(180deg);
}

.refresh:active {
  transform: scale(0.92);
}

.refresh:focus-visible {
  outline: 2px solid var(--accent-border);
  outline-offset: 2px;
}

/* Tooltip: shown on hover and on keyboard focus. */
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

.refresh:hover .tip,
.refresh:focus-visible .tip {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}
</style>
