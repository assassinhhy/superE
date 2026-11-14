<script setup>
import { onMounted, onUnmounted, ref } from "vue";
import { subscribeNotices } from "../notify.js";

const items = ref([]);
let stop = () => {};
const timers = new Map();

onMounted(() => {
  stop = subscribeNotices((notice) => {
    items.value = [{ ...notice, key: `${notice.id}-${Date.now()}` }, ...items.value].slice(0, 3);
    const key = items.value[0].key;
    const timer = setTimeout(() => {
      items.value = items.value.filter((item) => item.key !== key);
      timers.delete(key);
    }, 6000);
    timers.set(key, timer);
  });
});

onUnmounted(() => {
  stop();
  for (const timer of timers.values()) clearTimeout(timer);
});

function dismiss(key) {
  items.value = items.value.filter((item) => item.key !== key);
}
</script>

<template>
  <div class="toasts" aria-live="polite">
    <button
      v-for="item in items"
      :key="item.key"
      type="button"
      class="toast"
      :class="item.tone || 'warn'"
      @click="dismiss(item.key)"
    >
      <strong>{{ item.title }}</strong>
      <span>{{ item.body }}</span>
    </button>
  </div>
</template>
