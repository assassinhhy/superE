<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import { formatUtc8 } from "../indicators.js";

defineProps({
  short: { type: Boolean, default: false },
});

const text = ref(formatUtc8(Date.now()));
let timer = 0;

onMounted(() => {
  text.value = formatUtc8(Date.now());
  timer = setInterval(() => {
    text.value = formatUtc8(Date.now());
  }, 1000);
});

onBeforeUnmount(() => {
  if (timer) clearInterval(timer);
  timer = 0;
});
</script>

<template>
  <span class="clock-text">{{ short ? text.slice(11, 19) : text }}</span>
</template>
