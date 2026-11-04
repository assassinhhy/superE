<script setup>
import { computed, onMounted, ref } from "vue";
import { api } from "../api.js";
import { state } from "../live.js";
import { formatUtc8 } from "../indicators.js";

const now = ref(Date.now());
const polled = ref({});
setInterval(() => {
  now.value = Date.now();
}, 1000);

onMounted(async () => {
  try {
    polled.value = await api.status();
  } catch {
    polled.value = {};
  }
});

const status = computed(() => ({ ...polled.value, ...state.status }));

function lamp(name) {
  const value = status.value.sockets?.[name];
  if (value === "connected") return { text: "已连接", cls: "ok" };
  if (value === "overlap") return { text: "切换中", cls: "warn" };
  if (value === "preview") return { text: "预览", cls: "warn" };
  return { text: "未连接", cls: "bad" };
}

const age = computed(() => {
  const sockets = status.value.sockets || {};
  const ts = Number(status.value.ts || 0);
  const last = Math.max(
    Number(sockets.last_market_ms || 0),
    Number(sockets.last_public_ms || 0),
    Number(sockets.last_spot_ms || 0),
  );
  if (!ts || !last) return "—";
  const seconds = Math.max(0, Math.round((ts - last) / 1000));
  return `${seconds} 秒前`;
});

const weight = computed(() => {
  const used = status.value.self_weight_minute;
  const budget = status.value.weight_budget;
  if (used == null) return "—";
  return budget ? `${used} / ${budget}` : String(used);
});

const clock = computed(() => formatUtc8(now.value));
</script>

<template>
  <footer class="status">
    <span>UTC+8 {{ clock }}</span>
    <span :class="state.ws ? 'ok' : 'bad'">界面 {{ state.ws ? "已连接" : "未连接" }}</span>
    <span>代理 {{ status.proxy || "—" }}</span>
    <span :class="lamp('market').cls">合约行情 {{ lamp("market").text }}</span>
    <span :class="lamp('public').cls">盘口 {{ lamp("public").text }}</span>
    <span :class="lamp('spot').cls">现货 {{ lamp("spot").text }}</span>
    <span>最近消息 {{ age }}</span>
    <span :class="status.rest_halted ? 'bad' : ''">REST 权重 {{ weight }}</span>
    <span v-if="status.header_weight != null" class="muted">响应头 {{ status.header_weight }}</span>
    <span :class="status.clock_warn ? 'warn' : ''">
      时钟 {{ status.clock_offset_ms == null ? "—" : status.clock_offset_ms + " ms" }}
    </span>
    <span v-if="status.disk_free_gb != null" :class="status.disk_level === 'ok' ? '' : 'warn'">
      磁盘 {{ status.disk_free_gb }} GB
    </span>
    <span v-if="state.version" class="muted">v{{ state.version }}</span>
    <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">KLineChart · TradingView Lightweight Charts</a>
  </footer>
</template>
