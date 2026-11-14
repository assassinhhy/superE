<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";
import BoardPanel from "./components/BoardPanel.vue";
import ChartPane from "./components/ChartPane.vue";
import NewsView from "./components/NewsView.vue";
import SettingsDrawer from "./components/SettingsDrawer.vue";
import SidePanel from "./components/SidePanel.vue";
import StatusBar from "./components/StatusBar.vue";
import Toasts from "./components/Toasts.vue";
import TrainView from "./components/TrainView.vue";
import { startBoardDesk } from "./board.js";
import { INTERVALS, formatUtc8, isBackfilling } from "./indicators.js";
import { startLive, state } from "./live.js";
import { bootNotify } from "./notify.js";
import { startSignalDesk } from "./signalDesk.js";

const tab = ref("market");
const side = ref("board");
const market = ref("um");
const interval = ref("15m");
const settings = ref(false);
const clock = ref(formatUtc8(Date.now()));
const divergence = ref(false);
const focusPlan = ref("");
const board = reactive({ view: null, offline: false, error: "" });

try {
  divergence.value = localStorage.getItem("supere.divergence") === "1";
} catch {
  divergence.value = false;
}

const filling = computed(() => isBackfilling(state.events, Date.now()));

function setSpot() {
  market.value = "spot";
  interval.value = "1m";
}

function setDivergence(on) {
  divergence.value = Boolean(on);
  try {
    localStorage.setItem("supere.divergence", divergence.value ? "1" : "0");
  } catch {
    /* ignore */
  }
}

watch(market, (value) => {
  if (value === "spot") interval.value = "1m";
});

onMounted(() => {
  bootNotify();
  startLive();
  startSignalDesk(() => divergence.value);
  startBoardDesk(board);
  setInterval(() => {
    clock.value = formatUtc8(Date.now());
  }, 1000);
});
</script>

<template>
  <div class="app">
    <header class="top">
      <div class="brand">superE <small>ETHUSDT</small></div>
      <div class="markets">
        <button type="button" :class="{ on: market === 'um' }" @click="market = 'um'">合约</button>
        <button type="button" :class="{ on: market === 'spot' }" @click="setSpot">现货参考</button>
      </div>
      <div class="intervals">
        <button
          v-for="item in INTERVALS"
          :key="item.id"
          type="button"
          :disabled="market === 'spot' && item.id !== '1m'"
          :class="{ on: interval === item.id }"
          @click="interval = item.id"
        >
          {{ item.id }}
        </button>
      </div>
      <div class="clock">{{ clock }} · UTC+8</div>
      <nav>
        <button type="button" :class="{ on: tab === 'market' }" @click="tab = 'market'">行情</button>
        <button type="button" :class="{ on: tab === 'news' }" @click="tab = 'news'">消息</button>
        <button type="button" :class="{ on: tab === 'train' }" @click="tab = 'train'">训练</button>
        <button type="button" @click="settings = true">设置</button>
      </nav>
    </header>

    <p v-if="state.preview" class="banner">当前是预览数据，没有连接币安。K 线和成交都是合成的。</p>
    <p v-if="filling" class="banner warn">补数据中</p>
    <p v-if="state.error" class="banner warn">{{ state.error }}</p>

    <main v-if="tab === 'market'" class="desk">
      <ChartPane :market="market" :interval="interval" :board="board.view" :focus-plan="focusPlan" />
      <aside class="side">
        <div class="side-tabs">
          <button type="button" :class="{ on: side === 'board' }" @click="side = 'board'">盘面</button>
          <button type="button" :class="{ on: side === 'signal' }" @click="side = 'signal'">信号</button>
          <span class="side-clock">{{ clock.slice(11, 19) }} UTC+8</span>
        </div>
        <BoardPanel
          v-show="side === 'board'"
          :view="board.view"
          :offline="board.offline"
          :error="board.error"
          @focus="focusPlan = $event"
        />
        <SidePanel
          v-show="side === 'signal'"
          :market="market"
          :divergence="divergence"
          @divergence="setDivergence"
        />
      </aside>
    </main>
    <NewsView v-else-if="tab === 'news'" />
    <TrainView v-else />
    <Toasts />
    <StatusBar />
    <SettingsDrawer v-if="settings" @close="settings = false" />
  </div>
</template>
