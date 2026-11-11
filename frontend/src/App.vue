<script setup>
import { computed, onMounted, ref, watch } from "vue";
import ChartPane from "./components/ChartPane.vue";
import SettingsDrawer from "./components/SettingsDrawer.vue";
import SidePanel from "./components/SidePanel.vue";
import StatusBar from "./components/StatusBar.vue";
import TrainView from "./components/TrainView.vue";
import { INTERVALS, formatUtc8, isBackfilling } from "./indicators.js";
import { startLive, state } from "./live.js";

const tab = ref("predict");
const market = ref("um");
const interval = ref("15m");
const settings = ref(false);
const clock = ref(formatUtc8(Date.now()));

const filling = computed(() => isBackfilling(state.events, Date.now()));

function setSpot() {
  market.value = "spot";
  interval.value = "1m";
}

watch(market, (value) => {
  if (value === "spot") interval.value = "1m";
});

onMounted(() => {
  startLive();
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
        <button type="button" :class="{ on: tab === 'predict' }" @click="tab = 'predict'">预测</button>
        <button type="button" :class="{ on: tab === 'train' }" @click="tab = 'train'">训练</button>
        <button type="button" @click="settings = true">设置</button>
      </nav>
    </header>

    <p v-if="state.preview" class="banner">当前是预览数据，没有连接币安。K 线、盘口和成交都是合成的。</p>
    <p v-if="filling" class="banner warn">补数据中</p>
    <p v-if="state.error" class="banner warn">{{ state.error }}</p>

    <main v-if="tab === 'predict'" class="predict">
      <ChartPane :market="market" :interval="interval" />
      <SidePanel :market="market" :interval="interval" />
    </main>
    <TrainView v-else />
    <StatusBar />
    <SettingsDrawer v-if="settings" @close="settings = false" />
  </div>
</template>
