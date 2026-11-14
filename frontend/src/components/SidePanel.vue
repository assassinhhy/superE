<script setup>
import { computed, ref } from "vue";
import { formatUtc8 } from "../indicators.js";
import { state } from "../live.js";
import { setSoundEnabled, soundEnabled, systemBridgeReady } from "../notify.js";
import { desk } from "../signalDesk.js";
import { SIGNAL_GROUPS, SIGNAL_INTERVALS, filterSignals } from "../signals.js";

const props = defineProps({
  market: { type: String, required: true },
  divergence: { type: Boolean, required: true },
});
const emit = defineEmits(["divergence"]);

const interval = ref("全部");
const group = ref("全部");
const sound = ref(soundEnabled());
const bridge = ref(systemBridgeReady());

const price = computed(() => {
  const forming = state.forming.find((bar) => bar.market === props.market && bar.interval === "15m");
  return forming?.c || (props.market === "um" ? state.ticker?.close : null) || "—";
});

const rows = computed(() => filterSignals(desk.items, interval.value, group.value));

function toggleSound() {
  sound.value = !sound.value;
  setSoundEnabled(sound.value);
}

function toggleDivergence() {
  emit("divergence", !props.divergence);
}
</script>

<template>
  <div class="side-body">
    <section>
      <h2>实时</h2>
      <p class="px">{{ price }}</p>
      <p class="muted">信号按 ETHUSDT 永续的收盘 K 线计算，不跟未收盘的那根。</p>
    </section>

    <section class="signal-head">
      <h2>信号通知</h2>
      <div class="signal-switches">
        <button type="button" :class="{ on: sound }" @click="toggleSound">
          {{ sound ? "提示音开" : "提示音关" }}
        </button>
        <button type="button" :class="{ on: divergence }" @click="toggleDivergence">
          {{ divergence ? "RSI 背离开" : "RSI 背离关" }}
        </button>
      </div>
      <p class="fine">
        {{ bridge ? "系统通知走桌面窗口。" : "当前没有桌面通知桥，只用页面提示和提示音。Windows 上重启桌面窗口后会多一条气球通知。" }}
      </p>
      <p v-if="desk.error" class="fine down">{{ desk.error }}</p>
      <div class="news-filters">
        <button
          v-for="name in ['全部', ...SIGNAL_INTERVALS]"
          :key="name"
          type="button"
          :class="{ on: interval === name }"
          @click="interval = name"
        >
          {{ name }}
        </button>
      </div>
      <div class="news-filters">
        <button
          v-for="name in SIGNAL_GROUPS"
          :key="name"
          type="button"
          :class="{ on: group === name }"
          @click="group = name"
        >
          {{ name }}
        </button>
      </div>
      <p v-if="!rows.length" class="muted">这几个周期还没有触发信号。</p>
      <ul v-else class="signal-list">
        <li v-for="item in rows" :key="item.id" :class="item.tone">
          <div class="signal-top">
            <strong>{{ item.label }}</strong>
            <span>{{ item.interval }}</span>
          </div>
          <p>{{ formatUtc8(item.timeMs) }} · {{ Number(item.price).toFixed(2) }}</p>
        </li>
      </ul>
    </section>
  </div>
</template>
