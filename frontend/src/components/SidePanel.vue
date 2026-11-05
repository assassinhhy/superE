<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { state } from "../live.js";
import { api } from "../api.js";
import { PREDICT_INTERVALS, formatUtc8, formatUtc8Clock } from "../indicators.js";
import { forecastRows } from "../forecastRows.js";

const props = defineProps({
  market: { type: String, required: true },
  interval: { type: String, required: true },
});

const formingBar = computed(() =>
  state.forming.find((bar) => bar.market === props.market && bar.interval === props.interval),
);

const price = computed(() => formingBar.value?.c || (props.market === "um" ? state.ticker?.close : null) || "—");

const change = computed(() => {
  if (props.market !== "um") return null;
  const open = Number(state.ticker?.open);
  const close = Number(state.ticker?.close);
  if (!open || !Number.isFinite(close)) return null;
  return (close - open) / open;
});

// Debug predictions (调试，勿实盘). Polled once a minute; a 5m bar closes every 5 minutes.
const PREDICT_POLL_MS = 60_000;
const predictions = ref({});
const predictError = ref("");
let predictTimer = null;

async function loadPredictions() {
  try {
    const payload = await api.predict();
    predictions.value = Object.fromEntries((payload.items || []).map((item) => [item.interval, item]));
    predictError.value = "";
  } catch (err) {
    // Never keep the last good rows on screen: they would read as current.
    predictions.value = {};
    predictError.value = err?.message || String(err);
  }
}

onMounted(() => {
  loadPredictions();
  predictTimer = setInterval(loadPredictions, PREDICT_POLL_MS);
});
onUnmounted(() => clearInterval(predictTimer));

// One row per interval (5m → 4h). Missing, non-ok, or null forecasts show as 暂无预测.
const forecast = computed(() => {
  if (predictError.value) return [];
  return forecastRows(Object.values(predictions.value), PREDICT_INTERVALS);
});

const forecastAsof = computed(() => {
  const row = forecast.value.find((r) => r.detail?.asof);
  return row ? row.detail.asof : null;
});

const kinds = {
  volume: "放量",
  price_jump: "跳价",
  liquidation: "强平",
  oi: "持仓量",
  funding: "资金费率",
};

</script>

<template>
  <aside class="side">
    <section>
      <h2>实时</h2>
      <p class="px" :class="change == null ? '' : change >= 0 ? 'up' : 'down'">{{ price }}</p>
      <p v-if="market === 'um'" class="muted">
        24h
        <template v-if="change != null">{{ (change * 100).toFixed(2) }}%</template>
        <template v-else>—</template>
        · 成交额 {{ state.ticker?.quote_volume || "—" }}
      </p>
      <p v-else class="muted">现货只录 1 分钟，用来对照基差。24h 行情是合约的。</p>
    </section>

    <section>
      <h2>最新成交</h2>
      <p v-if="!state.trades.length" class="muted">还没有成交。</p>
      <ul v-else class="tape">
        <li
          v-for="trade in state.trades.slice(0, 18)"
          :key="trade.agg_id + '-' + trade.ts"
          :class="trade.is_buyer_maker ? 'down' : 'up'"
        >
          <span>{{ trade.price }}</span>
          <span>{{ trade.qty }}</span>
          <span>{{ formatUtc8Clock(trade.ts) }}</span>
        </li>
      </ul>
    </section>

    <section v-if="market === 'um'">
      <h2>预测区间</h2>
      <p class="debug-banner">调试预测，样本外弱，勿实盘</p>
      <p class="fine">价格区间预测：主行为中位数，括号内为 25%–75% 区间。不是方向，也不是交易点位。</p>
      <p v-if="predictError" class="fine down">预测接口出错：{{ predictError }}</p>
      <ul v-else class="forecast">
        <li v-for="row in forecast" :key="row.interval" class="forecast-row">
          <template v-if="row.hasForecast">
            <div class="forecast-head">
              <strong>{{ row.headline }}</strong>
              <span class="forecast-status" :class="{ ok: row.status === '达标' }">{{ row.status }}</span>
            </div>
            <p class="forecast-explain">{{ row.explain }}</p>
            <details class="explain-fold">
              <summary>80%区间与宽度（展开）</summary>
              <div class="explain-line">{{ row.detail.interval80 }}</div>
              <div class="explain-line">{{ row.detail.width50 }}</div>
              <div class="explain-line">{{ row.detail.width80 }}</div>
            </details>
          </template>
          <template v-else>
            <div class="forecast-head">
              <strong>{{ row.headline }}</strong>
            </div>
            <p class="fine">{{ row.message }}</p>
          </template>
        </li>
      </ul>
      <p v-if="forecastAsof" class="fine">截至 {{ forecastAsof }}（UTC+8），取已收盘的最新一根。</p>
    </section>

    <section>
      <h2>异动</h2>
      <p v-if="!state.anomalies.length" class="muted">还没有异动。消息源第二阶段才采集，这里只显示市场异动。</p>
      <ul v-else class="news">
        <li v-for="item in state.anomalies" :key="item.id">
          <strong>{{ kinds[item.kind] || item.kind }}</strong>
          <span>{{ item.ts_utc8 || formatUtc8(item.ts) }}</span>
          <em>{{ item.value }}</em>
        </li>
      </ul>
    </section>
  </aside>
</template>
