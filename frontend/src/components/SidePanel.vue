<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { state } from "../live.js";
import { api } from "../api.js";
import { PREDICT_INTERVALS, formatUtc8, formatUtc8Clock } from "../indicators.js";
import { formatLevels } from "../predictLevels.js";
import { buildNarrative } from "../predictNarrative.js";

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

const DIR_TEXT = { up: "多", down: "空", flat: "观望" };
const DIR_CLASS = { up: "up", down: "down", flat: "" };
const STATE_TEXT = {
  ok: "调试",
  stale: "数据过期",
  no_model: "未接入",
  no_bars: "无 K 线",
  gap: "有缺口",
  no_features: "特征不全",
  no_data: "无数据",
  bad_model: "模型异常",
};

// Top 3 SHAP contributions for the most likely class. Contributions are raw-score
// units: "+" pushes that class up, "−" pushes it down. They are not probabilities.
const EXPLAIN_TEXT = { up: "多", down: "空", flat: "观望" };
function explainBlock(p) {
  const feats = Array.isArray(p.top_features) ? p.top_features.slice(0, 3) : [];
  if (!feats.length) return null;
  return {
    head: `推动「${EXPLAIN_TEXT[p.explain_for] || p.explain_for || "—"}」的前三特征`,
    items: feats.map((f) => {
      const c = Number(f.contribution);
      const sign = c > 0 ? "+" : c < 0 ? "−" : "";
      return {
        key: f.name,
        text: `${f.name} ${sign}${Math.abs(c).toFixed(3)}`,
        value: f.value == null ? "—" : Number(f.value).toPrecision(4),
      };
    }),
  };
}

const signalRows = computed(() =>
  PREDICT_INTERVALS.map((id) => {
    const p = predictions.value[id];
    if (!p) {
      return {
        interval: id,
        dirText: "—",
        dirClass: "",
        probText: "—",
        stateText: predictError.value ? "预测服务不可用" : "加载中",
        tip: predictError.value,
        levels: null,
        top: null,
      };
    }
    const dir = p.status === "ok" ? p.direction : null;
    const probKey = dir ?? (p.status === "ok" ? "flat" : null);
    const prob = probKey && p.probs ? p.probs[probKey] : null;
    const tip = p.model_id
      ? `模型 ${p.model_id}；训练于 ${p.trained_at_utc8}（UTC+8）；样本外准确率 ${
          p.oos_accuracy == null ? "—" : (p.oos_accuracy * 100).toFixed(1) + "%"
        }。${p.message || ""}`
      : p.message || "";
    return {
      interval: id,
      dirText: dir ? DIR_TEXT[dir] : "—",
      dirClass: dir ? DIR_CLASS[dir] : "",
      probText: prob == null ? "—" : `${(prob * 100).toFixed(1)}%`,
      stateText: STATE_TEXT[p.status] || p.status || "—",
      tip,
      levels: formatLevels(p),
      top: explainBlock(p),
    };
  }),
);

// The default read-out: a short Chinese explanation in a fixed order (结论 → 现价/关键位 →
// 分周期 → 资金面 → 点位 → 高杠杆提醒 → 调试勿实盘). Feature contributions stay folded.
const narrative = computed(() =>
  buildNarrative({
    items: PREDICT_INTERVALS.map((id) => predictions.value[id]),
    price: price.value === "—" ? null : price.value,
  }),
);

const signalAsof = computed(() => {
  const first = Object.values(predictions.value).find((p) => p.asof_utc8);
  return first ? first.asof_utc8 : null;
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
      <h2>信号</h2>
      <p class="debug-banner">调试预测，样本外弱，勿实盘</p>
      <p class="fine">五个周期的调试模型都显示出来，效果好坏都在这里。没有模型的周期才标「未接入」。</p>
      <p v-if="predictError" class="fine down">预测接口出错：{{ predictError }}</p>
      <ol v-if="!predictError" class="narrative">
        <li v-for="line in narrative.lines" :key="line.key">
          <strong>{{ line.label }}</strong>
          <span>{{ line.text }}</span>
        </li>
      </ol>
      <table class="signals">
        <thead>
          <tr>
            <th>周期</th>
            <th>方向</th>
            <th>概率</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody>
          <template v-for="row in signalRows" :key="row.interval">
            <tr :title="row.tip">
              <td>{{ row.interval }}</td>
              <td :class="row.dirClass">{{ row.dirText }}</td>
              <td>{{ row.probText }}</td>
              <td>{{ row.stateText }}</td>
            </tr>
            <tr v-if="row.levels || row.top" class="explain-row">
              <td colspan="4">
                <details class="explain-fold">
                  <summary>点位与特征（展开）</summary>
                  <div v-if="row.levels" class="levels-line" :class="{ muted: row.levels.muted }">{{ row.levels.text }}</div>
                  <template v-if="row.top">
                    <div class="explain-head">{{ row.top.head }}</div>
                    <ul class="explain-list">
                      <li v-for="f in row.top.items" :key="f.key">
                        <span>{{ f.text }}</span>
                        <span class="explain-value">= {{ f.value }}</span>
                      </li>
                    </ul>
                  </template>
                </details>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
      <p v-if="signalAsof" class="fine">截至 {{ signalAsof }}（UTC+8），取已收盘的最新一根。</p>
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
