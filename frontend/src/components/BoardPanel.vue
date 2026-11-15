<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { estimatePosition } from "../boardCalc.js";
import {
  boardFreshLine,
  boardStaleSeconds,
  bumpLoss,
  loadCalc,
  loadLoss,
  loadPlanHistory,
  lossState,
  planHistoryExportURL,
  planOutcomeLine,
  planValidity,
  saveCalc,
  saveLoss,
} from "../board.js";

const props = defineProps({
  view: { type: Object, default: null },
  offline: { type: Boolean, default: false },
  error: { type: String, default: "" },
  fetchedAt: { default: null },
  since: { default: null },
});
const emit = defineEmits(["focus"]);

const calcOpen = ref(false);
const historyOpen = ref(false);
const history = ref([]);
const historyError = ref("");
const calc = ref(loadCalc());
const loss = ref(loadLoss());
const now = ref(Date.now());
const clock = ref(Date.now());
let clockTimer = null;

onMounted(() => {
  clockTimer = setInterval(() => {
    clock.value = Date.now();
  }, 1000);
});
onUnmounted(() => {
  if (clockTimer) clearInterval(clockTimer);
});

watch(calc, (value) => saveCalc(value), { deep: true });
watch(loss, (value) => saveLoss(value), { deep: true });

const losses = computed(() => lossState(loss.value, now.value).losses);

function changeLoss(delta) {
  now.value = Date.now();
  loss.value = bumpLoss(loss.value, delta, now.value);
}

function money(value) {
  if (value == null || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return value.toFixed(digits);
}

function px(value) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(2);
}

const exportHref = computed(() => planHistoryExportURL());

async function refreshHistory() {
  try {
    const payload = await loadPlanHistory();
    history.value = payload.plans || [];
    historyError.value = "";
  } catch {
    historyError.value = "历史计划未取到";
  }
}

function toggleHistory() {
  historyOpen.value = !historyOpen.value;
  if (historyOpen.value) refreshHistory();
}

watch(
  () => props.view?.basis?.boundary_ms,
  () => {
    if (historyOpen.value) refreshHistory();
  },
);

function planCalc(plan) {
  if (!plan || plan.stop == null) return null;
  const quoted = Number(plan.entry);
  const entry = Number.isFinite(quoted) ? quoted : (Number(plan.entry_low) + Number(plan.entry_high)) / 2;
  return estimatePosition({
    entry,
    stop: Number(plan.stop),
    side: plan.side,
    targets: plan.targets || [],
    atr: props.view?.atr?.["15m"],
    capital: Number(calc.value.capital),
    maxLossU: Number(calc.value.maxLossU),
    maxLossPct: Number(calc.value.maxLossPct) || 0,
    leverage: Number(calc.value.leverage),
    sizeRatio: plan.size_ratio || 1,
  });
}

const priceText = computed(() => px(props.view?.price?.value));
const freshLine = computed(() => boardFreshLine(props.view, Number.isFinite(props.fetchedAt) ? props.fetchedAt : null));
const staleSeconds = computed(() => {
  const anchor = Number.isFinite(props.fetchedAt) ? props.fetchedAt : props.since;
  return boardStaleSeconds(Number.isFinite(anchor) ? anchor : null, clock.value);
});
const oiAmount = computed(() => {
  const value = Number(props.view?.oi?.value);
  if (!Number.isFinite(value)) return props.view?.oi?.label || "未获取";
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
});
const liveMeta = computed(() => {
  const parts = [];
  const priceLabel = props.view?.price?.label;
  if (priceLabel && priceLabel !== "未获取") parts.push(`价 ${priceLabel}`);
  const funding = props.view?.funding;
  if (typeof funding?.utc8 === "string" && funding.utc8.length >= 19) parts.push(`费率 ${funding.utc8.slice(11, 19)}`);
  else if (funding?.source) parts.push(funding.source);
  const oi = props.view?.oi;
  if (oi?.label) parts.push(`OI ${oi.label}`);
  if (oi?.source) parts.push(oi.source);
  return parts.join(" · ");
});
const changeText = computed(() => {
  const ticker = props.view?.ticker;
  if (!ticker || ticker.status !== "ok" || ticker.value?.change_pct == null) return "";
  const value = ticker.value.change_pct;
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
});
const tone = computed(() => {
  const direction = props.view?.trend?.direction;
  if (direction === "空头") return "down";
  if (direction === "多头") return "up";
  return "";
});

function statusClass(status) {
  if (status === "可进场") return "ready";
  if (status === "已作废" || status === "已触发") return "down";
  return "wait";
}

function levelText(levels) {
  return (levels || [])
    .map((level) => `${px(level.price)}(${level.label || level.sources?.[0] || ""}·测${level.touches || 0})`)
    .join(" ");
}
</script>

<template>
  <div class="board">
    <p v-if="staleSeconds != null" class="board-stale">数据已过期 {{ staleSeconds }} 秒</p>
    <p v-if="offline" class="board-alert">盘面服务未启动</p>
    <p v-else-if="error" class="board-alert">{{ error }}</p>
    <template v-if="view">
      <p v-if="view.crosscheck?.status === 'anomaly'" class="board-alert">{{ view.crosscheck.message }}</p>
      <p v-else-if="view.crosscheck?.status === 'missing'" class="fine">交叉核对未获取</p>
      <p class="px" :class="tone">
        ETHUSDT {{ priceText }}
        <small v-if="changeText" :class="changeText.startsWith('-') ? 'down' : 'up'">{{ changeText }}</small>
      </p>
      <p v-if="view.ticker?.status === 'ok'" class="fine">
        高 {{ px(view.ticker.value.high) }} 低 {{ px(view.ticker.value.low) }}
        <span class="board-meta">{{ view.ticker.utc8 }} {{ view.ticker.source }}</span>
      </p>
      <p v-if="view.limit?.active" class="board-alert">{{ view.limit.message }}</p>

      <h2>结论</h2>
      <p class="board-conclusion" :class="tone">{{ view.conclusion?.text }}</p>
      <p class="fine">{{ view.trend?.text }} · 位置：{{ view.position?.text || "未获取" }}</p>

      <h2 class="board-head">走势要点 <span class="board-fresh">{{ freshLine }}</span></h2>
      <ul class="board-list">
        <li v-for="(item, index) in view.bullets || []" :key="index">
          {{ item.text }}
          <span class="board-meta">{{ item.source }}</span>
        </li>
        <li v-if="!(view.bullets || []).length" class="wait">这一截没有新的要点。</li>
      </ul>

      <h2 class="board-head">合约情绪 <span class="board-fresh">{{ freshLine }}</span></h2>
      <p v-if="view.sentiment?.status !== 'ok'" class="wait">未获取 <span class="board-meta">{{ view.sentiment?.label || view.metrics?.label }}</span></p>
      <template v-else>
        <p>{{ view.sentiment.text }}</p>
        <p class="fine">全市场多空比 {{ view.sentiment.global_lsr ?? "—" }} · 大户持仓 {{ view.sentiment.top_pos_lsr ?? "—" }}</p>
        <p class="fine">
          主动买卖比 {{ view.sentiment.taker ? view.sentiment.taker.ratio.toFixed(2) : "—" }}
          {{ view.sentiment.taker?.pressure || "" }}
          <span class="board-meta">{{ view.sentiment.taker?.utc8 }} {{ view.sentiment.taker?.source }}</span>
        </p>
      </template>
      <p class="fine">
        实时 最新价 {{ priceText }}
        · {{ view.funding?.text || "未获取" }}
        · OI {{ oiAmount }}
        <span v-if="liveMeta" class="board-meta">{{ liveMeta }}</span>
      </p>

      <h2>技术</h2>
      <table class="board-tech">
        <thead>
          <tr><th>周期</th><th>RSI</th><th>MACD柱</th><th>价 vs EMA20/50/200</th></tr>
        </thead>
        <tbody>
          <tr v-for="row in view.technical?.rows || []" :key="row.frame">
            <td>{{ row.frame }}</td>
            <td v-if="row.status !== 'ok'" colspan="3" class="wait">未获取</td>
            <template v-else>
              <td>{{ row.rsi == null ? "—" : row.rsi.toFixed(0) }}</td>
              <td>{{ row.hist == null ? "—" : row.hist.toFixed(1) }} {{ row.hist_state }}</td>
              <td>{{ row.vs_ema }} <span v-if="row.warn" class="ready">{{ row.warn }}</span></td>
            </template>
          </tr>
        </tbody>
      </table>
      <p class="fine">阻力 {{ levelText(view.levels?.resistance) || "—" }}</p>
      <p class="fine">支撑 {{ levelText(view.levels?.support) || "—" }}</p>

      <h2 class="board-head">情景计划 <span class="board-fresh">{{ freshLine }}</span></h2>
      <p class="fine">{{ view.plans_meta?.validity || ((view.plans_meta?.generated_utc8 || "—") + " 生成，" + view.plans_meta?.valid_bars + " 根 15m 内有效") }}</p>
      <p v-if="view.plans_meta?.paused" class="board-alert">{{ view.plans_meta.pause_reason || "情景计划暂停" }}</p>
      <article v-for="(plan, index) in view.plans || []" :key="plan.id" class="board-plan">
        <p>
          <strong>{{ index + 1 }}. {{ plan.title }}</strong>
          <span :class="statusClass(plan.status)">{{ plan.status }}</span>
          <button v-if="plan.id !== 'flip'" type="button" @click="emit('focus', plan.id)">图上看</button>
        </p>
        <p v-if="plan.id !== 'flip'" class="board-entry">
          进场 {{ px(plan.entry) }}
          <span class="fine">
            止损 {{ px(plan.stop) }}
            <span v-for="(target, targetIndex) in plan.targets" :key="target"> T{{ targetIndex + 1 }} {{ px(target) }}</span>
          </span>
        </p>
        <p v-if="plan.id !== 'flip'" class="fine">区间 {{ px(plan.entry_low) }}–{{ px(plan.entry_high) }}</p>
        <p class="fine">{{ planValidity(plan) }}</p>
        <p v-if="plan.note" class="fine">{{ plan.note }}</p>
        <p v-if="planCalc(plan)" class="fine" :class="{ bad: planCalc(plan).warnings.length }">
          名义 {{ money(planCalc(plan).notional) }} U · 保证金 {{ money(planCalc(plan).margin) }} U
          · 强平≈{{ px(planCalc(plan).liq) }}（估算）
          <span v-for="(leg, legIndex) in planCalc(plan).legs" :key="leg.target">
            T{{ legIndex + 1 }} {{ leg.pnlU >= 0 ? "+" : "" }}{{ money(leg.pnlU) }}U / 盈亏比 {{ leg.rr.toFixed(1) }}
          </span>
          <span v-for="note in planCalc(plan).notes" :key="note" class="hot">{{ note }}</span>
          <span v-for="warning in planCalc(plan).warnings" :key="warning" class="bad">{{ warning }}</span>
        </p>
      </article>

      <h2 class="board-head">
        <button type="button" @click="toggleHistory">历史计划</button>
        <a v-if="historyOpen" class="fine" :href="exportHref">导出 CSV</a>
      </h2>
      <div v-if="historyOpen" class="board-history">
        <p v-if="historyError" class="board-alert">{{ historyError }}</p>
        <p v-else-if="!history.length" class="wait">还没有记录。</p>
        <article v-for="row in history" :key="row.plan_id" class="board-plan">
          <p>
            <strong>{{ row.generated_utc8 }} {{ row.title }}</strong>
            {{ row.direction === "short" ? "空" : row.direction === "long" ? "多" : "" }}
          </p>
          <p v-if="row.entry != null" class="fine">
            进场 {{ px(row.entry) }}
            <span v-if="row.entry_low != null">区间 {{ px(row.entry_low) }}–{{ px(row.entry_high) }}</span>
          </p>
          <p class="fine">{{ planOutcomeLine(row) }}</p>
        </article>
      </div>

      <button type="button" @click="calcOpen = !calcOpen">计算器</button>
      <div v-if="calcOpen" class="board-calc">
        <label>本金 <input v-model.number="calc.capital" type="number" min="1" /></label>
        <label>亏损 U <input v-model.number="calc.maxLossU" type="number" min="0" step="0.5" /></label>
        <label>亏损 % <input v-model.number="calc.maxLossPct" type="number" min="0" step="0.01" /></label>
        <label>杠杆 <input v-model.number="calc.leverage" type="number" min="1" max="50" /></label>
        <p class="fine">逐仓，维持保证金率 0.5%。强平价是估算。</p>
      </div>

      <h2>风控</h2>
      <p>
        今日连亏
        <button type="button" @click="changeLoss(-1)">−</button>
        {{ losses }}
        <button type="button" @click="changeLoss(1)">+</button>
        <span v-if="losses >= (view.risk?.loss_streak_stop || 3)" class="bad">今天停手</span>
      </p>
      <p v-for="note in view.risk?.notes || []" :key="note" class="fine">{{ note }}</p>
      <p v-for="item in view.risk?.events || []" :key="item.ts_event" class="fine" :class="{ bad: item.blocking }">
        {{ item.utc8 }} {{ item.title }}：{{ item.window }} {{ item.blocking ? "不开新单" : "" }}
      </p>
    </template>
    <p class="board-disclaimer">概率判断，仅供参考，不构成投资建议。</p>
  </div>
</template>
