<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { state } from "../live.js";
import { api } from "../api.js";
import { PREDICT_INTERVALS, formatUtc8, formatUtc8Clock } from "../indicators.js";
import { forecastRows } from "../forecastRows.js";
import {
  NEWS_FILTERS,
  NEWS_POLL_MS,
  filterNews,
  formatNewsRows,
  importantRows,
  layoutNews,
  refreshButtonState,
  refreshCountdown,
  summaryHeadline,
} from "../news.js";
import { createPredictController } from "../predictController.js";

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

// Passive predictions: the controller only calls the prediction endpoint from
// the button below. Loading cached news is a different call and never predicts.
// The controller is plain JS, so publish immutable snapshots for Vue reactivity.
const view = ref(null);
const controller = createPredictController({
  api,
  intervals: PREDICT_INTERVALS,
  onChange: (state) => {
    view.value = {
      ...state,
      predictions: { ...state.predictions },
      news: { ...state.news },
    };
  },
});
view.value = {
  ...controller.state,
  predictions: { ...controller.state.predictions },
  news: { ...controller.state.news },
};

const predicting = computed(() => view.value.predicting);
const hasRun = computed(() => view.value.hasRun);
const predictError = computed(() => view.value.error);
const newsError = computed(() => view.value.newsError);
const forecast = computed(() => forecastRows(Object.values(view.value.predictions), PREDICT_INTERVALS));
const forecastAsof = computed(() => forecast.value.find((row) => row.asof)?.asof || "");

const nowMs = ref(Date.now());
const newsFilter = ref("全部");
const pageLimit = ref(15);
const showOlder = ref(false);
let newsPoll = 0;
let clockTick = 0;

const newsRows = computed(() => formatNewsRows(view.value.news.items, nowMs.value));
const otherRows = computed(() => formatNewsRows(view.value.news.other, nowMs.value));
const filteredRows = computed(() => filterNews(newsRows.value, newsFilter.value));
const laidOut = computed(() => layoutNews(filteredRows.value, nowMs.value, {
  limit: pageLimit.value,
  showOlder: showOlder.value,
}));
const highlightRows = computed(() => importantRows(newsRows.value));
const headline = computed(() => summaryHeadline(view.value.news.summary));
const prose1 = computed(() => view.value.news.summary?.prose_1h || "");
const prose24 = computed(() => view.value.news.summary?.prose_24h || "");
const newsAggregate = computed(() => view.value.news.aggregate || {});
const sourceEntries = computed(() => Object.entries(view.value.news.sources || {}));
const scoreCounts = computed(() => view.value.news.scoreCounts || null);
const buttonState = computed(() => refreshButtonState({
  refreshing: view.value.news.refreshing,
  cooldownUntilMs: view.value.news.cooldownUntilMs,
  cooldownRemainingSec: view.value.news.cooldownRemainingSec,
  nowMs: nowMs.value,
}));
const countdownText = computed(() => refreshCountdown(view.value.news.nextRefreshAt, nowMs.value));
const lastRefreshText = computed(() => {
  const ts = view.value.news.lastRefreshAt;
  return ts ? formatUtc8(ts) : "尚未刷新";
});

function runPredict() {
  controller.run();
}

function refreshNews() {
  controller.refreshNews();
}

function showMoreNews() {
  pageLimit.value += 15;
}

function revealOlderNews() {
  showOlder.value = true;
}

onMounted(() => {
  controller.loadCachedNews();
  newsPoll = setInterval(() => controller.loadCachedNews(), NEWS_POLL_MS);
  clockTick = setInterval(() => { nowMs.value = Date.now(); }, 1000);
});
onUnmounted(() => {
  clearInterval(newsPoll);
  clearInterval(clockTick);
});

function num(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function signed(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
}

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
      <button type="button" class="predict-btn" :disabled="predicting" @click="runPredict">
        {{ predicting ? "预测中…" : "预测" }}
      </button>
      <p class="fine">只有点击「预测」才会运行模型。消息用缓存；缓存超过 1 小时才会在这次点击时再抓一次。</p>
      <p v-if="predictError" class="fine down">预测失败：{{ predictError }}（保留上一次结果）</p>
      <p v-if="!hasRun && !predictError" class="muted">点击上方按钮后显示结果。</p>

      <template v-if="hasRun">
        <ul class="forecast">
          <li v-for="row in forecast" :key="row.interval" class="forecast-row">
            <template v-if="row.hasForecast">
              <div class="forecast-line">
                <span class="forecast-tag">模型原值</span>
                <strong>{{ row.raw.headline }}</strong>
              </div>
              <div v-if="row.adjusted" class="forecast-line adjusted">
                <span class="forecast-tag">加消息面后</span>
                <strong>{{ row.adjusted.headline }}</strong>
              </div>
              <p v-if="row.overlay && !row.overlay.applied" class="fine">
                消息面未触发，中位价和区间不调整。
              </p>
              <p v-else-if="row.overlay" class="fine">
                消息面已叠加：区间放宽，中位价有方向性调整。
              </p>
              <p class="forecast-explain">{{ row.explain }}</p>

              <div v-if="row.topNews.length" class="forecast-news">
                <p class="forecast-news-head">贡献最大的新闻</p>
                <ul>
                  <li v-for="news in row.topNews" :key="(news.id || '') + news.title">
                    <span class="news-dir" :class="news.direction > 0.12 ? 'up' : news.direction < -0.12 ? 'down' : 'neutral'">
                      {{ news.label }}
                    </span>
                    <a v-if="news.url" :href="news.url" target="_blank" rel="noopener noreferrer">{{ news.title }}</a>
                    <span v-else>{{ news.title }}</span>
                    <span class="fine">{{ news.time }} · {{ news.source }}</span>
                  </li>
                </ul>
              </div>

              <details class="explain-fold">
                <summary>80%区间与技术细节（展开）</summary>
                <div class="explain-line">模型原值 {{ row.raw.interval80 }}；{{ row.raw.width50 }}</div>
                <div v-if="row.adjusted" class="explain-line">
                  加消息面后 {{ row.adjusted.interval80 }}；{{ row.adjusted.width50 }}
                </div>
                <template v-if="row.overlay">
                  <div class="explain-line">
                    U {{ num(row.overlay.u, 3) }} · D {{ num(row.overlay.d, 3) }} · R {{ num(row.overlay.r, 3) }} ·
                    U×R {{ num(row.overlay.u * row.overlay.r, 3) }}（门限 0.15）
                  </div>
                  <div class="explain-line">
                    宽度倍率 ×{{ num(row.overlay.widthMultiplier, 3) }}；中位价 Δ {{ signed(row.overlay.medianDelta) }}
                    （上限 ±{{ num(row.overlay.medianDeltaCap) }}）
                  </div>
                  <div class="explain-line">
                    新闻 {{ row.overlay.counts.count || 0 }} 条；高紧急 1h/4h/24h =
                    {{ row.overlay.counts.high_urgency_1h || 0 }}/{{ row.overlay.counts.high_urgency_4h || 0 }}/{{ row.overlay.counts.high_urgency_24h || 0 }}
                  </div>
                </template>
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
        <p v-if="view.clickAt" class="fine">本次点击：{{ view.clickAt }}（UTC+8）。</p>
      </template>
    </section>

    <section>
      <h2>以太坊消息</h2>
      <p v-if="newsError" class="fine down">消息读取失败：{{ newsError }}</p>

      <p class="news-summary">{{ headline }}</p>
      <p v-if="prose1" class="news-prose">{{ prose1 }}</p>
      <p v-if="prose24" class="news-prose">{{ prose24 }}</p>
      <div class="news-toolbar">
        <span class="fine">上次刷新 {{ lastRefreshText }}</span>
        <span class="fine">下次刷新 {{ countdownText }}</span>
        <button type="button" class="news-refresh" :disabled="buttonState.disabled" @click="refreshNews">
          <i v-if="buttonState.spinning" class="news-spin"></i>
          {{ buttonState.label }}
        </button>
      </div>
      <div class="news-filters">
        <button
          v-for="name in NEWS_FILTERS"
          :key="name"
          type="button"
          :class="{ on: newsFilter === name }"
          @click="newsFilter = name"
        >
          {{ name }}
        </button>
      </div>

      <div class="news-highlights" :class="{ empty: !highlightRows.length }">
        <h3>重要消息</h3>
        <p v-if="!highlightRows.length" class="muted">暂无重要消息</p>
        <ul v-else>
          <li v-for="item in highlightRows" :key="'h-' + (item.id || item.contentHash || item.title)">
            <span class="news-dir" :class="item.directionClass">{{ item.directionLabel }}</span>
            <p>{{ item.impact || item.summary }}</p>
          </li>
        </ul>
      </div>

      <p v-if="!filteredRows.length" class="muted">最近 24 小时没有符合筛选的以太坊消息。</p>
      <ul v-else class="news-feed">
        <li
          v-for="item in laidOut.rows"
          :key="item.id || item.contentHash || item.title"
          class="news-item"
          :class="{ urgent: item.important }"
        >
          <div class="news-main" :title="item.reason || undefined">
            <span class="news-dir" :class="item.directionClass">{{ item.directionLabel }}</span>
            <p class="news-summary-text">{{ item.summary }}</p>
          </div>
          <p class="news-original">
            <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
            <span v-else>{{ item.title }}</span>
            <span class="news-source">{{ item.source }}</span>
            <span class="news-when">{{ item.relative }}</span>
          </p>
          <details v-if="item.reason" class="news-reason">
            <summary>原因</summary>
            <p>{{ item.reason }}</p>
          </details>
        </li>
      </ul>
      <details v-if="otherRows.length" class="news-other">
        <summary>其它（{{ otherRows.length }}）</summary>
        <ul class="news-feed">
          <li v-for="item in otherRows" :key="'o-' + (item.id || item.contentHash || item.title)" class="news-item">
            <div class="news-main">
              <span class="news-dir" :class="item.directionClass">{{ item.directionLabel }}</span>
              <p class="news-summary-text">{{ item.summary }}</p>
            </div>
            <p class="news-original">
              <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
              <span v-else>{{ item.title }}</span>
              <span class="news-source">{{ item.source }}</span>
              <span class="news-when">{{ item.relative }}</span>
            </p>
          </li>
        </ul>
      </details>

      <div v-if="laidOut.hiddenRecent || laidOut.olderHidden" class="news-more">
        <button v-if="laidOut.hiddenRecent" type="button" @click="showMoreNews">更多（还有 {{ laidOut.hiddenRecent }} 条）</button>
        <button v-if="laidOut.olderHidden" type="button" @click="revealOlderNews">展开 6 小时前的 {{ laidOut.olderCount }} 条</button>
      </div>

      <details v-if="newsAggregate.u != null || sourceEntries.length || scoreCounts" class="explain-fold">
        <summary>新闻数值与来源（调试）</summary>
        <div class="explain-line">
          U {{ num(newsAggregate.u, 3) }} · D {{ num(newsAggregate.d, 3) }} · R {{ num(newsAggregate.r, 3) }}
        </div>
        <div class="explain-line">
          24h 条数 {{ newsAggregate.count || 0 }}；利多/利空/中性 =
          {{ newsAggregate.positive || 0 }}/{{ newsAggregate.negative || 0 }}/{{ newsAggregate.neutral || 0 }}
        </div>
        <div v-if="scoreCounts" class="explain-line">
          打分：模型 {{ scoreCounts.llm || 0 }} · 规则 {{ scoreCounts.rule || scoreCounts.failed || 0 }} · 缓存 {{ scoreCounts.cached || 0 }}
        </div>
        <div v-for="[name, meta] in sourceEntries" :key="name" class="explain-line">
          {{ name }}：{{ meta.ok ? `成功 ${meta.count} 条` : `失败：${meta.error || '未知错误'}` }}
        </div>
        <div v-if="view.news.scorerVersion" class="explain-line">打分版本：{{ view.news.scorerVersion }}</div>
      </details>

      <p class="fine">每小时整点自动更新，也可以手动刷新。列表每 5 分钟读取一次缓存。</p>
    </section>

    <section>
      <h2>异动</h2>
      <p v-if="!state.anomalies.length" class="muted">还没有市场异动。</p>
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
