<script setup>
import { computed, onMounted, ref } from "vue";
import { state } from "../live.js";
import { api } from "../api.js";
import { PREDICT_INTERVALS, formatUtc8, formatUtc8Clock } from "../indicators.js";
import { forecastRows } from "../forecastRows.js";
import { formatAlert, formatNewsRows, summaryLine } from "../news.js";
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

const newsRows = computed(() => formatNewsRows(view.value.news.items));
const newsLine = computed(() => summaryLine(view.value.news.summary));
const alert = computed(() => formatAlert(view.value.news.alert));
const newsAggregate = computed(() => view.value.news.aggregate || {});
const sourceEntries = computed(() => Object.entries(view.value.news.sources || {}));

function runPredict() {
  controller.run();
}

onMounted(() => {
  // Cache only: no forecast call, no timer.
  controller.loadCachedNews();
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
        {{ predicting ? "抓取新闻并预测…" : "预测" }}
      </button>
      <p class="fine">只有点击「预测」才会依次抓取当前 ETH 新闻、打分并运行模型；页面不会自动预测。</p>
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
      <h2>新闻与异常</h2>
      <p v-if="newsError" class="fine down">缓存新闻读取失败：{{ newsError }}</p>

      <div class="news-alert" :class="alert.level">
        <div class="news-alert-head">
          <strong>{{ alert.title }}</strong>
          <span class="news-alert-level">{{ alert.urgency }}紧急</span>
        </div>
        <p v-if="alert.what"><b>发生了什么：</b>{{ alert.what }}</p>
        <p v-if="alert.why"><b>为何影响 ETH：</b>{{ alert.why }}</p>
        <p v-if="alert.consistency"><b>方向一致性：</b>{{ alert.consistency }}</p>
        <p v-if="alert.sources || alert.recency"><b>来源与时效：</b>{{ alert.sources }}<template v-if="alert.sources && alert.recency">，</template>{{ alert.recency }}</p>
        <p><b>关注窗口：</b>{{ alert.watchWindow }}</p>
        <p v-if="alert.uncertain" class="fine">{{ alert.note || "部分条目只有标题和规则打分，细节有限。" }}</p>
      </div>

      <p class="news-summary">{{ newsLine || "暂无新闻摘要" }}</p>
      <p v-if="!newsRows.length" class="muted">
        还没有新闻。面板先读缓存；点击「预测」会现场抓取最新 ETH 相关消息。
      </p>
      <ul v-else class="news-feed">
        <li
          v-for="item in newsRows"
          :key="item.id || item.contentHash || item.title"
          class="news-item"
          :class="{ urgent: item.urgent }"
        >
          <div class="news-meta">
            <span>{{ item.time }}</span>
            <span class="news-source">{{ item.source }}</span>
            <span class="news-dir" :class="item.directionClass">{{ item.directionLabel }}</span>
            <span>{{ item.urgencyLabel }}紧急</span>
            <span v-if="item.urgent" class="news-urgent-tag">高紧急</span>
          </div>
          <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
          <span v-else>{{ item.title }}</span>
          <p class="news-summary-text">{{ item.summary }}</p>
        </li>
      </ul>

      <details v-if="newsAggregate.u != null || sourceEntries.length" class="explain-fold">
        <summary>新闻数值与来源（调试）</summary>
        <div class="explain-line">
          U {{ num(newsAggregate.u, 3) }} · D {{ num(newsAggregate.d, 3) }} · R {{ num(newsAggregate.r, 3) }}
        </div>
        <div class="explain-line">
          24h 条数 {{ newsAggregate.count || 0 }}；利多/利空/中性 =
          {{ newsAggregate.positive || 0 }}/{{ newsAggregate.negative || 0 }}/{{ newsAggregate.neutral || 0 }}
        </div>
        <div v-for="[name, meta] in sourceEntries" :key="name" class="explain-line">
          {{ name }}：{{ meta.ok ? `成功 ${meta.count} 条` : `失败：${meta.error || '未知错误'}` }}
        </div>
        <div v-if="view.news.scorerVersion" class="explain-line">打分版本：{{ view.news.scorerVersion }}</div>
      </details>

      <p class="fine">利多/利空是新闻情绪标签，不代表价格一定跟随。面板先读缓存；点击「预测」才会抓取最新新闻。</p>
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
