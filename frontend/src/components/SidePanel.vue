<script setup>
import { computed } from "vue";
import { state } from "../live.js";
import { PREDICT_INTERVALS, formatUtc8, formatUtc8Clock } from "../indicators.js";

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

    <section>
      <h2>信号</h2>
      <p class="fine">预测信号在第二阶段才接入。这里不计算方向，也不放占位概率。</p>
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
          <tr v-for="id in PREDICT_INTERVALS" :key="id">
            <td>{{ id }}</td>
            <td>—</td>
            <td>—</td>
            <td>未接入</td>
          </tr>
        </tbody>
      </table>
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
