<script setup>
import { onMounted, ref } from "vue";
import { api } from "../api.js";
import { INTERVALS, formatUtc8 } from "../indicators.js";

const pages = [
  { id: "data", label: "数据管理" },
  { id: "features", label: "特征" },
  { id: "train", label: "训练" },
  { id: "backtest", label: "回测" },
  { id: "models", label: "模型" },
  { id: "monitor", label: "重训监控" },
];
const page = ref("data");
const catalog = ref(null);
const error = ref("");
const checkInterval = ref("5m");
const check = ref(null);
const checking = ref(false);
const command = ref(null);

const markets = [
  { id: "um", label: "合约 K 线" },
  { id: "um_mark", label: "标记价" },
  { id: "um_index", label: "指数价" },
  { id: "um_premium", label: "溢价" },
  { id: "spot", label: "现货" },
];

function cell(market, interval) {
  const row = catalog.value?.coverage?.find(
    (item) => item.market === market && item.interval === interval,
  );
  if (!row || row.range_from == null) return null;
  return row;
}

function cellText(row) {
  if (!row) return "—";
  const from = row.range_from_utc8 || formatUtc8(row.range_from);
  const to = row.range_to_utc8 || formatUtc8(row.range_to);
  const gaps = row.gap_count ? ` · 缺口 ${row.gap_count}` : "";
  return `${from.slice(0, 10)} → ${to.slice(0, 16)}${gaps}`;
}

async function load() {
  try {
    catalog.value = await api.catalog();
    error.value = "";
  } catch {
    error.value = "覆盖表还读不出来。先启动采集器，或用预览数据。";
  }
}

async function runCheck() {
  checking.value = true;
  try {
    check.value = await api.aggregate(checkInterval.value, 4);
  } catch (exc) {
    check.value = { ok: false, detail: String(exc.message || exc) };
  } finally {
    checking.value = false;
  }
}

onMounted(load);
</script>

<template>
  <section class="train">
    <nav class="subnav">
      <button
        v-for="item in pages"
        :key="item.id"
        type="button"
        :class="{ on: page === item.id }"
        @click="page = item.id"
      >
        {{ item.label }}
      </button>
    </nav>

    <div v-if="page === 'data'" class="sheet">
      <p v-if="error" class="chart-note">{{ error }}</p>
      <h2>可回溯上限</h2>
      <ul class="floors">
        <li v-for="floor in catalog?.floors || []" :key="floor.id">
          <strong>{{ floor.since }}</strong>
          {{ floor.note }}
        </li>
      </ul>

      <h2>周期 × 数据类型</h2>
      <div class="matrix-wrap">
        <table class="matrix">
          <thead>
            <tr>
              <th>周期</th>
              <th v-for="market in markets" :key="market.id">{{ market.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in INTERVALS" :key="item.id">
              <th>{{ item.id }}</th>
              <td v-for="market in markets" :key="market.id">
                {{ cellText(cell(market.id, item.id)) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2>自录流</h2>
      <table class="matrix" v-if="catalog?.raw?.length">
        <thead>
          <tr>
            <th>流</th>
            <th>日期</th>
            <th>条数</th>
            <th>缺口</th>
            <th>最近</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in catalog.raw" :key="row.stream + row.date">
            <td>{{ row.stream }}</td>
            <td>{{ row.date }}</td>
            <td>{{ row.msg_count }}</td>
            <td>{{ row.gap_count }}</td>
            <td>{{ row.last_utc8 || "—" }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">还没有自录的盘口、逐笔或强平。盘口和强平断了不能回补。</p>

      <h2>操作</h2>
      <div class="actions">
        <button
          v-for="item in catalog?.commands || []"
          :key="item.id"
          type="button"
          @click="command = item"
        >
          {{ item.label }}
        </button>
        <label>
          校验周期
          <select v-model="checkInterval">
            <option v-for="item in INTERVALS" :key="item.id" :value="item.id">{{ item.id }}</option>
          </select>
        </label>
        <button type="button" :disabled="checking" @click="runCheck">
          {{ checking ? "校验中" : "校验（1 分钟聚合对比）" }}
        </button>
      </div>
      <p v-if="check" class="check" :class="check.ok ? 'up' : 'down'">
        {{ check.ok ? "通过" : "未通过" }}
        · 对比 {{ check.compared ?? 0 }} 根
        · {{ check.detail }}
      </p>
      <p v-if="command" class="command">
        <strong>{{ command.label }}</strong>
        <code>{{ command.command }}</code>
        <span>{{ command.detail }}</span>
        <button type="button" @click="command = null">关闭</button>
      </p>
    </div>

    <div v-else-if="page === 'features'" class="sheet placeholder">
      <h2>特征</h2>
      <p>第二阶段才做特征库。分组会包括价格动量、波动、成交量与订单流、合约情绪、多周期、时间和默认关闭的 BTC 参考。</p>
      <p>特征体检（缺失率、相关、共线、未来函数）和特征哈希也在那一阶段。这一页不假装已经算出特征。</p>
    </div>
    <div v-else-if="page === 'train'" class="sheet placeholder">
      <h2>训练</h2>
      <p>第二阶段：按周期排队，标签用 triple-barrier，模型用 LightGBM、XGBoost 和逻辑回归基线。验证固定为 purged walk-forward 加 embargo，不提供随机打乱。</p>
      <p>5 分钟到 4 小时做机器学习。12 小时只做实验。1 日、1 周、1 月不做方向模型。</p>
    </div>
    <div v-else-if="page === 'backtest'" class="sheet placeholder">
      <h2>回测</h2>
      <p>第二阶段用样本外预测。成交价是信号之后那根 K 线的开盘价，并计入手续费、滑点和资金费。超不过基线的模型会标出来。</p>
    </div>
    <div v-else-if="page === 'models'" class="sheet placeholder">
      <h2>模型</h2>
      <p>第二阶段才有冠军、挑战者和归档。模型用原生格式加 meta.json，不用 pickle。现在模型表是空的。</p>
    </div>
    <div v-else class="sheet placeholder">
      <h2>定期重训与监控</h2>
      <p>第三阶段才做定时重训、漂移监控和冠军替换。增量学习默认不用。</p>
    </div>
  </section>
</template>
