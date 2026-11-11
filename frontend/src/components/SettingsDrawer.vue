<script setup>
import { onMounted, ref } from "vue";
import { api } from "../api.js";

const emit = defineEmits(["close"]);
const settings = ref(null);
const error = ref("");

onMounted(async () => {
  try {
    settings.value = await api.settings();
  } catch {
    error.value = "设置没有读出来";
  }
});
</script>

<template>
  <div class="drawer-back" @click.self="emit('close')">
    <aside class="drawer">
      <header>
        <h2>设置</h2>
        <button type="button" @click="emit('close')">关闭</button>
      </header>
      <p v-if="error">{{ error }}</p>
      <template v-else-if="settings">
        <p>{{ settings.note }}</p>
        <dl>
          <dt>时区</dt>
          <dd>{{ settings.timezone }}，K 线仍按币安 UTC 边界</dd>
          <dt>代理</dt>
          <dd>{{ settings.proxy }}</dd>
          <dt>不走代理</dt>
          <dd>{{ settings.no_proxy }}</dd>
          <dt>数据目录</dt>
          <dd>{{ settings.data_dir }}</dd>
          <dt>手续费</dt>
          <dd>
            挂单 {{ (settings.fees.maker * 100).toFixed(2) }}% ·
            吃单 {{ (settings.fees.taker * 100).toFixed(2) }}% ·
            默认 {{ settings.fees.default_side }} ·
            滑点 {{ settings.fees.slippage_ticks }} tick
          </dd>
          <dt>异动阈值</dt>
          <dd>
            放量 {{ settings.anomaly.volume_median_mult }} 倍 ·
            跳价 {{ settings.anomaly.price_atr_mult }} 倍 ATR ·
            10 秒 {{ settings.anomaly.price_10s_pct }}% ·
            强平单笔 {{ settings.anomaly.liq_single_usd }} ·
            一分钟 {{ settings.anomaly.liq_minute_usd }} ·
            持仓量 {{ settings.anomaly.oi_change_pct }}%
          </dd>
          <dt>磁盘</dt>
          <dd>低于 {{ settings.disk.warn_free_gb }} GB 警告，低于 {{ settings.disk.critical_free_gb }} GB 严重</dd>
        </dl>
      </template>
    </aside>
  </div>
</template>
