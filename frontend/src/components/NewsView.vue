<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";
import { api } from "../api.js";
import { formatUtc8 } from "../indicators.js";
import {
  NEWS_POLL_MS,
  filterNews,
  formatNewsRows,
  isAlertNews,
  overviewSentence,
  refreshButtonState,
  refreshCountdown,
  urgentBrief,
} from "../news.js";
import { pushNotice } from "../notify.js";

const READ_KEY = "supere.news.read";

const payload = ref(null);
const error = ref("");
const direction = ref("全部");
const source = ref("全部");
const openId = ref("");
const nowMs = ref(Date.now());
const read = ref(loadRead());
const seenHashes = new Set();
let primed = false;
let poll = 0;
let clock = 0;

const rows = computed(() => formatNewsRows(payload.value?.items || [], nowMs.value));
const sources = computed(() => ["全部", ...new Set(rows.value.map((row) => row.source).filter(Boolean))]);
const filtered = computed(() => filterNews(rows.value, direction.value, source.value));
const urgentRows = computed(() => rows.value.filter(isAlertNews));
const brief = computed(() => urgentBrief(rows.value));
const overview = computed(() => overviewSentence(payload.value?.summary));
const buttonState = computed(() => refreshButtonState({
  refreshing: Boolean(payload.value?.refreshing),
  cooldownRemainingSec: payload.value?.cooldown_remaining_s || 0,
  nowMs: nowMs.value,
}));
const lastText = computed(() => {
  const ts = payload.value?.last_refresh_at;
  return ts ? formatUtc8(ts) : "尚未刷新";
});
const nextText = computed(() => refreshCountdown(payload.value?.next_refresh_at, nowMs.value));

function loadRead() {
  try {
    const raw = JSON.parse(localStorage.getItem(READ_KEY) || "[]");
    return new Set(Array.isArray(raw) ? raw : []);
  } catch {
    return new Set();
  }
}

function saveRead() {
  localStorage.setItem(READ_KEY, JSON.stringify([...read.value].slice(-500)));
}

function rowKey(row) {
  return String(row.contentHash || row.id || row.title);
}

function isUnread(row) {
  return !read.value.has(rowKey(row));
}

function apply(body) {
  payload.value = body || {};
  error.value = "";
  const fresh = formatNewsRows(body?.items || [], Date.now()).filter(isAlertNews);
  for (const row of fresh) {
    const id = `news:${rowKey(row)}`;
    if (!primed) {
      seenHashes.add(id);
      continue;
    }
    if (seenHashes.has(id)) continue;
    seenHashes.add(id);
    pushNotice({
      id,
      title: `紧急${row.directionLabel}`,
      body: row.summary,
      tone: row.directionClass === "down" ? "down" : "up",
    });
  }
  primed = true;
}

async function load() {
  try {
    apply(await api.news({ limit: 80, since_hours: 24 }));
  } catch {
    error.value = "新闻服务没连上。在仓库根目录另开一个窗口执行 python -m supere news，不用重启行情。";
  }
}

async function refresh() {
  try {
    const body = await api.refreshNews();
    apply(body);
    if (body?.status === "error") error.value = body.message || "这次没有抓到新消息，下面仍是上次的缓存。";
    else if (body?.cooling) error.value = "";
  } catch {
    error.value = "手动刷新没有发出去。确认 8769 上的新闻服务还在。";
  }
}

function toggle(row) {
  const key = rowKey(row);
  openId.value = openId.value === key ? "" : key;
  if (!read.value.has(key)) {
    read.value = new Set(read.value).add(key);
    saveRead();
  }
}

function markAll() {
  const next = new Set(read.value);
  for (const row of rows.value) next.add(rowKey(row));
  read.value = next;
  saveRead();
}

onMounted(() => {
  load();
  poll = setInterval(load, NEWS_POLL_MS);
  clock = setInterval(() => { nowMs.value = Date.now(); }, 1000);
});
onUnmounted(() => {
  clearInterval(poll);
  clearInterval(clock);
});
</script>

<template>
  <section class="news-page">
    <header class="news-hero">
      <p class="news-kicker">以太坊消息</p>
      <h2>{{ overview }}</h2>
      <div class="news-meta">
        <span>上次刷新 {{ lastText }}</span>
        <span>下次刷新 {{ nextText }}</span>
        <button type="button" class="news-refresh" :disabled="buttonState.disabled" @click="refresh">
          <i v-if="buttonState.spinning" class="news-spin"></i>
          {{ buttonState.label }}
        </button>
        <button type="button" @click="markAll">全部已读</button>
      </div>
      <p v-if="error" class="news-error">{{ error }}</p>
    </header>

    <article v-if="urgentRows.length" class="news-urgent">
      <p class="news-kicker">紧急</p>
      <p class="news-brief">{{ brief }}</p>
      <ul>
        <li v-for="row in urgentRows.slice(0, 3)" :key="'u-' + rowKey(row)">
          <span class="news-dir" :class="row.directionClass">{{ row.directionLabel }}</span>
          <span>{{ row.summary }}</span>
        </li>
      </ul>
    </article>
    <p v-else class="news-quiet">现在没有需要马上看的高紧急利多或利空。</p>

    <div class="news-filters">
      <button
        v-for="name in ['全部', '利多', '利空', '中性', '只看重要']"
        :key="name"
        type="button"
        :class="{ on: direction === name }"
        @click="direction = name"
      >
        {{ name }}
      </button>
      <span class="split">来源</span>
      <button
        v-for="name in sources"
        :key="name"
        type="button"
        :class="{ on: source === name }"
        @click="source = name"
      >
        {{ name }}
      </button>
    </div>

    <p v-if="!filtered.length" class="news-quiet">这个筛选下没有消息。</p>
    <ol v-else class="news-timeline">
      <li
        v-for="row in filtered"
        :key="rowKey(row)"
        :class="[row.directionClass, { unread: isUnread(row), open: openId === rowKey(row) }]"
      >
        <button type="button" class="news-card" @click="toggle(row)">
          <span class="news-time">
            <strong>{{ row.relative }}</strong>
            <em>{{ row.time }}</em>
          </span>
          <span class="news-dir" :class="row.directionClass">{{ row.directionLabel }}</span>
          <span class="news-copy">
            <span v-if="isUnread(row)" class="news-new">未读</span>
            <span class="news-urgency">紧急度{{ row.urgencyLabel }}</span>
            <span class="news-source">{{ row.source }}</span>
            <p>{{ row.summary }}</p>
          </span>
        </button>
        <div v-if="openId === rowKey(row)" class="news-detail">
          <p v-if="row.impact">{{ row.impact }}</p>
          <p v-if="row.reason" class="news-reason-line">{{ row.reason }}</p>
          <p class="news-original">
            <a v-if="row.url" :href="row.url" target="_blank" rel="noopener noreferrer">查看原文</a>
            <span v-else>没有原文链接</span>
            <span>{{ row.title }}</span>
          </p>
        </div>
      </li>
    </ol>
  </section>
</template>
