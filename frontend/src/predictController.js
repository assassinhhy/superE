// Pure state controller for the passive forecast flow.
//
// Only run() calls the prediction endpoint. Loading the panel reads cached news
// through a different API call; it never triggers or refreshes a forecast.
// There are no timers here. A click is guarded against re-entry, and a failed
// click keeps the previous rows on screen instead of replacing them with blanks.

function emptyNews() {
  return {
    items: [], summary: null, alert: null, status: "empty", sources: {}, aggregate: null,
    scorerVersion: "", lastRefreshAt: null, nextRefreshAt: null, refreshing: false,
    cooldownRemainingSec: 0, cooldownUntilMs: 0, highlights: [], other: [], scoreCounts: null,
  };
}

function newsState(payload, nowMs = Date.now()) {
  const remain = Math.max(0, Math.ceil(Number(payload?.cooldown_remaining_s) || 0));
  return {
    items: payload?.items || [],
    summary: payload?.summary || null,
    alert: payload?.alert || null,
    status: payload?.status || "ok",
    sources: payload?.sources || {},
    aggregate: payload?.aggregate || null,
    scorerVersion: payload?.scorer_version || "",
    lastRefreshAt: payload?.last_refresh_at ?? null,
    nextRefreshAt: payload?.next_refresh_at ?? null,
    refreshing: Boolean(payload?.refreshing),
    cooldownRemainingSec: remain,
    cooldownUntilMs: remain > 0 ? nowMs + remain * 1000 : 0,
    highlights: payload?.highlights || [],
    other: payload?.other || [],
    scoreCounts: payload?.score_counts || null,
  };
}

export function createPredictController({ api, intervals, onChange } = {}) {
  const state = {
    predictions: {},
    news: emptyNews(),
    predicting: false,
    hasRun: false,
    error: "",
    newsError: "",
    clickAt: "",
  };
  let running = false;

  function update(patch) {
    Object.assign(state, patch);
    if (onChange) onChange(state);
  }

  async function loadCachedNews() {
    try {
      const payload = await api.news({ limit: 80, since_hours: 24 });
      update({ news: newsState(payload), newsError: "" });
      return payload;
    } catch (err) {
      update({ newsError: err?.message || String(err) });
      return null;
    }
  }

  async function refreshNews() {
    if (state.news.refreshing) return null;
    if (state.news.cooldownUntilMs && Date.now() < state.news.cooldownUntilMs) return null;
    update({ news: { ...state.news, refreshing: true }, newsError: "" });
    try {
      const payload = await api.refreshNews();
      update({ news: newsState(payload), newsError: "" });
      return payload;
    } catch (err) {
      update({
        news: { ...state.news, refreshing: false },
        newsError: err?.message || String(err),
      });
      return null;
    }
  }

  async function run() {
    if (running) return null;
    running = true;
    update({ predicting: true, error: "" });
    try {
      const payload = await api.predictOnClick();
      const items = payload?.items || [];
      const predictions = items.length
        ? Object.fromEntries(items.map((item) => [item.interval, item]))
        : state.predictions;
      // A failed news fetch must not wipe the cached panel; keep it and show a note.
      let nextNews = state.news;
      let newsError = state.newsError;
      if (payload?.news) {
        const failed = payload.news.status === "error" && !(payload.news.items || []).length;
        if (failed) {
          newsError = payload.news.message || "新闻抓取失败，已保留缓存";
        } else {
          nextNews = newsState(payload.news);
          newsError = "";
        }
      }
      update({
        predictions,
        news: nextNews,
        clickAt: payload?.click_at_utc8 || "",
        hasRun: true,
        newsError,
      });
      return payload;
    } catch (err) {
      // Keep the last good rows: a failed refresh must not look like "no model".
      update({ error: err?.message || String(err) });
      return null;
    } finally {
      running = false;
      update({ predicting: false });
    }
  }

  return { state, loadCachedNews, refreshNews, run };
}
