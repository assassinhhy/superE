// Pure state controller for the passive forecast flow.
//
// Only run() calls the prediction endpoint. Loading the panel reads cached news
// through a different API call; it never triggers or refreshes a forecast.
// There are no timers here. A click is guarded against re-entry, and a failed
// click keeps the previous rows on screen instead of replacing them with blanks.

function emptyNews() {
  return { items: [], summary: null, alert: null, status: "empty", sources: {}, aggregate: null, scorerVersion: "" };
}

function newsState(payload) {
  return {
    items: payload?.items || [],
    summary: payload?.summary || null,
    alert: payload?.alert || null,
    status: payload?.status || "ok",
    sources: payload?.sources || {},
    aggregate: payload?.aggregate || null,
    scorerVersion: payload?.scorer_version || "",
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
      const payload = await api.news({ limit: 50, since_hours: 72 });
      update({ news: newsState(payload), newsError: "" });
      return payload;
    } catch (err) {
      update({ newsError: err?.message || String(err) });
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
      const predictions = Object.fromEntries(items.map((item) => [item.interval, item]));
      const nextNews = payload?.news ? newsState(payload.news) : state.news;
      update({
        predictions,
        news: nextNews,
        clickAt: payload?.click_at_utc8 || "",
        hasRun: true,
        newsError: "",
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

  return { state, loadCachedNews, run };
}
