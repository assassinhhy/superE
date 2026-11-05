// Display rows for the price-range forecast (分位区间预测). Pure helpers, no DOM.
// The raw model quantiles stay untouched; the news overlay is a separate display copy.

import { sourceLabel } from "./news.js";

const DASH = "—";

function finite(v) {
  return v != null && Number.isFinite(Number(v));
}

function round(v) {
  return String(Math.round(Number(v)));
}

function bps(v) {
  return finite(v) ? `${Number(v).toFixed(1)} bps` : DASH;
}

function range(f, key25 = "p25", key75 = "p75") {
  return finite(f?.[key25]) && finite(f?.[key75])
    ? `${round(f[key25])}–${round(f[key75])}`
    : DASH;
}

function newsRow(item) {
  return {
    id: item?.id,
    title: item?.title || "(无标题)",
    url: item?.url || "",
    source: sourceLabel(item?.source),
    time: item?.time_utc8 || "—",
    label: item?.sentiment_label || "中性",
    summary: item?.summary_zh || "",
    urgency: finite(item?.urgency) ? Number(item.urgency) : 0,
    relevance: finite(item?.relevance) ? Number(item.relevance) : 0,
    direction: finite(item?.direction) ? Number(item.direction) : 0,
    contribution: finite(item?.contribution) ? Number(item.contribution) : 0,
  };
}

// One display row per interval. item may be undefined (not loaded yet).
// Returns a raw model row plus an optional "加消息面后" row. Not trading advice.
export function forecastRow(interval, item) {
  const f = item?.price_forecast;
  const usable = f && finite(f.median) && finite(f.p25) && finite(f.p75) && item.status === "ok";
  if (!usable) {
    return {
      interval,
      hasForecast: false,
      headline: `${interval}：暂无预测`,
      raw: null,
      adjusted: null,
      overlay: null,
      topNews: [],
      explain: "",
      message: item?.message || "尚未加载",
    };
  }
  const raw = {
    median: Number(f.median),
    headline: `${interval}后 ≈ ${round(f.median)}（${range(f)}）`,
    interval80: finite(f.p10) && finite(f.p90) ? `80%区间：${round(f.p10)}–${round(f.p90)}` : `80%区间：${DASH}`,
    width50: `50%区间宽度：${bps(f.width50_bps)}`,
    width80: `80%区间宽度：${bps(f.width80_bps)}`,
  };
  const ov = item?.news_overlay || null;
  let adjusted = null;
  if (ov && ov.adjusted && finite(ov.adjusted.median)) {
    const a = ov.adjusted;
    adjusted = {
      median: Number(a.median),
      headline: `${interval}后 ≈ ${round(a.median)}（${range(a)}）`,
      interval80: finite(a.p10) && finite(a.p90) ? `80%区间：${round(a.p10)}–${round(a.p90)}` : `80%区间：${DASH}`,
      width50: `50%区间宽度：${bps(ov.width50_bps)}`,
      width80: `80%区间宽度：${bps(ov.width80_bps)}`,
      applied: Boolean(ov.applied),
      widthMultiplier: finite(ov.width_multiplier) ? Number(ov.width_multiplier) : 1,
      medianDelta: finite(ov.median_delta) ? Number(ov.median_delta) : 0,
    };
  }
  const overlay = ov ? {
    applied: Boolean(ov.applied),
    u: finite(ov.u) ? Number(ov.u) : 0,
    d: finite(ov.d) ? Number(ov.d) : 0,
    r: finite(ov.r) ? Number(ov.r) : 0,
    gate: ov.gate || null,
    widthMultiplier: finite(ov.width_multiplier) ? Number(ov.width_multiplier) : 1,
    medianDelta: finite(ov.median_delta) ? Number(ov.median_delta) : 0,
    medianDeltaCap: finite(ov.median_delta_cap) ? Number(ov.median_delta_cap) : 0,
    halfLifeHours: ov.half_life_hours ?? null,
    windowHours: ov.window_hours ?? null,
    counts: ov.counts || {},
    sources: ov.sources || {},
    topNews: (ov.top_news || []).map(newsRow),
  } : null;
  return {
    interval,
    hasForecast: true,
    headline: raw.headline,
    raw,
    adjusted,
    overlay,
    topNews: overlay?.topNews || [],
    explain: f.explain || "",
    asof: f.asof_utc8 || "",
    horizonEnd: f.horizon_end_utc8 || "",
    message: "",
  };
}

// items: the API items array; intervals: display order.
export function forecastRows(items, intervals) {
  const byInterval = Object.fromEntries((items || []).map((item) => [item.interval, item]));
  return intervals.map((id) => forecastRow(id, byInterval[id]));
}
