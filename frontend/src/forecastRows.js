// Display rows for the price-range forecast (分位区间预测). Pure helpers, no DOM.
// Every number comes from the /api/predict payload. Not an order and not a trading signal.

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

// One display row per interval. item may be undefined (not loaded yet).
// Returns { interval, hasForecast, headline, status, explain, detail, message }.
export function forecastRow(interval, item) {
  const f = item?.price_forecast;
  const usable = f && finite(f.median) && finite(f.p25) && finite(f.p75) && item.status === "ok";
  if (!usable) {
    return {
      interval,
      hasForecast: false,
      headline: `${interval}：暂无预测`,
      status: "",
      explain: "",
      detail: null,
      message: item?.message || "尚未加载",
    };
  }
  const range = `${round(f.p25)}–${round(f.p75)}`;
  return {
    interval,
    hasForecast: true,
    headline: `${interval}后 ≈ ${round(f.median)}（${range}）`,
    status: f.status || "",
    explain: f.explain || "",
    detail: {
      interval80: finite(f.p10) && finite(f.p90) ? `80%区间：${round(f.p10)}–${round(f.p90)}` : `80%区间：${DASH}`,
      width50: `50%区间宽度：${bps(f.width50_bps)}`,
      width80: `80%区间宽度：${bps(f.width80_bps)}`,
      asof: f.asof_utc8 || "",
      horizonEnd: f.horizon_end_utc8 || "",
    },
    message: "",
  };
}

// items: the API items array; intervals: display order.
export function forecastRows(items, intervals) {
  const byInterval = Object.fromEntries((items || []).map((item) => [item.interval, item]));
  return intervals.map((id) => forecastRow(id, byInterval[id]));
}
