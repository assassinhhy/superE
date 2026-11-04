// Trade-level text for a debug prediction row. Pure helpers, no DOM.
// Levels are informational only: this is not an order and not a trading signal.

function price(v) {
  const n = Number(v);
  return v == null || !Number.isFinite(n) ? "—" : String(n);
}

function signedPct(v) {
  const n = Number(v);
  if (v == null || !Number.isFinite(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${(n * 100).toFixed(2)}%`;
}

// Returns null when there is no prediction row at all (nothing to annotate).
// Otherwise returns { text, muted }: the level line, or the "no levels" reason.
export function formatLevels(p) {
  if (!p) return null;
  const lv = p.levels;
  if (!lv) {
    const reason = p.levels_reason ? `（${p.levels_reason}）` : "";
    return { text: `点位：无${reason}`, muted: true };
  }
  const targets = Array.isArray(lv.targets) ? lv.targets : [];
  const tPct = Array.isArray(lv.target_pct) ? lv.target_pct : [];
  const backtested = Array.isArray(lv.backtested_targets) ? lv.backtested_targets : [];
  const tag = (i) => (backtested[i] === true ? "已回测口径，毛收益未扣费" : "未回测");
  const text =
    `入场 ${price(lv.entry)} · 止损 ${price(lv.stop)}（${signedPct(lv.stop_pct)}）` +
    ` · 止盈 ${price(targets[0])}（${signedPct(tPct[0])}，${tag(0)}）` +
    ` / ${price(targets[1])}（${signedPct(tPct[1])}，${tag(1)}）`;
  return { text, muted: false };
}
