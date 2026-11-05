// Badge for a debug prediction row the API marks as not tradable (tradable === false).
// Only an explicit false shows a badge; true or a missing field keeps the old display.
// Pure helper, no DOM. A badge is a warning, not a trading signal.

export const TRADABLE_FALLBACK_TEXT = "未达标/仅参考";

// Returns null when no badge should be shown. Otherwise { text, reason }:
// text is tradable_status (fallback when absent), reason is tradable_reason ("" when absent).
export function tradableBadge(p) {
  if (!p || p.tradable !== false) return null;
  const status = typeof p.tradable_status === "string" ? p.tradable_status.trim() : "";
  const reason = typeof p.tradable_reason === "string" ? p.tradable_reason.trim() : "";
  return { text: status || TRADABLE_FALLBACK_TEXT, reason };
}
