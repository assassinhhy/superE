// Short Chinese read-out of the five debug predictions, in a fixed order:
// 结论 → 现价/关键位 → 分周期 → 资金面 → 点位 → 高杠杆提醒 → 调试勿实盘.
// Pure helpers, no DOM. Every number comes from the prediction payload or the live price.
// This is informational text, not an order and not a trading signal.

import { formatLevels } from "./predictLevels.js";

const DIR_SHORT = { up: "多", down: "空", flat: "观望" };
const DIR_LEAN = { up: "偏多", down: "偏空" };
const CHANCE_LINE = "约 33% 为三分类瞎猜的水平";

function pct(v, digits = 1) {
  const n = Number(v);
  return v == null || !Number.isFinite(n) ? "—" : `${(n * 100).toFixed(digits)}%`;
}

function num(v) {
  const n = Number(v);
  return v == null || !Number.isFinite(n) ? "—" : String(n);
}

// Probability shown for a row: the chosen direction, or the flat class when flat.
function rowProb(p) {
  const key = p.direction ?? "flat";
  return p.probs ? p.probs[key] : null;
}

function leadOf(directional) {
  return directional.reduce((best, p) =>
    !best || (p.probs?.[p.direction] ?? 0) > (best.probs?.[best.direction] ?? 0) ? p : best,
  null);
}

function conclusionLine(ok, directional, firstMessage) {
  if (!ok.length) return `结论：没有可用的预测（${firstMessage || "数据或模型不可用"}），先不看。`;
  if (!directional.length) {
    const thr = ok[0].threshold == null ? "" : `${Math.round(Number(ok[0].threshold) * 100)}% `;
    return `结论：五个周期里没有方向过 ${thr}阈值的，先观望，不给点位。`;
  }
  const lead = leadOf(directional);
  const conflict = directional.some((p) => p.direction !== lead.direction);
  const tail = conflict ? "；周期方向有分歧，可信度低。" : "。";
  return `结论：${DIR_LEAN[lead.direction]}，主看 ${lead.interval}（${DIR_SHORT[lead.direction]} ${pct(rowProb(lead))}）${tail}`;
}

// items: the five prediction rows in display order (undefined when not loaded yet).
// price: the live price text for the header of the 现价 line.
export function buildNarrative({ items, price }) {
  const rows = (items || []).filter(Boolean);
  const ok = rows.filter((p) => p.status === "ok");
  const directional = ok.filter((p) => p.direction === "up" || p.direction === "down");
  const lead = leadOf(directional);
  const withCtx = ok.find((p) => p.context);
  const ctxRow = lead?.context ? lead : withCtx;
  const ctx = ctxRow?.context;

  const lines = [];

  lines.push({ key: "conclusion", label: "结论", text: conclusionLine(ok, directional, rows[0]?.message) });

  if (ctx) {
    const keyPart = `关键位（${ctxRow.interval} 近 ${ctx.window_bars} 根）：高 ${num(ctx.high)} / 低 ${num(ctx.low)}`;
    lines.push({ key: "price", label: "现价/关键位", text: `现价 ${price ?? num(ctx.ref_close)}；${keyPart}。` });
  } else {
    lines.push({ key: "price", label: "现价/关键位", text: `现价 ${price ?? "—"}；关键位暂缺。` });
  }

  const perInterval = rows.map((p) => {
    if (p.status !== "ok") return `${p.interval} 不可用`;
    return `${p.interval} ${DIR_SHORT[p.direction] ?? DIR_SHORT.flat}（${pct(rowProb(p), 0)}）`;
  });
  lines.push({ key: "intervals", label: "分周期", text: perInterval.join(" ｜ ") || "—" });

  const fund = ctx?.funding;
  if (!fund || fund.rate == null) {
    lines.push({ key: "funding", label: "资金面", text: "资金面：没有可用的资金费率数据。" });
  } else {
    const r = Number(fund.rate);
    const who = r > 0 ? "多头付给空头" : r < 0 ? "空头付给多头" : "费率为零";
    const sign = r > 0 ? "+" : "";
    lines.push({
      key: "funding",
      label: "资金面",
      text: `资金面：最新费率 ${sign}${(r * 100).toFixed(4)}%（${fund.time_utc8} 结算），${who}。持仓量和多空比还没接进解释。`,
    });
  }

  if (lead) {
    const lv = formatLevels(lead);
    const text = lv.muted ? lv.text : `点位（${lead.interval}）：${lv.text}`;
    lines.push({ key: "levels", label: "进场/止损/止盈", text });
  } else {
    lines.push({ key: "levels", label: "进场/止损/止盈", text: "点位：观望，不给进场、止损、止盈。" });
  }

  lines.push({
    key: "leverage",
    label: "高杠杆提醒",
    text: "高杠杆提醒：点位是价格，和杠杆无关。杠杆越高，同样一段反向波动吃掉的本金越多；按 20 倍算，1% 反向就约 20% 本金，还没算手续费和资金费。",
  });

  const accs = ok.map((p) => Number(p.oos_accuracy)).filter((x) => Number.isFinite(x));
  const accText = accs.length
    ? `样本外准确率 ${pct(Math.min(...accs))}–${pct(Math.max(...accs))}，${CHANCE_LINE}`
    : "没有可用的样本外准确率";
  lines.push({ key: "debug", label: "调试", text: `调试勿实盘：模型样本外弱，${accText}。` });

  return { lines, lead: lead ? { interval: lead.interval, direction: lead.direction } : null };
}
