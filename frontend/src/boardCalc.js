// Position size for a conditional plan. Isolated liquidation ignores fees and the margin ladder.

export function estimatePosition({
  entry,
  stop,
  side,
  targets = [],
  atr = null,
  capital = 100,
  maxLossU = 4,
  maxLossPct = 0,
  leverage = 20,
  leverageMax = 50,
  mmr = 0.005,
  takerFee = 0.0005,
  liqMinAtr = 1,
  feeMinMult = 3,
  rrMin = 1,
  sizeRatio = 1,
}) {
  const warnings = [];
  const notes = [];
  let lev = Number(leverage);
  if (!(lev > 0)) lev = 1;
  if (lev > leverageMax) {
    warnings.push("杠杆超过上限");
    lev = leverageMax;
  } else if (lev > 20) {
    notes.push("杠杆高于 20x，强平更靠近止损");
  }
  const riskBase = maxLossPct > 0 ? capital * maxLossPct : maxLossU;
  const risk = riskBase * (sizeRatio > 0 ? sizeRatio : 1);
  const stopPct = entry ? Math.abs(entry - stop) / entry : 0;
  const notional = stopPct > 0 ? risk / stopPct : null;
  const margin = notional == null || lev <= 0 ? null : notional / lev;
  const liq = side === "long"
    ? entry * (1 - 1 / lev + mmr)
    : entry * (1 + 1 / lev - mmr);
  const outside = side === "long" ? liq < stop : liq > stop;
  const distance = side === "long" ? stop - liq : liq - stop;
  if (atr != null && (!outside || distance < atr * liqMinAtr)) {
    warnings.push("杠杆过高/保证金不足");
  }
  const legs = [];
  for (const target of targets) {
    if (notional == null || risk <= 0) break;
    const move = side === "long" ? target - entry : entry - target;
    const pnl = (move / entry) * notional;
    const rr = pnl / risk;
    legs.push({ target, pnlU: pnl, rr });
    if (rr < rrMin) {
      warnings.push("不划算");
      break;
    }
  }
  const fee = notional == null ? null : notional * takerFee * 2;
  if (fee != null && legs.length && legs[0].pnlU < fee * feeMinMult) {
    warnings.push("波动太小，手续费吃利润");
  }
  return {
    entry,
    riskU: risk,
    stopPct,
    notional,
    margin,
    liq,
    feeU: fee,
    legs,
    warnings,
    notes,
    leverage: lev,
  };
}
