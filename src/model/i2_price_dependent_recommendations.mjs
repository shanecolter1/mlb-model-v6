export function americanBreakEven(americanOdds) {
  const a = Number(americanOdds);
  if (!Number.isFinite(a) || a === 0) return null;
  return a > 0 ? 100 / (a + 100) : (-a) / ((-a) + 100);
}

export function profitPerUnit(americanOdds) {
  const a = Number(americanOdds);
  if (!Number.isFinite(a) || a === 0) return null;
  return a > 0 ? a / 100 : 100 / (-a);
}

export function expectedValuePerUnit(probability, americanOdds) {
  const p = Number(probability);
  const b = profitPerUnit(americanOdds);
  if (!(p >= 0 && p <= 1) || b === null) return null;
  return p * b - (1 - p);
}

export function fullKellyFraction(probability, americanOdds) {
  const p = Number(probability);
  const b = profitPerUnit(americanOdds);
  if (!(p > 0 && p < 1) || !(b > 0)) return null;
  return (b * p - (1 - p)) / b;
}

export function validQuoteTimestamp(value) {
  const s = String(value || '').trim();
  return Boolean(s) && !Number.isNaN(Date.parse(s));
}

export function recommendationGate({
  projectionFrozen,
  bookmaker,
  americanOdds,
  priceTimestamp,
  modelAvailable,
  exactMarketMatch,
  baseballEligibility,
} = {}) {
  const reasons = [];
  if (baseballEligibility?.eligible !== true) reasons.push(...(baseballEligibility?.reasons?.length ? baseballEligibility.reasons : ['BASEBALL_INPUT_GOVERNANCE_MISSING']));
  if (projectionFrozen !== true) reasons.push('PROJECTION_NOT_FROZEN');
  if (!String(bookmaker || '').trim()) reasons.push('SPORTSBOOK_MISSING');
  const odds = Number(americanOdds);
  if (!Number.isFinite(odds) || odds === 0) reasons.push('PRICE_MISSING');
  if (!validQuoteTimestamp(priceTimestamp)) reasons.push('PRICE_TIMESTAMP_MISSING');
  if (modelAvailable !== true) reasons.push('MODEL_UNAVAILABLE');
  if (exactMarketMatch !== true) reasons.push('EXACT_MARKET_MATCH_FAILED');

  return {
    eligible: reasons.length === 0,
    status: reasons.length === 0 ? 'ACTIONABLE' : 'NO_RECOMMENDATION',
    reasons,
  };
}

function pct(value) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? null
    : Number((100 * Number(value)).toFixed(2));
}

export function enrichPriceDependentRecommendation(row, { projectionFrozen } = {}) {
  const side = String(row?.side || '');
  const marketType = String(row?.marketType || '');
  const standardTotalMatch = Boolean(
    row &&
    row.gamePk != null &&
    ['full','top','bottom'].includes(String(row.segment)) &&
    ['over','under'].includes(side) &&
    Number.isFinite(Number(row.line))
  );
  const modeledThreeWayDrawMatch = Boolean(
    row &&
    row.gamePk != null &&
    String(row.segment) === 'full' &&
    marketType === 'FULL_INNING_3WAY' &&
    side === 'draw' &&
    row.fallbackMarket === true
  );
  const exactMarketMatch = standardTotalMatch || modeledThreeWayDrawMatch;

  const gate = recommendationGate({
    projectionFrozen,
    bookmaker: row?.bookmaker,
    americanOdds: row?.americanOdds,
    priceTimestamp: row?.lastUpdatedAt,
    modelAvailable: row?.modelAvailable,
    exactMarketMatch,
    baseballEligibility:row?.baseballEligibility,
  });

  const pProd = Number(row?.productionConditionalPct) / 100;
  const pComb = Number(row?.combinedConditionalPct) / 100;
  const pEmp = Number(row?.empiricalConditionalPct) / 100;

  const productionKelly = gate.eligible ? fullKellyFraction(pProd, row.americanOdds) : null;
  const combinedKelly = gate.eligible ? fullKellyFraction(pComb, row.americanOdds) : null;
  const empiricalKelly = gate.eligible ? fullKellyFraction(pEmp, row.americanOdds) : null;

  return {
    ...row,
    exactMarketMatch,
    recommendationEligible: gate.eligible,
    recommendationStatus: gate.status,
    recommendationBlockReasons: gate.reasons,
    productionKellyPct: pct(productionKelly),
    combinedKellyPct: pct(combinedKelly),
    empiricalKellyPct: pct(empiricalKelly),
    priceDependentRankingMetric: gate.eligible ? 'COMBINED_EV_PCT' : null,
  };
}

export function rankPriceDependentRecommendations(rows = []) {
  return rows
    .filter(row => row?.recommendationEligible === true)
    .slice()
    .sort((a, b) => {
      const robust = Number(Boolean(b.robustPositiveEV)) - Number(Boolean(a.robustPositiveEV));
      if (robust) return robust;
      const structure = Number(a.marketStructurePriority ?? 99) - Number(b.marketStructurePriority ?? 99);
      if (structure) return structure;
      const ev = Number(b.combinedEVPct ?? -Infinity) - Number(a.combinedEVPct ?? -Infinity);
      if (ev) return ev;
      const minEv = Number(b.rangeMinEVPct ?? -Infinity) - Number(a.rangeMinEVPct ?? -Infinity);
      if (minEv) return minEv;
      return Number(b.combinedKellyPct ?? -Infinity) - Number(a.combinedKellyPct ?? -Infinity);
    });
}
