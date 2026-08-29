const EPS = 1e-12;

function clampProbability(value) {
  const p = Number(value);
  if (!Number.isFinite(p)) throw new TypeError('probability must be finite');
  return Math.min(1 - EPS, Math.max(EPS, p));
}

function logit(p) {
  const x = clampProbability(p);
  return Math.log(x / (1 - x));
}

function logistic(x) {
  return 1 / (1 + Math.exp(-Number(x)));
}

export function fairAmericanOdds(probability) {
  const p = clampProbability(probability);
  if (Math.abs(p - 0.5) < 1e-12) return 100;
  return p > 0.5 ? -100 * p / (1 - p) : 100 * (1 - p) / p;
}

export function conditionFullI2Over({ rawOver, totalPriorOver, broadOver }) {
  const raw = clampProbability(rawOver);
  const totalPrior = clampProbability(totalPriorOver);
  const broad = clampProbability(broadOver);
  const baseballLogitDelta = logit(raw) - logit(broad);
  const conditionedOver = logistic(logit(totalPrior) + baseballLogitDelta);
  return {
    conditionedOver,
    conditionedUnder: 1 - conditionedOver,
    baseballLogitDelta,
  };
}

export function solveEqualHalfLogitShift({ rawTopOver, rawBottomOver, targetFullOver }) {
  const top = clampProbability(rawTopOver);
  const bottom = clampProbability(rawBottomOver);
  const target = clampProbability(targetFullOver);
  const fullAt = shift => {
    const t = logistic(logit(top) + shift);
    const b = logistic(logit(bottom) + shift);
    return 1 - (1 - t) * (1 - b);
  };

  let lo = -20;
  let hi = 20;
  for (let i = 0; i < 100; i += 1) {
    const mid = (lo + hi) / 2;
    if (fullAt(mid) < target) lo = mid;
    else hi = mid;
  }
  const shift = (lo + hi) / 2;
  const topOver = logistic(logit(top) + shift);
  const bottomOver = logistic(logit(bottom) + shift);
  return { shift, topOver, bottomOver };
}

export function reweightExactDistribution(rawExact, targetOver) {
  const target = clampProbability(targetOver);
  const positiveKeys = ['1', '2', '3', '4+'];
  const positiveRaw = positiveKeys.reduce((sum, key) => sum + Math.max(0, Number(rawExact?.[key] || 0)), 0);
  if (!(positiveRaw > 0)) throw new RangeError('raw exact distribution must contain positive-run mass');
  const exact = { '0': 1 - target };
  for (const key of positiveKeys) {
    exact[key] = target * Math.max(0, Number(rawExact?.[key] || 0)) / positiveRaw;
  }
  const cumulative = {
    '1+': target,
    '2+': exact['2'] + exact['3'] + exact['4+'],
    '3+': exact['3'] + exact['4+'],
    '4+': exact['4+'],
  };
  return { exact, cumulative };
}

export function conditionI2Projection({
  rawOver,
  rawTopOver,
  rawBottomOver,
  rawExact,
  totalPriorOver,
  broadOver,
}) {
  const full = conditionFullI2Over({ rawOver, totalPriorOver, broadOver });
  const halves = solveEqualHalfLogitShift({
    rawTopOver,
    rawBottomOver,
    targetFullOver: full.conditionedOver,
  });
  const distribution = reweightExactDistribution(rawExact, full.conditionedOver);
  return {
    over05: full.conditionedOver,
    under05: full.conditionedUnder,
    fairOver: fairAmericanOdds(full.conditionedOver),
    fairUnder: fairAmericanOdds(full.conditionedUnder),
    top2Score: halves.topOver,
    bottom2Score: halves.bottomOver,
    fullI2: distribution,
    audit: {
      method: 'TOTAL_PRIOR_PLUS_BASEBALL_LOGIT_DELTA',
      baseballLogitDelta: full.baseballLogitDelta,
      equalHalfLogitShift: halves.shift,
    },
  };
}

// requested daily run trigger 2026-08-29T19:05Z
