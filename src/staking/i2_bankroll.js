// Downstream-only I2 staking. This module never fetches odds or changes predictions.
const frozenInputs = new WeakSet();
export const MAX_EXACT_HORIZON = 1000;

function finite(name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError(`${name} must be a finite number`);
  return value;
}

function probability(name, value) {
  const p = finite(name, value);
  if (p < 0 || p > 1) throw new RangeError(`${name} must be between 0 and 1`);
  return p;
}

function positive(name, value) {
  const n = finite(name, value);
  if (n <= 0) throw new RangeError(`${name} must be positive`);
  return n;
}

export function americanOddsToNetPayout(odds) {
  const n = typeof odds === 'string' && /^\s*[+-]?\d+\s*$/.test(odds) ? Number(odds) : odds;
  if (!Number.isSafeInteger(n) || Math.abs(n) < 100) throw new RangeError('american_odds must be an integer with absolute value at least 100');
  return n < 0 ? 100 / -n : n / 100;
}

export function expectedLogGrowth(p, b, f) {
  probability('win probability', p);
  positive('net payout', b);
  if (finite('staking fraction', f) < 0 || f > 1) throw new RangeError('staking fraction must be between 0 and 1');
  return (p ? p * Math.log1p(f * b) : 0) + (p < 1 ? (1 - p) * Math.log1p(-f) : 0);
}

// Absorbed probability is never placed back into the surviving (t, wins) states.
// Log balances avoid overflow and underflow, but the probabilities are summed exactly
// over the binomial state graph (up to ordinary floating point arithmetic).
export function finiteHorizonFirstPassage({
  current_bankroll, operational_ruin_floor, calibrated_win_probability,
  net_payout, staking_fraction, risk_horizon_bets,
}) {
  const bankroll = positive('current_bankroll', current_bankroll);
  const floor = positive('operational_ruin_floor', operational_ruin_floor);
  const p = probability('calibrated_win_probability', calibrated_win_probability);
  const b = positive('net_payout', net_payout);
  const f = finite('staking_fraction', staking_fraction);
  if (f < 0 || f > 1) throw new RangeError('staking_fraction must be between 0 and 1');
  if (!Number.isSafeInteger(risk_horizon_bets) || risk_horizon_bets < 1 || risk_horizon_bets > MAX_EXACT_HORIZON) {
    throw new RangeError(`risk_horizon_bets must be an integer from 1 to ${MAX_EXACT_HORIZON}`);
  }
  if (bankroll <= floor) return 1;
  if (f === 0) return 0;

  const logBankroll = Math.log(bankroll);
  const logFloor = Math.log(floor);
  const logWin = Math.log1p(f * b);
  const logLoss = Math.log1p(-f);
  let surviving = [1];
  let ruined = 0;
  for (let t = 1; t <= risk_horizon_bets; t += 1) {
    const next = new Float64Array(t + 1);
    for (let w = 0; w < surviving.length; w += 1) {
      const mass = surviving[w];
      if (mass === 0) continue;
      const winMass = mass * p;
      const lossMass = mass * (1 - p);
      if (winMass) {
        const logBalance = f === 1
          ? (w + 1 === t ? logBankroll + t * logWin : -Infinity)
          : logBankroll + (w + 1) * logWin + (t - w - 1) * logLoss;
        if (logBalance <= logFloor) ruined += winMass;
        else next[w + 1] += winMass;
      }
      if (lossMass) {
        const logBalance = f === 1 ? -Infinity : logBankroll + w * logWin + (t - w) * logLoss;
        if (logBalance <= logFloor) ruined += lossMass;
        else next[w] += lossMass;
      }
    }
    surviving = next;
  }
  return Math.max(0, Math.min(1, ruined));
}

export function findRiskLimit({
  current_bankroll, operational_ruin_floor, calibrated_win_probability,
  net_payout, risk_horizon_bets, max_ruin_probability,
}) {
  const maximum = probability('max_ruin_probability', max_ruin_probability);
  const args = {current_bankroll, operational_ruin_floor, calibrated_win_probability, net_payout, risk_horizon_bets};
  const ror = f => finiteHorizonFirstPassage({...args, staking_fraction: f});
  if (ror(0) > maximum) return {fraction: 0, firstPassageProbability: 1, feasible: false};
  if (ror(1) <= maximum) return {fraction: 1, firstPassageProbability: ror(1), feasible: true};
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 64 && hi - lo > 1e-13; i += 1) {
    const mid = (lo + hi) / 2;
    if (ror(mid) <= maximum) lo = mid;
    else hi = mid;
  }
  return {fraction: lo, firstPassageProbability: ror(lo), feasible: true};
}

const SIDES = Object.freeze({'I2_UNDER_0.5': 'under05Pct', 'I2_OVER_0.5': 'over05Pct'});

// The existing v0.4 Local-CV output supplies the validated probability. No raw
// projection is substituted for missing calibration and no extra shrinkage is used.
export function freezeI2StakingProbability(artifact, gameIdentifier, marketIdentifier) {
  if (artifact?.predictionFrozenBeforeDerivativeMarketRetrieval !== true ||
      artifact?.derivativeMarketDataUsed !== false || artifact?.i2PriceDataUsed !== false) {
    throw new Error('I2 prediction must be frozen before derivative prices are read');
  }
  if (!Object.hasOwn(SIDES, marketIdentifier)) throw new RangeError('market_identifier must be I2_UNDER_0.5 or I2_OVER_0.5');
  const matches = (artifact.ranking || []).filter(r => String(r.gamePk) === String(gameIdentifier));
  if (matches.length !== 1) throw new Error('Expected exactly one matching frozen I2 game');
  const row = matches[0];
  const game = (artifact.games || []).find(g => String(g.gamePk) === String(gameIdentifier));
  const calibrated = row[SIDES[marketIdentifier]] / 100;
  const valid = artifact.localCvProduction === true &&
    row.v04Calibration?.status === 'APPLIED' &&
    game?.runEnvironmentConditioned === true &&
    ['FROZEN_RESEARCH_PROJECTION', 'PROVISIONAL_RESEARCH_PROJECTION'].includes(game.modelStatus) &&
    typeof row[SIDES[marketIdentifier]] === 'number' &&
    Number.isFinite(calibrated) && calibrated >= 0 && calibrated <= 1;
  // The parent total-conditioned projection is raw with respect to Local-CV.
  // baseballOnlyRaw precedes the mandatory run-environment conditioning.
  const raw = marketIdentifier === 'I2_UNDER_0.5' ? game?.under05 : game?.over05;
  const token = Object.freeze({
    status: valid ? 'FROZEN' : 'CALIBRATION INPUT MISSING',
    gameIdentifier: String(gameIdentifier), marketIdentifier,
    game: row.matchup || `${game?.away} @ ${game?.home}`,
    calibratedProbability: valid ? calibrated : null,
    rawProbability: Number.isFinite(raw) && raw >= 0 && raw <= 1 ? raw : null,
    calibrationSource: valid ? row.v04Calibration.model : null,
    predictionClass: row.predictionClass || null,
    predictionGeneratedAt: artifact.generatedAt || null,
  });
  frozenInputs.add(token);
  return token;
}

function configFor(input) {
  const current = positive('current_bankroll', input.current_bankroll);
  const session = positive('session_start_bankroll', input.session_start_bankroll);
  const floor = input.operational_ruin_floor ?? session * 0.1;
  positive('operational_ruin_floor', floor);
  const horizon = input.risk_horizon_bets ?? 100;
  const maxRuin = probability('max_ruin_probability', input.max_ruin_probability ?? 0.1);
  const multiplier = finite('kelly_multiplier', input.kelly_multiplier ?? 1);
  if (multiplier < 0) throw new RangeError('kelly_multiplier must be nonnegative');
  return {current, session, floor, horizon, maxRuin, multiplier};
}

function roundedDownCents(amount) {
  // The small offset only compensates for a binary representation just below a cent.
  return Math.floor(amount * 100 + 1e-9) / 100;
}

export function sizeI2Wager(input) {
  const frozen = input.frozen_staking_probability;
  if (!frozenInputs.has(frozen)) throw new Error('A verified frozen I2 staking probability is required before accessing odds');
  if (String(input.game_identifier) !== frozen.gameIdentifier || input.market_identifier !== frozen.marketIdentifier) {
    throw new Error('Wager game/market does not match the frozen probability');
  }
  const {current, session, floor, horizon, maxRuin, multiplier} = configFor(input);
  if (!input.timestamp || !Number.isFinite(Date.parse(input.timestamp))) throw new RangeError('timestamp must be an ISO date-time');
  const base = {
    game: frozen.game, gameIdentifier: frozen.gameIdentifier, market: frozen.marketIdentifier,
    betIdentifier: input.bet_identifier || null,
    sportsbook: input.sportsbook || null, timestamp: input.timestamp,
    currentBankroll: current, sessionStartBankroll: session,
    rawModelProbability: frozen.rawProbability,
    calibratedProbability: frozen.calibratedProbability,
    stakingProbability: frozen.calibratedProbability,
    calibrationSource: frozen.calibrationSource, predictionClass: frozen.predictionClass,
    riskHorizonBets: horizon, operationalRuinFloor: floor,
    maximumRuinProbability: maxRuin, kellyMultiplier: multiplier,
    mathematicalZeroRuinProbability: 0,
    mathematicalZeroRuinDescription: '0% for continuous proportional staking at fractions strictly below 100%',
    portfolioCorrelationStatus: 'PORTFOLIO CORRELATION NOT MODELED',
  };
  if (frozen.status !== 'FROZEN') {
    return {...base, sportsbookPrice: null, status: 'CALIBRATION INPUT MISSING', finalStakeFraction: 0, finalStakeDollars: 0,
      riskConstraintStatus: 'NOT_CALCULATED', warnings: ['CALIBRATION INPUT MISSING']};
  }
  if (input.calibrated_win_probability !== undefined &&
      Math.abs(probability('calibrated_win_probability', input.calibrated_win_probability) - frozen.calibratedProbability) > 1e-12) {
    throw new Error('calibrated_win_probability differs from frozen artifact');
  }
  const rawProbability = input.raw_model_probability === undefined ? frozen.rawProbability : probability('raw_model_probability', input.raw_model_probability);
  if (frozen.rawProbability !== null && input.raw_model_probability !== undefined &&
      Math.abs(rawProbability - frozen.rawProbability) > 1e-12) {
    throw new Error('raw_model_probability differs from frozen artifact');
  }
  // Intentional order: no access to the odds property until both freezes and the
  // calibrated probability have passed the checks above.
  const sportsbookPrice = input.american_odds;
  const b = americanOddsToNetPayout(sportsbookPrice);
  const p = frozen.calibratedProbability;
  const rawEv = p * b - (1 - p);
  const ev = Math.abs(rawEv) < 1e-12 ? 0 : rawEv;
  const fullKelly = Math.max(0, ev / b);
  const growthTarget = fullKelly * multiplier;
  const breakEven = 1 / (1 + b);
  let risk;
  try {
    risk = findRiskLimit({current_bankroll: current, operational_ruin_floor: floor,
      calibrated_win_probability: p, net_payout: b, risk_horizon_bets: horizon, max_ruin_probability: maxRuin});
  } catch (error) {
    return {...base, rawModelProbability: rawProbability, sportsbookPrice, marketBreakEvenProbability: breakEven,
      probabilityEdge: p - breakEven, expectedValue: ev, expectedValuePercent: 100 * ev, netPayout: b,
      fullKellyFraction: fullKelly, growthTargetStakeFraction: growthTarget, riskLimitStakeFraction: null,
      finalStakeFraction: 0, finalStakeDollars: 0, status: 'RISK CALCULATION UNAVAILABLE',
      riskConstraintStatus: 'UNAVAILABLE', riskCalculationError: String(error.message),
      warnings: ['RISK CALCULATION UNAVAILABLE']};
  }
  const target = ev > 0 ? Math.min(growthTarget, risk.fraction) : 0;
  const proposedLogGrowth = expectedLogGrowth(p, b, Math.min(target, 1));
  const warnings = [];
  if (growthTarget > fullKelly + 1e-12 && expectedLogGrowth(p, b, Math.min(growthTarget, 1)) < expectedLogGrowth(p, b, fullKelly)) {
    warnings.push('OVERBETTING — LOWER EXPECTED COMPOUND GROWTH');
  }
  if (proposedLogGrowth < 0) warnings.push('NEGATIVE GEOMETRIC GROWTH — DO NOT USE THIS STAKE');
  const dollars = proposedLogGrowth < 0 ? 0 : roundedDownCents(current * Math.min(target, 1));
  const actualFraction = dollars / current;
  const capActive = ev > 0 && risk.fraction + 1e-12 < growthTarget;
  let status = ev <= 0 ? 'PASS — NON-POSITIVE EV' : capActive ? 'RISK CAP ACTIVE' : multiplier > 1 ? 'OVERBETTING WARNING' : multiplier < 1 ? 'KELLY MULTIPLIER ACTIVE' : 'KELLY OPTIMAL';
  if (proposedLogGrowth < 0) status = 'OVERBETTING WARNING';
  const riskConstraintStatus = !risk.feasible ? 'ALREADY_AT_OR_BELOW_FLOOR' : capActive ? 'RISK CAP ACTIVE' : 'NOT BINDING';
  return {
    ...base, rawModelProbability: rawProbability,
    rawModelProbabilitySource: input.raw_model_probability === undefined || frozen.rawProbability !== null ? 'FROZEN ARTIFACT' : 'USER SUPPLIED — DISPLAY ONLY',
    sportsbookPrice,
    marketBreakEvenProbability: breakEven, probabilityEdge: p - breakEven,
    expectedValue: ev, expectedValuePercent: ev * 100, netPayout: b,
    fullKellyFraction: fullKelly, growthTargetStakeFraction: growthTarget,
    riskLimitStakeFraction: risk.fraction, riskLimitFirstPassageProbability: risk.firstPassageProbability,
    finalStakeFraction: actualFraction, finalStakeDollars: dollars,
    expectedLogGrowthPerBet: expectedLogGrowth(p, b, actualFraction),
    logGrowthComparisons: {
      proposedFinal: expectedLogGrowth(p, b, actualFraction),
      halfKelly: expectedLogGrowth(p, b, fullKelly / 2),
      fullKelly: expectedLogGrowth(p, b, fullKelly),
      riskLimit: expectedLogGrowth(p, b, risk.fraction),
    },
    riskLimitStakeWarning: expectedLogGrowth(p, b, risk.fraction) < 0
      ? 'NEGATIVE GEOMETRIC GROWTH — DO NOT USE THIS STAKE' : null,
    expectedBankrollGrowthContribution: dollars * ev,
    expectedBankrollGrowthFraction: actualFraction * ev,
    operationalFirstPassageProbabilityAtFinalStake: finiteHorizonFirstPassage({
      current_bankroll: current, operational_ruin_floor: floor, calibrated_win_probability: p,
      net_payout: b, staking_fraction: actualFraction, risk_horizon_bets: horizon,
    }),
    mathematicalZeroRuinProbability: actualFraction === 1 ? 1 - p ** horizon : 0,
    status, riskConstraintStatus, warnings,
  };
}

export function settleI2Wager({current_bankroll, stake_dollars, american_odds, outcome}) {
  const bankroll = positive('current_bankroll', current_bankroll);
  const stake = finite('stake_dollars', stake_dollars);
  if (stake < 0 || stake > bankroll) throw new RangeError('stake_dollars must be between zero and current bankroll');
  if (outcome !== 'WIN' && outcome !== 'LOSS') throw new RangeError('outcome must be WIN or LOSS');
  const b = americanOddsToNetPayout(american_odds);
  return outcome === 'WIN' ? bankroll + stake * b : bankroll - stake;
}

export function summarizeI2Portfolio({current_bankroll, wagers = [], open_wagers = []}) {
  const bankroll = positive('current_bankroll', current_bankroll);
  const all = [...open_wagers, ...wagers].filter(w => {
    const amount = finite('portfolio stake', Number(w.stake_dollars ?? w.finalStakeDollars));
    if (amount < 0) throw new RangeError('portfolio stake must be nonnegative');
    return amount > 0;
  });
  const groups = new Map();
  let total = 0;
  for (const wager of all) {
    const dollars = finite('portfolio stake', Number(wager.stake_dollars ?? wager.finalStakeDollars));
    if (dollars < 0) throw new RangeError('portfolio stake must be nonnegative');
    total += dollars;
    const key = String(wager.correlation_group || wager.game_identifier || wager.gameIdentifier || 'UNIDENTIFIED');
    const current = groups.get(key) || {group: key, wagers: 0, dollarsAtRisk: 0};
    current.wagers += 1;
    current.dollarsAtRisk += dollars;
    groups.set(key, current);
  }
  return {
    simultaneousWagers: all.length, totalDollarsAtRisk: total,
    totalCurrentBankrollExposure: total / bankroll, totalBankrollPercentageAtRisk: 100 * total / bankroll,
    identifiableCorrelatedExposure: [...groups.values()].filter(g => g.wagers > 1),
    correlationStatus: 'PORTFOLIO CORRELATION NOT MODELED',
    ...(total > bankroll ? {warning: 'TOTAL EXPOSURE EXCEEDS CURRENT BANKROLL'} : {}),
  };
}

export async function sizeI2Batch({predictionArtifact, selections, loadQuotes}) {
  if (!Array.isArray(selections?.wagers) || typeof loadQuotes !== 'function') throw new TypeError('selections.wagers and loadQuotes are required');
  if (!selections.bankroll || typeof selections.bankroll !== 'object') throw new TypeError('selections.bankroll is required');
  for (const selection of selections.wagers) {
    if (Object.keys(selection).some(key => /odds|price|sportsbook|quote/i.test(key))) {
      throw new Error('Pre-freeze selection must not contain sportsbook prices or quotes');
    }
  }
  // Freeze every selection before invoking the callback that can access prices.
  const frozen = selections.wagers.map(w => freezeI2StakingProbability(predictionArtifact, w.game_identifier, w.market_identifier));
  if (frozen.some(x => x.status !== 'FROZEN')) {
    return {status: 'CALIBRATION INPUT MISSING', wagers: frozen.map((x, i) => ({gameIdentifier: x.gameIdentifier,
      market: x.marketIdentifier, status: x.status, finalStakeDollars: 0, finalStakeFraction: 0})),
    priceSourceAccessed: false};
  }
  for (let i = 0; i < frozen.length; i += 1) {
    if (selections.wagers[i].calibrated_win_probability !== undefined &&
        Math.abs(probability('calibrated_win_probability', selections.wagers[i].calibrated_win_probability) - frozen[i].calibratedProbability) > 1e-12) {
      throw new Error('calibrated_win_probability differs from frozen artifact');
    }
  }
  const quotes = await loadQuotes();
  if (!Array.isArray(quotes)) throw new TypeError('The price source must return an array of quotes');
  const results = selections.wagers.map((selection, i) => {
    const matches = quotes.filter(q => String(q.game_identifier) === frozen[i].gameIdentifier &&
      q.market_identifier === frozen[i].marketIdentifier &&
      (selection.bet_identifier === undefined || String(q.bet_identifier) === String(selection.bet_identifier)));
    if (matches.length !== 1) throw new Error(`Expected exactly one quote for ${frozen[i].gameIdentifier} ${frozen[i].marketIdentifier}`);
    const {american_odds, sportsbook, timestamp} = matches[0];
    return sizeI2Wager({...selections.bankroll, ...selections.risk, ...selection,
      american_odds, sportsbook, timestamp,
      frozen_staking_probability: frozen[i]});
  });
  return {status: 'SIZED', wagers: results, priceSourceAccessed: true,
    portfolio: summarizeI2Portfolio({current_bankroll: selections.bankroll.current_bankroll,
      wagers: results, open_wagers: selections.open_wagers || []})};
}
