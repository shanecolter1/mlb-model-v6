import test from 'node:test';
import assert from 'node:assert/strict';
import {
  americanOddsToNetPayout, expectedLogGrowth, finiteHorizonFirstPassage,
  findRiskLimit, freezeI2StakingProbability, sizeI2Wager, settleI2Wager,
  summarizeI2Portfolio, sizeI2Batch,
} from '../src/staking/i2_bankroll.js';

const b = 100 / 165;
const p4 = 1.04 / (1 + b);
const timestamp = '2026-09-23T15:00:00Z';

function artifact(p = p4, {calibrated = true, frozen = true} = {}) {
  return {
    generatedAt: '2026-09-23T14:00:00Z',
    predictionFrozenBeforeDerivativeMarketRetrieval: frozen,
    derivativeMarketDataUsed: false, i2PriceDataUsed: false,
    localCvProduction: calibrated,
    ranking: [{gamePk: 123, matchup: 'Away @ Home', under05Pct: 100 * p,
      over05Pct: 100 * (1 - p), v04Calibration: {status: calibrated ? 'APPLIED' : 'SKIPPED_MISSING_TOTAL_PRIOR', model: 'I2 v0.4 Local-CV'}}],
    games: [{gamePk: 123, modelStatus: 'FROZEN_RESEARCH_PROJECTION', runEnvironmentConditioned: true,
      under05: p + 0.001, over05: 1 - p - 0.001,
      baseballOnlyRaw: {under05: 0.62, over05: 0.38}}],
  };
}

function wager(bankroll = 10000, p = p4, extra = {}) {
  return sizeI2Wager({
    frozen_staking_probability: freezeI2StakingProbability(artifact(p), 123, 'I2_UNDER_0.5'),
    current_bankroll: bankroll, session_start_bankroll: 10000, american_odds: -165,
    game_identifier: 123, market_identifier: 'I2_UNDER_0.5', timestamp, ...extra,
  });
}

test('-165 and +4% EV yields 6.60% Kelly and $660, risk ceiling does not bind', () => {
  assert.equal(americanOddsToNetPayout(-165), b);
  assert.equal(americanOddsToNetPayout('+120'), 1.2);
  const result = wager();
  assert.ok(Math.abs(result.expectedValue - 0.04) < 1e-12);
  assert.ok(Math.abs(result.fullKellyFraction - 0.066) < 1e-12);
  assert.equal(result.finalStakeDollars, 660);
  assert.ok(Math.abs(result.rawModelProbability - (p4 + 0.001)) < 1e-12);
  assert.ok(Math.abs(result.stakingProbability - p4) < 1e-12);
  assert.equal(result.status, 'KELLY OPTIMAL');
  assert.ok(result.riskLimitStakeFraction > result.fullKellyFraction);
  assert.ok(Math.abs(result.riskLimitStakeFraction - 0.16578) < 0.001);
  assert.ok(result.operationalFirstPassageProbabilityAtFinalStake <= 0.1);
  assert.ok(result.riskLimitStakeWarning?.includes('NEGATIVE GEOMETRIC GROWTH'));
  assert.ok(Math.abs(result.expectedBankrollGrowthContribution - 26.4) < 1e-10);
});

test('zero and negative EV pass at zero stake', () => {
  for (const p of [1 / (1 + b), 0.6]) {
    const result = wager(10000, p);
    assert.equal(result.finalStakeDollars, 0);
    assert.equal(result.finalStakeFraction, 0);
    assert.equal(result.status, 'PASS — NON-POSITIVE EV');
  }
});

test('win and loss settlement recalculate dollar stakes from CURRENT bankroll', () => {
  const first = wager();
  const afterLoss = settleI2Wager({current_bankroll: 10000, stake_dollars: first.finalStakeDollars, american_odds: -165, outcome: 'LOSS'});
  const afterWin = settleI2Wager({current_bankroll: 10000, stake_dollars: first.finalStakeDollars, american_odds: -165, outcome: 'WIN'});
  assert.equal(afterLoss, 9340);
  assert.equal(afterWin, 10400);
  assert.equal(wager(afterLoss).finalStakeDollars, 616.44);
  assert.equal(wager(afterWin).finalStakeDollars, 686.4);
  assert.equal(wager(afterLoss).operationalRuinFloor, 1000);
});

test('first passage counts recovery paths that end above the floor', () => {
  const args = {current_bankroll: 100, operational_ruin_floor: 90,
    calibrated_win_probability: 0.5, net_payout: 1, staking_fraction: 0.2};
  assert.equal(finiteHorizonFirstPassage({...args, risk_horizon_bets: 2}), 0.5);
  assert.equal(finiteHorizonFirstPassage({...args, risk_horizon_bets: 3}), 0.625);
  // With two bets, only LL ends at/below 90 (0.25), but LW ruined at bet 1.
  assert.ok(0.5 > 0.25);
});

test('absorbing ruin is never reintroduced even if later wins could recover', () => {
  const args = {current_bankroll: 100, operational_ruin_floor: 90,
    calibrated_win_probability: 1, net_payout: 1, staking_fraction: 0.2, risk_horizon_bets: 3};
  assert.equal(finiteHorizonFirstPassage(args), 0);
  assert.equal(finiteHorizonFirstPassage({...args, calibrated_win_probability: 0}), 1);
  assert.equal(finiteHorizonFirstPassage({...args, calibrated_win_probability: 0.5}), 0.625);
});

test('first-passage operational risk is nondecreasing over candidate fractions', () => {
  const args = {current_bankroll: 100, operational_ruin_floor: 60, calibrated_win_probability: 0.65,
    net_payout: 0.6, risk_horizon_bets: 8};
  const values = [0.02, 0.04, 0.08, 0.12, 0.2, 0.3].map(staking_fraction =>
    finiteHorizonFirstPassage({...args, staking_fraction}));
  values.slice(1).forEach((x, i) => assert.ok(x + 1e-12 >= values[i]));
});

test('full Kelly maximizes log growth locally; stakes above Kelly lose growth', () => {
  const kelly = 0.066;
  const peak = expectedLogGrowth(p4, b, kelly);
  assert.ok(peak > expectedLogGrowth(p4, b, kelly / 2));
  assert.ok(peak > expectedLogGrowth(p4, b, kelly - 0.01));
  assert.ok(peak > expectedLogGrowth(p4, b, kelly + 0.01));
  const overbet = wager(10000, p4, {kelly_multiplier: 2});
  assert.ok(overbet.logGrowthComparisons.proposedFinal < overbet.logGrowthComparisons.fullKelly);
  assert.ok(overbet.warnings.includes('OVERBETTING — LOWER EXPECTED COMPOUND GROWTH'));
  assert.equal(overbet.status, 'OVERBETTING WARNING');
});

test('negative log growth is rejected even when multiplier explicitly exceeds Kelly', () => {
  const result = wager(10000, p4, {kelly_multiplier: 15, max_ruin_probability: 1});
  assert.equal(result.finalStakeDollars, 0);
  assert.ok(result.warnings.includes('NEGATIVE GEOMETRIC GROWTH — DO NOT USE THIS STAKE'));
});

test('literal $0 risk remains separate from operational first-passage risk', () => {
  const result = wager(10000, 0.8, {operational_ruin_floor: 9000, risk_horizon_bets: 2, max_ruin_probability: 0.9});
  assert.equal(result.mathematicalZeroRuinProbability, 0);
  assert.match(result.mathematicalZeroRuinDescription, /strictly below 100%/);
  assert.ok(result.operationalFirstPassageProbabilityAtFinalStake > 0);
  assert.equal(result.operationalRuinFloor, 9000);
});

test('100% staking reaches mathematical zero on a loss and handles wins without NaN', () => {
  const args = {current_bankroll: 100, operational_ruin_floor: 150, calibrated_win_probability: 0.5,
    net_payout: 0.1, staking_fraction: 1, risk_horizon_bets: 2};
  assert.equal(finiteHorizonFirstPassage(args), 1); // Even the first win gives only $110.
  assert.equal(finiteHorizonFirstPassage({...args, operational_ruin_floor: 10}), 0.75);
  assert.equal(finiteHorizonFirstPassage({...args, operational_ruin_floor: 10,
    calibrated_win_probability: 1}), 0);
});

test('risk ceiling can reduce Kelly but cannot enlarge it', () => {
  const restricted = wager(10000, 0.8, {operational_ruin_floor: 9000,
    risk_horizon_bets: 100, max_ruin_probability: 0.1});
  assert.equal(restricted.status, 'RISK CAP ACTIVE');
  assert.ok(restricted.riskLimitStakeFraction < restricted.fullKellyFraction);
  assert.ok(restricted.finalStakeFraction <= restricted.fullKellyFraction);
  assert.ok(restricted.operationalFirstPassageProbabilityAtFinalStake <= 0.1);
  assert.ok(findRiskLimit({current_bankroll: 10000, operational_ruin_floor: 1000,
    calibrated_win_probability: p4, net_payout: b, risk_horizon_bets: 100,
    max_ruin_probability: 0.1}).fraction > 0.066);
  const base = {current_bankroll: 10000, operational_ruin_floor: 1000,
    calibrated_win_probability: p4, net_payout: b, risk_horizon_bets: 100};
  const limit = findRiskLimit({...base, max_ruin_probability: 0.1}).fraction;
  assert.ok(finiteHorizonFirstPassage({...base, staking_fraction: limit}) <= 0.1);
  assert.ok(finiteHorizonFirstPassage({...base, staking_fraction: limit + 1e-6}) > 0.1);
});

test('missing calibration and missing prediction freeze cannot read sportsbook price', () => {
  assert.throws(() => freezeI2StakingProbability(artifact(p4, {frozen: false}), 123, 'I2_UNDER_0.5'), /frozen/);
  let oddsAccesses = 0;
  const invalidInput = {frozen_staking_probability: {}, game_identifier: 123,
    market_identifier: 'I2_UNDER_0.5'};
  Object.defineProperty(invalidInput, 'american_odds', {get() {oddsAccesses += 1; throw Error('Odds were accessed');}});
  assert.throws(() => sizeI2Wager(invalidInput), /verified frozen/);
  const uncalibrated = freezeI2StakingProbability(artifact(p4, {calibrated: false}), 123, 'I2_UNDER_0.5');
  const input = {frozen_staking_probability: uncalibrated, current_bankroll: 10000,
    session_start_bankroll: 10000, game_identifier: 123, market_identifier: 'I2_UNDER_0.5', timestamp};
  Object.defineProperty(input, 'american_odds', {get() {oddsAccesses += 1; throw Error('Odds were accessed');}});
  assert.equal(sizeI2Wager(input).status, 'CALIBRATION INPUT MISSING');
  assert.equal(oddsAccesses, 0);
});

test('batch loads no quotes until all selected probabilities are frozen and calibrated', async () => {
  const selections = {bankroll: {current_bankroll: 10000, session_start_bankroll: 10000},
    wagers: [{game_identifier: 123, market_identifier: 'I2_UNDER_0.5'}]};
  let priceAccesses = 0;
  const loadQuotes = async () => {priceAccesses += 1; return [{game_identifier: 123,
    market_identifier: 'I2_UNDER_0.5', american_odds: -165, timestamp}];};
  await assert.rejects(sizeI2Batch({predictionArtifact: artifact(p4, {frozen: false}), selections, loadQuotes}), /frozen/);
  assert.equal(priceAccesses, 0);
  const missing = await sizeI2Batch({predictionArtifact: artifact(p4, {calibrated: false}), selections, loadQuotes});
  assert.equal(missing.status, 'CALIBRATION INPUT MISSING');
  assert.equal(priceAccesses, 0);
  const sized = await sizeI2Batch({predictionArtifact: artifact(), selections, loadQuotes});
  assert.equal(priceAccesses, 1);
  assert.equal(sized.wagers[0].finalStakeDollars, 660);
  assert.equal(sized.portfolio.totalDollarsAtRisk, 660);
});

test('quote payload cannot overwrite bankroll, probability, or risk settings', async () => {
  const selections = {bankroll: {current_bankroll: 10000, session_start_bankroll: 10000},
    wagers: [{game_identifier: 123, market_identifier: 'I2_UNDER_0.5', bet_identifier: 'book-a'}]};
  const result = await sizeI2Batch({predictionArtifact: artifact(), selections,
    loadQuotes: async () => [{game_identifier: 123, market_identifier: 'I2_UNDER_0.5',
      bet_identifier: 'book-a', american_odds: -165, timestamp, current_bankroll: 1000000,
      calibrated_win_probability: 0.99, kelly_multiplier: 10}]});
  assert.equal(result.wagers[0].finalStakeDollars, 660);
  assert.equal(result.wagers[0].currentBankroll, 10000);
  assert.equal(result.wagers[0].betIdentifier, 'book-a');
  selections.wagers[0].american_odds = -165;
  await assert.rejects(sizeI2Batch({predictionArtifact: artifact(), selections,
    loadQuotes: async () => { throw Error('Must not access price'); }}), /Pre-freeze selection/);
});

test('frozen probability cannot be silently replaced with an unvalidated value', () => {
  assert.throws(() => wager(10000, p4, {calibrated_win_probability: 0.9}), /differs from frozen/);
});

test('risk calculation unavailable fails closed, without an uncapped Kelly stake', () => {
  const result = wager(10000, p4, {risk_horizon_bets: 1001});
  assert.equal(result.status, 'RISK CALCULATION UNAVAILABLE');
  assert.equal(result.finalStakeDollars, 0);
});

test('portfolio reports aggregate and same-game exposure without an arbitrary multiplier', () => {
  const result = wager();
  const exposure = summarizeI2Portfolio({current_bankroll: 10000,
    wagers: [{...result, gameIdentifier: '123'}, {...result, finalStakeDollars: 200, gameIdentifier: '123'}],
    open_wagers: [{stake_dollars: 100, game_identifier: 999}]});
  assert.equal(exposure.simultaneousWagers, 3);
  assert.equal(exposure.totalDollarsAtRisk, 960);
  assert.equal(exposure.totalBankrollPercentageAtRisk, 9.6);
  assert.deepEqual(exposure.identifiableCorrelatedExposure, [{group: '123', wagers: 2, dollarsAtRisk: 860}]);
  assert.equal(exposure.correlationStatus, 'PORTFOLIO CORRELATION NOT MODELED');
});
