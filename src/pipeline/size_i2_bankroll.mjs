#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import {sizeI2Batch} from '../staking/i2_bankroll.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, item, i, all) => {
  if (i % 2 === 0) pairs.push([item, all[i + 1]]);
  return pairs;
}, []));
if (!args['--predictions'] || !args['--selections'] || !args['--quotes'] || process.argv.length % 2 !== 0) {
  throw new Error('Usage: node src/pipeline/size_i2_bankroll.mjs --predictions FROZEN_V04.json --selections SELECTIONS.json --quotes QUOTES.json [--output REPORT.json]');
}

const predictions = JSON.parse(await fs.readFile(args['--predictions'], 'utf8'));
const selections = JSON.parse(await fs.readFile(args['--selections'], 'utf8'));
const report = await sizeI2Batch({predictionArtifact: predictions, selections,
  loadQuotes: async () => JSON.parse(await fs.readFile(args['--quotes'], 'utf8'))});

function percent(value, decimals = 4) { return value == null ? 'unavailable' : `${(value * 100).toFixed(decimals)}%`; }
function dollars(value) { return value == null ? 'unavailable' : `$${value.toFixed(2)}`; }
function decimal(value) { return value == null ? 'unavailable' : String(value); }
for (const w of report.wagers) {
  console.log([
    `Game: ${w.game || w.gameIdentifier}`,
    `Market: ${w.market}`,
    `Bet identifier: ${w.betIdentifier ?? 'unavailable'}`,
    `Sportsbook price: ${w.sportsbookPrice ?? 'unavailable'}`,
    `Current bankroll: ${dollars(w.currentBankroll)}`,
    `Raw model probability: ${percent(w.rawModelProbability)}`,
    `Calibrated/staking probability: ${percent(w.stakingProbability)}`,
    `Market break-even probability: ${percent(w.marketBreakEvenProbability)}`,
    `Probability edge: ${percent(w.probabilityEdge)}`,
    `Expected value: ${decimal(w.expectedValue)} (${w.expectedValuePercent?.toFixed(4) ?? 'unavailable'}%)`,
    `Net payout b: ${decimal(w.netPayout)}`,
    `Full Kelly: ${percent(w.fullKellyFraction)}`,
    `Configured Kelly multiplier: ${decimal(w.kellyMultiplier)}`,
    `Growth-target stake: ${percent(w.growthTargetStakeFraction)}`,
    `Risk horizon: ${w.riskHorizonBets ?? 'unavailable'} bets`,
    `Operational ruin floor: ${dollars(w.operationalRuinFloor)}`,
    `Maximum permitted operational ruin probability: ${percent(w.maximumRuinProbability)}`,
    `Calculated risk-limit stake: ${percent(w.riskLimitStakeFraction)}`,
    `FINAL STAKE: ${percent(w.finalStakeFraction)} / ${dollars(w.finalStakeDollars)}`,
    `Expected logarithmic growth per bet: ${decimal(w.expectedLogGrowthPerBet)}`,
    `Log growth — proposed final / half Kelly / full Kelly / risk limit: ${[
      w.logGrowthComparisons?.proposedFinal, w.logGrowthComparisons?.halfKelly,
      w.logGrowthComparisons?.fullKelly, w.logGrowthComparisons?.riskLimit,
    ].map(decimal).join(' / ')}`,
    ...(w.riskLimitStakeWarning ? [`Risk-limit stake comparison: ${w.riskLimitStakeWarning}`] : []),
    `Expected bankroll growth contribution: ${dollars(w.expectedBankrollGrowthContribution)}`,
    `Operational first-passage probability at final stake: ${percent(w.operationalFirstPassageProbabilityAtFinalStake, 8)}`,
    `Literal $0 ruin probability: ${percent(w.mathematicalZeroRuinProbability)}`,
    `Risk-constraint status: ${w.riskConstraintStatus ?? 'unavailable'}`,
    `Status: ${w.status}`,
    ...(w.warnings?.length ? [`Warnings: ${w.warnings.join('; ')}`] : []),
  ].join('\n'));
  console.log('');
}
if (report.portfolio) {
  console.log(`Simultaneous wagers: ${report.portfolio.simultaneousWagers}`);
  console.log(`Total dollars at risk: ${dollars(report.portfolio.totalDollarsAtRisk)}`);
  console.log(`Total current bankroll exposure: ${percent(report.portfolio.totalCurrentBankrollExposure)}`);
  console.log(`Identifiable same-game/group exposure: ${JSON.stringify(report.portfolio.identifiableCorrelatedExposure)}`);
  console.log(report.portfolio.correlationStatus);
}

if (args['--output']) {
  const output = args['--output'];
  await fs.mkdir(path.dirname(output), {recursive: true});
  const temporary = `${output}.${process.pid}.tmp`;
  await fs.writeFile(temporary, JSON.stringify(report, (_, value) =>
    value === -Infinity ? '-Infinity' : value, 2) + '\n', {flag: 'wx', mode: 0o600});
  await fs.rename(temporary, output);
}
