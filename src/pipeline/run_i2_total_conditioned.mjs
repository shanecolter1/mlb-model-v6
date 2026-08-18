import fs from 'node:fs';
import { conditionI2Projection } from '../model/i2_total_conditioning.js';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const OUTPUT = process.env.I2_OUTPUT || `data/runtime/i2/${DATE}_frozen_predictions.json`;
const TOTALS_PATH = process.env.I2_RUN_ENVIRONMENT || `data/runtime/i2/${DATE}_run_environment.json`;
const PRIOR_PATH = process.env.I2_TOTAL_PRIOR || 'data/derived/i2/i2_total_conditioned_prior.json';
const OVERRIDES = String(process.env.I2_LINEUP_OVERRIDES || '').trim();

if (OVERRIDES && fs.existsSync(OVERRIDES)) await import('./run_i2_full_slate_override.mjs');
else await import('./run_i2_today_upstream_wrapper.mjs');

if (!fs.existsSync(OUTPUT)) throw new Error(`Missing I2 prediction artifact: ${OUTPUT}`);
if (!fs.existsSync(TOTALS_PATH)) throw new Error(`Missing required I2 run-environment artifact: ${TOTALS_PATH}`);
if (!fs.existsSync(PRIOR_PATH)) throw new Error(`Missing I2 total-conditioned prior: ${PRIOR_PATH}`);

const payload = JSON.parse(fs.readFileSync(OUTPUT, 'utf8'));
const totals = JSON.parse(fs.readFileSync(TOTALS_PATH, 'utf8'));
const prior = JSON.parse(fs.readFileSync(PRIOR_PATH, 'utf8'));

if (totals?.scope !== 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES') {
  throw new Error(`Run-environment artifact exposed an unapproved scope: ${totals?.scope || 'missing'}`);
}
if (!prior?.governance?.exactBucketRequired || prior?.governance?.broadPriorFallbackAllowed) {
  throw new Error('I2 total prior governance must require exact buckets and prohibit broad-prior fallback');
}

function norm(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function teamKey(value) {
  const n = norm(value);
  if (n === 'oakland athletics' || n === 'athletics') return 'athletics';
  return n;
}

function findEnvironment(game) {
  const away = teamKey(game.away);
  const home = teamKey(game.home);
  const gameMs = Date.parse(game.gameDate || 0);
  const candidates = (totals.events || []).filter(event =>
    teamKey(event.awayTeam) === away && teamKey(event.homeTeam) === home
  );
  if (!candidates.length) return null;
  candidates.sort((a, b) => Math.abs(Date.parse(a.commenceTime || 0) - gameMs) - Math.abs(Date.parse(b.commenceTime || 0) - gameMs));
  const best = candidates[0];
  const timeDiffHours = Math.abs(Date.parse(best.commenceTime || 0) - gameMs) / 3600000;
  if (!Number.isFinite(timeDiffHours) || timeDiffHours > 12) return null;
  if (best.firstSeenAt && Date.parse(best.firstSeenAt) >= gameMs) return null;
  return best;
}

function pct(x) { return Math.round(Number(x) * 10000) / 100; }
function odds(x) { return x == null ? null : Math.round(Number(x)); }
function fractionExact(game) {
  return Object.fromEntries(Object.entries(game.fullI2Exact || {}).map(([k, v]) => [k, Number(v) / 100]));
}

const rawRanking = Array.isArray(payload.ranking) ? JSON.parse(JSON.stringify(payload.ranking)) : [];
const rawRankingByGame = new Map(rawRanking.map(row => [String(row.gamePk), row]));
let conditionedCount = 0;
let missingTotalCount = 0;
let missingBucketCount = 0;

for (const game of payload.games || []) {
  if (!['FROZEN_RESEARCH_PROJECTION', 'PROVISIONAL_RESEARCH_PROJECTION'].includes(game.modelStatus)) continue;

  const raw = {
    under05: Number(game.under05),
    over05: Number(game.over05),
    under05Pct: Number(game.under05Pct),
    over05Pct: Number(game.over05Pct),
    fairUnder: game.fairUnder,
    fairOver: game.fairOver,
    fullI2Exact: JSON.parse(JSON.stringify(game.fullI2Exact || {})),
    fullI2Cumulative: JSON.parse(JSON.stringify(game.fullI2Cumulative || {})),
    top2ScorePct: Number(game.top2ScorePct),
    bottom2ScorePct: Number(game.bottom2ScorePct),
  };
  game.baseballOnlyRaw = raw;

  const env = findEnvironment(game);
  if (!env) {
    missingTotalCount += 1;
    game.runEnvironmentConditioned = false;
    game.runEnvironment = { status: 'MISSING_APPROVED_PREGAME_FULL_GAME_TOTAL' };
    game.modelStatus = 'PENDING_FULL_GAME_TOTAL';
    for (const key of ['under05','over05','under05Pct','over05Pct','fairUnder','fairOver','fullI2Exact','fullI2Cumulative','top2ScorePct','bottom2ScorePct']) game[key] = null;
    continue;
  }

  const total = Number(env.fullGameTotal);
  const bucketKey = Number.isFinite(total) ? total.toFixed(1) : 'NaN';
  const bucket = prior.buckets?.[bucketKey];
  if (!bucket) {
    missingBucketCount += 1;
    game.runEnvironmentConditioned = false;
    game.runEnvironment = {
      status: 'MISSING_EXACT_TOTAL_PRIOR_BUCKET',
      fullGameTotal: total,
      source: totals.source || null,
      definition: totals.totalDefinition || null,
    };
    game.modelStatus = 'PENDING_TOTAL_PRIOR_BUCKET';
    for (const key of ['under05','over05','under05Pct','over05Pct','fairUnder','fairOver','fullI2Exact','fullI2Cumulative','top2ScorePct','bottom2ScorePct']) game[key] = null;
    continue;
  }

  const conditioned = conditionI2Projection({
    rawOver: raw.over05,
    rawTopOver: raw.top2ScorePct / 100,
    rawBottomOver: raw.bottom2ScorePct / 100,
    rawExact: fractionExact({ fullI2Exact: raw.fullI2Exact }),
    totalPriorOver: Number(bucket.over05),
    broadOver: Number(prior.broad.over05),
  });

  game.under05 = conditioned.under05;
  game.over05 = conditioned.over05;
  game.under05Pct = pct(conditioned.under05);
  game.over05Pct = pct(conditioned.over05);
  game.fairUnder = odds(conditioned.fairUnder);
  game.fairOver = odds(conditioned.fairOver);
  game.fullI2Exact = Object.fromEntries(Object.entries(conditioned.fullI2.exact).map(([k, v]) => [k, pct(v)]));
  game.fullI2Cumulative = Object.fromEntries(Object.entries(conditioned.fullI2.cumulative).map(([k, v]) => [k, pct(v)]));
  game.top2ScorePct = pct(conditioned.top2Score);
  game.bottom2ScorePct = pct(conditioned.bottom2Score);
  game.runEnvironmentConditioned = true;
  game.runEnvironment = {
    status: 'APPLIED',
    fullGameTotal: total,
    bucket: bucketKey,
    totalPriorOverPct: pct(bucket.over05),
    totalPriorUnderPct: pct(bucket.under05),
    totalPriorN: bucket.n,
    broadOverPct: pct(prior.broad.over05),
    broadUnderPct: pct(prior.broad.under05),
    totalSource: totals.source || null,
    totalDefinition: totals.totalDefinition || null,
    firstSeenAt: env.firstSeenAt || null,
    latestObservedTotal: env.latestObservedTotal ?? null,
    adjustmentMethod: conditioned.audit.method,
    baseballLogitDelta: conditioned.audit.baseballLogitDelta,
    equalHalfLogitShift: conditioned.audit.equalHalfLogitShift,
  };
  conditionedCount += 1;
}

const eligible = (payload.games || []).filter(game => game.runEnvironmentConditioned === true);
payload.baseballOnlyRawRanking = rawRanking;
payload.ranking = eligible
  .map(game => {
    const priorRow = rawRankingByGame.get(String(game.gamePk)) || {};
    return {
      ...priorRow,
      gamePk: game.gamePk,
      matchup: `${game.away} @ ${game.home}`,
      gameDate: game.gameDate,
      under05Pct: game.under05Pct,
      over05Pct: game.over05Pct,
      fairUnder: game.fairUnder,
      fairOver: game.fairOver,
      top2ScorePct: game.top2ScorePct,
      bottom2ScorePct: game.bottom2ScorePct,
      runEnvironment: game.runEnvironment,
    };
  })
  .sort((a, b) => Number(b.under05Pct) - Number(a.under05Pct))
  .map((row, index) => ({ ...row, rank: index + 1 }));

payload.model = 'MLB I2 Under/Over v0.3 Total-Conditioned Research Build';
payload.marketDataUsed = true;
payload.marketUseScope = 'FULL_GAME_TOTAL_POINT_ONLY';
payload.derivativeMarketDataUsed = false;
payload.i2PriceDataUsed = false;
payload.predictionFrozenBeforeDerivativeMarketRetrieval = true;
payload.runEnvironmentConditioning = {
  required: true,
  appliedGames: conditionedCount,
  missingApprovedTotalGames: missingTotalCount,
  missingPriorBucketGames: missingBucketCount,
  totalArtifact: TOTALS_PATH,
  priorArtifact: PRIOR_PATH,
  totalDefinition: totals.totalDefinition || null,
  totalSourceScope: totals.scope || null,
  broadPriorFallbackAllowed: false,
  method: 'Historical total-conditioned prior + baseball-only logit delta; exact positive-run tail reweighted proportionally; equal logit shift applied to Top2/Bottom2 scoring probabilities.',
};
payload.projectedGames = conditionedCount;
payload.pendingOrErroredGames = (payload.games || []).length - conditionedCount;
payload.knownResearchLimitations = Array.from(new Set([
  ...(payload.knownResearchLimitations || []),
  'Live conditioning currently uses the first observed DraftKings pregame total captured by the isolated run-environment endpoint; it is locked across reruns but is not guaranteed to equal the historical archive opening tick.',
]));

fs.writeFileSync(OUTPUT, JSON.stringify(payload, null, 2));
console.log(JSON.stringify({
  output: OUTPUT,
  conditionedCount,
  missingTotalCount,
  missingBucketCount,
  ranking: payload.ranking.map(r => ({ rank: r.rank, matchup: r.matchup, under05Pct: r.under05Pct, fairUnder: r.fairUnder, total: r.runEnvironment?.fullGameTotal }))
}, null, 2));
