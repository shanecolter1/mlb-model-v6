import fs from 'node:fs';
import path from 'node:path';
import { collectSources, resolveGameInputs, applyResolvedInputs } from '../inputs/i2_baseball_sources.mjs';
import { projectionGate, compactAudit, lineupDelta } from '../inputs/i2_source_governance.mjs';
import { simulateFullSecondInning, fairAmericanOdds } from '../model/i2_inning_model.js';
import { predictI2EventVector } from '../model/i2_vnext_event_model.js';
import { selectI2VenueProfile } from '../model/i2_venue_profile.js';
import { applyEnvironmentalEventVector } from '../event_probability_engine.js';
import { createSeededRandom, seedFromGameId } from '../model/seeded_random.js';
import { applyHalfScoreContrast, validateHalfContrastArtifact } from '../model/i2_half_contrast.js';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const CUTOFF = process.env.I2_CUTOFF || new Date().toISOString();
const SEASON = Number(DATE.slice(0, 4));
const TRIALS = Number(process.env.I2_TRIALS || 50000);
const OUTPUT = process.env.I2_OUTPUT || `data/runtime/i2/${DATE}_vnext_predictions.json`;
const I1_PRIOR_PATH = process.env.I2_I1_PRIOR || 'data/derived/i2/i2_play_calibration.json';
const TRANSITION_PATH = process.env.I2_TRANSITIONS || 'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json';
const VENUE_PATH = process.env.I2_VENUE_PROFILES || `data/runtime/i2/savant_venue_profiles_${SEASON}_3yr.json`;
const VNEXT_MODEL_PATH = process.env.I2_VNEXT_MODEL || 'data/derived/i2_vnext/i2_vnext_event_model_live.json';
const VNEXT_ARSENAL_PATH = process.env.I2_VNEXT_ARSENAL || 'data/derived/i2_vnext/live_arsenal_profile.json';
const VNEXT_CALIBRATION_PATH = process.env.I2_VNEXT_CALIBRATION || 'data/derived/i2_vnext/i2_vnext_full_calibration.json';
const VNEXT_HALF_CONTRAST_PATH = process.env.I2_VNEXT_HALF_CONTRAST || 'data/derived/i2_vnext/i2_vnext_half_contrast.json';

let baseballSources;
const priorFrozen = fs.existsSync(OUTPUT) ? JSON.parse(fs.readFileSync(OUTPUT, 'utf8')) : null;

const EVENT_KEYS = ['single','double','triple','home_run','walk','hit_by_pitch','strikeout','ball_in_play_out'];
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function fetchJson(url, attempts = 3) {
  let last;
  for (let i = 0; i < attempts; i += 1) {
    try {
      const r = await fetch(url, {headers:{accept:'application/json','user-agent':'MLB-I2-Research/0.2'}});
      if (!r.ok) throw new Error(`${r.status} ${r.statusText} ${url}`);
      return await r.json();
    } catch (e) {
      last = e;
      if (i + 1 < attempts) await sleep(300 * (i + 1));
    }
  }
  throw last;
}

function loadJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

const i1PriorCalibration = loadJson(I1_PRIOR_PATH);
const rawTransitionArtifact = loadJson(TRANSITION_PATH);
function adaptPlayCalibration(payload, pitchCountPmf = {}) {
  if (payload?.base_transitions) return payload;
  if (!payload?.states) throw new Error('Unsupported I2 transition artifact');
  const eventMap = {
    strikeout:'out',
    ball_in_play_out:'out',
    walk:'bb',
    hit_by_pitch:'bb',
    single:'single',
    double:'double',
    triple:'triple',
    home_run:'hr',
  };
  const base_transitions = {};
  for (const [event, source] of Object.entries(eventMap)) {
    for (let outs=0; outs<3; outs+=1) {
      for (let mask=0; mask<8; mask+=1) {
        base_transitions[`${event}|${outs}|${mask}`] = payload.states[`${source}|${outs}|${mask}`] || [];
      }
    }
  }
  return {
    version:`adapted-${payload.version || 'validated-transitions'}`,
    base_transitions,
    pitch_count_pmf:pitchCountPmf || {},
    governance:{
      sourceTrainingYears:payload.training_years || null,
      selectionYear:payload.selection_year || null,
      lockedValidationYear:payload.locked_validation_year || null,
    },
  };
}
const playCalibration = adaptPlayCalibration(
  rawTransitionArtifact,
  i1PriorCalibration.pitch_count_pmf || {}
);
const vnextEventModel = loadJson(VNEXT_MODEL_PATH);
const vnextArsenal = loadJson(VNEXT_ARSENAL_PATH);
const vnextFullCalibration = loadJson(VNEXT_CALIBRATION_PATH);
const vnextHalfContrast = fs.existsSync(VNEXT_HALF_CONTRAST_PATH)
  ? validateHalfContrastArtifact(loadJson(VNEXT_HALF_CONTRAST_PATH))
  : null;
const prospectiveValidationStart = [
  vnextFullCalibration?.holdout_policy?.prospective_validation_start,
  vnextHalfContrast?.enabled !== false ? vnextHalfContrast?.prospective_validation_start : null,
].filter(Boolean).sort().at(-1) || null;

function clamp01(x) { return Math.max(1e-9, Math.min(1 - 1e-9, Number(x))); }
function applyFinalCalibration(rawProbability) {
  const raw = clamp01(rawProbability);
  const curve = vnextFullCalibration?.final_curve || vnextFullCalibration?.selected || {type:'identity'};
  const type = String(curve.type || curve.method || 'identity');
  if (type === 'identity' || type === 'none') return raw;
  if (type === 'sigmoid' || type === 'sigmoid_logit') {
    const intercept = Number(curve.intercept || 0);
    const slope = Number(curve.slope ?? 1);
    const z = Math.log(raw / (1 - raw));
    return clamp01(1 / (1 + Math.exp(-(intercept + slope * z))));
  }
  if (type === 'isotonic') {
    const xs = curve.x_thresholds || [];
    const ys = curve.y_thresholds || [];
    if (!xs.length || xs.length !== ys.length) throw new Error('Invalid vNext isotonic calibration artifact');
    if (raw <= xs[0]) return clamp01(ys[0]);
    if (raw >= xs[xs.length - 1]) return clamp01(ys[ys.length - 1]);
    for (let i = 1; i < xs.length; i += 1) {
      if (raw <= xs[i]) {
        const t = (raw - xs[i - 1]) / Math.max(1e-12, xs[i] - xs[i - 1]);
        return clamp01(ys[i - 1] + t * (ys[i] - ys[i - 1]));
      }
    }
  }
  throw new Error(`Unsupported vNext calibration type ${type}`);
}
const totalCalPAs = Object.values(i1PriorCalibration.event_counts || {}).reduce((a,b)=>a+Number(b||0),0);
const league = Object.fromEntries(EVENT_KEYS.map(k => [k, Number(i1PriorCalibration.event_counts?.[k] || 0) / totalCalPAs]));

let venueProfiles = [];
if (fs.existsSync(VENUE_PATH)) {
  try { venueProfiles = loadJson(VENUE_PATH); } catch {}
}

function venueProfileFor(game) {
  return selectI2VenueProfile(game, venueProfiles);
}

function venueStatusFor(profile) {
  if (!venueProfiles.length) return 'SAVANT_PROFILE_SET_MISSING';
  return profile ? 'SAVANT_PROFILE_MATCHED' : 'SAVANT_PROFILE_UNMATCHED_NEUTRAL_FALLBACK';
}

function sanitizeEnvironment(profile) {
  if (!profile) return null;
  return {
    multipliers: {
      single: profile.multipliers?.single ?? 1,
      double: profile.multipliers?.double ?? 1,
      triple: profile.multipliers?.triple ?? 1,
      hr: profile.multipliers?.hr ?? 1,
    },
    handedness: {
      L: {
        single: profile.handedness?.L?.single ?? null,
        double: profile.handedness?.L?.double ?? null,
        triple: profile.handedness?.L?.triple ?? null,
        hr: profile.handedness?.L?.hr ?? null,
      },
      R: {
        single: profile.handedness?.R?.single ?? null,
        double: profile.handedness?.R?.double ?? null,
        triple: profile.handedness?.R?.triple ?? null,
        hr: profile.handedness?.R?.hr ?? null,
      },
    },
    audit: profile.audit || null,
    venue_name: profile.venue_name,
  };
}

function getStatBlock(payload) {
  return payload?.stats?.[0]?.splits?.[0]?.stat || {};
}

function n(stat, ...keys) {
  for (const key of keys) {
    const v = stat?.[key];
    if (v !== undefined && v !== null && v !== '') {
      const x = Number(v);
      if (Number.isFinite(x)) return x;
    }
  }
  return 0;
}

function maybeNumber(stat, ...keys) {
  for (const key of keys) {
    const v = stat?.[key];
    if (v !== undefined && v !== null && v !== '') {
      const x = Number(v);
      if (Number.isFinite(x)) return x;
    }
  }
  return null;
}

function maybeRaw(stat, ...keys) {
  for (const key of keys) {
    const v = stat?.[key];
    if (v !== undefined && v !== null && v !== '') return v;
  }
  return null;
}

function ratesFromCounts(counts, denom, priorStrength) {
  const d = Math.max(0, Number(denom || 0));
  const strength = Math.max(0, priorStrength);
  const out = {};
  for (const k of EVENT_KEYS) {
    const c = Math.max(0, Number(counts[k] || 0));
    out[k] = (c + league[k] * strength) / (d + strength || 1);
  }
  const sum = EVENT_KEYS.reduce((s,k)=>s+out[k],0);
  for (const k of EVENT_KEYS) out[k] /= sum;
  return out;
}

function hitterCounts(stat) {
  const pa = n(stat,'plateAppearances');
  const h = n(stat,'hits');
  const d2 = n(stat,'doubles');
  const d3 = n(stat,'triples');
  const hr = n(stat,'homeRuns');
  const bb = n(stat,'baseOnBalls','walks');
  const hbp = n(stat,'hitByPitch');
  const so = n(stat,'strikeOuts');
  const single = Math.max(0, h - d2 - d3 - hr);
  const residual = Math.max(0, pa - (single+d2+d3+hr+bb+hbp+so));
  return {denom:pa, counts:{single,double:d2,triple:d3,home_run:hr,walk:bb,hit_by_pitch:hbp,strikeout:so,ball_in_play_out:residual}};
}

function pitcherCounts(stat) {
  const bf = n(stat,'battersFaced');
  const h = n(stat,'hits');
  let d2 = n(stat,'doubles');
  let d3 = n(stat,'triples');
  const hr = n(stat,'homeRuns');
  const bb = n(stat,'baseOnBalls','walks');
  const hbp = n(stat,'hitBatsmen','hitByPitch');
  const so = n(stat,'strikeOuts');
  let fallbackExtraBaseSplit = false;
  if (h > 0 && d2 === 0 && d3 === 0) {
    const nonHrHits = Math.max(0, h - hr);
    const denom = Math.max(league.single + league.double + league.triple, 1e-9);
    d2 = nonHrHits * league.double / denom;
    d3 = nonHrHits * league.triple / denom;
    fallbackExtraBaseSplit = true;
  }
  const single = Math.max(0, h - d2 - d3 - hr);
  const residual = Math.max(0, bf - (single+d2+d3+hr+bb+hbp+so));
  return {denom:bf, fallbackExtraBaseSplit, counts:{single,double:d2,triple:d3,home_run:hr,walk:bb,hit_by_pitch:hbp,strikeout:so,ball_in_play_out:residual}};
}

function starterSeasonSnapshot(stat, pc) {
  const wins = maybeNumber(stat, 'wins');
  const losses = maybeNumber(stat, 'losses');
  return {
    wins,
    losses,
    record: wins !== null && losses !== null ? `${wins}-${losses}` : null,
    era: maybeRaw(stat, 'era'),
    whip: maybeRaw(stat, 'whip'),
    inningsPitched: maybeRaw(stat, 'inningsPitched'),
    strikeOuts: maybeNumber(stat, 'strikeOuts'),
    walks: maybeNumber(stat, 'baseOnBalls', 'walks'),
    homeRunsAllowed: maybeNumber(stat, 'homeRuns'),
    battersFaced: pc?.denom ?? maybeNumber(stat, 'battersFaced'),
    gamesStarted: maybeNumber(stat, 'gamesStarted'),
  };
}

const statCache = new Map();
async function seasonStats(personId, group) {
  const key = `${personId}:${group}`;
  if (!statCache.has(key)) {
    const url = `https://statsapi.mlb.com/api/v1/people/${personId}/stats?stats=season&group=${group}&season=${SEASON}`;
    statCache.set(key, fetchJson(url).then(getStatBlock).catch(e => ({__error:String(e)})));
  }
  return statCache.get(key);
}

const gameLogCache = new Map();
async function pitchingGameLog(personId) {
  const key = `${personId}:pitchingGameLog:${SEASON}`;
  if (!gameLogCache.has(key)) {
    const url = `https://statsapi.mlb.com/api/v1/people/${personId}/stats?stats=gameLog&group=pitching&season=${SEASON}`;
    gameLogCache.set(key, fetchJson(url).then(x => x?.stats?.[0]?.splits || []).catch(() => []));
  }
  return gameLogCache.get(key);
}
function inningsOuts(raw) {
  const [wholeText='0', outsText='0'] = String(raw ?? '0').split('.');
  const whole = Number(wholeText) || 0;
  const outs = Math.max(0, Math.min(2, Number(outsText) || 0));
  return whole * 3 + outs;
}
async function estimateOpenerI2Survival(personId) {
  const rows = await pitchingGameLog(personId);
  const cutoff = Date.parse(`${DATE}T00:00:00Z`);
  const recent = rows
    .filter(x => {
      const t = Date.parse(x.date || x.game?.gameDate || '');
      const gs = Number(x.stat?.gamesStarted || 0);
      const outs = inningsOuts(x.stat?.inningsPitched);
      const ageDays = Number.isFinite(t) ? (cutoff - t) / 86400000 : Infinity;
      return gs > 0 && ageDays > 0 && ageDays <= 90 && outs <= 9;
    })
    .sort((a,b)=>Date.parse(b.date || b.game?.gameDate || '')-Date.parse(a.date || a.game?.gameDate || ''))
    .slice(0,8);
  if (recent.length < 3) {
    return {status:'INSUFFICIENT_OPENER_HISTORY',n:recent.length,probability:null};
  }
  const reached = recent.filter(x => inningsOuts(x.stat?.inningsPitched) > 3).length;
  return {
    status:'EMPIRICAL_RECENT_SHORT_STARTS',
    n:recent.length,
    reachedI2:reached,
    probability:reached/recent.length,
    sample:recent.map(x=>({date:x.date || x.game?.gameDate || null,inningsPitched:x.stat?.inningsPitched ?? null})),
  };
}

function lineupFromFeed(feed, side) {
  const team = feed?.liveData?.boxscore?.teams?.[side];
  const players = Object.values(team?.players || {});
  return players.filter(p => Number(p.battingOrder) > 0).sort((a,b)=>Number(a.battingOrder)-Number(b.battingOrder)).slice(0,9);
}

function probableStarter(feed, side) {
  return feed?.gameData?.probablePitchers?.[side] || null;
}

async function buildLineup(feed, side) {
  const rows = lineupFromFeed(feed, side);
  if (rows.length !== 9) return {confirmed:false, lineup:[], names:[]};
  const lineup = [];
  for (const p of rows) {
    const id = p.person?.id;
    const stat = await seasonStats(id, 'hitting');
    const hc = hitterCounts(stat);
    const sideCode = feed?.gameData?.players?.[`ID${id}`]?.batSide?.code || 'R';
    lineup.push({id,name:p.person?.fullName,side:sideCode,eventRates:ratesFromCounts(hc.counts, hc.denom, 100),seasonPA:hc.denom,rawStatsError:stat.__error || null});
  }
  return {confirmed:true, lineup, names: lineup.map(x=>x.name)};
}

async function buildPitcherById(feed, id, name = null) {
  if (!id) return null;
  const stat = await seasonStats(id, 'pitching');
  const pc = pitcherCounts(stat);
  const seasonSnapshot = starterSeasonSnapshot(stat, pc);
  const person = feed?.gameData?.players?.[`ID${id}`] || {};
  return {
    id:Number(id),
    name:name || person.fullName || String(id),
    throws:person?.pitchHand?.code || 'R',
    eventRatesAllowed:ratesFromCounts(pc.counts, pc.denom, 180),
    seasonBF:pc.denom,
    fallbackExtraBaseSplit:pc.fallbackExtraBaseSplit,
    rawStatsError:stat.__error || null,
    gamesStarted:seasonSnapshot.gamesStarted ?? n(stat,'gamesStarted'),
    era:seasonSnapshot.era,
    seasonStats:seasonSnapshot,
  };
}
async function buildStarter(feed, side) {
  const p = probableStarter(feed, side);
  return p?.id ? buildPitcherById(feed, p.id, p.fullName) : null;
}

async function buildI2PitchingPlan(feed, side, starter, auditSide) {
  const plan = auditSide?.pitchingPlan || {role:'NORMAL_STARTER'};
  const restriction = (auditSide?.news || []).find(n =>
    /innings limit|pitch.?count|rehab|abbreviated start|return.*(IL|injur)/i.test(String(n.reason || ''))
  );
  if (restriction && plan.role === 'NORMAL_STARTER') {
    return {
      status:'STARTER_RESTRICTION_UNRESOLVED',
      eligible:false,
      mixture:null,
      audit:{restriction:restriction.reason || null},
    };
  }
  if (plan.role !== 'OPENER_BULK') {
    return {status:'NORMAL_STARTER',eligible:true,mixture:null,audit:null};
  }
  const openerId = Number(plan.resolvedOpenerMlbId || starter?.id || 0);
  const bulkId = Number(plan.resolvedBulkMlbId || 0);
  if (!openerId || !bulkId) {
    return {
      status:'OPENER_BULK_IDENTITY_UNRESOLVED',
      eligible:false,
      mixture:null,
      audit:{openerId:openerId || null,bulkId:bulkId || null},
    };
  }
  const survival = await estimateOpenerI2Survival(openerId);
  if (survival.probability == null) {
    return {
      status:survival.status,
      eligible:false,
      mixture:null,
      audit:{...survival,openerId,bulkId},
    };
  }
  const opener = openerId === starter?.id
    ? starter
    : await buildPitcherById(feed, openerId, plan.resolvedOpenerMlbName || plan.opener);
  const bulk = await buildPitcherById(feed, bulkId, plan.resolvedBulkMlbName || plan.primaryBulkPitcher);
  if (!opener || !bulk) {
    return {
      status:'OPENER_BULK_PITCHER_BUILD_FAILED',
      eligible:false,
      mixture:null,
      audit:{...survival,openerId,bulkId},
    };
  }
  const p = Math.max(0, Math.min(1, Number(survival.probability)));
  return {
    status:'OPENER_BULK_EMPIRICAL_MIXTURE',
    eligible:true,
    mixture:[
      {weight:p,pitcher:opener},
      {weight:1-p,pitcher:bulk},
    ].filter(x=>x.weight>0),
    audit:{...survival,openerId,bulkId,opener:opener.name,bulk:bulk.name},
  };
}

function pct(x){ return Math.round(x*10000)/100; }
function odds(x){ return x == null ? null : Math.round(x); }

async function runGame(game) {
  const gamePk = game.gamePk;
  const originalFeed = await fetchJson(`https://statsapi.mlb.com/api/v1.1/game/${gamePk}/feed/live`);
  const previous = priorFrozen?.games?.find(g => String(g.gamePk) === String(gamePk))?.inputAudit;
  const inputAudit = await resolveGameInputs(game, originalFeed, baseballSources, previous);
  inputAudit.previousProjectedVsCurrent = previous ? Object.fromEntries(['away','home'].map(side => [side,{previousStatus:previous[side].lineup.status,previousTimestamp:previous[side].lineup.timestamp,currentStatus:inputAudit[side].lineup.status,...lineupDelta(previous[side].lineup.players,inputAudit[side].lineup.players)}])) : null;
  let feed;
  try { feed = await applyResolvedInputs(originalFeed, inputAudit); }
  catch (error) { return {gamePk, gameDate:game.gameDate, away:game.teams?.away?.team?.name, home:game.teams?.home?.team?.name, inputAudit, modelStatus:'PENDING_INPUT_IDENTITY', bettingEligibility:{eligible:false,status:'NO_ACTIONABLE_RECOMMENDATION',reasons:['UNRESOLVED_MLB_IDENTITY']}, error:String(error)}; }
  const awayBuilt = await buildLineup(feed,'away');
  const homeBuilt = await buildLineup(feed,'home');
  const awayStarter = await buildStarter(feed,'away');
  const homeStarter = await buildStarter(feed,'home');
  const awayPitchingPlan = awayStarter ? await buildI2PitchingPlan(feed,'away',awayStarter,inputAudit.away) : null;
  const homePitchingPlan = homeStarter ? await buildI2PitchingPlan(feed,'home',homeStarter,inputAudit.home) : null;
  const venueProfile = venueProfileFor(game);
  const venueStatus = venueStatusFor(venueProfile);
  const environmentalContext = sanitizeEnvironment(venueProfile);
  const base = {
    gamePk,
    gameDate:game.gameDate,
    away:game.teams?.away?.team?.name,
    home:game.teams?.home?.team?.name,
    venue:game.venue?.name,
    lineupConfirmed:['away','home'].every(side => inputAudit[side].lineup.status.startsWith('CONFIRMED')),
    inputAudit,
    predictionClass:['away','home'].every(side=>inputAudit[side].lineup.status.startsWith('CONFIRMED')) ? 'CONFIRMED_INPUTS' : 'PROVISIONAL_EXPECTED_INPUTS',
    sourceEligibility:inputAudit.gate,
    bettingEligibility:{
      eligible:false,
      status:'SHADOW_ONLY_NOT_PROMOTED',
      reasons:['VNEXT_PROSPECTIVE_VALIDATION_REQUIRED'],
    },
    awayLineup:awayBuilt.names,
    homeLineup:homeBuilt.names,
    awayStarter:awayStarter?.name || null,
    homeStarter:homeStarter?.name || null,
    awayStarterStats:awayStarter?.seasonStats || null,
    homeStarterStats:homeStarter?.seasonStats || null,
    venueProfile:venueProfile?.venue_name || null,
    venueStatus,
    status:game.status?.detailedState || null,
    dataAudit:{
      awayStarterBF:awayStarter?.seasonBF ?? null,
      homeStarterBF:homeStarter?.seasonBF ?? null,
      awayStarterGamesStarted:awayStarter?.gamesStarted ?? null,
      homeStarterGamesStarted:homeStarter?.gamesStarted ?? null,
      awayStarterExtraBaseFallback:awayStarter?.fallbackExtraBaseSplit ?? null,
      homeStarterExtraBaseFallback:homeStarter?.fallbackExtraBaseSplit ?? null,
      awayStarterStatsError:awayStarter?.rawStatsError ?? null,
      homeStarterStatsError:homeStarter?.rawStatsError ?? null,
      venueProfileMatched:Boolean(venueProfile),
      venueMatchRule:'EXACT_NORMALIZED_VENUE_NAME_ONLY',
      venueStatus,
      venueFallbackApplied:!venueProfile,
      awayI2PitchingPlan:awayPitchingPlan?.status || null,
      homeI2PitchingPlan:homePitchingPlan?.status || null,
      awayI2PitchingPlanAudit:awayPitchingPlan?.audit || null,
      homeI2PitchingPlanAudit:homePitchingPlan?.audit || null,
    }
  };
  if (!awayBuilt.confirmed || !homeBuilt.confirmed) return {...base, modelStatus:'PENDING_CONFIRMED_LINEUP'};
  if (!awayStarter || !homeStarter) return {...base, modelStatus:'PENDING_PROBABLE_STARTER'};
  if (!awayPitchingPlan?.eligible || !homePitchingPlan?.eligible) {
    return {
      ...base,
      modelStatus:'PENDING_I2_PITCHING_PLAN',
      bettingEligibility:{
        eligible:false,
        status:'NO_ACTIONABLE_RECOMMENDATION',
        reasons:[
          ...(!awayPitchingPlan?.eligible ? [awayPitchingPlan?.status || 'AWAY_I2_PITCHING_PLAN_UNRESOLVED'] : []),
          ...(!homePitchingPlan?.eligible ? [homePitchingPlan?.status || 'HOME_I2_PITCHING_PLAN_UNRESOLVED'] : []),
        ],
      },
    };
  }

  const random = createSeededRandom(seedFromGameId(String(gamePk), Number(DATE.replaceAll('-',''))));
  const i2EventVectorProvider = ({batter, pitcher}) => {
    const pitcherThrows = pitcher.throws || 'R';
    const batterSide = batter.side === 'S'
      ? (pitcherThrows === 'L' ? 'R' : 'L')
      : (batter.side || 'R');
    const neutral = predictI2EventVector({
      batterId: batter.id,
      pitcherId: pitcher.id,
      batterSide,
      pitcherThrows,
      model: vnextEventModel,
      arsenalProfile: vnextArsenal,
    });
    return applyEnvironmentalEventVector({
      neutralVector: neutral,
      environmentalContext,
      batterSide,
    }).probabilities;
  };
  const result = simulateFullSecondInning({
    away:{lineup:awayBuilt.lineup, starter:awayStarter, i2PitcherMixture:awayPitchingPlan.mixture},
    home:{lineup:homeBuilt.lineup, starter:homeStarter, i2PitcherMixture:homePitchingPlan.mixture},
    league,
    environmentalContext,
    weights:{batter:0.5,pitcher:0.5},
    trials:TRIALS,
    random,
    playCalibration,
    i2EventVectorProvider,
  });
  const rawUnder05 = result.under05;
  const rawTopScoreProbability = result.top2.cumulative['1+'];
  const rawBottomScoreProbability = result.bottom2.cumulative['1+'];
  const halfContrastApplicable = Boolean(
    vnextHalfContrast && vnextHalfContrast.enabled !== false && venueProfile
  );
  const halfContrast = halfContrastApplicable
    ? applyHalfScoreContrast({
        topScoreProbability:rawTopScoreProbability,
        bottomScoreProbability:rawBottomScoreProbability,
        artifact:vnextHalfContrast,
      })
    : {
        rawTopScoreProbability,
        rawBottomScoreProbability,
        adjustedTopScoreProbability:rawTopScoreProbability,
        adjustedBottomScoreProbability:rawBottomScoreProbability,
        adjustedUnder05:rawUnder05,
        h:0,
      };
  const halfAdjustedUnder05 = halfContrastApplicable ? halfContrast.adjustedUnder05 : rawUnder05;
  if (!venueProfile) {
    base.bettingEligibility = {
      eligible:false,
      status:'SHADOW_INPUT_INCOMPLETE',
      reasons:['SAVANT_VENUE_PROFILE_MISSING_NEUTRAL_FALLBACK'],
    };
  }
  const finalUnder05 = applyFinalCalibration(halfAdjustedUnder05);
  const finalOver05 = 1 - finalUnder05;
  const finalFairUnder = fairAmericanOdds(finalUnder05);
  const finalFairOver = fairAmericanOdds(finalOver05);
  // A fresh simulation of the resolved identities satisfies a prior invalidation.
  inputAudit.previousProjectionInvalidations = inputAudit.gate.invalidations;
  inputAudit.gate = projectionGate(inputAudit);
  base.sourceEligibility = inputAudit.gate;
  if (!venueProfile) {
    base.bettingEligibility = {
      eligible:false,
      status:'SHADOW_INPUT_INCOMPLETE',
      reasons:[
        'VNEXT_PROSPECTIVE_VALIDATION_REQUIRED',
        'SAVANT_VENUE_PROFILE_MISSING_NEUTRAL_FALLBACK',
      ],
    };
  } else {
    base.bettingEligibility = {
      eligible:false,
      status:'SHADOW_ONLY_NOT_PROMOTED',
      reasons:['VNEXT_PROSPECTIVE_VALIDATION_REQUIRED'],
    };
  }
  return {...base,modelStatus:'FROZEN_VNEXT_SHADOW_PROJECTION',trials:TRIALS,rawUnder05,rawOver05:1-rawUnder05,halfAdjustedUnder05,halfAdjustedOver05:1-halfAdjustedUnder05,halfContrastApplied:halfContrastApplicable,halfContrastH:halfContrast.h,under05:finalUnder05,over05:finalOver05,under05Pct:pct(finalUnder05),over05Pct:pct(finalOver05),rawUnder05Pct:pct(rawUnder05),halfAdjustedUnder05Pct:pct(halfAdjustedUnder05),fairUnder:odds(finalFairUnder),fairOver:odds(finalFairOver),rawFullI2Exact:Object.fromEntries(Object.entries(result.fullI2.exact).map(([k,v])=>[k,pct(v)])),rawFullI2Cumulative:Object.fromEntries(Object.entries(result.fullI2.cumulative).map(([k,v])=>[k,pct(v)])),rawTop2Exact:Object.fromEntries(Object.entries(result.top2.exact).map(([k,v])=>[k,pct(v)])),rawTop2Cumulative:Object.fromEntries(Object.entries(result.top2.cumulative).map(([k,v])=>[k,pct(v)])),rawBottom2Exact:Object.fromEntries(Object.entries(result.bottom2.exact).map(([k,v])=>[k,pct(v)])),rawBottom2Cumulative:Object.fromEntries(Object.entries(result.bottom2.cumulative).map(([k,v])=>[k,pct(v)])),rawTop2ScoreProbability,rawBottom2ScoreProbability,top2ScoreProbability:halfContrast.adjustedTopScoreProbability,bottom2ScoreProbability:halfContrast.adjustedBottomScoreProbability,rawTop2ScorePct:pct(rawTopScoreProbability),rawBottom2ScorePct:pct(rawBottomScoreProbability),top2ScorePct:pct(halfContrast.adjustedTopScoreProbability),bottom2ScorePct:pct(halfContrast.adjustedBottomScoreProbability),awayI2StartSlotPct:Object.fromEntries(Object.entries(result.stateDiagnostics.awayI2StartSlotProbability).map(([k,v])=>[k,pct(v)])),homeI2StartSlotPct:Object.fromEntries(Object.entries(result.stateDiagnostics.homeI2StartSlotProbability).map(([k,v])=>[k,pct(v)])),awayMeanPitchesEnteringI2:Math.round(result.stateDiagnostics.awayMeanPitchesEnteringI2*100)/100,homeMeanPitchesEnteringI2:Math.round(result.stateDiagnostics.homeMeanPitchesEnteringI2*100)/100};
}

async function main(){
  baseballSources = await collectSources(DATE);
  const schedule = await fetchJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${DATE}&hydrate=probablePitcher,team,venue`);
  const allGames = schedule?.dates?.flatMap(d=>d.games || []) || [];
  const cutoffMs = Date.parse(CUTOFF);
  const remaining = allGames.filter(g => Date.parse(g.gameDate) > cutoffMs && !['final','live'].includes(String(g.status?.abstractGameState || '').toLowerCase()));
  const games=[];
  for (const game of remaining) {
    try { games.push(await runGame(game)); }
    catch(e) { games.push({gamePk:game.gamePk,gameDate:game.gameDate,away:game.teams?.away?.team?.name,home:game.teams?.home?.team?.name,modelStatus:'ERROR',error:String(e)}); }
  }
  // Recheck immediately before freeze; changed inputs must get a clean subsequent rerun.
  const finalSources = await collectSources(DATE);
  for (const projected of games) {
    if (!projected.inputAudit) continue;
    try {
      const scheduled = remaining.find(g => g.gamePk === projected.gamePk);
      const freshFeed = await fetchJson(`https://statsapi.mlb.com/api/v1.1/game/${projected.gamePk}/feed/live`);
      const checked = await resolveGameInputs(scheduled, freshFeed, finalSources, projected.inputAudit);
      // Re-resolve current source names to team-scoped MLB IDs before deciding whether
      // the frozen inputs changed. Ambiguous/unresolved identities fail closed here.
      await applyResolvedInputs(freshFeed, checked);
      checked.gate = projectionGate({...checked,previous:projected.inputAudit});
      projected.inputAudit.freezeCheck = {checkedAt:new Date().toISOString(), gate:checked.gate};
      projected.sourceEligibility = checked.gate;
      projected.bettingEligibility = projected.modelStatus === 'FROZEN_VNEXT_SHADOW_PROJECTION'
        ? (projected.dataAudit?.venueProfileMatched
            ? {eligible:false,status:'SHADOW_ONLY_NOT_PROMOTED',reasons:['VNEXT_PROSPECTIVE_VALIDATION_REQUIRED']}
            : {eligible:false,status:'SHADOW_INPUT_INCOMPLETE',reasons:['VNEXT_PROSPECTIVE_VALIDATION_REQUIRED','SAVANT_VENUE_PROFILE_MISSING_NEUTRAL_FALLBACK']})
        : {eligible:false,status:'NO_ACTIONABLE_RECOMMENDATION',reasons:['MODEL_UNAVAILABLE']};
      if (checked.gate.requiresCleanRerun) projected.modelStatus = 'PROJECTION_INVALIDATED';
      projected.inputAudit.confirmationAudit = {away:checked.away.lineup.audit,home:checked.home.lineup.audit};
    } catch { projected.bettingEligibility = {eligible:false,status:'NO_ACTIONABLE_RECOMMENDATION',reasons:['PREFREEZE_RECHECK_FAILED']}; }
  }
  const ranked = games.filter(g=>g.modelStatus==='FROZEN_VNEXT_SHADOW_PROJECTION').sort((a,b)=>b.under05-a.under05);
  ranked.forEach((g,i)=>g.underRank=i+1);
  const payload={
    model:'MLB I2 vNext Direct Talent Shadow',
    promotionStatus:'SHADOW_ONLY_PROSPECTIVE_VALIDATION_REQUIRED',
    prospectiveValidationStart,
    baseballDataArchitecture:'SELF_CONTAINED_REPOSITORY_PIPELINE',
    primaryBaseballSource:'MLB Stats API direct',
    provisionalLineupSource:'RotoWire public daily-lineups HTML',
    netlifyMlbProxyUsed:false,
    paidRotowireApiUsed:false,
    date:DATE,
    cutoff:CUTOFF,
    generatedAt:new Date().toISOString(),
    trialsPerGame:TRIALS,
    marketDataUsed:false,
    transitionModel:playCalibration.version,
    transitionGovernance:playCalibration.governance || null,
    i1StateEngine:'existing V6 season-rate engine used only to simulate I1 lineup progression',
    i2TalentEngine:'direct I2 Statcast PA model; jointly regularized batter/pitcher/platoon/arsenal',
    halfCalibration:vnextHalfContrast ? {version:vnextHalfContrast.version,type:vnextHalfContrast.type,enabled:vnextHalfContrast.enabled !== false,h:vnextHalfContrast.zero_sum_half_contrast_h,status:vnextHalfContrast.status,matchedHomeVenueOnly:vnextHalfContrast.matched_home_venue_only === true,prospectiveValidationStart:vnextHalfContrast.enabled !== false ? vnextHalfContrast.prospective_validation_start : null} : null,
    finalCalibrationVersion:vnextFullCalibration?.version || null,
    probabilityStackVersion:vnextFullCalibration?.prospective_cohort_version || null,
    finalCalibration:vnextFullCalibration?.final_curve || vnextFullCalibration?.selected || null,
    leagueBaselineSource:'Retrosheet 2021-2025 pooled event counts from i2_play_calibration.json',
    parkSource:venueProfiles.length?'Baseball Savant 3-year Statcast park factors':'neutral fallback',
    venueMatchRule:'EXACT_NORMALIZED_VENUE_NAME_ONLY',
    currentPlayerSource:'I2: fitted direct Statcast I2 player effects + Savant arsenal; I1 only: MLB Stats API season-to-date rates',
    starterStatsPersisted:true,
    starterStatsFields:['wins','losses','record','era','whip','inningsPitched','strikeOuts','walks','homeRunsAllowed','battersFaced','gamesStarted'],
    knownResearchLimitations:[
      'Weather/roof is not yet applied.',
      'Missing Savant venue profiles use an explicit neutral fallback and are fail-closed for betting eligibility.',
      'Normal starters default to the probable starter for I2; confirmed opener/bulk identities must be supplied by the input workflow.',
      'Multiyear 2022-2026 validation rejected an added half-inning calibration layer; raw top/bottom probabilities feed the single final full-I2 calibration layer.',
      'Full-I2 calibration is validated on the normal-starter path; nonstandard pitching plans are not silently assigned the same calibration evidence.',
      'MLB season-to-date rates remain in I1 only to generate the I2 starting-position distribution; they are not I2 talent inputs.'
    ],
    remainingGamesAtCutoff:remaining.length,
    projectedGames:ranked.length,
    pendingOrErroredGames:games.length-ranked.length,
    ranking:ranked.map(g=>({
      rank:g.underRank,
      bettingEligibility:g.bettingEligibility,
      inputAudit:g.inputAudit,
      gamePk:g.gamePk,
      matchup:`${g.away} @ ${g.home}`,
      gameDate:g.gameDate,
      under05Pct:g.under05Pct,
      over05Pct:g.over05Pct,
      fairUnder:g.fairUnder,
      fairOver:g.fairOver,
      awayStarter:g.awayStarter,
      homeStarter:g.homeStarter,
      awayStarterStats:g.awayStarterStats,
      homeStarterStats:g.homeStarterStats,
      top2ScorePct:g.top2ScorePct,
      bottom2ScorePct:g.bottom2ScorePct
    })),
    inputSourceAudit:games.map(compactAudit),
    games
  };
  fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
  fs.writeFileSync(OUTPUT, JSON.stringify(payload,null,2));
  console.log(JSON.stringify(payload.ranking,null,2));
  console.log(`Wrote ${OUTPUT}`);
}

await main();
