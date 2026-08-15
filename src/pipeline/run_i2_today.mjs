import fs from 'node:fs';
import path from 'node:path';
import { simulateFullSecondInning } from '../model/i2_inning_model.js';
import { createSeededRandom, seedFromGameId } from '../model/seeded_random.js';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const CUTOFF = process.env.I2_CUTOFF || new Date().toISOString();
const SEASON = Number(DATE.slice(0, 4));
const TRIALS = Number(process.env.I2_TRIALS || 50000);
const OUTPUT = process.env.I2_OUTPUT || `data/runtime/i2/${DATE}_frozen_predictions.json`;
const CALIBRATION_PATH = process.env.I2_PLAY_CALIBRATION || 'data/derived/i2/i2_play_calibration.json';
const VENUE_PATH = process.env.I2_VENUE_PROFILES || `data/runtime/i2/savant_venue_profiles_${SEASON}_3yr.json`;

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

const playCalibration = loadJson(CALIBRATION_PATH);
const totalCalPAs = Object.values(playCalibration.event_counts || {}).reduce((a,b)=>a+Number(b||0),0);
const league = Object.fromEntries(EVENT_KEYS.map(k => [k, Number(playCalibration.event_counts?.[k] || 0) / totalCalPAs]));

let venueProfiles = [];
if (fs.existsSync(VENUE_PATH)) {
  try { venueProfiles = loadJson(VENUE_PATH); } catch {}
}

function norm(s) { return String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim(); }
function venueProfileFor(game) {
  if (!venueProfiles.length) return null;
  const venueName = norm(game.venue?.name);
  let p = venueProfiles.find(x => norm(x.venue_name) === venueName);
  if (p) return p;
  const homeName = norm(game.teams?.home?.team?.name || game.teams?.home?.team?.clubName);
  const homeAbbr = norm(game.teams?.home?.team?.abbreviation);
  p = venueProfiles.find(x => {
    const t = norm(x.team);
    return t && (homeName.endsWith(t) || t.endsWith(homeName) || (homeAbbr && t === homeAbbr));
  });
  return p || null;
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
        single: profile.handedness?.L?.single ?? 1,
        double: profile.handedness?.L?.double ?? 1,
        triple: profile.handedness?.L?.triple ?? 1,
        hr: profile.handedness?.L?.hr ?? 1,
      },
      R: {
        single: profile.handedness?.R?.single ?? 1,
        double: profile.handedness?.R?.double ?? 1,
        triple: profile.handedness?.R?.triple ?? 1,
        hr: profile.handedness?.R?.hr ?? 1,
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

const statCache = new Map();
async function seasonStats(personId, group) {
  const key = `${personId}:${group}`;
  if (!statCache.has(key)) {
    const url = `https://statsapi.mlb.com/api/v1/people/${personId}/stats?stats=season&group=${group}&season=${SEASON}`;
    statCache.set(key, fetchJson(url).then(getStatBlock).catch(e => ({__error:String(e)})));
  }
  return statCache.get(key);
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

async function buildStarter(feed, side) {
  const p = probableStarter(feed, side);
  if (!p?.id) return null;
  const stat = await seasonStats(p.id, 'pitching');
  const pc = pitcherCounts(stat);
  return {id:p.id,name:p.fullName,eventRatesAllowed:ratesFromCounts(pc.counts, pc.denom, 180),seasonBF:pc.denom,fallbackExtraBaseSplit:pc.fallbackExtraBaseSplit,rawStatsError:stat.__error || null,gamesStarted:n(stat,'gamesStarted'),era:stat?.era ?? null};
}

function pct(x){ return Math.round(x*10000)/100; }
function odds(x){ return x == null ? null : Math.round(x); }

async function runGame(game) {
  const gamePk = game.gamePk;
  const feed = await fetchJson(`https://statsapi.mlb.com/api/v1.1/game/${gamePk}/feed/live`);
  const awayBuilt = await buildLineup(feed,'away');
  const homeBuilt = await buildLineup(feed,'home');
  const awayStarter = await buildStarter(feed,'away');
  const homeStarter = await buildStarter(feed,'home');
  const venueProfile = venueProfileFor(game);
  const environmentalContext = sanitizeEnvironment(venueProfile);
  const base = {gamePk,gameDate:game.gameDate,away:game.teams?.away?.team?.name,home:game.teams?.home?.team?.name,venue:game.venue?.name,lineupConfirmed:awayBuilt.confirmed && homeBuilt.confirmed,awayLineup:awayBuilt.names,homeLineup:homeBuilt.names,awayStarter:awayStarter?.name || null,homeStarter:homeStarter?.name || null,venueProfile:venueProfile?.venue_name || null,status:game.status?.detailedState || null,dataAudit:{awayStarterBF:awayStarter?.seasonBF ?? null,homeStarterBF:homeStarter?.seasonBF ?? null,awayStarterGamesStarted:awayStarter?.gamesStarted ?? null,homeStarterGamesStarted:homeStarter?.gamesStarted ?? null,awayStarterExtraBaseFallback:awayStarter?.fallbackExtraBaseSplit ?? null,homeStarterExtraBaseFallback:homeStarter?.fallbackExtraBaseSplit ?? null,venueProfileMatched:Boolean(venueProfile)}};
  if (!awayBuilt.confirmed || !homeBuilt.confirmed) return {...base, modelStatus:'PENDING_CONFIRMED_LINEUP'};
  if (!awayStarter || !homeStarter) return {...base, modelStatus:'PENDING_PROBABLE_STARTER'};

  const random = createSeededRandom(seedFromGameId(String(gamePk), Number(DATE.replaceAll('-',''))));
  const result = simulateFullSecondInning({away:{lineup:awayBuilt.lineup, starter:awayStarter},home:{lineup:homeBuilt.lineup, starter:homeStarter},league,environmentalContext,weights:{batter:0.5,pitcher:0.5},trials:TRIALS,random,playCalibration});
  return {...base,modelStatus:'FROZEN_RESEARCH_PROJECTION',trials:TRIALS,under05:result.under05,over05:result.over05,under05Pct:pct(result.under05),over05Pct:pct(result.over05),fairUnder:odds(result.fairOdds.under05),fairOver:odds(result.fairOdds.over05),fullI2Exact:Object.fromEntries(Object.entries(result.fullI2.exact).map(([k,v])=>[k,pct(v)])),fullI2Cumulative:Object.fromEntries(Object.entries(result.fullI2.cumulative).map(([k,v])=>[k,pct(v)])),top2ScorePct:pct(result.top2.cumulative['1+']),bottom2ScorePct:pct(result.bottom2.cumulative['1+']),awayI2StartSlotPct:Object.fromEntries(Object.entries(result.stateDiagnostics.awayI2StartSlotProbability).map(([k,v])=>[k,pct(v)])),homeI2StartSlotPct:Object.fromEntries(Object.entries(result.stateDiagnostics.homeI2StartSlotProbability).map(([k,v])=>[k,pct(v)])),awayMeanPitchesEnteringI2:Math.round(result.stateDiagnostics.awayMeanPitchesEnteringI2*100)/100,homeMeanPitchesEnteringI2:Math.round(result.stateDiagnostics.homeMeanPitchesEnteringI2*100)/100};
}

async function main(){
  const schedule = await fetchJson(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${DATE}&hydrate=probablePitcher,team,venue`);
  const allGames = schedule?.dates?.flatMap(d=>d.games || []) || [];
  const cutoffMs = Date.parse(CUTOFF);
  const remaining = allGames.filter(g => Date.parse(g.gameDate) > cutoffMs && !['final','live'].includes(String(g.status?.abstractGameState || '').toLowerCase()));
  const games=[];
  for (const game of remaining) {
    try { games.push(await runGame(game)); }
    catch(e) { games.push({gamePk:game.gamePk,gameDate:game.gameDate,away:game.teams?.away?.team?.name,home:game.teams?.home?.team?.name,modelStatus:'ERROR',error:String(e)}); }
  }
  const ranked = games.filter(g=>g.modelStatus==='FROZEN_RESEARCH_PROJECTION').sort((a,b)=>b.under05-a.under05);
  ranked.forEach((g,i)=>g.underRank=i+1);
  const payload={model:'MLB I2 Under/Over v0.2 Research Build',date:DATE,cutoff:CUTOFF,generatedAt:new Date().toISOString(),trialsPerGame:TRIALS,marketDataUsed:false,weights:{batter:0.5,pitcher:0.5},leagueBaselineSource:'Retrosheet 2021-2025 pooled event counts from i2_play_calibration.json',parkSource:venueProfiles.length?'Baseball Savant 3-year Statcast park factors':'neutral fallback',currentPlayerSource:'MLB Stats API 2026 season-to-date basic batting/pitching statistics',knownResearchLimitations:['Advanced daily/as-of Statcast batter/pitcher interaction weights are not yet fitted.','Weather/roof is not yet applied by the v0.2 I2 simulator.','Conventional starters default to the probable starter for I2 unless an explicit opener/bulk mixture is supplied.','If MLB pitching stats omit doubles/triples allowed, non-HR hits are split using league event proportions and flagged in dataAudit.'],remainingGamesAtCutoff:remaining.length,projectedGames:ranked.length,pendingOrErroredGames:games.length-ranked.length,ranking:ranked.map(g=>({rank:g.underRank,gamePk:g.gamePk,matchup:`${g.away} @ ${g.home}`,gameDate:g.gameDate,under05Pct:g.under05Pct,over05Pct:g.over05Pct,fairUnder:g.fairUnder,fairOver:g.fairOver,awayStarter:g.awayStarter,homeStarter:g.homeStarter,top2ScorePct:g.top2ScorePct,bottom2ScorePct:g.bottom2ScorePct})),games};
  fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
  fs.writeFileSync(OUTPUT, JSON.stringify(payload,null,2));
  console.log(JSON.stringify(payload.ranking,null,2));
  console.log(`Wrote ${OUTPUT}`);
}

await main();
