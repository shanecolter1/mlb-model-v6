import fs from 'node:fs';

const overridePath = String(process.env.I2_LINEUP_OVERRIDES || '').trim();
const overrides = overridePath && fs.existsSync(overridePath)
  ? JSON.parse(fs.readFileSync(overridePath, 'utf8'))
  : null;
const games = overrides?.games || {};
const applied = new Map();

function norm(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function surname(value) {
  const parts = norm(value).split(' ').filter(Boolean);
  return parts.at(-1) || '';
}

function feedGamePk(url) {
  const statsMatch = url.pathname.match(/^\/api\/v1\.1\/game\/(\d+)\/feed\/live$/);
  if (statsMatch) return statsMatch[1];
  if (url.searchParams.get('type') === 'feed') return url.searchParams.get('gamePk');
  return null;
}

function allGameDataPlayers(payload) {
  return Object.values(payload?.gameData?.players || {});
}

function resolvePlayer(payload, teamBox, desiredName) {
  const rosterPlayers = Object.values(teamBox?.players || {});
  const globalPlayers = allGameDataPlayers(payload);
  const exact = norm(desiredName);

  let match = rosterPlayers.find(p => norm(p?.person?.fullName) === exact);
  if (match) return match;

  match = globalPlayers.find(p => norm(p?.fullName) === exact || norm(p?.person?.fullName) === exact);
  if (match) {
    const person = match.person || match;
    const row = { person: { id: person.id, fullName: person.fullName } };
    teamBox.players ||= {};
    teamBox.players[`ID${person.id}`] = row;
    return row;
  }

  const last = surname(desiredName);
  const byLastRoster = rosterPlayers.filter(p => surname(p?.person?.fullName) === last);
  if (byLastRoster.length === 1) return byLastRoster[0];

  const byLastGlobal = globalPlayers.filter(p => surname(p?.fullName || p?.person?.fullName) === last);
  if (byLastGlobal.length === 1) {
    const person = byLastGlobal[0].person || byLastGlobal[0];
    const row = { person: { id: person.id, fullName: person.fullName } };
    teamBox.players ||= {};
    teamBox.players[`ID${person.id}`] = row;
    return row;
  }

  throw new Error(`Unable to resolve lineup override player: ${desiredName}`);
}

function resolveStarter(payload, side, desiredName) {
  const teamPlayers = Object.values(payload?.liveData?.boxscore?.teams?.[side]?.players || {});
  const globalPlayers = allGameDataPlayers(payload);
  const exact = norm(desiredName);

  let match = teamPlayers.find(p => norm(p?.person?.fullName) === exact);
  if (match?.person?.id) return match.person;

  match = globalPlayers.find(p => norm(p?.fullName) === exact || norm(p?.person?.fullName) === exact);
  if (match) {
    const person = match.person || match;
    if (person?.id) return { id: person.id, fullName: person.fullName };
  }

  const last = surname(desiredName);
  const byLastTeam = teamPlayers.filter(p => surname(p?.person?.fullName) === last);
  if (byLastTeam.length === 1 && byLastTeam[0]?.person?.id) return byLastTeam[0].person;

  const byLastGlobal = globalPlayers.filter(p => surname(p?.fullName || p?.person?.fullName) === last);
  if (byLastGlobal.length === 1) {
    const person = byLastGlobal[0].person || byLastGlobal[0];
    if (person?.id) return { id: person.id, fullName: person.fullName };
  }

  throw new Error(`Unable to resolve probable-starter override: ${desiredName}`);
}

function applySide(payload, gamePk, side, desiredNames) {
  if (!Array.isArray(desiredNames) || desiredNames.length !== 9) return false;
  const teamBox = payload?.liveData?.boxscore?.teams?.[side];
  if (!teamBox) throw new Error(`Missing ${side} boxscore for override game ${gamePk}`);

  const existing = Object.values(teamBox.players || {})
    .filter(p => Number(p?.battingOrder) > 0)
    .sort((a, b) => Number(a.battingOrder) - Number(b.battingOrder));
  if (existing.length === 9) return false; // MLB feed-confirmed order always wins.

  for (const p of Object.values(teamBox.players || {})) p.battingOrder = '';
  const resolved = desiredNames.map(name => resolvePlayer(payload, teamBox, name));
  const ids = resolved.map(p => p?.person?.id);
  if (new Set(ids).size !== 9 || ids.some(id => !id)) {
    throw new Error(`Override for ${gamePk} ${side} did not resolve to 9 unique MLB player ids`);
  }
  resolved.forEach((p, idx) => { p.battingOrder = String((idx + 1) * 100); });
  return true;
}

function starterSpec(spec, side) {
  const raw = spec?.[`${side}Starter`];
  if (!raw) return null;
  if (typeof raw === 'string') return { name: raw, source: spec?.[`${side}StarterSource`] || null, role: 'probable starter' };
  if (typeof raw === 'object' && raw.name) return raw;
  return null;
}

function applyStarter(payload, gamePk, side, spec) {
  const wanted = starterSpec(spec, side);
  if (!wanted?.name) return null;
  payload.gameData ||= {};
  payload.gameData.probablePitchers ||= {};
  const existing = payload.gameData.probablePitchers[side];
  if (existing?.id) return null; // MLB feed-confirmed/probable starter always wins.

  const person = resolveStarter(payload, side, wanted.name);
  payload.gameData.probablePitchers[side] = { id: person.id, fullName: person.fullName };
  return {
    applied: true,
    name: person.fullName,
    source: wanted.source || null,
    role: wanted.role || 'probable starter',
    note: wanted.note || null,
  };
}

function patchFeed(payload, gamePk) {
  const spec = games[String(gamePk)];
  if (!spec) return payload;
  const awayApplied = applySide(payload, gamePk, 'away', spec.away);
  const homeApplied = applySide(payload, gamePk, 'home', spec.home);
  const awayStarter = applyStarter(payload, gamePk, 'away', spec);
  const homeStarter = applyStarter(payload, gamePk, 'home', spec);
  if (awayApplied || homeApplied || awayStarter || homeStarter) {
    applied.set(String(gamePk), {
      awayApplied,
      homeApplied,
      awayStarter,
      homeStarter,
      matchup: spec.matchup || null,
      lineupMethod: spec.lineupMethod || null,
    });
    console.log(`[I2 override] ${gamePk}: awayLineup=${awayApplied} homeLineup=${homeApplied} awayStarter=${Boolean(awayStarter)} homeStarter=${Boolean(homeStarter)}`);
  }
  return payload;
}

const baseFetch = globalThis.fetch.bind(globalThis);
globalThis.fetch = async function lineupOverrideFetch(input, init = {}) {
  const response = await baseFetch(input, init);
  if (!overrides || !response.ok) return response;

  const raw = typeof input === 'string' || input instanceof URL ? String(input) : String(input?.url || input);
  let url;
  try { url = new URL(raw); } catch { return response; }
  const gamePk = feedGamePk(url);
  if (!gamePk || !games[String(gamePk)]) return response;

  const payload = await response.clone().json();
  patchFeed(payload, gamePk);
  return new Response(JSON.stringify(payload), {
    status: response.status,
    statusText: response.statusText,
    headers: { 'content-type': 'application/json' },
  });
};

await import('./run_i2_today_upstream_wrapper.mjs');

const output = process.env.I2_OUTPUT || `data/runtime/i2/${process.env.I2_DATE || new Date().toISOString().slice(0, 10)}_frozen_predictions.json`;
if (fs.existsSync(output)) {
  const payload = JSON.parse(fs.readFileSync(output, 'utf8'));
  const appliedIds = new Set(applied.keys());

  for (const game of payload.games || []) {
    const detail = applied.get(String(game.gamePk));
    if (detail) {
      game.predictionClass = 'PROVISIONAL_EXPECTED_INPUTS';
      game.lineupSource = detail.awayApplied || detail.homeApplied
        ? 'preliminary weekday/game-series lineup override because MLB live feed lacked a complete batting order'
        : 'MLB live feed batting order';
      game.lineupFeedConfirmed = !(detail.awayApplied || detail.homeApplied);
      game.lineupSimulationReady = true;
      game.probableStarterFallbacks = {
        away: detail.awayStarter || null,
        home: detail.homeStarter || null,
      };
      if (game.modelStatus === 'FROZEN_RESEARCH_PROJECTION') game.modelStatus = 'PROVISIONAL_RESEARCH_PROJECTION';
    } else if (game.modelStatus === 'FROZEN_RESEARCH_PROJECTION') {
      game.predictionClass = 'FROZEN_FEED_CONFIRMED';
      game.lineupSource = 'MLB live feed batting order';
      game.lineupFeedConfirmed = true;
      game.lineupSimulationReady = true;
    }
  }

  for (const rank of payload.ranking || []) {
    const detail = applied.get(String(rank.gamePk));
    if (detail) {
      rank.predictionClass = 'PROVISIONAL_EXPECTED_INPUTS';
      rank.lineupSource = detail.awayApplied || detail.homeApplied ? 'preliminary weekday/game-series lineup override' : 'MLB live feed';
      rank.probableStarterFallbacks = { away: detail.awayStarter || null, home: detail.homeStarter || null };
    } else {
      rank.predictionClass = 'FROZEN_FEED_CONFIRMED';
      rank.lineupSource = 'MLB live feed';
    }
  }

  payload.lineupOverrideAudit = {
    overrideFile: overridePath || null,
    sourceStatus: overrides?.governance?.sourceStatus || null,
    note: overrides?.governance?.note || null,
    requestedGames: Object.keys(games).length,
    appliedGames: [...applied.entries()].map(([gamePk, detail]) => ({ gamePk: Number(gamePk), ...detail })),
    appliedGameCount: applied.size,
    feedConfirmedProjectedGames: (payload.ranking || []).filter(r => r.predictionClass === 'FROZEN_FEED_CONFIRMED').length,
    provisionalProjectedGames: (payload.ranking || []).filter(r => r.predictionClass === 'PROVISIONAL_EXPECTED_INPUTS').length,
  };
  payload.fullSlateProjectionMode = appliedIds.size ? 'CONFIRMED_PLUS_PROVISIONAL_LINEUP_AND_SOURCED_PROBABLE_OVERRIDES' : 'FEED_CONFIRMED_ONLY';
  fs.writeFileSync(output, JSON.stringify(payload, null, 2));
}
