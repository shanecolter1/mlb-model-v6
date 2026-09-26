/**
 * Lean direct-I2 event model inference for I2 vNext.
 *
 * Inputs are baseball-only and must be frozen before market retrieval.
 * The statistical model is trained on Statcast inning-2 terminal PAs with
 * jointly regularized batter, pitcher, platoon, and arsenal-matchup effects.
 */

const EVENT_KEYS = Object.freeze([
  'single', 'double', 'triple', 'home_run', 'walk', 'hit_by_pitch',
  'strikeout', 'ball_in_play_out',
]);

function finite(value, fallback = 0) {
  const x = Number(value);
  return Number.isFinite(x) ? x : fallback;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const e = logits.map(x => Math.exp(x - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

export function arsenalMatchupScore({ batterId, pitcherId, arsenalProfile }) {
  const globalX = finite(arsenalProfile?.league_global_xwoba, 0.320);
  const leagueUsage = arsenalProfile?.league_pitch_usage || {};
  const leagueX = arsenalProfile?.league_xwoba_by_pitch || {};
  const batter = arsenalProfile?.batter_xwoba_by_pitch?.[String(batterId)] || {};
  const pitcher = arsenalProfile?.pitcher_usage_by_pitch?.[String(pitcherId)] || null;
  const usage = pitcher && Object.keys(pitcher).length ? pitcher : leagueUsage;
  const entries = Object.entries(usage).filter(([, w]) => finite(w, 0) > 0);
  const total = entries.reduce((s, [, w]) => s + finite(w, 0), 0);
  if (!(total > 0)) return globalX;
  return entries.reduce((s, [pitchType, w]) => {
    const xwoba = finite(batter[pitchType], finite(leagueX[pitchType], globalX));
    return s + (finite(w, 0) / total) * xwoba;
  }, 0);
}

export function predictI2EventVector({
  batterId,
  pitcherId,
  batterSide,
  pitcherThrows,
  model,
  arsenalProfile,
}) {
  if (!model?.classes?.length) throw new Error('Missing I2 vNext model classes');
  const score = arsenalMatchupScore({ batterId, pitcherId, arsenalProfile });
  const mean = finite(model?.arsenal_feature?.mean, 0.320);
  const sd = Math.max(1e-9, finite(model?.arsenal_feature?.sd, 1));
  const z = (score - mean) / sd;
  const featureKeys = [
    `cat__batter_${String(batterId)}`,
    `cat__pitcher_${String(pitcherId)}`,
    `cat__platoon_${String(batterSide || '?')}v${String(pitcherThrows || '?')}`,
  ];

  const logits = model.classes.map(cls => {
    const coef = model.coefficients?.[cls] || {};
    let value = finite(model.intercepts?.[cls], 0);
    for (const key of featureKeys) value += finite(coef[key], 0);
    value += finite(coef['num__arsenal_z'], 0) * z;
    return value;
  });

  const probabilities = softmax(logits);
  const vector = Object.fromEntries(EVENT_KEYS.map(k => [k, 0]));
  model.classes.forEach((cls, i) => {
    if (cls in vector) vector[cls] = probabilities[i];
  });
  const total = EVENT_KEYS.reduce((s, k) => s + finite(vector[k], 0), 0);
  if (!(total > 0)) throw new Error('I2 vNext event vector has zero mass');
  for (const key of EVENT_KEYS) vector[key] = finite(vector[key], 0) / total;
  return vector;
}

export const I2_VNEXT_EVENT_KEYS = EVENT_KEYS;
