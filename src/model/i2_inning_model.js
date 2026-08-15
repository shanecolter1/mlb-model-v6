/**
 * I2 full-inning probability engine.
 *
 * Market isolation: this module accepts baseball inputs only. It has no odds,
 * sportsbook, or market client and must be run/frozen before market retrieval.
 *
 * Core architecture:
 * 1. Simulate I1 from the confirmed batting order to obtain the I2 starting slot
 *    and a distribution of pitches entering I2.
 * 2. Simulate T2 and B2 separately with batter/pitcher event vectors.
 * 3. Combine the two half-inning outcomes into exact 0/1/2/3/4+ and cumulative
 *    1+/2+/3+/4+ full-I2 probabilities.
 *
 * The event probabilities are supplied by the existing Version 6 event engine.
 */

import {
  buildNeutralEventVector,
  applyEnvironmentalEventVector,
  validateEventVector,
} from "../event_probability_engine.js";

const EXACT_KEYS = Object.freeze(["0", "1", "2", "3", "4+"]);
const CUM_KEYS = Object.freeze(["1+", "2+", "3+", "4+"]);

function clamp01(x) {
  return Math.max(0, Math.min(1, Number(x)));
}

function drawEvent(vector, random) {
  let x = random();
  for (const [event, p] of Object.entries(vector)) {
    x -= p;
    if (x <= 0) return event;
  }
  return "ball_in_play_out";
}

function advance(event, bases, random) {
  let runs = 0;
  const [first, second, third] = bases;
  switch (event) {
    case "walk":
    case "hit_by_pitch": {
      if (first && second && third) runs += 1;
      return {
        bases: [true, Boolean(first), Boolean(third || (first && second))],
        runs,
        out: false,
      };
    }
    case "single": {
      if (third) runs += 1;
      let scoreSecond = false;
      if (second) scoreSecond = random() < 0.62;
      if (scoreSecond) runs += 1;
      const newThird = Boolean(second && !scoreSecond);
      const newSecond = Boolean(first);
      return { bases: [true, newSecond, newThird], runs, out: false };
    }
    case "double": {
      if (third) runs += 1;
      if (second) runs += 1;
      let scoreFirst = false;
      if (first) scoreFirst = random() < 0.45;
      if (scoreFirst) runs += 1;
      return {
        bases: [false, true, Boolean(first && !scoreFirst)],
        runs,
        out: false,
      };
    }
    case "triple":
      runs += Number(first) + Number(second) + Number(third);
      return { bases: [false, false, true], runs, out: false };
    case "home_run":
      runs += 1 + Number(first) + Number(second) + Number(third);
      return { bases: [false, false, false], runs, out: false };
    case "strikeout":
    case "ball_in_play_out":
      return { bases, runs: 0, out: true };
    default:
      throw new Error(`Unknown event ${event}`);
  }
}

function defaultPitchCountDraw(event, random) {
  // Pregame-safe pitch count simulation. These are intentionally simple
  // starting distributions; the historical I2 state dataset is the calibration
  // source for replacing them with fitted event/count distributions.
  const ranges = {
    strikeout: [3, 6],
    walk: [4, 7],
    hit_by_pitch: [1, 5],
    single: [1, 6],
    double: [1, 6],
    triple: [1, 6],
    home_run: [1, 6],
    ball_in_play_out: [1, 5],
  };
  const [lo, hi] = ranges[event] ?? [1, 6];
  return lo + Math.floor(random() * (hi - lo + 1));
}

function selectPitcher(mixture, random) {
  if (!Array.isArray(mixture) || mixture.length === 0) {
    throw new RangeError("pitcher mixture must contain at least one pitcher");
  }
  const total = mixture.reduce((s, x) => s + Number(x.weight ?? 0), 0);
  if (!(total > 0)) throw new RangeError("pitcher mixture weights must sum above zero");
  let x = random() * total;
  for (const item of mixture) {
    x -= Number(item.weight ?? 0);
    if (x <= 0) return item.pitcher;
  }
  return mixture[mixture.length - 1].pitcher;
}

function batterEventVector({ batter, pitcher, league, environmentalContext, weights }) {
  const neutral = buildNeutralEventVector({
    batter: batter.eventRates,
    pitcher: pitcher.eventRatesAllowed,
    league,
    weights,
  });
  const adjusted = applyEnvironmentalEventVector({
    neutralVector: neutral,
    environmentalContext,
    batterSide: batter.side,
  });
  validateEventVector(adjusted.probabilities);
  return adjusted.probabilities;
}

export function simulateHalfInningWithLineup({
  lineup,
  startSlot,
  pitcherMixture,
  league,
  environmentalContext = null,
  weights = { batter: 0.5, pitcher: 0.5 },
  random = Math.random,
  pitchCountDraw = defaultPitchCountDraw,
}) {
  if (!Array.isArray(lineup) || lineup.length !== 9) {
    throw new RangeError("lineup must contain exactly nine hitters in batting-order sequence");
  }
  if (!Number.isInteger(startSlot) || startSlot < 1 || startSlot > 9) {
    throw new RangeError("startSlot must be an integer from 1 through 9");
  }

  let outs = 0;
  let bases = [false, false, false];
  let runs = 0;
  let plateAppearances = 0;
  let pitches = 0;
  let slot = startSlot;

  while (outs < 3) {
    const batter = lineup[slot - 1];
    const pitcher = selectPitcher(pitcherMixture, random);
    const vector = batterEventVector({
      batter,
      pitcher,
      league,
      environmentalContext,
      weights,
    });
    const event = drawEvent(vector, random);
    pitches += Math.max(1, Number(pitchCountDraw(event, random)) || 1);
    const result = advance(event, bases, random);
    bases = result.bases;
    runs += result.runs;
    if (result.out) outs += 1;
    plateAppearances += 1;
    slot = slot === 9 ? 1 : slot + 1;

    // Defensive stop against malformed vectors / RNGs.
    if (plateAppearances > 40) throw new Error("half inning exceeded 40 plate appearances");
  }

  return { runs, nextSlot: slot, plateAppearances, pitches };
}

export function simulateSideToI2({
  lineup,
  starter,
  i2PitcherMixture = null,
  league,
  environmentalContext = null,
  weights = { batter: 0.5, pitcher: 0.5 },
  random = Math.random,
  pitchCountDraw = defaultPitchCountDraw,
}) {
  const starterMixture = [{ weight: 1, pitcher: starter }];
  const i1 = simulateHalfInningWithLineup({
    lineup,
    startSlot: 1,
    pitcherMixture: starterMixture,
    league,
    environmentalContext,
    weights,
    random,
    pitchCountDraw,
  });

  // Conventional starter default. Opener/bulk games should supply an explicit
  // I2 mixture from the shared upstream baseball data layer.
  const i2Mix = i2PitcherMixture ?? starterMixture;
  const i2 = simulateHalfInningWithLineup({
    lineup,
    startSlot: i1.nextSlot,
    pitcherMixture: i2Mix,
    league,
    environmentalContext,
    weights,
    random,
    pitchCountDraw,
  });

  return {
    i1,
    i2,
    i2StartSlot: i1.nextSlot,
    pitchesEnteringI2: i1.pitches,
  };
}

function emptyCounts() {
  return Object.fromEntries(EXACT_KEYS.map((k) => [k, 0]));
}

function bucket(runs) {
  return runs >= 4 ? "4+" : String(runs);
}

function probabilities(counts, trials) {
  const exact = Object.fromEntries(EXACT_KEYS.map((k) => [k, counts[k] / trials]));
  const cumulative = {
    "1+": 1 - exact["0"],
    "2+": exact["2"] + exact["3"] + exact["4+"],
    "3+": exact["3"] + exact["4+"],
    "4+": exact["4+"],
  };
  return { exact, cumulative };
}

export function fairAmericanOdds(probability) {
  const p = clamp01(probability);
  if (p <= 0 || p >= 1) return null;
  if (Math.abs(p - 0.5) < 1e-12) return 100;
  return p > 0.5 ? -100 * p / (1 - p) : 100 * (1 - p) / p;
}

export function simulateFullSecondInning({
  away,
  home,
  league,
  environmentalContext = null,
  weights = { batter: 0.5, pitcher: 0.5 },
  trials = 100000,
  random = Math.random,
  pitchCountDraw = defaultPitchCountDraw,
}) {
  if (!Number.isInteger(trials) || trials <= 0) {
    throw new RangeError("trials must be a positive integer");
  }

  const topCounts = emptyCounts();
  const bottomCounts = emptyCounts();
  const fullCounts = emptyCounts();
  const awayStartSlots = Array(10).fill(0);
  const homeStartSlots = Array(10).fill(0);
  let awayPitchesEnteringI2 = 0;
  let homePitchesEnteringI2 = 0;

  for (let i = 0; i < trials; i += 1) {
    // Away offense faces home pitching; home offense faces away pitching.
    const top = simulateSideToI2({
      lineup: away.lineup,
      starter: home.starter,
      i2PitcherMixture: home.i2PitcherMixture,
      league,
      environmentalContext,
      weights,
      random,
      pitchCountDraw,
    });
    const bottom = simulateSideToI2({
      lineup: home.lineup,
      starter: away.starter,
      i2PitcherMixture: away.i2PitcherMixture,
      league,
      environmentalContext,
      weights,
      random,
      pitchCountDraw,
    });

    topCounts[bucket(top.i2.runs)] += 1;
    bottomCounts[bucket(bottom.i2.runs)] += 1;
    fullCounts[bucket(top.i2.runs + bottom.i2.runs)] += 1;
    awayStartSlots[top.i2StartSlot] += 1;
    homeStartSlots[bottom.i2StartSlot] += 1;
    awayPitchesEnteringI2 += top.pitchesEnteringI2;
    homePitchesEnteringI2 += bottom.pitchesEnteringI2;
  }

  const top = probabilities(topCounts, trials);
  const bottom = probabilities(bottomCounts, trials);
  const full = probabilities(fullCounts, trials);
  const under05 = full.exact["0"];
  const over05 = 1 - under05;

  return {
    trials,
    top2: top,
    bottom2: bottom,
    fullI2: full,
    under05,
    over05,
    fairOdds: {
      under05: fairAmericanOdds(under05),
      over05: fairAmericanOdds(over05),
    },
    stateDiagnostics: {
      awayI2StartSlotProbability: Object.fromEntries(
        Array.from({ length: 9 }, (_, j) => [String(j + 1), awayStartSlots[j + 1] / trials])
      ),
      homeI2StartSlotProbability: Object.fromEntries(
        Array.from({ length: 9 }, (_, j) => [String(j + 1), homeStartSlots[j + 1] / trials])
      ),
      awayMeanPitchesEnteringI2: awayPitchesEnteringI2 / trials,
      homeMeanPitchesEnteringI2: homePitchesEnteringI2 / trials,
    },
    governance: {
      marketDataUsed: false,
      predictionMustBeFrozenBeforeMarketRetrieval: true,
      exactBuckets: EXACT_KEYS,
      cumulativeBuckets: CUM_KEYS,
    },
  };
}
