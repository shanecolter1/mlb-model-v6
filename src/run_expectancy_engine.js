/**
 * Preliminary run-expectancy adapter for shadow testing.
 *
 * This is deliberately not presented as a final baseball simulator. It provides
 * two auditable bridges:
 * 1. a simple linear event-value expectation for smoke tests;
 * 2. a Monte Carlo half-inning simulator with explicit base/out states.
 */

import { validateEventVector } from "./event_probability_engine.js";

const DEFAULT_LINEAR_WEIGHTS = Object.freeze({
  single: 0.47,
  double: 0.78,
  triple: 1.09,
  home_run: 1.40,
  walk: 0.33,
  hit_by_pitch: 0.34,
  strikeout: -0.10,
  ball_in_play_out: -0.08,
});

export function expectedRunsPerPlateAppearance(
  eventVector,
  weights = DEFAULT_LINEAR_WEIGHTS
) {
  validateEventVector(eventVector);
  return Object.entries(eventVector).reduce(
    (total, [event, probability]) =>
      total + probability * (weights[event] ?? 0),
    0
  );
}

function drawEvent(vector, random) {
  let x = random();
  for (const [event, p] of Object.entries(vector)) {
    x -= p;
    if (x <= 0) return event;
  }
  return "ball_in_play_out";
}

function advance(event, bases) {
  let runs = 0;
  const [first, second, third] = bases;

  switch (event) {
    case "walk":
    case "hit_by_pitch": {
      const newThird = third || (second && first);
      const newSecond = second || first;
      const newFirst = true;
      if (third && second && first) runs += 1;
      return { bases: [newFirst, newSecond, newThird], runs, out: false };
    }
    case "single":
      runs += third ? 1 : 0;
      runs += second ? 1 : 0;
      return { bases: [true, first, false], runs, out: false };
    case "double":
      runs += third ? 1 : 0;
      runs += second ? 1 : 0;
      runs += first ? 0.55 : 0;
      return { bases: [false, true, Boolean(first && runs < 1)], runs, out: false };
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
      throw new Error(`Unknown event: ${event}`);
  }
}

export function simulateHalfInning(eventVector, random = Math.random) {
  validateEventVector(eventVector);
  let outs = 0;
  let bases = [false, false, false];
  let runs = 0;
  while (outs < 3) {
    const result = advance(drawEvent(eventVector, random), bases);
    bases = result.bases;
    runs += result.runs;
    if (result.out) outs += 1;
  }
  return runs;
}

export function simulateTeamRuns({
  eventVector,
  games = 10000,
  innings = 9,
  random = Math.random,
}) {
  validateEventVector(eventVector);
  if (!Number.isInteger(games) || games <= 0) throw new RangeError("games must be a positive integer");

  const distribution = new Map();
  let total = 0;
  for (let g = 0; g < games; g += 1) {
    let runs = 0;
    for (let inning = 0; inning < innings; inning += 1) {
      runs += simulateHalfInning(eventVector, random);
    }
    total += runs;
    distribution.set(runs, (distribution.get(runs) ?? 0) + 1);
  }

  return {
    meanRuns: total / games,
    distribution: Object.fromEntries(
      [...distribution.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([runs, count]) => [runs, count / games])
    ),
  };
}

export { DEFAULT_LINEAR_WEIGHTS };
