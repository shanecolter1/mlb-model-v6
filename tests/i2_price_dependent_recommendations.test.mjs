import assert from 'node:assert/strict';
import {
  americanBreakEven,
  fullKellyFraction,
  recommendationGate,
  enrichPriceDependentRecommendation,
  rankPriceDependentRecommendations,
} from '../src/model/i2_price_dependent_recommendations.mjs';

assert.equal(Number(americanBreakEven(128).toFixed(6)), Number((100/228).toFixed(6)));
assert(fullKellyFraction(0.4899, 128) > 0);

assert.equal(recommendationGate({
  projectionFrozen:true, bookmaker:'fanduel', americanOdds:128,
  priceTimestamp:'2026-09-25T19:39:59.797Z', modelAvailable:true, exactMarketMatch:true,
}).eligible, true);

assert.deepEqual(recommendationGate({
  projectionFrozen:true, bookmaker:'', americanOdds:null,
  priceTimestamp:null, modelAvailable:true, exactMarketMatch:true,
}).reasons, ['SPORTSBOOK_MISSING','PRICE_MISSING','PRICE_TIMESTAMP_MISSING']);

assert(recommendationGate({
  projectionFrozen:false, bookmaker:'fanduel', americanOdds:128,
  priceTimestamp:'2026-09-25T19:39:59.797Z', modelAvailable:true, exactMarketMatch:true,
}).reasons.includes('PROJECTION_NOT_FROZEN'));

assert(recommendationGate({
  projectionFrozen:true, bookmaker:'fanduel', americanOdds:128,
  priceTimestamp:'2026-09-25T19:39:59.797Z', modelAvailable:true, exactMarketMatch:false,
}).reasons.includes('EXACT_MARKET_MATCH_FAILED'));

const base = {
  gamePk:1, matchup:'A @ B', marketType:'TEAM_HALF_INNING_TOTAL',
  segment:'bottom', line:0.5, side:'over', modelAvailable:true,
  lastUpdatedAt:'2026-09-25T19:39:59.797Z',
  productionConditionalPct:30, combinedConditionalPct:30, empiricalConditionalPct:30,
  rangeMinEVPct:0, robustPositiveEV:false,
};

const higherProbBadPrice = enrichPriceDependentRecommendation({
  ...base, bookmaker:'book-a', americanOdds:180,
  combinedConditionalPct:35, combinedEVPct:-2,
}, {projectionFrozen:true});

const lowerProbGoodPrice = enrichPriceDependentRecommendation({
  ...base, bookmaker:'book-b', americanOdds:300,
  combinedConditionalPct:30, combinedEVPct:20,
}, {projectionFrozen:true});

const ranked = rankPriceDependentRecommendations([higherProbBadPrice, lowerProbGoodPrice]);
assert.equal(ranked[0].bookmaker, 'book-b', 'Ranking must be price/EV dependent, not probability-only');

const noTimestamp = enrichPriceDependentRecommendation({
  ...base, bookmaker:'book-c', americanOdds:300, lastUpdatedAt:null, combinedEVPct:20,
}, {projectionFrozen:true});
assert.equal(noTimestamp.recommendationEligible, false);

console.log('i2_price_dependent_recommendations tests passed');
