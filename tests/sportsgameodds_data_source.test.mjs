import { strict as assert } from 'node:assert';
import {
  assertPostFreezeContext,
  buildMlbInningOddIds,
  classifyMlbOdd,
  filterEventsByLocalDate,
  normalizeSgoEvent,
} from '../src/market/sportsgameodds_data_source.mjs';

const ids = buildMlbInningOddIds();
assert(ids.includes('points-all-2i-ou-over'));
assert(ids.includes('points-away-2i-ou-over'));
assert(ids.includes('points-home-2i-ou-under'));
assert(ids.includes('points-away-2i-ml3way-away'));
assert(ids.includes('points-all-2i-ml3way-draw'));
assert(ids.includes('points-home-2i-ml3way-home'));
assert(ids.includes('points-away-9i-ou-over'));
assert(ids.includes('points-home-9i-ou-under'));
assert(!ids.includes('points-all-9i-ou-over'));
assert(!ids.includes('points-away-9i-ml3way-away'));

assert.throws(() => assertPostFreezeContext({ projectionFrozen: false }));
assert.equal(assertPostFreezeContext({ projectionFrozen: true, frozenAt: '2026-09-25T14:00:00Z' }), true);

assert.deepEqual(
  classifyMlbOdd({ statID:'points', statEntityID:'away', periodID:'2i', betTypeID:'ou', sideID:'over' }),
  { marketType:'TEAM_HALF_INNING_TOTAL', inning:2, segment:'top', teamSide:'away', side:'over' }
);

const event = {
  eventID: 'evt1', leagueID: 'MLB', startTime: '2026-09-25T23:10:00Z',
  teams: {
    away: { teamID:'CHC_MLB', names:{ long:'Chicago Cubs', short:'CHC' } },
    home: { teamID:'BOS_MLB', names:{ long:'Boston Red Sox', short:'BOS' } },
  },
  odds: {
    'points-all-2i-ou-over': {
      oddID:'points-all-2i-ou-over', statID:'points', statEntityID:'all', periodID:'2i', betTypeID:'ou', sideID:'over',
      fairOdds:'+125', bookOdds:'+118', fairOverUnder:'0.5', bookOverUnder:'0.5',
      byBookmaker: {
        draftkings:{ odds:'+150', overUnder:'0.5', available:true, lastUpdatedAt:'2026-09-25T14:05:00Z' },
        pinnacle:{ odds:'+145', overUnder:'0.5', available:true },
      },
    },
  },
};
const normalized = normalizeSgoEvent(event, { bookmakerIDs:['draftkings'] });
assert.equal(normalized.markets.length, 1);
assert.equal(normalized.markets[0].marketType, 'FULL_INNING_TOTAL');
assert.equal(normalized.markets[0].prices.length, 1);
assert.equal(normalized.markets[0].prices[0].americanOdds, 150);
assert.equal(normalized.markets[0].providerFairOdds, 125);
assert.equal(normalized.markets[0].providerFairLine, 0.5);

const dated = filterEventsByLocalDate([normalized], '2026-09-25', 'America/Chicago');
assert.equal(dated.length, 1);

console.log('SportsGameOdds data source tests passed.');
