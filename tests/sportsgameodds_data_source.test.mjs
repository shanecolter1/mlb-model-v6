import { strict as assert } from 'node:assert';
import {
  assertPostFreezeContext,
  assertPreFreezeIsolation,
  extractMlbDraftKingsFullGameTotalPoints,
  buildMlbInningOddIds,
  buildMlbNinthInningCandidateOddIds,
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
const ninthCandidates = buildMlbNinthInningCandidateOddIds();
assert(ninthCandidates.includes('points-all-9i-ou-over'));
assert(ninthCandidates.includes('points-away-9i-ml3way-away'));

assert.throws(() => assertPostFreezeContext({ projectionFrozen: false }));
assert.equal(assertPostFreezeContext({ projectionFrozen: true, frozenAt: '2026-09-25T14:00:00Z' }), true);

assert.deepEqual(
  classifyMlbOdd({ statID:'points', statEntityID:'away', periodID:'2i', betTypeID:'ou', sideID:'over' }),
  { marketType:'TEAM_HALF_INNING_TOTAL', inning:2, segment:'top', teamSide:'away', side:'over' }
);

const event = {
  eventID: 'evt1', leagueID: 'MLB', status: { started:false, startsAt:'2026-09-25T23:10:00Z' },
  teams: {
    away: { teamID:'CHC_MLB', names:{ long:'Chicago Cubs', short:'CHC' } },
    home: { teamID:'BOS_MLB', names:{ long:'Boston Red Sox', short:'BOS' } },
  },
  odds: {
    'points-all-2i-ou-over': {
      oddID:'points-all-2i-ou-over', statID:'points', statEntityID:'all', periodID:'2i', betTypeID:'ou', sideID:'over',
      fairOdds:'+125', bookOdds:'+118', fairOverUnder:'0.5', bookOverUnder:'0.5',
      byBookmaker: {
        draftkings:{ odds:'-115', overUnder:'1.5', available:true, lastUpdatedAt:'2026-09-25T14:05:00Z', altLines:[{ odds:'+150', overUnder:'0.5', available:true, lastUpdatedAt:'2026-09-25T14:04:00Z' }] },
        pinnacle:{ odds:'+145', overUnder:'0.5', available:true },
      },
    },
  },
};
const normalized = normalizeSgoEvent(event, { bookmakerIDs:['draftkings'] });
assert.equal(normalized.markets.length, 1);
assert.equal(normalized.markets[0].marketType, 'FULL_INNING_TOTAL');
assert.equal(normalized.markets[0].prices.length, 2);
const dkHalf = normalized.markets[0].prices.find(x => x.line === 0.5);
assert.equal(dkHalf.americanOdds, 150);
assert.equal(dkHalf.isAlternateLine, true);
assert.equal(normalized.markets[0].providerFairOdds, 125);
assert.equal(normalized.markets[0].providerFairLine, 0.5);

const dated = filterEventsByLocalDate([normalized], '2026-09-25', 'America/Chicago');
assert.equal(dated.length, 1);

console.log('SportsGameOdds data source tests passed.');


const preFreezePayload = {
  data: [{
    eventID:'mlb-evt-1',
    status:{ startsAt:'2026-09-25T23:10:00Z' },
    teams:{
      away:{ names:{ long:'Chicago Cubs' } },
      home:{ names:{ long:'Boston Red Sox' } },
    },
    odds:{
      'points-all-game-ou-over':{
        byBookmaker:{
          draftkings:{
            available:true,
            overUnder:'8.5',
            odds:'-108',
            lastUpdatedAt:'2026-09-25T15:00:00Z',
            deeplink:'https://example.invalid/price'
          }
        }
      }
    }
  }]
};
const preFreeze = extractMlbDraftKingsFullGameTotalPoints(preFreezePayload);
assert.equal(preFreeze.length, 1);
assert.deepEqual(preFreeze[0], {
  eventId:'mlb-evt-1',
  commenceTime:'2026-09-25T23:10:00Z',
  awayTeam:'Chicago Cubs',
  homeTeam:'Boston Red Sox',
  fullGameTotal:8.5,
  bookmaker:'draftkings',
  lastUpdate:'2026-09-25T15:00:00Z',
});
assert.equal(assertPreFreezeIsolation(preFreeze[0]), true);
assert(!('odds' in preFreeze[0]));
assert(!('deeplink' in preFreeze[0]));
