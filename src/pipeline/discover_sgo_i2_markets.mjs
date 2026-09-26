import fs from 'node:fs/promises';
import {
  fetchMlbSecondInningMarketCatalog,
  fetchMlbRawEventsForOddIds,
} from '../market/sportsgameodds_data_source.mjs';

const date = process.env.I2_DATE || new Intl.DateTimeFormat('en-CA', {
  timeZone: process.env.I2_TIME_ZONE || 'America/Chicago',
  year:'numeric', month:'2-digit', day:'2-digit',
}).format(new Date());
const frozenPath = process.env.FROZEN_PROJECTION_PATH || `data/runtime/i2/${date}_frozen_predictions.json`;
const outPath = process.env.SGO_I2_DISCOVERY_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_i2_exhaustive_discovery.json`;

const frozen = JSON.parse(await fs.readFile(frozenPath,'utf8'));
const frozenAt = frozen.generatedAt || frozen.frozenAt || frozen.freezeTimestamp || new Date().toISOString();
const freezeContext = { projectionFrozen:true, frozenAt };

const catalog = await fetchMlbSecondInningMarketCatalog({});

function text(m) {
  return [
    m?.oddID,m?.marketGroupID,m?.marketGroupName,m?.statID,m?.statEntityID,
    m?.periodID,m?.betTypeID,m?.sideID,
    ...Object.values(m?.marketGroupNameBySport || {}),
  ].filter(Boolean).join(' ').toLowerCase();
}
function supportBooks(m) {
  const mlb=m?.support?.MLB || {};
  return Object.entries(mlb).filter(([,v])=>v?.supported===true).map(([k])=>k).sort();
}
function candidateClass(m) {
  const t=text(m);
  const period=String(m?.periodID||'').toLowerCase();
  if(period!=='2i' && !/(2nd\s+inning|second\s+inning|inning\s*2|\b2i\b)/i.test(t)) return null;

  const stat=String(m?.statID||'').toLowerCase();
  const entity=String(m?.statEntityID||'').toLowerCase();
  const bet=String(m?.betTypeID||'').toLowerCase();
  const side=String(m?.sideID||'').toLowerCase();

  if(stat==='points' && entity==='all' && bet==='ou' && side==='under')
    return {tier:1,equivalent:'UNDER_0.5_IF_LINE_0.5',reason:'full-inning total under'};
  if(stat==='points' && entity==='all' && bet==='ou' && side==='over')
    return {tier:1,equivalent:'OVER_0.5_IF_LINE_0.5',reason:'full-inning total over'};
  if(stat==='points' && entity==='all' && bet==='yn' && side==='no')
    return {tier:1,equivalent:'UNDER_0.5',reason:'any runs no'};
  if(stat==='points' && entity==='all' && bet==='yn' && side==='yes')
    return {tier:1,equivalent:'OVER_0.5',reason:'any runs yes'};

  const exactZero = /(exact|exactly|number|total).*run|run.*(exact|exactly|number|total)|inning runs/.test(t) &&
    /(^|[^0-9])0([^0-9]|$)|zero|none|no runs/.test(t);
  const onePlus = /(1\+|1 or more|one or more|at least 1|at least one|yes.*run)/.test(t);
  if(exactZero) return {tier:1,equivalent:'UNDER_0.5_CANDIDATE',reason:'zero/exact inning-run candidate'};
  if(onePlus) return {tier:1,equivalent:'OVER_0.5_CANDIDATE',reason:'one-plus inning-run candidate'};

  if(bet==='ml3way' && side==='draw')
    return {tier:2,equivalent:'THREE_WAY_DRAW_FALLBACK',reason:'draw includes 0-0 plus higher ties; model separately'};

  if(/run|point|score/.test(t))
    return {tier:3,equivalent:'REVIEW',reason:'second-inning scoring market requiring settlement review'};
  return null;
}

const discovered = catalog.markets.map(m=>({
  oddID:m.oddID,
  marketGroupID:m.marketGroupID||null,
  marketGroupName:m.marketGroupName||null,
  marketGroupNameBySport:m.marketGroupNameBySport||null,
  statID:m.statID||null,
  statEntityID:m.statEntityID||null,
  periodID:m.periodID||null,
  betTypeID:m.betTypeID||null,
  sideID:m.sideID||null,
  supported:m.isSupported,
  bookmakers:supportBooks(m),
  classification:candidateClass(m),
})).filter(x=>x.classification);

const oddIDs=[...new Set(discovered.map(x=>x.oddID).filter(Boolean))];
const raw = oddIDs.length ? await fetchMlbRawEventsForOddIds({
  freezeContext,
  oddIDs,
  bookmakerIDs:[],
  includeOpenCloseOdds:false,
  includeAltLines:true,
}) : {events:[]};

const observed=[];
for(const event of raw.events||[]){
  for(const [oddID,odd] of Object.entries(event?.odds||{})){
    const meta=discovered.find(x=>x.oddID===oddID);
    for(const [bookmakerID,book] of Object.entries(odd?.byBookmaker||{})){
      const entries=[{...book,isAlternateLine:false},...(book?.altLines||[]).map(x=>({...x,isAlternateLine:true}))];
      for(const b of entries){
        if(b?.available!==true) continue;
        observed.push({
          eventID:event.eventID,
          startTime:event.startTime||event?.status?.startsAt||null,
          away:event?.teams?.away?.names?.long||event?.teams?.away?.name||null,
          home:event?.teams?.home?.names?.long||event?.teams?.home?.name||null,
          oddID,bookmakerID,
          odds:b.odds??null,overUnder:b.overUnder??null,
          isAlternateLine:Boolean(b.isAlternateLine),
          lastUpdatedAt:b.lastUpdatedAt||book?.lastUpdatedAt||null,
          classification:meta?.classification||null,
        });
      }
    }
  }
}

const allBooks=[...new Set([
  ...discovered.flatMap(x=>x.bookmakers||[]),
  ...observed.map(x=>x.bookmakerID),
])].sort();
const byBook=Object.fromEntries(allBooks.map(book=>[book,{
  catalogCandidates:discovered.filter(x=>x.bookmakers.includes(book)),
  observedPrices:observed.filter(x=>x.bookmakerID===book),
}]));

const output={
  generatedAt:new Date().toISOString(),date,
  governance:{postFreezeOnly:true,frozenAt,marketIsolationPreserved:true},
  catalogSummary:{
    discoveryScope:catalog.discoveryScope,
    allMarketCount:catalog.allMarketCount,
    secondInningCandidateCount:catalog.secondInningCandidateCount,
    classifiedCandidateCount:discovered.length,
    requestedOddIDCount:oddIDs.length,
    bookmakerCount:allBooks.length,
    observedPriceRows:observed.length,
  },
  discoveredMarkets:discovered,
  bookmakers:byBook,
  observedPrices:observed,
};
await fs.mkdir('data/runtime/i2',{recursive:true});
await fs.writeFile(outPath,JSON.stringify(output,null,2)+'\n');
console.log(JSON.stringify({outPath,...output.catalogSummary,books:allBooks},null,2));
