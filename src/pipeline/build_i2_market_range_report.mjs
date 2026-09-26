import fs from 'node:fs';
import path from 'node:path';
import { reweightExactDistribution } from '../model/i2_total_conditioning.js';
import { enrichPriceDependentRecommendation, rankPriceDependentRecommendations } from '../model/i2_price_dependent_recommendations.mjs';

const DATE=String(process.env.I2_DATE||new Date().toISOString().slice(0,10));
const PARENT=String(process.env.I2_PARENT_INPUT||`data/runtime/i2/${DATE}_scratch_parent.json`);
const PROD=String(process.env.I2_PRODUCTION_INPUT||`data/runtime/i2/${DATE}_scratch_v04.json`);
const EMP=String(process.env.I2_EMPIRICAL_INPUT||`data/runtime/i2/${DATE}_scratch_empirical.json`);
const MARKETS=String(process.env.I2_MARKETS_INPUT||`data/runtime/i2/${DATE}_scratch_markets.json`);
const OUTPUT=String(process.env.I2_RANGE_OUTPUT||`data/runtime/i2/${DATE}_i2_market_range.json`);
const CSV=String(process.env.I2_RANGE_CSV||`docs/inning_markets/${DATE}_i2_market_range.csv`);

const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const parent=read(PARENT), prod=read(PROD), emp=read(EMP), markets=read(MARKETS);
const projectionFrozen=markets?.marketIsolation?.frozenProjection?.projectionFrozen===true;

const KEYS=['0','1','2','3','4+'];
const vals={'0':0,'1':1,'2':2,'3':3,'4+':4};
function norm(s){return String(s||'').normalize('NFD').replace(/\p{Diacritic}/gu,'').toLowerCase().replace(/[^a-z0-9]+/g,' ').trim();}
function teamKey(s){const n=norm(s); return ['oakland athletics','athletics'].includes(n)?'athletics':n;}
function matchKey(a,h){return `${teamKey(a)}@${teamKey(h)}`;}
function exactFrac(obj){
  if(!obj)return null;
  const e={};
  for(const k of KEYS){const v=Number(obj[k]); if(!Number.isFinite(v))return null; e[k]=v/100;}
  const s=KEYS.reduce((z,k)=>z+e[k],0);
  if(!(s>0))return null;
  for(const k of KEYS)e[k]/=s;
  return e;
}
function reweightPct(rawPct,targetOver){
  const e=exactFrac(rawPct); if(!e||!(targetOver>=0&&targetOver<=1))return null;
  return reweightExactDistribution(e,targetOver).exact;
}
function avgExact(a,b){
  if(!a||!b)return null;
  return Object.fromEntries(KEYS.map(k=>[k,(a[k]+b[k])/2]));
}
function outcome(exact,line,side){
  if(!exact||!Number.isFinite(line)||line<0||line>3.5)return null;
  const isInt=Math.abs(line-Math.round(line))<1e-9;
  let win=0,push=0;
  for(const k of KEYS){
    const r=vals[k], p=Number(exact[k]||0);
    if(k==='4+' && line>=4)return null;
    if(side==='over'){
      if(r>line)win+=p; else if(isInt&&r===line)push+=p;
    } else if(side==='under'){
      if(r<line)win+=p; else if(isInt&&r===line)push+=p;
    } else return null;
  }
  const loss=Math.max(0,1-win-push);
  const resolved=win+loss;
  return {win,push,loss,conditionalWin:resolved>0?win/resolved:null};
}
function fairAmerican(p){
  if(!(p>0&&p<1))return null;
  return p>=.5?Math.round(-100*p/(1-p)):Math.round(100*(1-p)/p);
}
function breakEven(american){
  const a=Number(american); if(!Number.isFinite(a)||a===0)return null;
  return a>0?100/(a+100):(-a)/((-a)+100);
}
function profitPer1(american){
  const a=Number(american); if(!Number.isFinite(a)||a===0)return null;
  return a>0?a/100:100/(-a);
}
function ev(out,american){
  if(!out)return null; const b=profitPer1(american); if(b==null)return null;
  return out.win*b-out.loss;
}
function pct(x){return x==null?null:Number((100*x).toFixed(2));}
function pct1(x){return x==null?null:Number((100*x).toFixed(1));}

const parentBy=new Map((parent.games||[]).map(g=>[String(g.gamePk),g]));
const prodBy=new Map((prod.ranking||[]).map(g=>[String(g.gamePk),g]));
const empBy=new Map((emp.games||[]).map(g=>[String(g.gamePk),g]));
const gameByMatch=new Map();
for(const g of parent.games||[])gameByMatch.set(matchKey(g.away,g.home),String(g.gamePk));

const modelByGame=new Map();
for(const [gamePk,g] of parentBy){
  if(!g.runEnvironmentConditioned)continue;
  const pr=prodBy.get(gamePk), eg=empBy.get(gamePk);
  if(!pr||!eg||eg.error)continue;

  const prodFull=reweightPct(g.fullI2Exact,Number(pr.over05Pct)/100);
  const prodTop=reweightPct(g.top2Exact,Number(g.top2ScorePct)/100);
  const prodBottom=reweightPct(g.bottom2Exact,Number(g.bottom2ScorePct)/100);
  const empFull=exactFrac(eg.fullI2?.exact);
  const empTop=exactFrac(eg.top2?.exact);
  const empBottom=exactFrac(eg.bottom2?.exact);
  const combinedFull=avgExact(prodFull,empFull);
  const combinedTop=avgExact(prodTop,empTop);
  const combinedBottom=avgExact(prodBottom,empBottom);
  if(!prodFull||!prodTop||!prodBottom||!empFull||!empTop||!empBottom)continue;

  modelByGame.set(gamePk,{
    gamePk:Number(gamePk),matchup:`${g.away} @ ${g.home}`,away:g.away,home:g.home,
    production:{full:prodFull,top:prodTop,bottom:prodBottom},
    empirical:{full:empFull,top:empTop,bottom:empBottom},
    combined:{full:combinedFull,top:combinedTop,bottom:combinedBottom},
    basis:{
      full:'v0.4 0.5 calibration extended across the frozen conditioned positive-run tail proportionally for alternate totals',
      top:'v0.3 total-conditioned Top-2 0.5 probability with the raw simulated positive-run tail reweighted proportionally',
      bottom:'v0.3 total-conditioned Bottom-2 0.5 probability with the raw simulated positive-run tail reweighted proportionally',
      empirical:'locked 65% pitcher-season / 35% offense-last25 model at 0.5; same weights applied bucket-by-bucket for alternate totals',
      combined:'50/50 arithmetic mean of production and empirical run distributions'
    },
    productionUnder05Pct:Number(pr.under05Pct),
    productionOver05Pct:Number(pr.over05Pct),
    empiricalUnder05Pct:Number(eg.empiricalUnder05Pct),
    empiricalOver05Pct:Number(eg.empiricalOver05Pct),
    combinedUnder05Pct:Number(((Number(pr.under05Pct)+Number(eg.empiricalUnder05Pct))/2).toFixed(2)),
    combinedOver05Pct:Number(((Number(pr.over05Pct)+Number(eg.empiricalOver05Pct))/2).toFixed(2)),
  });
}

const marketRows=(markets.rows||[]).filter(r=>Number(r.inning)===2&&['FULL_INNING_TOTAL','TEAM_HALF_INNING_TOTAL'].includes(r.marketType));
const rows=[];
for(const r of marketRows){
  const gamePk=gameByMatch.get(matchKey(r.awayTeam,r.homeTeam));
  const m=gamePk?modelByGame.get(gamePk):null;
  const segment=r.marketType==='FULL_INNING_TOTAL'?'full':String(r.segment);
  const line=Number(r.line), side=String(r.side).toLowerCase();
  const prodOut=m?outcome(m.production[segment],line,side):null;
  const empOut=m?outcome(m.empirical[segment],line,side):null;
  const combOut=m?outcome(m.combined[segment],line,side):null;
  const cands=[prodOut?.conditionalWin,combOut?.conditionalWin,empOut?.conditionalWin].filter(Number.isFinite);
  const prodEv=ev(prodOut,r.americanOdds), combEv=ev(combOut,r.americanOdds), empEv=ev(empOut,r.americanOdds);
  const evs=[prodEv,combEv,empEv].filter(Number.isFinite);
  const comparison={
    gamePk:gamePk?Number(gamePk):null,matchup:r.matchup,startTime:r.startTime,
    marketType:r.marketType,segment,line,side,bookmaker:r.bookmakerID,
    americanOdds:Number(r.americanOdds),isAlternateLine:Boolean(r.isAlternateLine),lastUpdatedAt:r.lastUpdatedAt||null,
    breakEvenPct:pct(breakEven(r.americanOdds)),
    productionWinPct:pct(prodOut?.win),productionPushPct:pct(prodOut?.push),productionConditionalPct:pct(prodOut?.conditionalWin),productionFairOdds:fairAmerican(prodOut?.conditionalWin),productionEVPct:pct(prodEv),
    combinedWinPct:pct(combOut?.win),combinedPushPct:pct(combOut?.push),combinedConditionalPct:pct(combOut?.conditionalWin),combinedFairOdds:fairAmerican(combOut?.conditionalWin),combinedEVPct:pct(combEv),
    empiricalWinPct:pct(empOut?.win),empiricalPushPct:pct(empOut?.push),empiricalConditionalPct:pct(empOut?.conditionalWin),empiricalFairOdds:fairAmerican(empOut?.conditionalWin),empiricalEVPct:pct(empEv),
    rangeLowPct:cands.length?pct(Math.min(...cands)):null,rangeHighPct:cands.length?pct(Math.max(...cands)):null,
    rangeMinEVPct:evs.length?pct(Math.min(...evs)):null,rangeMaxEVPct:evs.length?pct(Math.max(...evs)):null,
    robustPositiveEV:evs.length===3&&Math.min(...evs)>0,
    modelAvailable:Boolean(m&&prodOut&&empOut&&combOut),
  };
  rows.push(enrichPriceDependentRecommendation(comparison,{projectionFrozen}));
}
rows.sort((a,b)=>String(a.startTime).localeCompare(String(b.startTime))||String(a.matchup).localeCompare(String(b.matchup))||String(a.segment).localeCompare(String(b.segment))||a.line-b.line||String(a.side).localeCompare(String(b.side))||String(a.bookmaker).localeCompare(String(b.bookmaker)));

const actionableRecommendations=rankPriceDependentRecommendations(rows);
const modeledMarketInventory=[];
for(const m of modelByGame.values()){
  for(const segment of ['full','top','bottom']){
    for(const line of [0.5,1.5]){
      for(const side of ['over','under']){
        const quotes=rows.filter(r=>r.gamePk===m.gamePk&&r.segment===segment&&r.line===line&&r.side===side);
        const actionable=quotes.filter(r=>r.recommendationEligible);
        modeledMarketInventory.push({
          gamePk:m.gamePk,matchup:m.matchup,
          marketType:segment==='full'?'FULL_INNING_TOTAL':'TEAM_HALF_INNING_TOTAL',
          segment,line,side,
          quoteCount:quotes.length,
          actionableQuoteCount:actionable.length,
          priceStatus:quotes.length===0?'UNPRICED':(actionable.length?'PRICED_ACTIONABLE':'PRICED_BLOCKED'),
          bestActionableBook:actionable.slice().sort((a,b)=>Number(b.combinedEVPct??-Infinity)-Number(a.combinedEVPct??-Infinity))[0]?.bookmaker||null,
          bestActionableOdds:actionable.slice().sort((a,b)=>Number(b.combinedEVPct??-Infinity)-Number(a.combinedEVPct??-Infinity))[0]?.americanOdds??null,
        });
      }
    }
  }
}

const summaries=[...modelByGame.values()].map(m=>{
  const vals=[m.productionUnder05Pct,m.combinedUnder05Pct,m.empiricalUnder05Pct];
  const prices=rows.filter(r=>r.gamePk===m.gamePk&&r.segment==='full'&&r.line===0.5);
  const underPrices=prices.filter(r=>r.side==='under').sort((a,b)=>b.americanOdds-a.americanOdds);
  const overPrices=prices.filter(r=>r.side==='over').sort((a,b)=>b.americanOdds-a.americanOdds);
  return {
    gamePk:m.gamePk,matchup:m.matchup,
    productionUnder05Pct:m.productionUnder05Pct,combinedUnder05Pct:m.combinedUnder05Pct,empiricalUnder05Pct:m.empiricalUnder05Pct,
    underRangeLowPct:Number(Math.min(...vals).toFixed(2)),underRangeHighPct:Number(Math.max(...vals).toFixed(2)),
    bestUnder05Book:underPrices[0]?.bookmaker||null,bestUnder05Odds:underPrices[0]?.americanOdds??null,
    bestOver05Book:overPrices[0]?.bookmaker||null,bestOver05Odds:overPrices[0]?.americanOdds??null,
  };
}).sort((a,b)=>b.combinedUnder05Pct-a.combinedUnder05Pct);

const books=[...new Set(marketRows.map(r=>r.bookmakerID))].sort();
const linesBySegment={};
for(const seg of ['full','top','bottom']){
  linesBySegment[seg]=[...new Set(rows.filter(r=>r.segment===seg).map(r=>r.line).filter(Number.isFinite))].sort((a,b)=>a-b);
}
const availability={
  allApiBookmakers:books,
  inning2TotalPriceRows:marketRows.length,
  fullInningRows:rows.filter(r=>r.segment==='full').length,
  topHalfRows:rows.filter(r=>r.segment==='top').length,
  bottomHalfRows:rows.filter(r=>r.segment==='bottom').length,
  linesBySegment,
  modeledGames:modelByGame.size,
  productionProjectedGames:(prod.ranking||[]).length,
  empiricalModeledGames:(emp.games||[]).filter(x=>!x.error).length,
};

const output={
  date:DATE,generatedAt:new Date().toISOString(),
  definition:'Production / 50-50 combined / empirical are treated as an uncertainty range. Range low/high are the min/max resolved-market probabilities among the three models. Betting recommendations are post-freeze and price-mandatory.',
  productionModel:prod.model,empiricalMethod:emp.method,
  governance:{
    noManualAdjustments:true,
    freshScratchArtifacts:true,
    priceRetrievalPostFreeze:true,
    alternateTailDisclosure:'Production and empirical alternate-line tails are explicit distribution extensions described in each game model basis; the locked 0.5 probabilities are unchanged.',
    recommendationPolicy:{
      priceMandatory:true,
      probabilityOnlyRankingForbidden:true,
      requiredFields:['projectionFrozen','sportsbook','americanOdds','priceTimestamp','exactMarketMatch'],
      unpricedMarkets:'NO_RECOMMENDATION',
      rankingBasis:'Robust-positive-EV first, then combined EV, then range-min EV, then combined full Kelly',
    },
  },
  availability,gameSummary:summaries,modeledMarketInventory,actionableRecommendations,marketComparisons:rows,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});fs.mkdirSync(path.dirname(CSV),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(output,null,2)+'\n');

const cols=['gamePk','matchup','startTime','marketType','segment','line','side','bookmaker','americanOdds','isAlternateLine','lastUpdatedAt','breakEvenPct','productionWinPct','productionPushPct','productionConditionalPct','productionFairOdds','productionEVPct','combinedWinPct','combinedPushPct','combinedConditionalPct','combinedFairOdds','combinedEVPct','empiricalWinPct','empiricalPushPct','empiricalConditionalPct','empiricalFairOdds','empiricalEVPct','rangeLowPct','rangeHighPct','rangeMinEVPct','rangeMaxEVPct','robustPositiveEV','modelAvailable','exactMarketMatch','recommendationEligible','recommendationStatus','recommendationBlockReasons','productionKellyPct','combinedKellyPct','empiricalKellyPct','priceDependentRankingMetric'];
const q=v=>`"${String(v??'').replaceAll('"','""')}"`;
fs.writeFileSync(CSV,[cols.join(','),...rows.map(r=>cols.map(c=>q(r[c])).join(','))].join('\n')+'\n');
console.log(JSON.stringify({output:OUTPUT,csv:CSV,availability,summary:summaries},null,2));
