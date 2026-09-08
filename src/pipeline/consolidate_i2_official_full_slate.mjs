import fs from 'node:fs';

const date=process.env.I2_DATE||new Date().toISOString().slice(0,10),gate=61.5;
const pred=JSON.parse(fs.readFileSync(`data/runtime/i2/${date}_official_v04_predictions.json`,'utf8'));
const emp=JSON.parse(fs.readFileSync(`data/runtime/i2/${date}_official_empirical_challenger.json`,'utf8'));
const audit=JSON.parse(fs.readFileSync(`data/runtime/i2/${date}_official_lineup_audit.json`,'utf8'));
const mk=JSON.parse(fs.readFileSync(`data/runtime/i2/${date}_official_period_markets.json`,'utf8'));
const normalize=s=>String(s||'').toLowerCase().replace(/[^a-z0-9]/g,'');
const amerToDec=a=>a>0?1+a/100:1+100/Math.abs(a);
const calcEV=(p,a)=>p*amerToDec(a)-1;
function marketFor(matchup){
  const [away,home]=matchup.split(' @ '),aa=normalize(away),hh=normalize(home),candidates=[];
  for(const r of mk.rows||[]){
    const rm=normalize(r.matchup);if(!(rm.includes(aa)&&rm.includes(hh)))continue;
    const ml=String(r.market||'').toLowerCase();if(!(ml.includes('2')&&ml.includes('inning')))continue;
    for(const o of r.outcomes||[]){if(String(o.name||'').toLowerCase()==='under'&&Number(o.point)===0.5&&o.price!=null)candidates.push({sportsbook:r.sportsbook,market:r.market,price:Number(o.price),updatedAt:r.updated_at});}
  }
  return candidates.sort((a,b)=>b.price-a.price);
}
const rows=(pred.ranking||[]).map(r=>{
  const e=emp.games.find(x=>x.gamePk===r.gamePk),a=audit.games.find(x=>x.gamePk===r.gamePk),markets=marketFor(r.matchup),fd=markets.find(x=>/fan.?duel/i.test(x.sportsbook)),dk=markets.find(x=>/draft.?kings/i.test(x.sportsbook)),best=markets[0]||null,p=Number(r.under05Pct)/100;
  return {rank:r.rank,gamePk:r.gamePk,matchup:r.matchup,gameDate:r.gameDate,lineupsConfirmed:a?.lineupsConfirmed||false,awayLineup:a?.awayLineup||[],homeLineup:a?.homeLineup||[],awayStarter:r.awayStarter,homeStarter:r.homeStarter,awayStarterStats:a?.awayStarter?.stats||null,homeStarterStats:a?.homeStarter?.stats||null,total:r.runEnvironment?.fullGameTotal??null,productionUnderPct:r.under05Pct,productionFairUnder:r.fairUnder,underQualified:Number(r.under05Pct)>=gate,top2ScorePct:r.top2ScorePct,bottom2ScorePct:r.bottom2ScorePct,empiricalUnderPct:e?.fullUnder!=null?100*e.fullUnder:null,empiricalTop2ScorelessPct:e?.top2Scoreless!=null?100*e.top2Scoreless:null,empiricalBottom2ScorelessPct:e?.bottom2Scoreless!=null?100*e.bottom2Scoreless:null,awayStarterI2:e?.awayStarter||null,homeStarterI2:e?.homeStarter||null,awayOffense:e?.awayOffense||null,homeOffense:e?.homeOffense||null,bestI2UnderMarket:best?{...best,productionEVPct:100*calcEV(p,best.price),empiricalEVPct:e?.fullUnder!=null?100*calcEV(e.fullUnder,best.price):null}:null,fanduelI2Under:fd?{...fd,productionEVPct:100*calcEV(p,fd.price),empiricalEVPct:e?.fullUnder!=null?100*calcEV(e.fullUnder,fd.price):null}:null,draftkingsI2Under:dk?{...dk,productionEVPct:100*calcEV(p,dk.price),empiricalEVPct:e?.fullUnder!=null?100*calcEV(e.fullUnder,dk.price):null}:null};
});
const unresolved=(pred.games||[]).filter(g=>!rows.some(r=>r.gamePk===g.gamePk)).map(g=>({gamePk:g.gamePk,matchup:`${g.away} @ ${g.home}`,status:g.modelStatus||g.runEnvironment?.status||'UNRESOLVED',reason:g.error||g.reason||g.runEnvironment?.reason||null}));
const out={date,generatedAt:new Date().toISOString(),model:pred.model,productionGatePct:gate,scheduledGames:audit.scheduledGames,confirmedLineupGames:audit.confirmedGames,projectedGames:rows.length,qualifiedUnders:rows.filter(x=>x.underQualified),ranking:rows,unresolved,marketPullIsPostFreeze:true};
fs.writeFileSync(`data/runtime/i2/${date}_official_full_slate_summary.json`,JSON.stringify(out,null,2)+'\n');
console.log(JSON.stringify(out,null,2));
