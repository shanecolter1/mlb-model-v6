#!/usr/bin/env node
import fs from 'node:fs';
import {applyHalfScoreContrast, validateHalfContrastArtifact} from '../model/i2_half_contrast.js';

function arg(name) {
  const i=process.argv.indexOf(name);
  if (i<0 || !process.argv[i+1]) throw new Error(`Missing ${name}`);
  return process.argv[i+1];
}
function read(path){ return JSON.parse(fs.readFileSync(path,'utf8')); }
function mean(xs){ return xs.reduce((a,b)=>a+b,0)/xs.length; }
function logloss(y,p){
  const q=Math.max(1e-12,Math.min(1-1e-12,Number(p)));
  return -(y*Math.log(q)+(1-y)*Math.log1p(-q));
}
function close(actual, expected, label, tol=1e-12) {
  if (Math.abs(Number(actual)-Number(expected)) > tol) {
    throw new Error(`${label} mismatch: actual=${actual} expected=${expected}`);
  }
}

const replay=read(arg('--replay'));
const inputs=read(arg('--inputs'));
const artifact=validateHalfContrastArtifact(read(arg('--contrast')));
const phase19=read(arg('--phase19'));
const output=arg('--output');

if (replay.market_inputs_used !== false || inputs.market_inputs_used !== false || phase19.market_inputs_used !== false) {
  throw new Error('Market contamination');
}
if (Number(replay.season)!==2025 || Number(inputs.season)!==2025) {
  throw new Error('Expected canonical 2025 replay/input season');
}
if (Number(replay.trials_per_game)!==10000) throw new Error('Expected 10,000-trial replay');
close(artifact.zero_sum_half_contrast_h, phase19.candidate.zero_sum_half_contrast_h, 'h', 1e-15);

const games=new Map(inputs.games.map(g=>[String(g.gid),g]));
const rows=[];
for (const p of replay.predictions) {
  if (p.park_status !== 'RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT') continue;
  const g=games.get(String(p.gid));
  if (!g) throw new Error(`Missing input game ${p.gid}`);
  const rawTop=Number(p.top2_score_probability);
  const rawBottom=Number(p.bottom2_score_probability);
  const adjusted=applyHalfScoreContrast({
    topScoreProbability:rawTop,
    bottomScoreProbability:rawBottom,
    artifact,
  });
  rows.push({
    gid:p.gid,
    yTop:Number(g.observed.top2_runs>0),
    yBottom:Number(g.observed.bottom2_runs>0),
    yUnder:Number(p.observed_under05),
    rawTop,
    rawBottom,
    adjustedTop:adjusted.adjustedTopScoreProbability,
    adjustedBottom:adjusted.adjustedBottomScoreProbability,
    rawUnder:Number(p.raw_under05),
    adjustedUnder:adjusted.adjustedUnder05,
  });
}
if (rows.length!==2264) throw new Error(`Expected 2264 matched games; got ${rows.length}`);

const topRawResidual=mean(rows.map(r=>r.yTop-r.rawTop));
const topAdjustedResidual=mean(rows.map(r=>r.yTop-r.adjustedTop));
const bottomRawResidual=mean(rows.map(r=>r.yBottom-r.rawBottom));
const bottomAdjustedResidual=mean(rows.map(r=>r.yBottom-r.adjustedBottom));

const halfRawBrier=mean(rows.flatMap(r=>[(r.rawTop-r.yTop)**2,(r.rawBottom-r.yBottom)**2]));
const halfAdjustedBrier=mean(rows.flatMap(r=>[(r.adjustedTop-r.yTop)**2,(r.adjustedBottom-r.yBottom)**2]));
const halfRawLogloss=mean(rows.flatMap(r=>[logloss(r.yTop,r.rawTop),logloss(r.yBottom,r.rawBottom)]));
const halfAdjustedLogloss=mean(rows.flatMap(r=>[logloss(r.yTop,r.adjustedTop),logloss(r.yBottom,r.adjustedBottom)]));
const fullRawBrier=mean(rows.map(r=>(r.rawUnder-r.yUnder)**2));
const fullAdjustedBrier=mean(rows.map(r=>(r.adjustedUnder-r.yUnder)**2));
const fullRawLogloss=mean(rows.map(r=>logloss(r.yUnder,r.rawUnder)));
const fullAdjustedLogloss=mean(rows.map(r=>logloss(r.yUnder,r.adjustedUnder)));

const result={
  version:'i2-half-contrast-js-replay-verification-v1',
  market_inputs_used:false,
  season:2025,
  matched_games:rows.length,
  contrast_version:artifact.version,
  h:Number(artifact.zero_sum_half_contrast_h),
  residuals:{
    top_raw_actual_minus_predicted:topRawResidual,
    top_adjusted_actual_minus_predicted:topAdjustedResidual,
    bottom_raw_actual_minus_predicted:bottomRawResidual,
    bottom_adjusted_actual_minus_predicted:bottomAdjustedResidual,
  },
  scores:{
    half_raw_brier:halfRawBrier,
    half_adjusted_brier:halfAdjustedBrier,
    half_brier_delta:halfAdjustedBrier-halfRawBrier,
    half_raw_logloss:halfRawLogloss,
    half_adjusted_logloss:halfAdjustedLogloss,
    half_logloss_delta:halfAdjustedLogloss-halfRawLogloss,
    full_i2_raw_brier:fullRawBrier,
    full_i2_adjusted_brier:fullAdjustedBrier,
    full_i2_brier_delta:fullAdjustedBrier-fullRawBrier,
    full_i2_raw_logloss:fullRawLogloss,
    full_i2_adjusted_logloss:fullAdjustedLogloss,
    full_i2_logloss_delta:fullAdjustedLogloss-fullRawLogloss,
  },
  phase19_exact_match:false,
};

const expected=phase19.test_results;
close(topRawResidual,expected.top_raw_actual_minus_predicted,'top raw residual');
close(topAdjustedResidual,expected.top_adjusted_actual_minus_predicted,'top adjusted residual');
close(bottomRawResidual,expected.bottom_raw_actual_minus_predicted,'bottom raw residual');
close(bottomAdjustedResidual,expected.bottom_adjusted_actual_minus_predicted,'bottom adjusted residual');
close(result.scores.half_brier_delta,expected.half_brier.delta_adjusted_minus_raw,'half Brier delta');
close(result.scores.half_logloss_delta,expected.half_logloss.delta_adjusted_minus_raw,'half logloss delta');
close(result.scores.full_i2_brier_delta,expected.full_i2_brier.delta_adjusted_minus_raw,'full I2 Brier delta');
close(result.scores.full_i2_logloss_delta,expected.full_i2_logloss.delta_adjusted_minus_raw,'full I2 logloss delta');
result.phase19_exact_match=true;

fs.mkdirSync(output.split('/').slice(0,-1).join('/'),{recursive:true});
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
