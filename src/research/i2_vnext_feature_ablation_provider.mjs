/**
 * Research-only direct-I2 feature ablation provider.
 * Production inference remains in src/model/i2_vnext_event_model.js.
 */
import { arsenalMatchupScore, predictI2EventVector } from '../model/i2_vnext_event_model.js';

const EVENT_KEYS = Object.freeze([
  'single','double','triple','home_run','walk','hit_by_pitch','strikeout','ball_in_play_out',
]);

function finite(value, fallback=0) {
  const x=Number(value);
  return Number.isFinite(x) ? x : fallback;
}

function softmax(logits) {
  const max=Math.max(...logits);
  const e=logits.map(x=>Math.exp(x-max));
  const sum=e.reduce((a,b)=>a+b,0);
  return e.map(x=>x/sum);
}

export function predictI2EventVectorResearch({
  batterId,
  pitcherId,
  batterSide,
  pitcherThrows,
  model,
  arsenalProfile,
  mode='baseline',
}) {
  if (mode === 'baseline') {
    return predictI2EventVector({
      batterId,pitcherId,batterSide,pitcherThrows,model,arsenalProfile,
    });
  }
  if (!['no_platoon'].includes(mode)) {
    throw new Error(`Unsupported research I2 feature mode: ${mode}`);
  }
  if (!model?.classes?.length) throw new Error('Missing I2 vNext model classes');

  const platoon=`${String(batterSide || '?')}v${String(pitcherThrows || '?')}`;
  const score=arsenalMatchupScore({batterId,pitcherId,batterSide,arsenalProfile});
  const arsenalFeature=`arsenal_x_${platoon}`;
  const arsenalScale=Math.max(
    1e-9,
    finite(model?.arsenal_feature?.scales?.[arsenalFeature],1),
  );
  const scaledArsenal=score/arsenalScale;

  const featureKeys=[
    `cat__batter_${String(batterId)}`,
    `cat__pitcher_${String(pitcherId)}`,
  ];

  const logits=model.classes.map(cls=>{
    const coef=model.coefficients?.[cls] || {};
    let value=finite(model.intercepts?.[cls],0);
    for (const key of featureKeys) value += finite(coef[key],0);
    value += finite(coef[`num__${arsenalFeature}`],0) * scaledArsenal;
    return value;
  });

  const probabilities=softmax(logits);
  const vector=Object.fromEntries(EVENT_KEYS.map(k=>[k,0]));
  model.classes.forEach((cls,i)=>{
    if (cls in vector) vector[cls]=probabilities[i];
  });
  const total=EVENT_KEYS.reduce((s,k)=>s+finite(vector[k],0),0);
  if (!(total>0)) throw new Error('Research I2 event vector has zero mass');
  for (const key of EVENT_KEYS) vector[key]=finite(vector[key],0)/total;
  return vector;
}
