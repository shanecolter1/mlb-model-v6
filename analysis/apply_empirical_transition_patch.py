#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'index.html'
MODEL=ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json'

def main():
    model=json.loads(MODEL.read_text(encoding='utf-8'))
    states=model.get('states',{})
    expected={f'{ev}|{o}|{m}' for ev in ['out','bb','single','double','triple','hr'] for o in range(3) for m in range(8)}
    missing=sorted(expected-set(states))
    if missing: raise SystemExit(f'Missing empirical transition states: {missing[:10]}')
    model_min={
      'version':model['version'],'market_inputs_used':model['market_inputs_used'],
      'training_years':model['training_years'],'selection_year':model['selection_year'],
      'locked_validation_year':model['locked_validation_year'],'shrinkage':model['shrinkage'],
      'validation_2025':model['validation_2025'],'states':states
    }
    payload=json.dumps(model_min,separators=(',',':'))
    text=INDEX.read_text(encoding='utf-8')
    pattern=re.compile(r"function walkTransition\(mask\)\{.*?\n\}\n\nfunction mixDistributions",re.S)
    replacement=f'''const EMPIRICAL_PA_TRANSITION_MODEL={payload};
function empiricalPATransitions(kind,outs,mask){{
  const key=`${{kind}}|${{outs}}|${{mask}}`;
  const rows=EMPIRICAL_PA_TRANSITION_MODEL.states?.[key];
  if(!Array.isArray(rows)||!rows.length) throw new Error(`Missing empirical PA transition state ${{key}}`);
  return rows;
}}
function runDistribution(lineup,startIdx,pitcherId,outs=0,mask=0,currentCount=null){{
  // V11 empirical transition engine: all PA state changes are driven by the
  // chronologically validated 2021-2024 Retrosheet transition table with
  // shrinkage selected on 2024 and locked validation on 2025.
  if(!lineup.length){{
    const fallback=[.70,.20,.07,.02,.006,.003,.001];
    return currentCount?applyHistoricalCountState(fallback,outs,mask,currentCount):fallback;
  }}
  const memo=new Map();
  const cap=6;
  function rec(o,m,i,depth){{
    if(o>=3||depth>=18) return [1,0,0,0,0,0,0];
    const key=`${{o}}|${{m}}|${{i}}|${{depth}}`;
    if(memo.has(key)) return memo.get(key);
    const probs=outcomeProb(lineup[i]?.id,pitcherId,null);
    const ni=(i+1)%lineup.length;
    const ans=[0,0,0,0,0,0,0];
    function add(weight,runs,next){{
      for(let k=0;k<next.length;k++) ans[Math.min(cap,runs+k)]+=weight*next[k];
    }}
    for(const [kind,pk] of [['out',probs.out],['bb',probs.bb],['single',probs.single],['double',probs.double],['triple',probs.triple],['hr',probs.hr]]){{
      for(const t of empiricalPATransitions(kind,o,m)){{
        const no=Math.min(3,o+Number(t.outs_added||0));
        const nm=no>=3?0:Number(t.post_mask||0);
        add(pk*Number(t.p||0),Number(t.runs||0),rec(no,nm,ni,depth+1));
      }}
    }}
    const total=ans.reduce((a,b)=>a+b,0)||1;
    for(let k=0;k<ans.length;k++) ans[k]/=total;
    memo.set(key,ans);
    return ans;
  }}
  const raw=rec(outs,mask,startIdx,0);
  return currentCount?applyHistoricalCountState(raw,outs,mask,currentCount):raw;
}}

function mixDistributions'''
    new,n=pattern.subn(replacement,text,count=1)
    if n!=1: raise SystemExit(f'Expected one legacy transition block, replaced {n}')
    INDEX.write_text(new,encoding='utf-8')
    print(f'Patched index.html with {len(states)} empirical PA states; model={model["version"]}')

if __name__=='__main__': main()
