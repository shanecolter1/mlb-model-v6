#!/usr/bin/env python3
"""Staged version of the joint multinomial PA calibration.

Uses the exact leakage-safe feature builder and locked-2025 evaluation in
batter_pitcher_blend_calibration.py, but reduces 2024 model-selection degrees of
freedom and runtime:
1) choose half-life/prior from 3x3 grid at predeclared C=0.3;
2) with that pair frozen, choose C from {0.1,0.3,1.0};
3) refit on 2021-2024 and evaluate exactly once on locked 2025.

For the low-dimensional (15-feature) multinomial fit, Newton-Cholesky solves the
same L2-regularized objective much more efficiently than LBFGS when n >> p.
"""
from __future__ import annotations
import csv,json,gc
import numpy as np
import batter_pitcher_blend_calibration as core

OUT=core.OUT

def fit_newton(X,y,C):
    m=core.LogisticRegression(C=C,solver='newton-cholesky',max_iter=60,tol=1e-6)
    m.fit(X,y)
    return m

# Pure numerical solver substitution; statistical objective is unchanged.
core.fit=fit_newton

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plays,prov=core.fetch_plays()
    if not plays: raise RuntimeError('No Retrosheet modeled PAs materialized')
    stage1=[]; best=None; fixed_C=0.3
    for h in core.HALF_LIVES:
        Xs,y,yrs,_=core.build_multi_prior(plays,h,core.PRIORS)
        tr=np.isin(yrs,[2021,2022,2023]); sel=yrs==2024
        for prior in core.PRIORS:
            X=Xs[prior];m=core.fit(X[tr],y[tr],fixed_C);met=core.metrics(y[sel],m.predict_proba(X[sel]))
            row={'stage':'decay_prior','half_life_days':h,'prior_strength':prior,'C':fixed_C,'logloss_2024':met['logloss'],'brier_2024':met['brier']};stage1.append(row)
            if best is None or (met['logloss'],met['brier'])<(best[0],best[1]): best=(met['logloss'],met['brier'],h,prior)
        del Xs,y,yrs;gc.collect()
    _,_,h,prior=best
    Xs,y,yrs,_=core.build_multi_prior(plays,h,[prior]);X=Xs[prior];tr=np.isin(yrs,[2021,2022,2023]);sel=yrs==2024
    stage2=[];bestC=None
    for C in core.CS:
        m=core.fit(X[tr],y[tr],C);met=core.metrics(y[sel],m.predict_proba(X[sel]));row={'stage':'regularization','half_life_days':h,'prior_strength':prior,'C':C,'logloss_2024':met['logloss'],'brier_2024':met['brier']};stage2.append(row)
        if bestC is None or (met['logloss'],met['brier'])<(bestC[0],bestC[1]):bestC=(met['logloss'],met['brier'],C)
    C=bestC[2]
    del Xs,y,yrs;gc.collect()
    Xs,y,yrs,bases=core.build_multi_prior(plays,h,[prior],collect_baseline_prior=prior,collect_baseline_year=2025);X=Xs[prior];train=np.isin(yrs,[2021,2022,2023,2024]);val=yrs==2025
    m=core.fit(X[train],y[train],C);pred=m.predict_proba(X[val])
    if bases is None or len(bases)!=int(val.sum()):raise RuntimeError('Locked-2025 baseline alignment failure')
    names=['league','legacy','batter','pitcher'];validation={'joint_multinomial':core.metrics(y[val],pred)}
    for idx,nm in enumerate(names):validation[nm]=core.metrics(y[val],bases[:,idx,:])
    one=np.eye(6)[y[val]];by_class={c:{'mean_pred':float(pred[:,i].mean()),'observed_rate':float(one[:,i].mean()),'brier':float(np.mean((pred[:,i]-one[:,i])**2)),'legacy_brier':float(np.mean((bases[:,1,i]-one[:,i])**2))} for i,c in enumerate(core.CLASSES)}
    pa_pass=validation['joint_multinomial']['logloss']<validation['legacy']['logloss'] and validation['joint_multinomial']['brier']<=validation['legacy']['brier']
    manifest={'component':'joint multinomial batter/pitcher PA model','architecture':'regularized multinomial logistic model on batter/pitcher relative log rates and league logits','selection_protocol':'predeclared staged search: 9 decay/prior pairs at C=.3, then 3 C values for selected pair','numerical_solver':'newton-cholesky; same L2 multinomial objective as LBFGS','modeled_outcomes':core.CLASSES,'unmodeled_pa_handling':'HBP/interference/reach-on-error/no-out PAs excluded from PA fit; materiality tested by half-inning gate','governance_status':'PA_GATE_PASS_HALF_INNING_PENDING' if pa_pass else 'BLOCKED_PA_GATE','production_eligible':False,'development_years':[2021,2022,2023],'selection_year':[2024],'locked_validation_year':[2025],'market_inputs_used':False,'same_game_updates_used_in_features':False,'rolling_history_crosses_season_boundaries':True,'selected_hyperparameters':{'half_life_days':h,'prior_strength':prior,'C':C},'validation_2025':validation,'validation_by_class':by_class,'provenance':prov,'promotion_rule':'Beat legacy 68/32 on locked-2025 multiclass log loss without worsening multiclass Brier, then pass locked-2025 half-inning scoring gate.'}
    params={'version':'joint-multinomial-pa-v1','classes':core.CLASSES,'feature_order':[f'batter_log_ratio_{e}' for e in core.NONOUT]+[f'pitcher_log_ratio_{e}' for e in core.NONOUT]+[f'league_logit_{e}_vs_out' for e in core.NONOUT],'selected_hyperparameters':manifest['selected_hyperparameters'],'coef':m.coef_.tolist(),'intercept':m.intercept_.tolist(),'sklearn_classes':m.classes_.tolist(),'validation_2025':validation,'production_eligible':False}
    (OUT/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'joint_multinomial_pa_model.json').write_text(json.dumps(params,indent=2))
    rows=stage1+stage2
    with (OUT/'multinomial_grid_2024.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['stage','half_life_days','prior_strength','C','logloss_2024','brier_2024']);w.writeheader();w.writerows(rows)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
