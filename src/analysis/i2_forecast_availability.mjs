import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

export function validateForecast(pred, date) {
  if (!pred || pred.date !== date || !Array.isArray(pred.games)) throw new Error('Malformed prediction artifact: date/games');
  const seen = new Set();
  for (const g of pred.games) {
    if (!g || !Number.isInteger(g.gamePk) || seen.has(g.gamePk) || typeof g.modelStatus !== 'string') throw new Error('Malformed prediction game identity/status');
    seen.add(g.gamePk);
    if (['FROZEN_RESEARCH_PROJECTION','PROVISIONAL_RESEARCH_PROJECTION'].includes(g.modelStatus) &&
        (typeof g.under05 !== 'number' || !Number.isFinite(g.under05) || g.under05 < 0 || g.under05 > 1 ||
         typeof g.under05Pct !== 'number' || !Number.isFinite(g.under05Pct) || Math.abs(g.under05Pct - g.under05*100) > 0.11)) {
      throw new Error('Malformed prediction probability');
    }
  }
  return pred;
}

export const producerPaths = ['.github/workflows/i2_daily_run.yml', '.github/workflows/i2_v04_today.yml'];
export function relevantRuns(runs, date) {
  return runs.filter(r => r.head_branch === 'main' && producerPaths.includes(r.path) && (
    // Dated titles on new runs disambiguate backfills; legacy dispatch inputs are not exposed by REST.
    r.display_title?.startsWith('I2 forecast date=')
      ? (r.display_title.slice(17) || r.created_at.slice(0,10)) === date
      : true
  ));
}

export async function loadForecast({date, file, mode='required', fetchImpl=fetch, env=process.env}) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || new Date(`${date}T00:00:00Z`).toISOString().slice(0,10) !== date) throw new Error('Invalid I2_DATE');
  if (fs.existsSync(file)) return validateForecast(JSON.parse(fs.readFileSync(file,'utf8')),date);
  if (mode !== 'auto') throw new Error(`Missing prediction artifact: ${file}`);
  // Only the verified manual-only era can be classified as intentionally ungenerated.
  const since='2026-10-01';
  if (date < since) throw new Error(`Missing historical prediction artifact: ${file}`);
  for (const workflow of producerPaths) {
    if (/^\s+schedule:/m.test(fs.readFileSync(workflow,'utf8'))) throw new Error('Scheduled producer requires a frozen forecast');
  }
  const history=execFileSync('git',['log','HEAD','--format=%H','--',file],{encoding:'utf8'}).trim();
  if (history) throw new Error(`Missing persisted prediction artifact: ${file}`);
  const siblings=fs.readdirSync(path.dirname(file)).filter(f=>f.startsWith(`${date}_`) && !f.endsWith('_postmortem.json') && /(?:predictions|run_environment|empirical_snapshot)/.test(f));
  if (siblings.length || fs.existsSync(file.replace('_frozen_predictions.json','_postmortem.json'))) throw new Error(`Missing frozen artifact with existing forecast evidence: ${siblings.join(', ')}`);
  if (!env.GITHUB_TOKEN || !env.GITHUB_REPOSITORY) throw new Error('Cannot verify no-forecast status without GitHub Actions history');
  // Fail closed on API errors, retention gaps, or pagination limits; never infer absence from a failed lookup.
  if (Date.now()-Date.parse(`${date}T00:00:00Z`) > 85*86400000) throw new Error('Requested date is outside reliable Actions retention; no-forecast status needs review');
  let count=0;
  for (let page=1;page<=10;page++) {
    const url=`https://api.github.com/repos/${env.GITHUB_REPOSITORY}/actions/runs?branch=main&created=%3E%3D${since}&per_page=100&page=${page}`;
    const res=await fetchImpl(url,{headers:{authorization:`Bearer ${env.GITHUB_TOKEN}`,accept:'application/vnd.github+json'}});
    if (!res.ok) throw new Error(`Cannot verify producer history: HTTP ${res.status}`);
    const body=await res.json();
    if (!Array.isArray(body.workflow_runs) || !Number.isInteger(body.total_count)) throw new Error('Invalid Actions history response');
    const runs=relevantRuns(body.workflow_runs,date);
    if (runs.length) throw new Error(`Missing expected frozen prediction; producer runs: ${runs.map(r=>`${r.id}:${r.conclusion||r.status}`).join(', ')}`);
    count+=body.workflow_runs.length;
    if (count>=body.total_count) return null;
    if (!body.workflow_runs.length) break;
  }
  throw new Error('Incomplete producer history; no-forecast status unverified');
}
