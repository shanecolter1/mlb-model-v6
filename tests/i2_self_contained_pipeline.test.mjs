import fs from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';

const read = file => fs.readFileSync(file, 'utf8');

test('production I2 pipeline has no retired Netlify MLB upstream dependency', () => {
  const files = [
    'src/pipeline/run_i2_total_conditioned.mjs',
    'src/pipeline/run_i2_full_slate_override.mjs',
    'src/pipeline/run_i2_today_upstream_wrapper.mjs',
    '.github/workflows/i2_daily_run.yml',
    '.github/workflows/i2_v04_today.yml',
    '.github/workflows/sportsgameodds_inning_market_sync.yml',
  ];
  for (const file of files) {
    const source = read(file);
    assert.equal(source.includes('MLB_OTHER_MODEL_BASE_URL'), false, `${file} must not use MLB_OTHER_MODEL_BASE_URL`);
    assert.equal(source.includes('lambent-brioche-ca05b1.netlify.app'), false, `${file} must not reference the retired Netlify host`);
    assert.equal(source.includes('/.netlify/functions/mlb'), false, `${file} must not call the Netlify MLB proxy`);
  }
});

test('total-conditioned production imports the direct repository runner', () => {
  const source = read('src/pipeline/run_i2_total_conditioned.mjs');
  assert.match(source, /import\('\.\/run_i2_today\.mjs'\)/);
  assert.doesNotMatch(source, /run_i2_today_upstream_wrapper/);
});

test('provisional RotoWire retrieval is public-page only', () => {
  const source = read('src/inputs/i2_baseball_sources.mjs');
  assert.doesNotMatch(source, /api\.rotowire\.com/i);
  assert.doesNotMatch(source, /ROTOWIRE_API_KEY/);
  assert.match(source, /fetchRotowirePublic/);
});

test('paid sportsbook workflows cannot run from code pushes', () => {
  for (const file of [
    '.github/workflows/i2_daily_run.yml',
    '.github/workflows/i2_v04_today.yml',
    '.github/workflows/sportsgameodds_inning_market_sync.yml',
  ]) {
    const source = read(file);
    assert.doesNotMatch(source, /^\s*push:/m, `${file} must not auto-run on push`);
  }
  const marketSync = read('.github/workflows/sportsgameodds_inning_market_sync.yml');
  assert.doesNotMatch(marketSync, /^\s*workflow_run:/m, 'SportsGameOdds sync must require explicit dispatch');
});

test('locked run environment is reused before any paid refresh', () => {
  const source = read('src/pipeline/fetch_i2_run_environment.mjs');
  assert.match(source, /I2_REFRESH_RUN_ENVIRONMENT/);
  assert.match(source, /VALIDATED_EXISTING_LOCKED_ARTIFACT/);
  assert.match(source, /paidApiRequestMade:\s*false/);
});
