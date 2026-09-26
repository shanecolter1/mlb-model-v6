import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import assert from 'node:assert/strict';

const read = file => fs.readFileSync(file, 'utf8');

function filesUnder(root, extensions) {
  const out = [];
  const walk = dir => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (extensions.some(ext => entry.name.endsWith(ext))) out.push(full);
    }
  };
  walk(root);
  return out;
}

test('production I2 source tree has no retired Netlify MLB upstream dependency', () => {
  const files = [
    ...filesUnder('src/pipeline', ['.mjs', '.js']),
    ...filesUnder('src/inputs', ['.mjs', '.js']),
    ...filesUnder('.github/workflows', ['.yml', '.yaml']),
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
  const files = filesUnder('src/inputs', ['.mjs', '.js']);
  for (const file of files) {
    const source = read(file);
    assert.doesNotMatch(source, /api\.rotowire\.com/i, `${file} must not call the paid RotoWire API`);
    assert.doesNotMatch(source, /ROTOWIRE_API_KEY/, `${file} must not depend on a RotoWire API key`);
  }
  assert.match(read('src/inputs/rotowire_public.mjs'), /fetchRotowirePublic/);
});

test('any workflow with a paid SportsGameOdds call is manual-only', () => {
  const workflows = filesUnder('.github/workflows', ['.yml', '.yaml']);
  for (const file of workflows) {
    const source = read(file);
    const canCallPaidSportsbook =
      /SPORTSGAMEODDS_API_KEY/.test(source) ||
      /fetch_i2_run_environment\.mjs/.test(source) ||
      /fetch_sgo_mlb_inning_markets\.mjs/.test(source);
    if (!canCallPaidSportsbook) continue;

    assert.doesNotMatch(source, /^\s{2}push:/m, `${file} must not auto-run paid sportsbook access on push`);
    assert.doesNotMatch(source, /^\s{2}schedule:/m, `${file} must not auto-run paid sportsbook access on schedule`);
    assert.doesNotMatch(source, /^\s{2}workflow_run:/m, `${file} must not auto-run paid sportsbook access after another workflow`);
    assert.doesNotMatch(source, /^\s{2}pull_request:/m, `${file} must not auto-run paid sportsbook access on pull requests`);
    assert.match(source, /^\s{2}workflow_dispatch:/m, `${file} paid sportsbook access must require explicit dispatch`);
  }
});

test('locked run environment is reused before any paid refresh', () => {
  const source = read('src/pipeline/fetch_i2_run_environment.mjs');
  assert.match(source, /I2_REFRESH_RUN_ENVIRONMENT/);
  assert.match(source, /VALIDATED_EXISTING_LOCKED_ARTIFACT/);
  assert.match(source, /paidApiRequestMade:\s*false/);
});
