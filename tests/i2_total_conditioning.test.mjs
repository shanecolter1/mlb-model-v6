import assert from 'node:assert/strict';
import {
  conditionFullI2Over,
  conditionI2Projection,
} from '../src/model/i2_total_conditioning.js';

const broadOver = 4668 / 10730;

{
  const out = conditionFullI2Over({ rawOver: broadOver, totalPriorOver: 907 / 2146, broadOver });
  assert.ok(Math.abs(out.conditionedOver - 907 / 2146) < 1e-12, 'neutral baseball delta must return the total-conditioned prior');
}

{
  const rawOver = 0.49;
  const out = conditionFullI2Over({ rawOver, totalPriorOver: broadOver, broadOver });
  assert.ok(Math.abs(out.conditionedOver - rawOver) < 1e-12, 'broad prior must preserve the baseball-only projection');
}

{
  const out = conditionI2Projection({
    rawOver: 0.45,
    rawTopOver: 0.26,
    rawBottomOver: 0.25,
    rawExact: { '0': 0.55, '1': 0.22, '2': 0.12, '3': 0.06, '4+': 0.05 },
    totalPriorOver: 0.3618320610687023,
    broadOver,
  });
  const combined = 1 - (1 - out.top2Score) * (1 - out.bottom2Score);
  assert.ok(Math.abs(combined - out.over05) < 1e-10, 'conditioned half-inning probabilities must recombine to the conditioned full inning');
  const exactSum = Object.values(out.fullI2.exact).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(exactSum - 1) < 1e-12, 'conditioned exact distribution must sum to one');
  assert.ok(Math.abs(out.fullI2.exact['0'] - out.under05) < 1e-12, 'P(0) must equal Under 0.5 probability');
  assert.ok(out.over05 < 0.45, 'a low-total prior should pull an above-baseline raw Over probability downward');
}

console.log('i2_total_conditioning tests passed');
