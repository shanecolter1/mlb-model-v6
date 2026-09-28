function clamp01(x) {
  return Math.max(1e-9, Math.min(1 - 1e-9, Number(x)));
}

function logit(p) {
  const q = clamp01(p);
  return Math.log(q / (1 - q));
}

function logistic(z) {
  return clamp01(1 / (1 + Math.exp(-Number(z))));
}

export function validateHalfContrastArtifact(artifact) {
  if (!artifact || artifact.type !== 'zero_sum_logit_contrast') {
    throw new Error('Invalid I2 half-contrast artifact type');
  }
  if (artifact.market_inputs_used !== false) {
    throw new Error('I2 half-contrast artifact is not market-isolated');
  }
  if (artifact.common_component_applied !== false) {
    throw new Error('I2 half-contrast artifact must not apply the common calibration component');
  }
  const h = Number(artifact.zero_sum_half_contrast_h);
  if (!Number.isFinite(h) || h < 0 || h > 1) {
    throw new Error('Invalid I2 half-contrast magnitude');
  }
  const top = Number(artifact.top_logit_delta);
  const bottom = Number(artifact.bottom_logit_delta);
  if (Math.abs(top + h) > 1e-12 || Math.abs(bottom - h) > 1e-12) {
    throw new Error('I2 half-contrast deltas are not zero-sum');
  }
  return artifact;
}

export function applyHalfScoreContrast({topScoreProbability, bottomScoreProbability, artifact}) {
  validateHalfContrastArtifact(artifact);
  const h = Number(artifact.zero_sum_half_contrast_h);
  const rawTop = clamp01(topScoreProbability);
  const rawBottom = clamp01(bottomScoreProbability);
  const adjustedTop = logistic(logit(rawTop) - h);
  const adjustedBottom = logistic(logit(rawBottom) + h);
  return {
    rawTopScoreProbability: rawTop,
    rawBottomScoreProbability: rawBottom,
    adjustedTopScoreProbability: adjustedTop,
    adjustedBottomScoreProbability: adjustedBottom,
    adjustedUnder05: clamp01((1 - adjustedTop) * (1 - adjustedBottom)),
    h,
  };
}
