export function brierScore(probabilities, outcomes) {
  if (probabilities.length !== outcomes.length || probabilities.length === 0) {
    throw new RangeError("Inputs must have equal non-zero length");
  }
  return probabilities.reduce(
    (sum, p, i) => sum + (p - outcomes[i]) ** 2, 0
  ) / probabilities.length;
}

export function logLoss(probabilities, outcomes, epsilon = 1e-15) {
  if (probabilities.length !== outcomes.length || probabilities.length === 0) {
    throw new RangeError("Inputs must have equal non-zero length");
  }
  return probabilities.reduce((sum, p, i) => {
    const clipped = Math.min(1 - epsilon, Math.max(epsilon, p));
    return sum - (
      outcomes[i] * Math.log(clipped) +
      (1 - outcomes[i]) * Math.log(1 - clipped)
    );
  }, 0) / probabilities.length;
}

export function meanAbsoluteError(predictions, actuals) {
  if (predictions.length !== actuals.length || predictions.length === 0) {
    throw new RangeError("Inputs must have equal non-zero length");
  }
  return predictions.reduce(
    (sum, p, i) => sum + Math.abs(p - actuals[i]), 0
  ) / predictions.length;
}

export function rootMeanSquaredError(predictions, actuals) {
  if (predictions.length !== actuals.length || predictions.length === 0) {
    throw new RangeError("Inputs must have equal non-zero length");
  }
  return Math.sqrt(predictions.reduce(
    (sum, p, i) => sum + (p - actuals[i]) ** 2, 0
  ) / predictions.length);
}
