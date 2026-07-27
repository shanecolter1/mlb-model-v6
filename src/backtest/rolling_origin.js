/**
 * Creates strictly time-ordered rolling-origin folds.
 */
export function rollingOriginFolds(records, {
  minTrainDates = 30,
  testDates = 7,
  stepDates = 7,
} = {}) {
  const sortedDates = [...new Set(records.map((r) => r.game_date))].sort();
  const folds = [];

  for (
    let trainEnd = minTrainDates;
    trainEnd + testDates <= sortedDates.length;
    trainEnd += stepDates
  ) {
    const trainDateSet = new Set(sortedDates.slice(0, trainEnd));
    const testDateSet = new Set(sortedDates.slice(trainEnd, trainEnd + testDates));
    const train = records.filter((r) => trainDateSet.has(r.game_date));
    const test = records.filter((r) => testDateSet.has(r.game_date));

    folds.push({
      train_start: sortedDates[0],
      train_end: sortedDates[trainEnd - 1],
      test_start: sortedDates[trainEnd],
      test_end: sortedDates[trainEnd + testDates - 1],
      train,
      test,
    });
  }
  return folds;
}
