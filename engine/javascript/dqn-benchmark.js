function percentile(sorted, fraction) {
  if (sorted.length === 0) return null;
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(sorted.length * fraction) - 1),
  );
  return sorted[index];
}

export function summarizeDurations(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error('At least one duration is required.');
  }
  if (values.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error('Durations must be finite non-negative numbers.');
  }

  const sorted = values.slice().sort((a, b) => a - b);
  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];

  return {
    samples: sorted.length,
    min: sorted[0],
    max: sorted.at(-1),
    mean,
    median,
    p95: percentile(sorted, 0.95),
  };
}

export async function runDqnBenchmark(client, options) {
  const {
    board,
    player,
    legalMoves,
    warmup = 3,
    iterations = 30,
  } = options;
  if (!Number.isInteger(warmup) || warmup < 0) {
    throw new Error('Warmup must be a non-negative integer.');
  }
  if (!Number.isInteger(iterations) || iterations <= 0) {
    throw new Error('Iterations must be a positive integer.');
  }

  const initializedAt = performance.now();
  const metadata = await client.ensureReady();
  const initializationMs = performance.now() - initializedAt;

  for (let index = 0; index < warmup; index += 1) {
    await client.chooseIndex(board, player, legalMoves);
  }

  const inferenceDurations = [];
  const roundTripDurations = [];
  let lastAction = null;
  for (let index = 0; index < iterations; index += 1) {
    const startedAt = performance.now();
    const result = await client.chooseIndex(board, player, legalMoves);
    roundTripDurations.push(performance.now() - startedAt);
    inferenceDurations.push(result.inferenceMs);
    lastAction = result.index;
  }

  return {
    initializationMs,
    warmup,
    iterations,
    lastAction,
    metadata,
    inference: summarizeDurations(inferenceDurations),
    roundTrip: summarizeDurations(roundTripDurations),
  };
}
