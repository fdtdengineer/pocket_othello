import assert from 'node:assert/strict';

import {
  runDqnBenchmark,
  summarizeDurations,
} from '../engine/javascript/dqn-benchmark.js';

const summary = summarizeDurations([4, 1, 3, 2, 100]);
assert.deepEqual(summary, {
  samples: 5,
  min: 1,
  max: 100,
  mean: 22,
  median: 3,
  p95: 100,
});
assert.throws(() => summarizeDurations([]), /At least one/);
assert.throws(() => summarizeDurations([1, -1]), /non-negative/);

let calls = 0;
const client = {
  async ensureReady() {
    return { onnx: { sizeBytes: 12345 } };
  },
  async chooseIndex() {
    calls += 1;
    return { index: 19, inferenceMs: calls };
  },
};

const result = await runDqnBenchmark(client, {
  board: Array(64).fill(0),
  player: 1,
  legalMoves: [{ index: 19 }],
  warmup: 2,
  iterations: 4,
});
assert.equal(calls, 6);
assert.equal(result.lastAction, 19);
assert.equal(result.warmup, 2);
assert.equal(result.iterations, 4);
assert.deepEqual(result.inference, {
  samples: 4,
  min: 3,
  max: 6,
  mean: 4.5,
  median: 4.5,
  p95: 6,
});
assert.equal(result.metadata.onnx.sizeBytes, 12345);

console.log('All DQN benchmark tests passed.');
