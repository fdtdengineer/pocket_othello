import assert from 'node:assert/strict';
import path from 'node:path';

import * as ort from 'onnxruntime-web';

import {
  DQN_INPUT_NAME,
  DQN_OUTPUT_NAME,
  encodeDqnObservation,
  selectLegalAction,
} from '../engine/javascript/dqn-core.js';

const modelPath = process.argv[2];
if (!modelPath) throw new Error('Pass an ONNX model path to the Web runtime test.');

ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create(path.resolve(modelPath), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});

const board = Array(64).fill(0);
board[27] = -1;
board[28] = 1;
board[35] = 1;
board[36] = -1;
const legalIndices = [19, 26, 37, 44];
const observation = encodeDqnObservation(board, 1, legalIndices);
const tensor = new ort.Tensor('float32', observation, [1, 4, 8, 8]);
const outputs = await session.run({ [DQN_INPUT_NAME]: tensor });
const qValues = outputs[DQN_OUTPUT_NAME].data;

assert.equal(qValues.length, 64);
const action = selectLegalAction(qValues, legalIndices);
assert.ok(legalIndices.includes(action));

console.log(`ONNX Runtime Web selected legal action ${action}.`);
