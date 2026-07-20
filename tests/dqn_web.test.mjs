import assert from 'node:assert/strict';

import { DqnClient } from '../engine/javascript/dqn-client.js';
import {
  encodeDqnObservation,
  selectLegalAction,
} from '../engine/javascript/dqn-core.js';

const initialBoard = Array(64).fill(0);
initialBoard[27] = -1;
initialBoard[28] = 1;
initialBoard[35] = 1;
initialBoard[36] = -1;
const legalIndices = [19, 26, 37, 44];

const observation = encodeDqnObservation(initialBoard, 1, legalIndices);
assert.equal(observation.length, 256);
assert.equal(observation[28], 1);
assert.equal(observation[64 + 27], 1);
assert.equal(observation[128 + 19], 1);
assert.equal(observation[128 + 20], 0);
assert.equal(observation[192], 4 / 64);
assert.equal(observation[255], 4 / 64);

const qValues = new Float32Array(64);
qValues[19] = 0.1;
qValues[26] = 0.8;
qValues[37] = 0.8;
qValues[44] = 0.2;
assert.equal(selectLegalAction(qValues, legalIndices), 26);
assert.equal(selectLegalAction(qValues, []), null);

class FakeWorker {
  constructor() {
    this.listeners = new Map();
    this.terminated = false;
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  postMessage(message) {
    queueMicrotask(() => {
      const listener = this.listeners.get('message');
      if (!listener) return;
      if (message.type === 'init') {
        listener({ data: { id: message.id, type: 'ready', metadata: { modelType: 'test' } } });
      } else if (message.type === 'infer') {
        listener({ data: { id: message.id, type: 'result', index: 37, inferenceMs: 1.25 } });
      }
    });
  }

  terminate() {
    this.terminated = true;
  }
}

const fakeWorker = new FakeWorker();
const client = new DqnClient({
  workerFactory: () => fakeWorker,
  timeoutMs: 1000,
});
const result = await client.chooseIndex(
  initialBoard,
  1,
  legalIndices.map((index) => ({ index })),
);
assert.equal(result.index, 37);
assert.equal(result.inferenceMs, 1.25);
assert.equal(client.metadata.modelType, 'test');
client.destroy();
assert.equal(fakeWorker.terminated, true);

console.log('All browser DQN core and worker-client tests passed.');
