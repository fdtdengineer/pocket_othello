import {
  DQN_CHANNELS,
  DQN_INPUT_NAME,
  DQN_OUTPUT_NAME,
  encodeDqnObservation,
  selectLegalAction,
} from './dqn-core.js';

let ortModule = null;
let sessionPromise = null;
let loadedMetadata = null;

async function loadSession(options = {}) {
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async () => {
    const runtimeModuleUrl = options.runtimeModuleUrl
      || new URL('../../vendor/onnxruntime-web/ort.wasm.min.mjs', import.meta.url).href;
    const wasmBaseUrl = options.wasmBaseUrl
      || new URL('../../vendor/onnxruntime-web/', import.meta.url).href;
    const modelUrl = options.modelUrl
      || new URL('../models/othello_dqn.onnx', import.meta.url).href;
    const metadataUrl = options.metadataUrl
      || new URL('../models/othello_dqn.json', import.meta.url).href;

    ortModule = await import(runtimeModuleUrl);
    ortModule.env.wasm.wasmPaths = wasmBaseUrl;
    ortModule.env.wasm.numThreads = 1;

    try {
      const response = await fetch(metadataUrl, { cache: 'no-cache' });
      if (response.ok) loadedMetadata = await response.json();
    } catch {
      loadedMetadata = null;
    }

    const session = await ortModule.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return session;
  })();

  try {
    return await sessionPromise;
  } catch (error) {
    sessionPromise = null;
    throw error;
  }
}

async function runInference(message) {
  const session = await loadSession(message.options);
  const observation = encodeDqnObservation(
    message.board,
    message.player,
    message.legalIndices,
  );
  const input = new ortModule.Tensor(
    'float32',
    observation,
    [1, DQN_CHANNELS, 8, 8],
  );

  const startedAt = performance.now();
  const outputs = await session.run({ [DQN_INPUT_NAME]: input });
  const inferenceMs = performance.now() - startedAt;
  const qValues = outputs[DQN_OUTPUT_NAME]?.data;
  const index = selectLegalAction(qValues, message.legalIndices);
  if (index === null) throw new Error('DQN was asked to move without a legal action.');

  return { index, inferenceMs };
}

self.addEventListener('message', async (event) => {
  const message = event.data;
  if (!message || typeof message !== 'object') return;

  try {
    if (message.type === 'init') {
      await loadSession(message.options);
      self.postMessage({
        id: message.id,
        type: 'ready',
        metadata: loadedMetadata,
      });
      return;
    }

    if (message.type === 'infer') {
      const result = await runInference(message);
      self.postMessage({
        id: message.id,
        type: 'result',
        ...result,
      });
    }
  } catch (error) {
    self.postMessage({
      id: message.id,
      type: 'error',
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
