const DEFAULT_TIMEOUT_MS = 30_000;

export class DqnClient {
  constructor(options = {}) {
    this.workerUrl = options.workerUrl
      || new URL('./dqn-worker.js', import.meta.url);
    this.runtimeModuleUrl = options.runtimeModuleUrl
      || new URL('../../vendor/onnxruntime-web/ort.wasm.min.mjs', import.meta.url).href;
    this.wasmBaseUrl = options.wasmBaseUrl
      || new URL('../../vendor/onnxruntime-web/', import.meta.url).href;
    this.modelUrl = options.modelUrl
      || new URL('../models/othello_dqn.onnx', import.meta.url).href;
    this.metadataUrl = options.metadataUrl
      || new URL('../models/othello_dqn.json', import.meta.url).href;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.workerFactory = options.workerFactory
      || ((url) => new Worker(url, { type: 'module', name: 'pocket-othello-dqn' }));
    this.worker = null;
    this.pending = new Map();
    this.nextId = 1;
    this.readyPromise = null;
    this.metadata = null;
  }

  _ensureWorker() {
    if (this.worker) return this.worker;
    this.worker = this.workerFactory(this.workerUrl);
    this.worker.addEventListener('message', (event) => this._handleMessage(event.data));
    this.worker.addEventListener('error', (event) => {
      this._rejectAll(new Error(event.message || 'DQN worker failed.'));
    });
    return this.worker;
  }

  _handleMessage(message) {
    if (!message || !this.pending.has(message.id)) return;
    const pending = this.pending.get(message.id);
    this.pending.delete(message.id);
    clearTimeout(pending.timer);

    if (message.type === 'error') {
      pending.reject(new Error(message.message || 'DQN inference failed.'));
      return;
    }
    pending.resolve(message);
  }

  _rejectAll(error) {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.pending.clear();
  }

  _request(type, payload = {}) {
    const worker = this._ensureWorker();
    const id = this.nextId;
    this.nextId += 1;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`DQN ${type} request timed out.`));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      worker.postMessage({ id, type, ...payload });
    });
  }

  async ensureReady() {
    if (!this.readyPromise) {
      this.readyPromise = this._request('init', {
        options: {
          runtimeModuleUrl: this.runtimeModuleUrl,
          wasmBaseUrl: this.wasmBaseUrl,
          modelUrl: this.modelUrl,
          metadataUrl: this.metadataUrl,
        },
      }).then((message) => {
        this.metadata = message.metadata || null;
        return this.metadata;
      }).catch((error) => {
        this.readyPromise = null;
        throw error;
      });
    }
    return this.readyPromise;
  }

  async chooseIndex(board, player, legalMoves) {
    const legalIndices = legalMoves.map((move) => (
      typeof move === 'number' ? move : move.index
    ));
    if (legalIndices.length === 0) return null;
    await this.ensureReady();
    const message = await this._request('infer', {
      board: board.slice(),
      player,
      legalIndices,
      options: {
        runtimeModuleUrl: this.runtimeModuleUrl,
        wasmBaseUrl: this.wasmBaseUrl,
        modelUrl: this.modelUrl,
        metadataUrl: this.metadataUrl,
      },
    });
    return {
      index: message.index,
      inferenceMs: message.inferenceMs,
    };
  }

  destroy() {
    if (this.worker) this.worker.terminate();
    this.worker = null;
    this.readyPromise = null;
    this.metadata = null;
    this._rejectAll(new Error('DQN client was destroyed.'));
  }
}
