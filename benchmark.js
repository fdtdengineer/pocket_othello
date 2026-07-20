import { DqnClient } from './engine/javascript/dqn-client.js';
import { runDqnBenchmark } from './engine/javascript/dqn-benchmark.js';
import {
  BLACK,
  createInitialBoard,
  getLegalMoves,
} from './engine/javascript/rules.js';

const elements = {
  deviceInfo: document.querySelector('#deviceInfo'),
  warmupInput: document.querySelector('#warmupInput'),
  iterationsInput: document.querySelector('#iterationsInput'),
  runButton: document.querySelector('#runButton'),
  status: document.querySelector('#benchmarkStatus'),
  resultCard: document.querySelector('#resultCard'),
  resultMetrics: document.querySelector('#resultMetrics'),
  resultJson: document.querySelector('#resultJson'),
  copyButton: document.querySelector('#copyButton'),
};

const client = new DqnClient({ timeoutMs: 60_000 });
let latestResult = null;

function addMetric(container, label, value) {
  const wrapper = document.createElement('div');
  const term = document.createElement('dt');
  const description = document.createElement('dd');
  term.textContent = label;
  description.textContent = value;
  wrapper.append(term, description);
  container.append(wrapper);
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(3)} ms` : 'n/a';
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(2)} MB`;
}

function coordinate(index) {
  if (!Number.isInteger(index)) return 'n/a';
  return `${String.fromCharCode(97 + (index % 8))}${Math.floor(index / 8) + 1}`;
}

function renderDeviceInfo() {
  elements.deviceInfo.replaceChildren();
  addMetric(elements.deviceInfo, 'User agent', navigator.userAgent);
  addMetric(elements.deviceInfo, 'Logical cores', String(navigator.hardwareConcurrency || 'unknown'));
  addMetric(
    elements.deviceInfo,
    'Device memory',
    navigator.deviceMemory ? `${navigator.deviceMemory} GB` : 'not reported',
  );
  addMetric(elements.deviceInfo, 'Cross-origin isolated', String(crossOriginIsolated));
  addMetric(elements.deviceInfo, 'Screen', `${screen.width} × ${screen.height} @ ${devicePixelRatio}x`);
}

function renderResult(result) {
  elements.resultMetrics.replaceChildren();
  const modelBytes = result.metadata?.onnx?.sizeBytes;
  addMetric(elements.resultMetrics, 'Initialization', formatMs(result.initializationMs));
  addMetric(elements.resultMetrics, 'Inference median', formatMs(result.inference.median));
  addMetric(elements.resultMetrics, 'Inference p95', formatMs(result.inference.p95));
  addMetric(elements.resultMetrics, 'Round-trip median', formatMs(result.roundTrip.median));
  addMetric(elements.resultMetrics, 'Round-trip p95', formatMs(result.roundTrip.p95));
  addMetric(elements.resultMetrics, 'Model size', formatBytes(modelBytes));
  addMetric(elements.resultMetrics, 'Selected move', coordinate(result.lastAction));
  addMetric(elements.resultMetrics, 'Samples', String(result.iterations));
  elements.resultJson.textContent = JSON.stringify(result, null, 2);
  elements.resultCard.hidden = false;
}

async function run() {
  const warmup = Number.parseInt(elements.warmupInput.value, 10);
  const iterations = Number.parseInt(elements.iterationsInput.value, 10);
  elements.runButton.disabled = true;
  elements.status.classList.remove('is-error');
  elements.status.textContent = 'Loading the local WASM runtime and ONNX model…';
  elements.resultCard.hidden = true;

  try {
    const board = createInitialBoard();
    const legalMoves = getLegalMoves(board, BLACK);
    const benchmark = await runDqnBenchmark(client, {
      board,
      player: BLACK,
      legalMoves,
      warmup,
      iterations,
    });
    latestResult = {
      capturedAt: new Date().toISOString(),
      environment: {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency || null,
        deviceMemoryGb: navigator.deviceMemory || null,
        crossOriginIsolated,
        screen: {
          width: screen.width,
          height: screen.height,
          devicePixelRatio,
        },
      },
      ...benchmark,
    };
    renderResult(latestResult);
    elements.status.textContent = 'Benchmark complete.';
  } catch (error) {
    latestResult = null;
    elements.status.classList.add('is-error');
    elements.status.textContent = error instanceof Error
      ? `Benchmark failed: ${error.message}`
      : 'Benchmark failed.';
  } finally {
    elements.runButton.disabled = false;
  }
}

elements.runButton.addEventListener('click', run);
elements.copyButton.addEventListener('click', async () => {
  if (!latestResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(latestResult, null, 2));
    elements.copyButton.textContent = 'Copied';
    setTimeout(() => { elements.copyButton.textContent = 'Copy JSON'; }, 1200);
  } catch {
    elements.copyButton.textContent = 'Copy failed';
  }
});

renderDeviceInfo();
window.addEventListener('pagehide', () => client.destroy());

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./service-worker.js').catch(() => {});
  });
}
