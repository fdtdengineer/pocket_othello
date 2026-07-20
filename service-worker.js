const CACHE_NAME = 'pocket-othello-v6';
const LOCAL_ASSETS = [
  './',
  './index.html',
  './benchmark.html',
  './style.css',
  './dqn.css',
  './benchmark.css',
  './app.js',
  './benchmark.js',
  './engine.js',
  './ai.js',
  './engine/javascript/rules.js',
  './engine/javascript/evaluation.js',
  './engine/javascript/search.js',
  './engine/javascript/index.js',
  './engine/javascript/dqn-core.js',
  './engine/javascript/dqn-client.js',
  './engine/javascript/dqn-worker.js',
  './engine/javascript/dqn-benchmark.js',
  './online.js',
  './manifest.webmanifest',
  './icon.svg',
];
const OPTIONAL_ASSETS = [
  './engine/models/othello_dqn.onnx',
  './engine/models/othello_dqn.json',
];
const RUNTIME_MANIFEST = './vendor/onnxruntime-web/manifest.json';

async function cacheOptionalAssets(cache) {
  await Promise.allSettled(OPTIONAL_ASSETS.map((asset) => cache.add(asset)));

  try {
    const response = await fetch(RUNTIME_MANIFEST, { cache: 'no-cache' });
    if (!response.ok) return;
    const manifest = await response.clone().json();
    await cache.put(RUNTIME_MANIFEST, response);
    if (!Array.isArray(manifest.files)) return;
    await Promise.allSettled(
      manifest.files.map((file) => cache.add(`./vendor/onnxruntime-web/${file}`)),
    );
  } catch {
    // DQN assets are optional. The browser falls back to the Hard CPU.
  }
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(async (cache) => {
      await cache.addAll(LOCAL_ASSETS);
      await cacheOptionalAssets(cache);
    }),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)),
    )),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
        }
        return response;
      })
      .catch(() => caches.match(request).then(
        (cached) => cached || caches.match('./index.html'),
      )),
  );
});
