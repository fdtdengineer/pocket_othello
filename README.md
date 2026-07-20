# Pocket Othello

A responsive Othello/Reversi web game for iPhone, Android, and desktop browsers. The game UI uses plain HTML, CSS, and JavaScript and is deployed with GitHub Pages.

## Play Online

https://fdtdengineer.github.io/pocket_othello/

## Features

- Standard 8×8 Othello rules, legal-move hints, pass handling, and score tracking
- Easy, Normal, Hard, and on-device DQN CPU opponents
- Time-bounded iterative-deepening alpha-beta search
- ONNX Runtime Web inference in a dedicated Web Worker
- Safe Hard-CPU fallback when a trained ONNX model is not deployed
- Local two-player and browser-to-browser online modes
- Responsive mobile layout with iPhone safe-area and Safari toolbar clearance
- Installable PWA with optional offline DQN runtime/model caching
- Standalone CodinGame validation bot
- JavaScript/Python rules parity tests
- PyTorch teacher-data, imitation, reinforcement, and ONNX export pipeline

## Engine Architecture

The reusable game and AI logic lives under `engine/`:

- `engine/javascript/rules.js`: legal moves, transitions, pass handling, and terminal rules
- `engine/javascript/evaluation.js`: heuristic position evaluation
- `engine/javascript/search.js`: Easy, Normal, and Hard agents
- `engine/javascript/dqn-core.js`: browser observation encoding and legal Q-value selection
- `engine/javascript/dqn-client.js`: asynchronous Web Worker client
- `engine/javascript/dqn-worker.js`: ONNX Runtime Web inference
- `engine/codingame/othello.js`: standalone CodinGame JavaScript submission
- `engine/python/`: verified rules, Dueling Double DQN, training, and ONNX export
- `engine/models/`: deployable ONNX model and metadata

The root `engine.js` and `ai.js` files remain compatibility entry points for the existing web app.

## CodinGame Validation

Open the CodinGame Othello arena:

https://www.codingame.com/multiplayer/bot-programming/othello-1

Select JavaScript and paste the complete contents of `engine/codingame/othello.js`. The standalone bot uses a 100 ms search budget under the normal 150 ms turn limit and validates its output against the referee-provided legal moves.

## DQN Pipeline

The DQN uses a player-relative `4 × 8 × 8` observation containing own discs, opponent discs, legal moves, and game progress. Its 64 outputs correspond to row-major board actions. The default residual dueling network has 56,978 trainable parameters, approximately 228 KB of FP32 weights before ONNX packaging.

### 1. Generate Hard-CPU teacher data

```bash
npm run generate:teacher -- \
  --games 1000 \
  --output engine/python/data/teacher.jsonl
```

### 2. Imitation pretraining

```bash
npm run train:imitation -- \
  --data engine/python/data/teacher.jsonl \
  --output engine/python/checkpoints/imitation.pt
```

### 3. Mixed-opponent reinforcement training

```bash
npm run train:reinforcement -- \
  --init-checkpoint engine/python/checkpoints/imitation.pt \
  --output engine/python/checkpoints/reinforcement.pt
```

### 4. Browser ONNX export

```bash
npm run export:onnx -- \
  --checkpoint engine/python/checkpoints/reinforcement.pt \
  --output engine/models/othello_dqn.onnx
```

The exporter also writes `engine/models/othello_dqn.json` with the model contract, byte size, SHA-256 hash, and PyTorch/ONNX Runtime parity result. Until both inference files are deployed, the DQN button automatically uses the Hard CPU instead.

## Run Locally

Install the browser runtime and prepare its static WASM files:

```bash
npm install
npm run prepare:web
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

PyTorch and ONNX tooling can be installed separately:

```bash
python -m pip install -e "engine/python[all]"
```

## Tests

```bash
npm test
npm run test:dqn
npm run test:teacher
npm run test:imitation
npm run test:reinforcement
npm run test:onnx
npm run check
```

CI additionally creates an ONNX fixture and executes it through the JavaScript `onnxruntime-web` WASM backend.

## GitHub Pages Deployment

`.github/workflows/pages.yml` installs `onnxruntime-web`, copies only the WASM execution assets into the static artifact, and deploys `_site/`. The public application and inference runtime are served from the same GitHub Pages origin.

## Project Structure

```text
.
├── index.html
├── style.css
├── dqn.css
├── app.js
├── engine.js
├── ai.js
├── engine/
│   ├── javascript/
│   │   ├── rules.js
│   │   ├── evaluation.js
│   │   ├── search.js
│   │   ├── dqn-core.js
│   │   ├── dqn-client.js
│   │   └── dqn-worker.js
│   ├── codingame/
│   │   └── othello.js
│   ├── models/
│   │   └── README.md
│   └── python/
│       ├── othello/
│       ├── dqn/
│       ├── tests/
│       └── README.md
├── scripts/
│   └── prepare-web.mjs
├── tests/
├── online.js
├── service-worker.js
└── .github/workflows/
```

## License

MIT License
