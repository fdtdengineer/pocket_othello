# Pocket Othello

A responsive Othello/Reversi web game for iPhone, Android, and desktop browsers. It is built with plain HTML, CSS, and JavaScript and is ready for GitHub Pages.

## Play Online

https://fdtdengineer.github.io/pocket_othello/

## Features

- Standard 8×8 Othello rules, legal-move hints, pass handling, and score tracking
- CPU opponents with Easy, Normal, and Hard difficulty levels
- Time-bounded iterative-deepening alpha-beta search
- Local two-player mode on one device
- Online two-player matches using a six-character room code
- Direct browser-to-browser game synchronization with PeerJS/WebRTC
- Responsive mobile layout with iPhone safe-area and Safari toolbar clearance
- Installable PWA and offline CPU/local play
- Standalone CodinGame validation bot
- Pure-Python rules engine with JavaScript transition parity tests
- Compact PyTorch Dueling Double-DQN building blocks
- No browser build process

## Engine Architecture

The reusable game and AI logic lives under `engine/`:

- `engine/javascript/rules.js`: legal moves, transitions, pass handling, and terminal rules
- `engine/javascript/evaluation.js`: heuristic position evaluation
- `engine/javascript/search.js`: Easy, Normal, and Hard agents
- `engine/javascript/index.js`: stable JavaScript entry point
- `engine/codingame/othello.js`: standalone CodinGame JavaScript submission
- `engine/python/othello/rules.py`: dependency-free Python rules with the same representation and transitions
- `engine/python/dqn/`: compact observation, model, replay-buffer, and Double-DQN utilities

The root `engine.js` and `ai.js` files are compatibility entry points for the existing web app. They contain no engine implementation and only re-export modules from `engine/javascript/`.

See [`engine/README.md`](engine/README.md) for the shared board representation, action-space contract, and cross-language validation design.

## CodinGame Validation

The heuristic search agent can be tested directly in the CodinGame Othello arena:

https://www.codingame.com/multiplayer/bot-programming/othello-1

1. Open the arena and select JavaScript.
2. Copy the complete contents of `engine/codingame/othello.js`.
3. Paste it into the CodinGame editor and run the test cases or submit it to the arena.

The file is dependency-free and requires no bundling. CodinGame allows 150 ms per normal turn; the bot uses a 100 ms search budget and validates its selected coordinate against the referee-provided legal-action list before printing it.

## Python/PyTorch DQN

The Python rules and initial DQN core are implemented under `engine/python/`. JavaScript, Python, and CodinGame implementations share these conventions:

- flat row-major board with 64 cells
- `0` for empty, `1` for black, and `-1` for white
- action index `row * 8 + column`, from `0` to `63`
- a fixed 64-action output with illegal-action masking
- automatic pass handling and the same terminal-state definition

The DQN input is a player-relative `4 × 8 × 8` tensor containing own discs, opponent discs, legal moves, and game progress. The default residual dueling network has 56,978 trainable parameters, approximately 228 KB of FP32 weights. It includes signed zero-sum Double-DQN targets, replay memory, epsilon-greedy legal action selection, target-network updates, and a tested optimizer step.

Training loops, teacher-data generation, self-play, evaluation, ONNX export, and browser integration are subsequent stages.

## Online Matches

One player selects **Online → Create Room** and shares the six-character code. The other player opens the same site, selects **Online**, enters the code, and joins.

Online matches use PeerJS Cloud for connection signaling and WebRTC for the game data. A small number of restrictive networks or symmetric NAT configurations may require a TURN server and can fail to connect.

## Run Locally

Serve the directory over HTTP:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Tests

Node.js and Python 3.10 or later are required for the lightweight engine suite:

```bash
npm test
npm run check
```

Install the optional PyTorch package and run the DQN tests:

```bash
python -m pip install -e "engine/python[dqn]"
npm run test:dqn
```

`npm test` runs the browser-engine tests, CodinGame source checks, Python rule tests, and JavaScript/Python transition parity checks. `npm run test:dqn` validates the compact network, legal masking, signed Double-DQN targets, replay buffer, and optimizer update.

## GitHub Pages Deployment

The repository includes `.github/workflows/pages.yml`.

After pushing to `main`, open:

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
```

The published URL will be:

```text
https://fdtdengineer.github.io/pocket_othello/
```

## Project Structure

```text
.
├── index.html
├── style.css
├── app.js
├── engine.js                     # Compatibility re-export
├── ai.js                         # Compatibility re-export
├── engine/
│   ├── README.md
│   ├── javascript/
│   │   ├── rules.js
│   │   ├── evaluation.js
│   │   ├── search.js
│   │   └── index.js
│   ├── codingame/
│   │   └── othello.js
│   └── python/
│       ├── pyproject.toml
│       ├── othello/
│       │   ├── __init__.py
│       │   └── rules.py
│       ├── dqn/
│       │   ├── __init__.py
│       │   ├── encoding.py
│       │   ├── model.py
│       │   ├── replay_buffer.py
│       │   └── agent.py
│       ├── tests/
│       │   ├── test_rules.py
│       │   └── test_dqn.py
│       └── README.md
├── online.js
├── manifest.webmanifest
├── service-worker.js
├── icon.svg
├── tests/
│   ├── engine.test.mjs
│   ├── codingame.test.mjs
│   └── export_js_rule_cases.mjs
└── .github/workflows/
    ├── ci.yml
    └── pages.yml
```

## License

MIT License
