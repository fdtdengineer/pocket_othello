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
- Engine layout prepared for a future Python/PyTorch DQN implementation
- No browser build process

## Engine Architecture

The reusable game and AI logic lives under `engine/`:

- `engine/javascript/rules.js`: legal moves, transitions, pass handling, and terminal rules
- `engine/javascript/evaluation.js`: heuristic position evaluation
- `engine/javascript/search.js`: Easy, Normal, and Hard agents
- `engine/javascript/index.js`: stable JavaScript entry point
- `engine/codingame/othello.js`: standalone CodinGame JavaScript submission
- `engine/python/`: reserved package boundary for the future PyTorch/DQN agent

The root `engine.js` and `ai.js` files are compatibility entry points for the existing web app. They contain no engine implementation and only re-export modules from `engine/javascript/`.

See [`engine/README.md`](engine/README.md) for the shared board representation and action-space contract.

## CodinGame Validation

The heuristic search agent can be tested directly in the CodinGame Othello arena:

https://www.codingame.com/multiplayer/bot-programming/othello-1

1. Open the arena and select JavaScript.
2. Copy the complete contents of `engine/codingame/othello.js`.
3. Paste it into the CodinGame editor and run the test cases or submit it to the arena.

The file is dependency-free and requires no bundling. CodinGame allows 150 ms per normal turn; the bot uses a 100 ms search budget and validates its selected coordinate against the referee-provided legal-action list before printing it.

## Future Python/PyTorch DQN

The future reinforcement-learning implementation should be added under `engine/python/`. JavaScript, Python, and CodinGame implementations share these conventions:

- flat row-major board with 64 cells
- `0` for empty, `1` for black, and `-1` for white
- action index `row * 8 + column`, from `0` to `63`
- a fixed 64-action output with illegal-action masking

This allows the DQN environment, trained model, browser heuristic agent, and CodinGame adapter to be evaluated against the same rules and action encoding.

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

```bash
npm test
npm run check
```

`npm test` validates the game rules, CPU agents, 150 ms search cap, and the standalone CodinGame source constraints.

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
│       └── README.md
├── online.js
├── manifest.webmanifest
├── service-worker.js
├── icon.svg
├── tests/
│   ├── engine.test.mjs
│   └── codingame.test.mjs
└── .github/workflows/pages.yml
```

## License

MIT License
