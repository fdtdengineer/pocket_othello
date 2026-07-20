# Othello Engine

The game UI and the decision engines are kept separate so that browser agents, CodinGame submissions, and future Python/PyTorch agents can share the same rules and action conventions.

## Directory layout

```text
engine/
├── README.md
├── javascript/
│   ├── rules.js        # Board transitions, legal moves, pass and terminal rules
│   ├── evaluation.js   # Heuristic board evaluation
│   ├── search.js       # Easy/Normal agents and iterative-deepening alpha-beta
│   └── index.js        # Public JavaScript entry point
├── codingame/
│   └── othello.js      # Dependency-free JavaScript submission
└── python/
    └── README.md       # Planned PyTorch/DQN package boundary
```

## Shared representation contract

All implementations should use the following representation unless a boundary adapter explicitly converts it:

- Board: flat array of 64 cells in row-major order
- Cell values: `0` empty, `1` black, `-1` white
- Action: integer index `row * 8 + column`, from `0` through `63`
- Initial player: black (`1`)
- Pass: no action is emitted when the player has no legal move; the environment advances the turn
- Terminal state: neither player has a legal move
- Reward convention for DQN: recommended final reward `+1` win, `0` draw, `-1` loss from the acting agent's perspective

Keeping the action space fixed at 64 allows a future DQN to output one Q-value per square and mask illegal actions before selecting `argmax`.

## JavaScript API

Import the browser engine from one stable entry point:

```js
import {
  BLACK,
  createInitialBoard,
  getLegalMoves,
  applyMove,
  evaluateBoard,
  chooseCpuMove,
} from './engine/javascript/index.js';
```

The hard agent uses iterative deepening with alpha-beta pruning. CodinGame allows 150 ms per normal turn, so the reusable search engine caps search at 120 ms and defaults to 100 ms, leaving time for parsing and output.

## CodinGame validation

Open the Othello bot-programming arena:

https://www.codingame.com/multiplayer/bot-programming/othello-1

Select JavaScript, then copy the complete contents of:

```text
engine/codingame/othello.js
```

The file is standalone: it contains no imports, exports, browser APIs, or build step. It parses the CodinGame board and legal-action input, searches for at most 100 ms, validates its selected coordinate against the referee-provided legal actions, and prints one move.

## Python/PyTorch boundary

The future Python implementation should live under `engine/python/` rather than being mixed into the browser code. It should mirror the representation contract above and expose an environment API similar to:

```python
reset() -> observation
legal_action_mask() -> bool[64]
step(action: int) -> observation, reward, terminated, info
```

Training, checkpoints, and neural-network code should depend on that environment API, not on UI or CodinGame input/output code.
