# Othello Engine

The game UI and the decision engines are kept separate so that browser agents, CodinGame submissions, and Python/PyTorch agents can share the same rules and action conventions.

## Directory layout

```text
engine/
├── README.md
├── javascript/
│   ├── rules.js        # Browser rules and board transitions
│   ├── evaluation.js   # Heuristic board evaluation
│   ├── search.js       # Easy/Normal agents and alpha-beta search
│   └── index.js        # Public JavaScript entry point
├── codingame/
│   └── othello.js      # Dependency-free JavaScript submission
└── python/
    ├── pyproject.toml
    ├── othello/
    │   ├── __init__.py
    │   └── rules.py    # Rules-compatible Python implementation
    ├── dqn/
    │   ├── encoding.py
    │   ├── model.py
    │   ├── replay_buffer.py
    │   └── agent.py
    └── tests/
        ├── test_rules.py
        └── test_dqn.py
```

## Shared representation contract

All implementations use the following representation unless a boundary adapter explicitly converts it:

- Board: flat array of 64 cells in row-major order
- Cell values: `0` empty, `1` black, `-1` white
- Action: integer index `row * 8 + column`, from `0` through `63`
- Initial player: black (`1`)
- Pass: no action is emitted when the player has no legal move; the environment advances the turn
- Terminal state: neither player has a legal move
- DQN action mask: 64 booleans, with `true` only for legal squares
- Recommended terminal reward: `+1` win, `0` draw, `-1` loss from the acting agent's perspective

Keeping the action space fixed at 64 allows the DQN to output one Q-value per square and mask illegal actions before selecting `argmax`.

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

## Python rules API

The Python package mirrors the JavaScript rule behavior and has no NumPy or PyTorch dependency:

```python
from othello import (
    BLACK,
    apply_move,
    create_initial_board,
    get_legal_moves,
    legal_action_mask,
)

board = create_initial_board()
move = get_legal_moves(board, BLACK)[0]
next_board = apply_move(board, move, BLACK)
mask = legal_action_mask(next_board, -BLACK)
```

Neural-network code depends on this rules API rather than implementing a second set of game transitions.

## Compact Dueling Double DQN

The Python DQN package currently provides:

- player-relative `4 × 8 × 8` observations
- a 56,978-parameter residual dueling Q-network
- one Q-value for each of the 64 board squares
- epsilon-greedy legal-action selection
- a CPU-backed replay buffer
- Double-DQN target computation
- hard and soft target-network updates
- Smooth L1 optimization with gradient clipping

The replay transition stores a signed bootstrap factor. It is `-1` when the opponent acts next and `+1` when an opponent pass lets the same player move again. This preserves the zero-sum meaning of player-relative Q-values.

Training loops, teacher-data generation, self-play, evaluation, and ONNX export remain separate stages and will build on these primitives.

## Cross-language validation

`tests/export_js_rule_cases.mjs` generates deterministic complete games from the JavaScript engine. The Python tests compare, for every generated position:

- piece counts
- legal moves and exact flip lists
- 64-element legal-action masks
- every legal resulting board
- pass and next-player decisions
- terminal and winner decisions

Run lightweight checks from the repository root:

```bash
npm test
npm run check
```

After installing the optional PyTorch dependency, run:

```bash
npm run test:dqn
```

## CodinGame validation

Open the Othello bot-programming arena:

https://www.codingame.com/multiplayer/bot-programming/othello-1

Select JavaScript, then copy the complete contents of:

```text
engine/codingame/othello.js
```

The file is standalone: it contains no imports, exports, browser APIs, or build step. It parses the CodinGame board and legal-action input, searches for at most 100 ms, validates its selected coordinate against the referee-provided legal actions, and prints one move.
