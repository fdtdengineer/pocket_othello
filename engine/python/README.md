# Python/PyTorch DQN Engine

This directory contains the Python side of the engine. The rules layer is implemented now; PyTorch training and browser-model export will be added in later stages.

## Current layout

```text
engine/python/
├── pyproject.toml
├── othello/
│   ├── __init__.py
│   └── rules.py          # Dependency-free rules matching JavaScript
├── tests/
│   └── test_rules.py     # Unit and JavaScript parity tests
└── README.md
```

Future DQN components should be added without coupling them to the browser UI:

```text
engine/python/
├── dqn/
│   ├── model.py
│   ├── replay_buffer.py
│   ├── agent.py
│   ├── self_play.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
└── checkpoints/          # Ignored model artifacts
```

## Shared contract

The Python implementation follows `engine/README.md` and the JavaScript rules exactly:

- flat row-major board with 64 cells
- `0` empty, `1` black, `-1` white
- action indices from `0` to `63`
- automatic pass handling
- terminal state when neither player has a legal move
- a 64-element boolean legal-action mask

The rules module intentionally has no NumPy or PyTorch dependency. Tensor encoding and neural-network code should depend on this API rather than reimplementing Othello transitions.

## Example

```python
from othello import BLACK, apply_move, create_initial_board, get_legal_moves

board = create_initial_board()
move = get_legal_moves(board, BLACK)[0]
next_board = apply_move(board, move, BLACK)
```

## Validation

From the repository root:

```bash
python -m unittest discover -s engine/python/tests -p "test_*.py"
```

The parity test asks the JavaScript engine to generate deterministic complete-game positions, every legal move, resulting boards, pass decisions, and terminal decisions. Python recomputes each result and requires an exact match.
