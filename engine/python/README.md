# Python/PyTorch DQN Engine

This directory contains the Python rules layer and the compact PyTorch building blocks for a future browser-deployable Othello agent.

## Current layout

```text
engine/python/
├── pyproject.toml
├── othello/
│   ├── __init__.py
│   └── rules.py          # Dependency-free rules matching JavaScript
├── dqn/
│   ├── __init__.py
│   ├── encoding.py       # Player-relative 4×8×8 observations
│   ├── model.py          # Compact residual dueling Q-network
│   ├── replay_buffer.py  # CPU-backed replay memory
│   └── agent.py          # Legal action selection and Double-DQN updates
├── tests/
│   ├── test_rules.py
│   └── test_dqn.py
└── README.md
```

Training, self-play, evaluation, and ONNX export will be added in later stages without coupling this package to the browser UI.

## Installation

Install the package and PyTorch extra from the repository root:

```bash
python -m pip install -e "engine/python[dqn]"
```

A CPU-only PyTorch wheel can also be installed explicitly before installing the local package:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e engine/python --no-deps
```

## Observation contract

`encode_observation(board, player)` returns a `4 × 8 × 8` tensor:

1. current player's discs
2. opponent's discs
3. legal-action mask
4. constant game-progress plane

The board is always encoded from the acting player's perspective. One model therefore handles both black and white.

## Compact model

The default `DuelingQNetwork` uses:

- 32 feature channels
- three residual blocks
- one scalar value head
- one spatial 8×8 advantage head
- 64 output Q-values

It has 56,978 trainable parameters, corresponding to about 228 KB of FP32 weights before ONNX packaging or quantization. The architecture uses standard convolution, ReLU, pooling, addition, mean, and linear operations to keep later browser export straightforward.

## Two-player Double DQN

The next state is encoded from the next acting player's perspective. Each replay transition therefore stores a `bootstrap_sign`:

- `-1` when play passes to the opponent
- `+1` when the opponent must pass and the same player moves again
- terminal states do not bootstrap

The target is computed as:

```text
target = reward + gamma * bootstrap_sign * next_value
```

The online network selects the next legal action and the target network evaluates it. Illegal moves are masked before every `argmax`.

## Shared rules contract

The Python implementation follows `engine/README.md` and the JavaScript rules exactly:

- flat row-major board with 64 cells
- `0` empty, `1` black, `-1` white
- action indices from `0` to `63`
- automatic pass handling
- terminal state when neither player has a legal move
- a 64-element boolean legal-action mask

The rules module intentionally has no NumPy or PyTorch dependency. Tensor and neural-network code depend on this API rather than reimplementing Othello transitions.

## Validation

Run the lightweight rules and parity tests:

```bash
npm test
```

After installing PyTorch, run the DQN tests:

```bash
npm run test:dqn
```

The DQN tests cover player-relative encoding, output shape, the model-size ceiling, legal-action selection, signed Double-DQN targets, replay-buffer behavior, and a real optimizer update.
