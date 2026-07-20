# Python/PyTorch DQN Engine

This directory contains the Python rules layer, compact PyTorch DQN building blocks, and the validated teacher-data interface for a future browser-deployable Othello agent.

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
│   ├── agent.py          # Legal action selection and Double-DQN updates
│   └── teacher_data.py   # Validated JSONL loader and imitation dataset
├── tests/
│   ├── test_rules.py
│   ├── test_dqn.py
│   └── test_teacher_data.py
└── README.md
```

Training, self-play, evaluation, and ONNX export remain independent from the browser UI.

## Installation

```bash
python -m pip install -e "engine/python[dqn]"
```

A CPU-only PyTorch wheel can also be installed explicitly:

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

The default `DuelingQNetwork` uses 32 feature channels, three residual blocks, one scalar value head, one spatial 8×8 advantage head, and 64 output Q-values. It has 56,978 trainable parameters, corresponding to about 228 KB of FP32 weights before ONNX packaging or quantization.

## Two-player Double DQN

Each replay transition stores a `bootstrap_sign`:

- `-1` when play passes to the opponent
- `+1` when the opponent must pass and the same player moves again
- terminal states do not bootstrap

```text
target = reward + gamma * bootstrap_sign * next_value
```

The online network selects the next legal action and the target network evaluates it. Illegal moves are masked before every `argmax`.

## Hard-CPU teacher data

Generate labeled positions from the current JavaScript Hard CPU:

```bash
npm run generate:teacher -- \
  --games 1000 \
  --output engine/python/data/teacher.jsonl \
  --seed 20260720 \
  --time-ms 100 \
  --max-depth 6 \
  --exploration 0.20
```

Each JSONL record contains the 64-cell board, acting player, Hard-CPU action, exact 64-element legal mask, game number, and ply. A neighboring `.meta.json` file records the generator settings and counts. Generated datasets are ignored by Git.

`TeacherExample` validates every record by recomputing the legal mask with the Python rules engine. `TeacherDataset` then exposes `(observation, action)` pairs suitable for supervised imitation learning:

```python
from dqn import TeacherDataset

dataset = TeacherDataset.from_jsonl("engine/python/data/teacher.jsonl")
observation, action = dataset[0]
```

Exploration controls only the move used to advance the generated game. The stored label always comes from `chooseHardMove`, allowing diverse positions without replacing the teacher target.

## Shared rules contract

The Python implementation follows `engine/README.md` and the JavaScript rules exactly:

- flat row-major board with 64 cells
- `0` empty, `1` black, `-1` white
- action indices from `0` to `63`
- automatic pass handling
- terminal state when neither player has a legal move
- a 64-element boolean legal-action mask

## Validation

```bash
npm test
npm run test:dqn
npm run test:teacher
```

The teacher-data test runs the JavaScript generator twice with the same seed, requires byte-identical JSONL and metadata, validates every label against Python rules, checks dataset tensor output, and rejects corrupted masks.
