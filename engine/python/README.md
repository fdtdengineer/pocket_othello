# Python/PyTorch DQN Engine

This directory contains the Python rules layer, compact PyTorch DQN components, validated Hard-CPU teacher data, and supervised imitation pretraining for a future browser-deployable Othello agent.

## Current layout

```text
engine/python/
├── pyproject.toml
├── othello/
│   └── rules.py
├── dqn/
│   ├── encoding.py       # Player-relative 4×8×8 observations
│   ├── model.py          # Compact residual dueling Q-network
│   ├── replay_buffer.py
│   ├── agent.py          # Legal action selection and Double-DQN updates
│   ├── teacher_data.py   # Validated JSONL loader
│   ├── symmetry.py       # Eight rotations/reflections
│   └── imitation.py      # Supervised pretraining CLI
└── tests/
    ├── test_rules.py
    ├── test_dqn.py
    ├── test_teacher_data.py
    └── test_imitation.py
```

## Installation

```bash
python -m pip install -e "engine/python[dqn]"
```

A CPU-only PyTorch wheel can also be installed explicitly:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e engine/python --no-deps
```

## Observation and model

`encode_observation(board, player)` returns a player-relative `4 × 8 × 8` tensor containing own discs, opponent discs, legal actions, and game progress. The default `DuelingQNetwork` has 56,978 trainable parameters, about 228 KB of FP32 weights before ONNX packaging or quantization.

Each replay transition stores a `bootstrap_sign`: `-1` when the opponent acts next and `+1` when an opponent pass lets the same player act again.

```text
target = reward + gamma * bootstrap_sign * next_value
```

## Hard-CPU teacher data

```bash
npm run generate:teacher -- \
  --games 1000 \
  --output engine/python/data/teacher.jsonl \
  --seed 20260720 \
  --time-ms 100 \
  --max-depth 6 \
  --exploration 0.20
```

Every JSONL record contains the board, acting player, Hard-CPU action, exact legal mask, game number, and ply. Exploration affects only the move used to advance the generated game; the stored label always comes from `chooseHardMove`.

`TeacherExample` recomputes the legal mask with the Python rules engine. `TeacherDataset` exposes validated `(observation, action)` pairs.

## Imitation pretraining

Train the compact network to reproduce Hard-CPU moves:

```bash
npm run train:imitation -- \
  --data engine/python/data/teacher.jsonl \
  --output engine/python/checkpoints/imitation.pt \
  --epochs 20 \
  --batch-size 128 \
  --augmentation random \
  --device auto
```

The trainer:

- splits complete games between training and validation when multiple games are available
- masks illegal logits before cross-entropy loss and accuracy calculation
- supports no augmentation, one random symmetry per sample, or all eight symmetries
- saves the best validation checkpoint with model dimensions, training settings, metrics, and weights

The eight transformations apply the same rotation/reflection to all observation planes and the row-major teacher action. The resulting action is tested to remain legal in the transformed legal-mask plane.

The imitation checkpoint is an initialization for later Double-DQN training, not the final reinforcement-learning policy.

## Validation

```bash
npm test
npm run test:dqn
npm run test:teacher
npm run test:imitation
```

The imitation smoke test generates teacher data, verifies symmetry/action consistency, performs real optimizer updates, and reloads the saved checkpoint.
