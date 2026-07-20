# Python/PyTorch DQN Engine

This directory contains the Python rules layer, compact PyTorch DQN components, Hard-CPU teacher data, imitation pretraining, and mixed-opponent reinforcement learning for a future browser-deployable Othello agent.

## Layout

```text
engine/python/
├── othello/
│   ├── rules.py
│   └── heuristic.py      # Fixed deterministic opponent
├── dqn/
│   ├── encoding.py
│   ├── model.py
│   ├── replay_buffer.py
│   ├── agent.py
│   ├── teacher_data.py
│   ├── symmetry.py
│   ├── imitation.py
│   ├── self_play.py      # Complete episode collection
│   └── reinforcement.py  # Mixed-opponent Double-DQN CLI
└── tests/
```

## Installation

```bash
python -m pip install -e "engine/python[dqn]"
```

## Observation and model

`encode_observation(board, player)` returns a player-relative `4 × 8 × 8` tensor containing own discs, opponent discs, legal actions, and game progress. The default `DuelingQNetwork` has 56,978 trainable parameters, about 228 KB of FP32 weights before ONNX packaging or quantization.

Each replay transition stores a signed bootstrap factor:

```text
target = reward + gamma * bootstrap_sign * next_value
```

The sign is `-1` when the opponent acts next and `+1` when an opponent pass lets the same player move again.

## 1. Generate Hard-CPU teacher data

```bash
npm run generate:teacher -- \
  --games 1000 \
  --output engine/python/data/teacher.jsonl \
  --seed 20260720 \
  --time-ms 100 \
  --max-depth 6 \
  --exploration 0.20
```

Every record is validated against the Python rules engine before training.

## 2. Imitation pretraining

```bash
npm run train:imitation -- \
  --data engine/python/data/teacher.jsonl \
  --output engine/python/checkpoints/imitation.pt \
  --epochs 20 \
  --batch-size 128 \
  --augmentation random \
  --device auto
```

The trainer splits complete games between training and validation, masks illegal logits, supports all eight board symmetries, and saves the best validation model.

## 3. Mixed-opponent reinforcement training

```bash
npm run train:reinforcement -- \
  --init-checkpoint engine/python/checkpoints/imitation.pt \
  --output engine/python/checkpoints/reinforcement.pt \
  --episodes 10000 \
  --self-play-weight 0.50 \
  --heuristic-weight 0.30 \
  --random-weight 0.20 \
  --device auto
```

The opponent pool contains:

- the current network on both colors for self-play
- a deterministic one-ply positional/mobility heuristic
- a random legal policy

All moves are retained as off-policy experience, including fixed-opponent actions. This ensures that terminal rewards remain observable when the fixed opponent makes the final move.

The reinforcement trainer provides:

- replay memory and configurable warmup
- signed Double-DQN targets
- linear epsilon decay
- soft target-network updates
- configurable opponent mixture
- periodic checkpoints with online/target weights, optimizer state, training settings, replay size, losses, and game results

The replay buffer itself is not serialized because it can become much larger than the mobile inference model.

## Validation

```bash
npm test
npm run test:dqn
npm run test:teacher
npm run test:imitation
npm run test:reinforcement
```

The reinforcement smoke test runs complete games against self-play, heuristic, and random opponents, verifies legal actions and terminal transitions, initializes from an imitation checkpoint, performs real Double-DQN updates, and reloads the reinforcement checkpoint.
