# Python/PyTorch DQN Engine

This directory contains the Python rules layer, compact PyTorch DQN training, and browser-ready ONNX export.

## Installation

```bash
python -m pip install -e "engine/python[all]"
```

For training only:

```bash
python -m pip install -e "engine/python[dqn]"
```

## Model contract

`encode_observation(board, player)` returns a player-relative `4 × 8 × 8` float tensor:

1. acting player's discs
2. opponent discs
3. legal-action mask
4. game-progress plane

The default `DuelingQNetwork` has 56,978 trainable parameters and 64 row-major action outputs.

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

## 2. Imitation pretraining

```bash
npm run train:imitation -- \
  --data engine/python/data/teacher.jsonl \
  --output engine/python/checkpoints/imitation.pt \
  --epochs 20 \
  --augmentation random \
  --device auto
```

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

The replay pool combines self-play, deterministic heuristic, and random-opponent games. All moves are retained as off-policy experience, with signed pass-aware Double-DQN targets.

## 4. Export ONNX for the browser

```bash
npm run export:onnx -- \
  --checkpoint engine/python/checkpoints/reinforcement.pt \
  --output engine/models/othello_dqn.onnx
```

The exporter accepts either an imitation checkpoint (`model_state_dict`) or a reinforcement checkpoint (`online_state_dict`). It writes:

```text
engine/models/othello_dqn.onnx
engine/models/othello_dqn.json
```

The metadata JSON records:

- model dimensions and parameter count
- input/output names, shapes, and data types
- channel order and 64-action row-major contract
- ONNX opset, byte size, and SHA-256 hash
- ONNX checker result and PyTorch/ONNX Runtime maximum absolute error

The ONNX graph has a dynamic batch dimension but fixed `4 × 8 × 8` spatial input. It excludes the optimizer, replay buffer, and training history.

## Validation

```bash
npm test
npm run test:dqn
npm run test:teacher
npm run test:imitation
npm run test:reinforcement
npm run test:onnx
```

The ONNX test exports both checkpoint formats, runs ONNX structural validation, compares ONNX Runtime output with PyTorch output, checks the browser input/output contract, and verifies that the default model remains below 1 MB.
