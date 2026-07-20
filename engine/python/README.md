# Python/PyTorch DQN Engine

This directory is reserved for the future reinforcement-learning implementation. Keep it independent from the browser UI and from CodinGame input/output adapters.

Recommended layout:

```text
engine/python/
├── othello_env.py      # Rules-compatible RL environment
├── model.py            # PyTorch Q-network
├── replay_buffer.py    # Experience replay
├── dqn_agent.py        # Action selection and optimization
├── train.py            # Self-play or opponent-pool training
├── evaluate.py         # Fixed-seed evaluation against baseline agents
├── tests/
└── checkpoints/        # Ignored model artifacts
```

The implementation should follow the shared contract in `engine/README.md`:

- flat row-major board with 64 cells
- `0` empty, `1` black, `-1` white
- action indices from `0` to `63`
- a 64-element legal-action mask
- terminal rewards from the acting agent's perspective

A useful first observation encoding is a three-channel tensor:

1. current player's discs
2. opponent's discs
3. legal-action mask

This keeps the policy player-relative and avoids training separate black and white networks.
