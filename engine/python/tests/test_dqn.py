from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ENGINE_ROOT = REPOSITORY_ROOT / "engine" / "python"
sys.path.insert(0, str(PYTHON_ENGINE_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from dqn import (  # noqa: E402
    DuelingQNetwork,
    ReplayBatch,
    ReplayBuffer,
    Transition,
    compute_double_dqn_targets,
    count_trainable_parameters,
    encode_observation,
    optimize_double_dqn,
    select_action,
)
from othello import BLACK, WHITE, create_initial_board, legal_action_mask  # noqa: E402


class FixedQNetwork(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(observations.shape[0], -1)


class DQNTest(unittest.TestCase):
    def test_player_relative_encoding(self) -> None:
        board = create_initial_board()
        black = encode_observation(board, BLACK)
        white = encode_observation(board, WHITE)

        self.assertEqual(tuple(black.shape), (4, 8, 8))
        self.assertTrue(torch.equal(black[0], white[1]))
        self.assertTrue(torch.equal(black[1], white[0]))
        self.assertEqual(int(black[2].sum().item()), 4)
        self.assertAlmostEqual(float(black[3, 0, 0]), 4 / 64)

    def test_compact_dueling_network(self) -> None:
        model = DuelingQNetwork()
        output = model(torch.zeros(2, 4, 8, 8))
        self.assertEqual(tuple(output.shape), (2, 64))
        self.assertLess(count_trainable_parameters(model), 100_000)

    def test_action_selection_respects_mask(self) -> None:
        values = [float(index) for index in range(64)]
        model = FixedQNetwork(values)
        state = torch.zeros(4, 8, 8)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[[3, 7, 12]] = True

        self.assertEqual(select_action(model, state, mask), 12)
        action = select_action(
            model,
            state,
            mask,
            epsilon=1.0,
            rng=random.Random(7),
        )
        self.assertIn(action, {3, 7, 12})

    def test_signed_double_dqn_targets(self) -> None:
        online_values = [0.0] * 64
        online_values[3] = 10.0
        target_values = [0.0] * 64
        target_values[3] = 5.0
        online = FixedQNetwork(online_values)
        target = FixedQNetwork(target_values)

        next_masks = torch.zeros(3, 64, dtype=torch.bool)
        next_masks[:2, 3] = True
        batch = ReplayBatch(
            states=torch.zeros(3, 4, 8, 8),
            actions=torch.zeros(3, dtype=torch.long),
            rewards=torch.tensor([1.0, 1.0, 1.0]),
            next_states=torch.zeros(3, 4, 8, 8),
            next_legal_masks=next_masks,
            terminated=torch.tensor([False, False, True]),
            bootstrap_signs=torch.tensor([-1.0, 1.0, -1.0]),
        )

        targets = compute_double_dqn_targets(
            online,
            target,
            batch,
            gamma=0.9,
        )
        self.assertTrue(torch.allclose(targets, torch.tensor([-3.5, 5.5, 1.0])))

    def test_replay_buffer_and_optimization(self) -> None:
        board = create_initial_board()
        state = encode_observation(board, BLACK)
        next_state = encode_observation(board, WHITE)
        next_mask = torch.tensor(legal_action_mask(board, WHITE))

        buffer = ReplayBuffer(capacity=4)
        for action in [19, 26, 37, 44, 19]:
            buffer.push(
                Transition(
                    state=state,
                    action=action,
                    reward=0.0,
                    next_state=next_state,
                    next_legal_mask=next_mask,
                    terminated=False,
                    bootstrap_sign=-1.0,
                )
            )

        self.assertEqual(len(buffer), 4)
        batch = buffer.sample(4, rng=random.Random(1))
        self.assertEqual(tuple(batch.states.shape), (4, 4, 8, 8))

        online = DuelingQNetwork(channels=8, residual_blocks=1, value_channels=4)
        target = DuelingQNetwork(channels=8, residual_blocks=1, value_channels=4)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        before = [parameter.detach().clone() for parameter in online.parameters()]
        loss = optimize_double_dqn(
            online,
            target,
            optimizer,
            batch,
            gamma=0.99,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, online.parameters(), strict=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
