from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ENGINE_ROOT = REPOSITORY_ROOT / "engine" / "python"
sys.path.insert(0, str(PYTHON_ENGINE_ROOT))

import torch  # noqa: E402

from dqn.model import DuelingQNetwork  # noqa: E402
from dqn.reinforcement import (  # noqa: E402
    ReinforcementConfig,
    train_reinforcement,
)
from dqn.self_play import play_episode  # noqa: E402
from othello import BLACK, create_initial_board  # noqa: E402
from othello.heuristic import choose_heuristic_move  # noqa: E402


class ReinforcementTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = DuelingQNetwork(
            channels=8,
            residual_blocks=1,
            value_channels=4,
        )

    def test_fixed_heuristic_move_is_legal(self) -> None:
        move = choose_heuristic_move(create_initial_board(), BLACK)
        self.assertIsNotNone(move)
        assert move is not None
        self.assertIn(move.index, {19, 26, 37, 44})

    def test_all_opponent_modes_collect_valid_complete_episodes(self) -> None:
        for index, mode in enumerate(("self", "heuristic", "random")):
            with self.subTest(mode=mode):
                result = play_episode(
                    self.model,
                    mode=mode,
                    epsilon=0.25,
                    rng=random.Random(100 + index),
                )
                self.assertGreater(result.moves, 0)
                self.assertLessEqual(result.moves, 60)
                occupied = result.black_discs + result.white_discs
                self.assertGreaterEqual(occupied, 4)
                self.assertLessEqual(occupied, 64)
                self.assertTrue(result.transitions[-1].terminated)
                self.assertIn(result.transitions[-1].reward, {-1.0, 0.0, 1.0})

                for transition in result.transitions:
                    self.assertTrue(
                        bool(transition.state[2].flatten()[transition.action].item())
                    )
                    if transition.terminated:
                        self.assertFalse(bool(transition.next_legal_mask.any().item()))
                    else:
                        self.assertTrue(bool(transition.next_legal_mask.any().item()))
                        self.assertIn(transition.bootstrap_sign, {-1.0, 1.0})

    def test_smoke_training_loads_imitation_weights_and_saves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_path = root / "imitation.pt"
            output_path = root / "reinforcement.pt"
            model_config = {
                "input_channels": 4,
                "channels": 8,
                "residual_blocks": 1,
                "value_channels": 4,
            }
            torch.save(
                {
                    "format_version": 1,
                    "stage": "imitation",
                    "model_config": model_config,
                    "model_state_dict": self.model.state_dict(),
                },
                init_path,
            )

            result = train_reinforcement(
                output_path=output_path,
                init_checkpoint=init_path,
                config=ReinforcementConfig(
                    episodes=3,
                    batch_size=16,
                    replay_capacity=512,
                    warmup_transitions=16,
                    updates_per_episode=1,
                    epsilon_start=0.20,
                    epsilon_end=0.10,
                    epsilon_decay_episodes=3,
                    self_play_weight=1.0,
                    heuristic_weight=0.0,
                    random_weight=0.0,
                    checkpoint_every=3,
                    seed=31415,
                ),
                device="cpu",
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(result["episodes"], 3)
            self.assertGreater(result["updates"], 0)
            self.assertGreaterEqual(result["replay_size"], 16)

            checkpoint = torch.load(
                output_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["stage"], "reinforcement")
            self.assertEqual(checkpoint["episode"], 3)
            self.assertEqual(checkpoint["initialized_from"], str(init_path))
            self.assertEqual(checkpoint["model_config"], model_config)
            self.assertEqual(len(checkpoint["history"]), 3)


if __name__ == "__main__":
    unittest.main()
