from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ENGINE_ROOT = REPOSITORY_ROOT / "engine" / "python"
sys.path.insert(0, str(PYTHON_ENGINE_ROOT))

import torch  # noqa: E402

from dqn.imitation import ImitationConfig, train_imitation  # noqa: E402
from dqn.model import DuelingQNetwork  # noqa: E402
from dqn.symmetry import (  # noqa: E402
    SYMMETRY_COUNT,
    transform_action,
    transform_observation,
)
from dqn.teacher_data import load_teacher_examples  # noqa: E402
from dqn.encoding import encode_observation  # noqa: E402


class ImitationTest(unittest.TestCase):
    def generate(self, output: Path) -> None:
        subprocess.run(
            [
                "node",
                "engine/javascript/generate_teacher_data.mjs",
                "--games",
                "2",
                "--output",
                str(output),
                "--seed",
                "98765",
                "--time-ms",
                "120",
                "--max-depth",
                "2",
                "--exploration",
                "0.40",
                "--max-examples",
                "32",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_symmetries_preserve_labeled_legality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "teacher.jsonl"
            self.generate(data_path)
            example = next(load_teacher_examples(data_path))
            observation = encode_observation(example.board, example.player)

            transformed_actions = set()
            for symmetry in range(SYMMETRY_COUNT):
                transformed = transform_observation(observation, symmetry)
                action = int(transform_action(example.action, symmetry))
                transformed_actions.add(action)
                self.assertTrue(bool(transformed[2].flatten()[action].item()))
                self.assertAlmostEqual(
                    float(transformed[3].mean()),
                    float(observation[3].mean()),
                )
            self.assertGreaterEqual(len(transformed_actions), 4)

    def test_imitation_training_writes_loadable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "teacher.jsonl"
            checkpoint_path = root / "imitation.pt"
            self.generate(data_path)
            examples = list(load_teacher_examples(data_path))

            seed = 2468
            torch.manual_seed(seed)
            initial = DuelingQNetwork(
                channels=8,
                residual_blocks=1,
                value_channels=4,
            )
            initial_state = {
                name: value.detach().clone()
                for name, value in initial.state_dict().items()
            }

            result = train_imitation(
                examples,
                output_path=checkpoint_path,
                config=ImitationConfig(
                    epochs=2,
                    batch_size=8,
                    learning_rate=2e-3,
                    validation_fraction=0.25,
                    augmentation="random",
                    seed=seed,
                ),
                model_config={
                    "channels": 8,
                    "residual_blocks": 1,
                    "value_channels": 4,
                },
                device="cpu",
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(len(result["history"]), 2)
            self.assertGreater(result["train_examples"], 0)
            self.assertGreater(result["validation_examples"], 0)
            self.assertTrue(math.isfinite(float(result["best_validation_loss"])))

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["format_version"], 1)
            self.assertEqual(checkpoint["stage"], "imitation")
            self.assertEqual(checkpoint["model_config"]["channels"], 8)
            self.assertTrue(
                any(
                    not torch.equal(initial_state[name], value)
                    for name, value in checkpoint["model_state_dict"].items()
                )
            )


if __name__ == "__main__":
    unittest.main()
