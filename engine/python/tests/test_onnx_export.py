from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ENGINE_ROOT = REPOSITORY_ROOT / "engine" / "python"
sys.path.insert(0, str(PYTHON_ENGINE_ROOT))

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from dqn.encoding import encode_observation  # noqa: E402
from dqn.export_onnx import (  # noqa: E402
    INPUT_NAME,
    OUTPUT_NAME,
    export_checkpoint_to_onnx,
)
from dqn.model import DuelingQNetwork  # noqa: E402
from othello import BLACK, create_initial_board  # noqa: E402


class OnnxExportTest(unittest.TestCase):
    def test_reinforcement_checkpoint_exports_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "reinforcement.pt"
            model_path = root / "othello_dqn.onnx"
            metadata_path = root / "othello_dqn.json"

            model_config = {
                "input_channels": 4,
                "channels": 32,
                "residual_blocks": 3,
                "value_channels": 8,
            }
            torch.manual_seed(123)
            model = DuelingQNetwork(**model_config)
            torch.save(
                {
                    "format_version": 1,
                    "stage": "reinforcement",
                    "model_config": model_config,
                    "online_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            metadata = export_checkpoint_to_onnx(
                checkpoint_path,
                model_path,
                metadata_path=metadata_path,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertLess(model_path.stat().st_size, 1_000_000)
            self.assertEqual(metadata["sourceStage"], "reinforcement")
            self.assertEqual(metadata["input"]["shape"], ["batch", 4, 8, 8])
            self.assertEqual(metadata["output"]["shape"], ["batch", 64])
            self.assertEqual(metadata["actionSpace"]["size"], 64)
            self.assertTrue(metadata["validation"]["runtimeCompared"])
            self.assertLess(metadata["validation"]["maxAbsoluteError"], 1e-4)
            self.assertEqual(len(metadata["onnx"]["sha256"]), 64)

            observation = encode_observation(
                create_initial_board(),
                BLACK,
            ).unsqueeze(0)
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            output = session.run(
                [OUTPUT_NAME],
                {INPUT_NAME: observation.numpy()},
            )[0]
            self.assertEqual(output.shape, (1, 64))

    def test_imitation_checkpoint_uses_model_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "imitation.pt"
            model_path = root / "imitation.onnx"
            model_config = {
                "input_channels": 4,
                "channels": 8,
                "residual_blocks": 1,
                "value_channels": 4,
            }
            model = DuelingQNetwork(**model_config)
            torch.save(
                {
                    "format_version": 1,
                    "stage": "imitation",
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            metadata = export_checkpoint_to_onnx(
                checkpoint_path,
                model_path,
            )
            self.assertEqual(metadata["sourceStage"], "imitation")
            self.assertEqual(metadata["modelConfig"], model_config)


if __name__ == "__main__":
    unittest.main()
