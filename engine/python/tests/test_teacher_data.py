from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ENGINE_ROOT = REPOSITORY_ROOT / "engine" / "python"
sys.path.insert(0, str(PYTHON_ENGINE_ROOT))

import torch  # noqa: E402

from dqn.teacher_data import (  # noqa: E402
    TeacherDataset,
    load_teacher_examples,
)


class TeacherDataTest(unittest.TestCase):
    def generate(self, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "node",
                "engine/javascript/generate_teacher_data.mjs",
                "--games",
                "3",
                "--output",
                str(output),
                "--seed",
                "123456",
                "--time-ms",
                "120",
                "--max-depth",
                "2",
                "--exploration",
                "0.35",
                "--max-examples",
                "16",
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_generation_is_valid_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"

            first_result = self.generate(first_path)
            second_result = self.generate(second_path)
            self.assertIn("Wrote 16 teacher examples", first_result.stderr)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

            first_metadata = json.loads(
                Path(f"{first_path}.meta.json").read_text(encoding="utf8")
            )
            second_metadata = json.loads(
                Path(f"{second_path}.meta.json").read_text(encoding="utf8")
            )
            self.assertEqual(first_metadata, second_metadata)
            self.assertEqual(first_metadata["examples"], 16)
            self.assertEqual(first_metadata["teacher"], "chooseHardMove")

            examples = list(load_teacher_examples(first_path))
            self.assertEqual(len(examples), 16)
            self.assertTrue(all(example.legal_mask[example.action] for example in examples))

            dataset = TeacherDataset(examples)
            observation, action = dataset[0]
            self.assertEqual(tuple(observation.shape), (4, 8, 8))
            self.assertEqual(observation.dtype, torch.float32)
            self.assertEqual(action.dtype, torch.long)
            self.assertTrue(bool(observation[2].flatten()[int(action)].item()))

    def test_invalid_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher.jsonl"
            self.generate(output)
            record = json.loads(output.read_text(encoding="utf8").splitlines()[0])
            record["legalMask"] = [False] * 64
            output.write_text(f"{json.dumps(record)}\n", encoding="utf8")

            with self.assertRaisesRegex(ValueError, "Invalid teacher data"):
                list(load_teacher_examples(output))


if __name__ == "__main__":
    unittest.main()
