from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from dqn.export_onnx import export_checkpoint_to_onnx
from dqn.model import DuelingQNetwork


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model_config = {
        "input_channels": 4,
        "channels": 8,
        "residual_blocks": 1,
        "value_channels": 4,
    }
    torch.manual_seed(20260720)
    model = DuelingQNetwork(**model_config)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "browser-fixture.pt"
        torch.save(
            {
                "format_version": 1,
                "stage": "reinforcement",
                "model_config": model_config,
                "online_state_dict": model.state_dict(),
            },
            checkpoint,
        )
        export_checkpoint_to_onnx(
            checkpoint,
            output,
            validate=False,
        )


if __name__ == "__main__":
    main()
