"""Export an imitation or reinforcement checkpoint to a browser-ready ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import DuelingQNetwork, count_trainable_parameters

INPUT_NAME = "observation"
OUTPUT_NAME = "q_values"


def _checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("online_state_dict")
    if state is None:
        state = checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError("Checkpoint does not contain model weights.")
    return state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    opset_version: int = 17,
    validate: bool = True,
) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("Install the export dependencies with engine/python[export].") from error

    checkpoint_source = Path(checkpoint_path)
    output = Path(output_path)
    metadata_output = (
        Path(metadata_path)
        if metadata_path is not None
        else output.with_suffix(".json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_source,
        map_location="cpu",
        weights_only=False,
    )
    model_config = dict(checkpoint["model_config"])
    model = DuelingQNetwork(**model_config)
    model.load_state_dict(_checkpoint_state(checkpoint))
    model.eval()

    input_channels = int(model_config.get("input_channels", 4))
    example = torch.zeros(1, input_channels, 8, 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    validation: dict[str, Any] = {
        "onnxChecker": "passed",
        "runtimeCompared": False,
        "maxAbsoluteError": None,
    }
    if validate:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "Install onnxruntime or call export_checkpoint_to_onnx(validate=False)."
            ) from error

        generator = torch.Generator().manual_seed(20260720)
        sample = torch.randn(
            3,
            input_channels,
            8,
            8,
            generator=generator,
            dtype=torch.float32,
        )
        with torch.no_grad():
            expected = model(sample).cpu().numpy()
        session = ort.InferenceSession(
            str(output),
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: sample.numpy()},
        )[0]
        max_error = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
            raise RuntimeError(
                f"ONNX output does not match PyTorch output; max error={max_error}."
            )
        validation = {
            "onnxChecker": "passed",
            "runtimeCompared": True,
            "maxAbsoluteError": max_error,
        }

    metadata = {
        "formatVersion": 1,
        "modelType": "othello-dueling-dqn",
        "sourceCheckpoint": checkpoint_source.name,
        "sourceStage": checkpoint.get("stage"),
        "modelConfig": model_config,
        "trainableParameters": count_trainable_parameters(model),
        "input": {
            "name": INPUT_NAME,
            "dtype": "float32",
            "shape": ["batch", input_channels, 8, 8],
            "channels": ["own", "opponent", "legal", "progress"],
        },
        "output": {
            "name": OUTPUT_NAME,
            "dtype": "float32",
            "shape": ["batch", 64],
        },
        "actionSpace": {
            "size": 64,
            "layout": "row-major",
            "illegalActionHandling": "mask before argmax",
        },
        "onnx": {
            "opsetVersion": opset_version,
            "file": output.name,
            "sizeBytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "validation": validation,
    }
    metadata_output.write_text(
        f"{json.dumps(metadata, indent=2)}\n",
        encoding="utf8",
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="engine/models/othello_dqn.onnx",
    )
    parser.add_argument("--metadata")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Run ONNX structural validation but skip ONNX Runtime comparison.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    metadata = export_checkpoint_to_onnx(
        arguments.checkpoint,
        arguments.output,
        metadata_path=arguments.metadata,
        opset_version=arguments.opset,
        validate=not arguments.skip_runtime_validation,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
