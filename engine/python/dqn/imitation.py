"""Supervised imitation pretraining from Hard-CPU teacher labels."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .agent import mask_illegal_actions
from .model import DuelingQNetwork
from .symmetry import AllSymmetriesDataset, SYMMETRY_COUNT, transform_batch
from .teacher_data import TeacherDataset, TeacherExample, load_teacher_examples


@dataclass(frozen=True, slots=True)
class ImitationConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.10
    augmentation: str = "random"
    seed: int = 20260720
    gradient_clip_norm: float = 10.0

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("Epochs and batch size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("Validation fraction must be strictly between 0 and 1.")
        if self.augmentation not in {"none", "random", "all"}:
            raise ValueError("Augmentation must be one of: none, random, all.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("Gradient clipping norm must be positive.")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def split_teacher_examples(
    examples: Sequence[TeacherExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[TeacherExample], list[TeacherExample]]:
    """Split by game when possible, avoiding adjacent-position leakage."""

    if len(examples) < 2:
        raise ValueError("At least two teacher examples are required.")
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1.")

    rng = random.Random(seed)
    groups: dict[int, list[TeacherExample]] = {}
    all_have_game = all(example.game is not None for example in examples)
    if all_have_game:
        for example in examples:
            assert example.game is not None
            groups.setdefault(example.game, []).append(example)

    if len(groups) >= 2:
        game_ids = list(groups)
        rng.shuffle(game_ids)
        validation_games = max(1, round(len(game_ids) * validation_fraction))
        validation_games = min(validation_games, len(game_ids) - 1)
        validation_ids = set(game_ids[:validation_games])
        train = [
            example
            for game_id, group in groups.items()
            if game_id not in validation_ids
            for example in group
        ]
        validation = [
            example
            for game_id, group in groups.items()
            if game_id in validation_ids
            for example in group
        ]
        return train, validation

    shuffled = list(examples)
    rng.shuffle(shuffled)
    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _masked_logits(model: DuelingQNetwork, observations: Tensor) -> Tensor:
    q_values = model(observations)
    legal_masks = observations[:, 2].flatten(1).bool()
    return mask_illegal_actions(q_values, legal_masks)


def _run_training_epoch(
    model: DuelingQNetwork,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    augmentation: str,
    augmentation_generator: torch.Generator,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    samples = 0

    for observations, actions in loader:
        if augmentation == "random":
            symmetries = torch.randint(
                SYMMETRY_COUNT,
                (observations.shape[0],),
                generator=augmentation_generator,
            )
            observations, actions = transform_batch(observations, actions, symmetries)

        observations = observations.to(device)
        actions = actions.to(device)
        logits = _masked_logits(model, observations)
        loss = F.cross_entropy(logits, actions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_size = int(actions.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        correct += int((logits.argmax(dim=1) == actions).sum().item())
        samples += batch_size

    return total_loss / samples, correct / samples


@torch.no_grad()
def evaluate_imitation(
    model: DuelingQNetwork,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    samples = 0

    for observations, actions in loader:
        observations = observations.to(device)
        actions = actions.to(device)
        logits = _masked_logits(model, observations)
        loss = F.cross_entropy(logits, actions)

        batch_size = int(actions.shape[0])
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == actions).sum().item())
        samples += batch_size

    return total_loss / samples, correct / samples


def train_imitation(
    examples: Sequence[TeacherExample],
    *,
    output_path: str | Path,
    config: ImitationConfig = ImitationConfig(),
    model_config: dict[str, int] | None = None,
    device: str = "auto",
) -> dict[str, object]:
    config.validate()
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    train_examples, validation_examples = split_teacher_examples(
        examples,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    train_dataset: Dataset[tuple[Tensor, Tensor]] = TeacherDataset(train_examples)
    if config.augmentation == "all":
        train_dataset = AllSymmetriesDataset(train_dataset)
    validation_dataset = TeacherDataset(validation_examples)

    loader_generator = torch.Generator().manual_seed(config.seed)
    augmentation_generator = torch.Generator().manual_seed(config.seed ^ 0x5A17)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    resolved_model_config = {
        "input_channels": 4,
        "channels": 32,
        "residual_blocks": 3,
        "value_channels": 8,
        **(model_config or {}),
    }
    resolved_device = _resolve_device(device)
    model = DuelingQNetwork(**resolved_model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[EpochMetrics] = []
    best_accuracy = float("-inf")
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_training_epoch(
            model,
            train_loader,
            optimizer,
            device=resolved_device,
            augmentation=config.augmentation,
            augmentation_generator=augmentation_generator,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        validation_loss, validation_accuracy = evaluate_imitation(
            model,
            validation_loader,
            device=resolved_device,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )
        history.append(metrics)

        if (
            validation_accuracy > best_accuracy
            or (
                validation_accuracy == best_accuracy
                and validation_loss < best_loss
            )
        ):
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    assert best_state is not None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "stage": "imitation",
        "model_config": resolved_model_config,
        "model_state_dict": best_state,
        "training_config": asdict(config),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "history": [asdict(item) for item in history],
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
    }
    torch.save(checkpoint, output)
    return {
        "checkpoint": str(output),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "history": checkpoint["history"],
        "best_validation_accuracy": best_accuracy,
        "best_validation_loss": best_loss,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Teacher JSONL path")
    parser.add_argument(
        "--output",
        default="engine/python/checkpoints/imitation.pt",
        help="Output checkpoint path",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument(
        "--augmentation",
        choices=("none", "random", "all"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--value-channels", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    examples = list(load_teacher_examples(arguments.data))
    result = train_imitation(
        examples,
        output_path=arguments.output,
        config=ImitationConfig(
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            validation_fraction=arguments.validation_fraction,
            augmentation=arguments.augmentation,
            seed=arguments.seed,
            gradient_clip_norm=arguments.gradient_clip_norm,
        ),
        model_config={
            "channels": arguments.channels,
            "residual_blocks": arguments.residual_blocks,
            "value_channels": arguments.value_channels,
        },
        device=arguments.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
