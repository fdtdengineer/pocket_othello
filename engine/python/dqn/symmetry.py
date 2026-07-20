"""Dihedral board symmetries for Othello observations and actions."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

SYMMETRY_COUNT = 8


def _validate_symmetry(symmetry: int) -> None:
    if not 0 <= symmetry < SYMMETRY_COUNT:
        raise ValueError("Symmetry must be between 0 and 7.")


def transform_observation(observation: Tensor, symmetry: int) -> Tensor:
    """Apply one of four rotations, optionally preceded by a horizontal flip."""

    _validate_symmetry(symmetry)
    transformed = observation
    if symmetry >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    rotations = symmetry % 4
    if rotations:
        transformed = torch.rot90(transformed, rotations, dims=(-2, -1))
    return transformed


def transform_action(action: int | Tensor, symmetry: int) -> int | Tensor:
    """Transform row-major action indices with the same board symmetry."""

    _validate_symmetry(symmetry)
    tensor = torch.as_tensor(action, dtype=torch.long)
    row = torch.div(tensor, 8, rounding_mode="floor")
    col = tensor % 8

    if symmetry >= 4:
        col = 7 - col
    for _ in range(symmetry % 4):
        row, col = 7 - col, row

    transformed = row * 8 + col
    if isinstance(action, Tensor):
        return transformed.to(action.device)
    return int(transformed.item())


def transform_batch(
    observations: Tensor,
    actions: Tensor,
    symmetries: Tensor,
) -> tuple[Tensor, Tensor]:
    if observations.ndim != 4 or observations.shape[-2:] != (8, 8):
        raise ValueError("Expected observations with shape [batch, channels, 8, 8].")
    if actions.ndim != 1 or symmetries.ndim != 1:
        raise ValueError("Actions and symmetries must be one-dimensional.")
    if not (observations.shape[0] == actions.shape[0] == symmetries.shape[0]):
        raise ValueError("Batch dimensions must match.")

    transformed_observations = []
    transformed_actions = []
    for observation, action, symmetry in zip(
        observations,
        actions,
        symmetries,
        strict=True,
    ):
        symmetry_value = int(symmetry.item())
        transformed_observations.append(
            transform_observation(observation, symmetry_value)
        )
        transformed_actions.append(transform_action(action, symmetry_value))
    return torch.stack(transformed_observations), torch.stack(transformed_actions)


class AllSymmetriesDataset(Dataset[tuple[Tensor, Tensor]]):
    """Deterministically expose all eight symmetries of another dataset."""

    def __init__(self, base: Dataset[tuple[Tensor, Tensor]]) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base) * SYMMETRY_COUNT

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        base_index, symmetry = divmod(index, SYMMETRY_COUNT)
        observation, action = self.base[base_index]
        transformed_observation = transform_observation(observation, symmetry)
        transformed_action = transform_action(action, symmetry)
        return transformed_observation, torch.as_tensor(
            transformed_action,
            dtype=torch.long,
        )
