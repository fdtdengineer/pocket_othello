"""Validated JSONL teacher data produced by the JavaScript Hard CPU."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from othello import BLACK, BOARD_CELLS, EMPTY, WHITE, legal_action_mask

from .encoding import encode_observation


@dataclass(frozen=True, slots=True)
class TeacherExample:
    board: tuple[int, ...]
    player: int
    action: int
    legal_mask: tuple[bool, ...]
    game: int | None = None
    ply: int | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], *, validate: bool = True) -> "TeacherExample":
        try:
            example = cls(
                board=tuple(int(cell) for cell in value["board"]),
                player=int(value["player"]),
                action=int(value["action"]),
                legal_mask=tuple(bool(item) for item in value["legalMask"]),
                game=int(value["game"]) if "game" in value else None,
                ply=int(value["ply"]) if "ply" in value else None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid teacher-data record.") from error
        if validate:
            example.validate()
        return example

    def validate(self) -> None:
        if len(self.board) != BOARD_CELLS:
            raise ValueError("Teacher board must contain exactly 64 cells.")
        if any(cell not in (WHITE, EMPTY, BLACK) for cell in self.board):
            raise ValueError("Teacher board contains an invalid cell value.")
        if self.player not in (BLACK, WHITE):
            raise ValueError("Teacher player must be BLACK (1) or WHITE (-1).")
        if not 0 <= self.action < BOARD_CELLS:
            raise ValueError("Teacher action must be between 0 and 63.")
        if len(self.legal_mask) != BOARD_CELLS:
            raise ValueError("Teacher legal mask must contain exactly 64 values.")

        expected_mask = legal_action_mask(self.board, self.player)
        if self.legal_mask != expected_mask:
            raise ValueError("Teacher legal mask does not match the Python rules engine.")
        if not self.legal_mask[self.action]:
            raise ValueError("Teacher action is not legal in the supplied position.")


def load_teacher_examples(
    path: str | Path,
    *,
    validate: bool = True,
) -> Iterator[TeacherExample]:
    source = Path(path)
    with source.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise ValueError("Teacher record must be a JSON object.")
                yield TeacherExample.from_mapping(value, validate=validate)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"Invalid teacher data at {source}:{line_number}.") from error


class TeacherDataset(Dataset[tuple[Tensor, Tensor]]):
    """Compact in-memory classification dataset for imitation learning."""

    def __init__(self, examples: Sequence[TeacherExample]) -> None:
        if not examples:
            raise ValueError("TeacherDataset requires at least one example.")
        for example in examples:
            example.validate()

        self._boards = torch.tensor(
            [example.board for example in examples],
            dtype=torch.int8,
        )
        self._players = torch.tensor(
            [example.player for example in examples],
            dtype=torch.int8,
        )
        self._actions = torch.tensor(
            [example.action for example in examples],
            dtype=torch.long,
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TeacherDataset":
        return cls(list(load_teacher_examples(path)))

    def __len__(self) -> int:
        return int(self._actions.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        board = self._boards[index].tolist()
        player = int(self._players[index].item())
        observation = encode_observation(board, player)
        return observation, self._actions[index]
