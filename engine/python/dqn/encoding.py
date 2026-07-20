"""Player-relative tensor encoding for the DQN."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from othello import BOARD_SIZE, EMPTY, legal_action_mask

OBSERVATION_CHANNELS = 4


def encode_observation(
    board: Sequence[int],
    player: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode one board as ``[own, opponent, legal, progress]``.

    The first two planes are binary and player-relative. The legal plane marks
    valid action indices. The progress plane is constant and ranges from 0 to 1.
    """

    if len(board) != BOARD_SIZE * BOARD_SIZE:
        raise ValueError("An Othello board must contain exactly 64 cells.")
    if player not in (-1, 1):
        raise ValueError("Player must be BLACK (1) or WHITE (-1).")

    values = torch.as_tensor(board, device=device).reshape(BOARD_SIZE, BOARD_SIZE)
    observation = torch.empty(
        (OBSERVATION_CHANNELS, BOARD_SIZE, BOARD_SIZE),
        dtype=dtype,
        device=device,
    )
    observation[0] = (values == player).to(dtype)
    observation[1] = (values == -player).to(dtype)

    mask = torch.as_tensor(
        legal_action_mask(board, player),
        dtype=dtype,
        device=device,
    ).reshape(BOARD_SIZE, BOARD_SIZE)
    observation[2] = mask

    occupied = sum(value != EMPTY for value in board)
    observation[3].fill_(occupied / (BOARD_SIZE * BOARD_SIZE))
    return observation


def encode_batch(
    positions: Iterable[tuple[Sequence[int], int]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    encoded = [
        encode_observation(board, player, dtype=dtype, device=device)
        for board, player in positions
    ]
    if not encoded:
        raise ValueError("At least one position is required.")
    return torch.stack(encoded)
