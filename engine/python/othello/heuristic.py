"""Deterministic one-ply heuristic opponent for reinforcement training."""

from __future__ import annotations

from collections.abc import Sequence

from .rules import (
    BLACK,
    BOARD_CELLS,
    EMPTY,
    Move,
    apply_move,
    count_pieces,
    get_legal_moves,
    opponent,
)

POSITION_WEIGHTS = (
    120, -28, 20, 8, 8, 20, -28, 120,
    -28, -55, -6, -4, -4, -6, -55, -28,
    20, -6, 14, 3, 3, 14, -6, 20,
    8, -4, 3, 3, 3, 3, -4, 8,
    8, -4, 3, 3, 3, 3, -4, 8,
    20, -6, 14, 3, 3, 14, -6, 20,
    -28, -55, -6, -4, -4, -6, -55, -28,
    120, -28, 20, 8, 8, 20, -28, 120,
)
CORNERS = (0, 7, 56, 63)
CORNER_NEIGHBORS = {
    0: (1, 8, 9),
    7: (6, 14, 15),
    56: (48, 49, 57),
    63: (54, 55, 62),
}


def _positional_score(board: Sequence[int], player: int) -> float:
    return float(sum(weight * value * player for weight, value in zip(POSITION_WEIGHTS, board, strict=True)))


def _frontier_count(board: Sequence[int], player: int) -> int:
    frontier = 0
    for index, value in enumerate(board):
        if value != player:
            continue
        row, col = divmod(index, 8)
        touches_empty = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                next_row = row + dr
                next_col = col + dc
                if (
                    0 <= next_row < 8
                    and 0 <= next_col < 8
                    and board[next_row * 8 + next_col] == EMPTY
                ):
                    touches_empty = True
                    break
            if touches_empty:
                break
        if touches_empty:
            frontier += 1
    return frontier


def _stable_corner_bonus(board: Sequence[int], player: int) -> float:
    score = 0.0
    for corner in CORNERS:
        if board[corner] == player:
            score += 1.0
        elif board[corner] == opponent(player):
            score -= 1.0
        elif board[corner] == EMPTY:
            for neighbor in CORNER_NEIGHBORS[corner]:
                if board[neighbor] == player:
                    score -= 0.45
                elif board[neighbor] == opponent(player):
                    score += 0.45
    return score


def evaluate_board(board: Sequence[int], player: int) -> float:
    if len(board) != BOARD_CELLS:
        raise ValueError("An Othello board must contain exactly 64 cells.")
    pieces = count_pieces(board)
    my_discs = pieces["black"] if player == BLACK else pieces["white"]
    their_discs = pieces["white"] if player == BLACK else pieces["black"]
    my_moves = len(get_legal_moves(board, player))
    their_moves = len(get_legal_moves(board, opponent(player)))
    disc_difference = my_discs - their_discs

    if pieces["empty"] == 0 or (my_moves == 0 and their_moves == 0):
        if disc_difference > 0:
            return 100_000.0 + disc_difference * 100.0
        if disc_difference < 0:
            return -100_000.0 + disc_difference * 100.0
        return 0.0

    phase = pieces["empty"] / BOARD_CELLS
    mobility = my_moves - their_moves
    frontier = _frontier_count(board, opponent(player)) - _frontier_count(board, player)
    corners = _stable_corner_bonus(board, player)
    return (
        _positional_score(board, player) * (0.65 + phase * 0.35)
        + mobility * (18.0 + phase * 20.0)
        + frontier * (7.0 + phase * 8.0)
        + corners * 85.0
        + disc_difference * (2.0 + (1.0 - phase) * 14.0)
    )


def choose_heuristic_move(board: Sequence[int], player: int) -> Move | None:
    legal_moves = get_legal_moves(board, player)
    if not legal_moves:
        return None
    return max(
        legal_moves,
        key=lambda move: (
            evaluate_board(apply_move(board, move, player), player)
            + len(move.flips) * 0.35,
            -move.index,
        ),
    )
