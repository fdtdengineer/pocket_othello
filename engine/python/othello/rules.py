"""Pure-Python Othello rules matching ``engine/javascript/rules.js``.

The module deliberately has no NumPy or PyTorch dependency. Training code can
build tensors on top of this stable row-major board/action contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

EMPTY = 0
BLACK = 1
WHITE = -1
BOARD_SIZE = 8
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE

_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass(frozen=True, slots=True)
class Move:
    """A legal action and the exact discs it flips."""

    index: int
    flips: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Turn:
    """Turn resolution after a move."""

    current_player: int
    passed_player: int | None
    game_over: bool


def opponent(player: int) -> int:
    return -player


def index_of(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def row_col(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_SIZE)


def is_inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def create_initial_board() -> list[int]:
    board = [EMPTY] * BOARD_CELLS
    board[index_of(3, 3)] = WHITE
    board[index_of(3, 4)] = BLACK
    board[index_of(4, 3)] = BLACK
    board[index_of(4, 4)] = WHITE
    return board


def get_flips(board: Sequence[int], index: int, player: int) -> tuple[int, ...]:
    if len(board) != BOARD_CELLS or not 0 <= index < BOARD_CELLS:
        return ()
    if board[index] != EMPTY:
        return ()

    start_row, start_col = row_col(index)
    all_flips: list[int] = []

    for dr, dc in _DIRECTIONS:
        row = start_row + dr
        col = start_col + dc
        line: list[int] = []

        while is_inside(row, col):
            value = board[index_of(row, col)]
            if value == opponent(player):
                line.append(index_of(row, col))
                row += dr
                col += dc
                continue
            if value == player and line:
                all_flips.extend(line)
            break

    return tuple(all_flips)


def get_legal_moves(board: Sequence[int], player: int) -> list[Move]:
    return [
        Move(index=index, flips=flips)
        for index in range(len(board))
        if board[index] == EMPTY
        and (flips := get_flips(board, index, player))
    ]


def legal_action_mask(board: Sequence[int], player: int) -> tuple[bool, ...]:
    mask = [False] * BOARD_CELLS
    for move in get_legal_moves(board, player):
        mask[move.index] = True
    return tuple(mask)


def apply_move(board: Sequence[int], move_or_index: Move | int, player: int) -> list[int]:
    if isinstance(move_or_index, int):
        index = move_or_index
        flips = get_flips(board, index, player)
    else:
        index = move_or_index.index
        flips = move_or_index.flips

    if not flips:
        raise ValueError("Illegal Othello move.")

    next_board = list(board)
    next_board[index] = player
    for flip_index in flips:
        next_board[flip_index] = player
    return next_board


def count_pieces(board: Sequence[int]) -> dict[str, int]:
    black = sum(value == BLACK for value in board)
    white = sum(value == WHITE for value in board)
    return {
        "black": black,
        "white": white,
        "empty": BOARD_CELLS - black - white,
    }


def get_next_turn(board: Sequence[int], player_who_moved: int) -> Turn:
    preferred = opponent(player_who_moved)
    if get_legal_moves(board, preferred):
        return Turn(current_player=preferred, passed_player=None, game_over=False)
    if get_legal_moves(board, player_who_moved):
        return Turn(
            current_player=player_who_moved,
            passed_player=preferred,
            game_over=False,
        )
    return Turn(current_player=EMPTY, passed_player=None, game_over=True)


def is_game_over(board: Sequence[int]) -> bool:
    return not get_legal_moves(board, BLACK) and not get_legal_moves(board, WHITE)


def winner(board: Sequence[int]) -> int:
    pieces = count_pieces(board)
    if pieces["black"] > pieces["white"]:
        return BLACK
    if pieces["white"] > pieces["black"]:
        return WHITE
    return EMPTY
