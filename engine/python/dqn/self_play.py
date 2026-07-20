"""Episode collection against self-play and fixed opponents."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from othello import (
    BLACK,
    WHITE,
    apply_move,
    count_pieces,
    create_initial_board,
    get_legal_moves,
    get_next_turn,
    legal_action_mask,
    opponent,
    winner,
)
from othello.heuristic import choose_heuristic_move

from .agent import select_action
from .encoding import encode_observation
from .replay_buffer import Transition

OpponentMode = Literal["self", "heuristic", "random"]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    mode: OpponentMode
    network_color: int | None
    winner: int
    black_discs: int
    white_discs: int
    moves: int
    transitions: tuple[Transition, ...]


def _terminal_reward(board: list[int], acting_player: int) -> float:
    game_winner = winner(board)
    if game_winner == acting_player:
        return 1.0
    if game_winner == opponent(acting_player):
        return -1.0
    return 0.0


def play_episode(
    model: nn.Module,
    *,
    mode: OpponentMode,
    epsilon: float,
    rng: random.Random,
) -> EpisodeResult:
    """Collect one complete off-policy episode.

    Every move is retained, including actions selected by fixed opponents. This
    is valid off-policy experience and makes terminal rewards observable even
    when a fixed opponent plays the final move.
    """

    if mode not in {"self", "heuristic", "random"}:
        raise ValueError(f"Unsupported opponent mode: {mode}")

    board = create_initial_board()
    current_player = BLACK
    network_color = None if mode == "self" else (BLACK if rng.random() < 0.5 else WHITE)
    transitions: list[Transition] = []

    for _ in range(60):
        legal_moves = get_legal_moves(board, current_player)
        if not legal_moves:
            raise RuntimeError("Turn resolution produced a player without a legal move.")

        state = encode_observation(board, current_player)
        state_mask = torch.tensor(
            legal_action_mask(board, current_player),
            dtype=torch.bool,
        )
        network_turn = mode == "self" or current_player == network_color

        if network_turn:
            action = select_action(
                model,
                state,
                state_mask,
                epsilon=epsilon,
                rng=rng,
            )
            move = next(move for move in legal_moves if move.index == action)
        elif mode == "heuristic":
            move = choose_heuristic_move(board, current_player)
            if move is None:
                raise RuntimeError("Heuristic opponent failed to return a legal move.")
            action = move.index
        else:
            move = rng.choice(legal_moves)
            action = move.index

        next_board = apply_move(board, move, current_player)
        turn = get_next_turn(next_board, current_player)
        terminated = turn.game_over
        next_player = opponent(current_player) if terminated else turn.current_player
        next_state = encode_observation(next_board, next_player)
        next_mask = torch.tensor(
            legal_action_mask(next_board, next_player),
            dtype=torch.bool,
        )
        bootstrap_sign = (
            -1.0
            if terminated or next_player != current_player
            else 1.0
        )
        reward = _terminal_reward(next_board, current_player) if terminated else 0.0

        transitions.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_legal_mask=next_mask,
                terminated=terminated,
                bootstrap_sign=bootstrap_sign,
            )
        )

        board = next_board
        if terminated:
            break
        current_player = next_player
    else:
        raise RuntimeError("Othello episode exceeded 60 moves.")

    pieces = count_pieces(board)
    return EpisodeResult(
        mode=mode,
        network_color=network_color,
        winner=winner(board),
        black_discs=pieces["black"],
        white_discs=pieces["white"],
        moves=len(transitions),
        transitions=tuple(transitions),
    )
