from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ENGINE_ROOT = REPOSITORY_ROOT / "engine" / "python"
sys.path.insert(0, str(PYTHON_ENGINE_ROOT))

from othello import (  # noqa: E402
    BLACK,
    EMPTY,
    WHITE,
    apply_move,
    count_pieces,
    create_initial_board,
    get_flips,
    get_legal_moves,
    get_next_turn,
    is_game_over,
    legal_action_mask,
    winner,
)


class PythonRulesTest(unittest.TestCase):
    def test_initial_position(self) -> None:
        board = create_initial_board()
        self.assertEqual(count_pieces(board), {"black": 2, "white": 2, "empty": 60})
        self.assertEqual(
            [move.index for move in get_legal_moves(board, BLACK)],
            [19, 26, 37, 44],
        )
        self.assertEqual(get_flips(board, 19, BLACK), (27,))
        self.assertEqual(sum(legal_action_mask(board, BLACK)), 4)

    def test_illegal_move_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Illegal"):
            apply_move(create_initial_board(), 0, BLACK)

    def test_javascript_parity(self) -> None:
        result = subprocess.run(
            ["node", "tests/export_js_rule_cases.mjs"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        fixture = json.loads(result.stdout)

        self.assertGreater(len(fixture["cases"]), 100)
        self.assertGreater(fixture["metadata"]["passTransitions"], 0)
        self.assertGreater(fixture["metadata"]["terminalTransitions"], 0)

        for case_number, case in enumerate(fixture["cases"]):
            with self.subTest(case=case_number, player=case["player"]):
                board = case["board"]
                player = case["player"]

                self.assertEqual(count_pieces(board), case["countPieces"])
                self.assertEqual(is_game_over(board), case["gameOver"])
                self.assertEqual(winner(board), case["winner"])
                self.assertEqual(
                    list(legal_action_mask(board, player)),
                    case["legalActionMask"],
                )

                python_moves = get_legal_moves(board, player)
                self.assertEqual(
                    [
                        {"index": move.index, "flips": list(move.flips)}
                        for move in python_moves
                    ],
                    case["legalMoves"],
                )

                moves_by_index = {move.index: move for move in python_moves}
                for transition in case["transitions"]:
                    move = moves_by_index[transition["index"]]
                    next_board = apply_move(board, move, player)
                    self.assertEqual(next_board, transition["board"])

                    next_turn = get_next_turn(next_board, player)
                    self.assertEqual(
                        {
                            "currentPlayer": next_turn.current_player,
                            "passedPlayer": next_turn.passed_player,
                            "gameOver": next_turn.game_over,
                        },
                        transition["turn"],
                    )

    def test_terminal_winner(self) -> None:
        full_black = [BLACK] * 64
        self.assertTrue(is_game_over(full_black))
        self.assertEqual(winner(full_black), BLACK)
        self.assertEqual(get_next_turn(full_black, BLACK).current_player, EMPTY)

        full_white = [WHITE] * 64
        self.assertTrue(is_game_over(full_white))
        self.assertEqual(winner(full_white), WHITE)


if __name__ == "__main__":
    unittest.main()
