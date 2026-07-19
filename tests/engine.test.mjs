import assert from 'node:assert/strict';
import {
  BLACK,
  WHITE,
  EMPTY,
  applyMove,
  countPieces,
  createInitialBoard,
  getFlips,
  getLegalMoves,
  getNextTurn,
  isGameOver,
  winner,
} from '../engine.js';
import { chooseCpuMove } from '../ai.js';

const board = createInitialBoard();
assert.deepEqual(countPieces(board), { black: 2, white: 2, empty: 60 });
assert.deepEqual(getLegalMoves(board, BLACK).map((move) => move.index), [19, 26, 37, 44]);
assert.deepEqual(getFlips(board, 19, BLACK), [27]);

const moved = applyMove(board, 19, BLACK);
assert.deepEqual(countPieces(moved), { black: 4, white: 1, empty: 59 });
assert.equal(moved[27], BLACK);
assert.throws(() => applyMove(board, 0, BLACK), /Illegal/);

const turn = getNextTurn(moved, BLACK);
assert.equal(turn.currentPlayer, WHITE);
assert.equal(turn.gameOver, false);

const fullBlack = Array(64).fill(BLACK);
assert.equal(isGameOver(fullBlack), true);
assert.equal(winner(fullBlack), BLACK);
assert.equal(getNextTurn(fullBlack, BLACK).currentPlayer, EMPTY);

for (const difficulty of ['easy', 'normal', 'hard']) {
  const move = chooseCpuMove(board, BLACK, difficulty, { random: () => 0.2, timeLimitMs: 120, maxDepth: 3 });
  assert.ok(move, `${difficulty} should return a move`);
  assert.ok([19, 26, 37, 44].includes(move.index), `${difficulty} returned an illegal move`);
}

// Find and verify a deterministic pass situation from legal play.
let passBoard = createInitialBoard();
let current = BLACK;
let foundPass = false;
for (let ply = 0; ply < 60 && !foundPass; ply += 1) {
  const legal = getLegalMoves(passBoard, current);
  if (legal.length === 0) break;
  passBoard = applyMove(passBoard, legal[0], current);
  const result = getNextTurn(passBoard, current);
  if (result.passedPlayer !== null) {
    foundPass = true;
    assert.equal(result.currentPlayer, current);
    assert.equal(getLegalMoves(passBoard, result.passedPlayer).length, 0);
    assert.ok(getLegalMoves(passBoard, result.currentPlayer).length > 0);
  }
  if (result.gameOver) break;
  current = result.currentPlayer;
}
assert.equal(foundPass, true, 'Expected deterministic legal play to produce a pass.');

console.log('All engine and CPU tests passed.');
