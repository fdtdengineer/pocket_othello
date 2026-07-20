import {
  BLACK,
  EMPTY,
  countPieces,
  getLegalMoves,
  opponent,
} from './rules.js';

export const POSITION_WEIGHTS = [
  120, -28, 20, 8, 8, 20, -28, 120,
  -28, -55, -6, -4, -4, -6, -55, -28,
  20, -6, 14, 3, 3, 14, -6, 20,
  8, -4, 3, 3, 3, 3, -4, 8,
  8, -4, 3, 3, 3, 3, -4, 8,
  20, -6, 14, 3, 3, 14, -6, 20,
  -28, -55, -6, -4, -4, -6, -55, -28,
  120, -28, 20, 8, 8, 20, -28, 120,
];

export const CORNERS = [0, 7, 56, 63];

const CORNER_NEIGHBORS = {
  0: [1, 8, 9],
  7: [6, 14, 15],
  56: [48, 49, 57],
  63: [54, 55, 62],
};

function positionalScore(board, player) {
  let score = 0;
  for (let i = 0; i < 64; i += 1) {
    score += POSITION_WEIGHTS[i] * board[i] * player;
  }
  return score;
}

function frontierCount(board, player) {
  let frontier = 0;
  for (let i = 0; i < 64; i += 1) {
    if (board[i] !== player) continue;
    const row = Math.floor(i / 8);
    const col = i % 8;
    let touchesEmpty = false;
    for (let dr = -1; dr <= 1 && !touchesEmpty; dr += 1) {
      for (let dc = -1; dc <= 1; dc += 1) {
        if (dr === 0 && dc === 0) continue;
        const r = row + dr;
        const c = col + dc;
        if (r >= 0 && r < 8 && c >= 0 && c < 8 && board[r * 8 + c] === EMPTY) {
          touchesEmpty = true;
          break;
        }
      }
    }
    if (touchesEmpty) frontier += 1;
  }
  return frontier;
}

function stableCornerBonus(board, player) {
  let score = 0;
  for (const corner of CORNERS) {
    if (board[corner] === player) score += 1;
    if (board[corner] === opponent(player)) score -= 1;
    if (board[corner] === EMPTY) {
      for (const neighbor of CORNER_NEIGHBORS[corner]) {
        if (board[neighbor] === player) score -= 0.45;
        if (board[neighbor] === opponent(player)) score += 0.45;
      }
    }
  }
  return score;
}

export function evaluateBoard(board, player) {
  const pieces = countPieces(board);
  const myDiscs = player === BLACK ? pieces.black : pieces.white;
  const theirDiscs = player === BLACK ? pieces.white : pieces.black;
  const myMoves = getLegalMoves(board, player).length;
  const theirMoves = getLegalMoves(board, opponent(player)).length;
  const discDiff = myDiscs - theirDiscs;
  const mobility = myMoves - theirMoves;
  const frontier = frontierCount(board, opponent(player)) - frontierCount(board, player);
  const corners = stableCornerBonus(board, player);
  const phase = pieces.empty / 64;

  if (pieces.empty === 0 || (myMoves === 0 && theirMoves === 0)) {
    if (discDiff > 0) return 100000 + discDiff * 100;
    if (discDiff < 0) return -100000 + discDiff * 100;
    return 0;
  }

  return (
    positionalScore(board, player) * (0.65 + phase * 0.35) +
    mobility * (18 + phase * 20) +
    frontier * (7 + phase * 8) +
    corners * 85 +
    discDiff * (2 + (1 - phase) * 14)
  );
}
