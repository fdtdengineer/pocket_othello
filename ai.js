import {
  BLACK,
  WHITE,
  EMPTY,
  applyMove,
  countPieces,
  getLegalMoves,
  getNextTurn,
  opponent,
} from './engine.js';

const POSITION_WEIGHTS = [
  120, -28, 20, 8, 8, 20, -28, 120,
  -28, -55, -6, -4, -4, -6, -55, -28,
  20, -6, 14, 3, 3, 14, -6, 20,
  8, -4, 3, 3, 3, 3, -4, 8,
  8, -4, 3, 3, 3, 3, -4, 8,
  20, -6, 14, 3, 3, 14, -6, 20,
  -28, -55, -6, -4, -4, -6, -55, -28,
  120, -28, 20, 8, 8, 20, -28, 120,
];

const CORNERS = [0, 7, 56, 63];
const CORNER_NEIGHBORS = {
  0: [1, 8, 9],
  7: [6, 14, 15],
  56: [48, 49, 57],
  63: [54, 55, 62],
};

function randomItem(items, random = Math.random) {
  return items[Math.floor(random() * items.length)];
}

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

export function chooseEasyMove(board, player, random = Math.random) {
  const legal = getLegalMoves(board, player);
  if (legal.length === 0) return null;
  const corners = legal.filter((move) => CORNERS.includes(move.index));
  if (corners.length && random() < 0.55) return randomItem(corners, random);
  return randomItem(legal, random);
}

export function chooseNormalMove(board, player, random = Math.random) {
  const legal = getLegalMoves(board, player);
  if (legal.length === 0) return null;

  const ranked = legal.map((move) => {
    const next = applyMove(board, move, player);
    return {
      move,
      score: evaluateBoard(next, player) + move.flips.length * 0.35 + random() * 2,
    };
  }).sort((a, b) => b.score - a.score);

  const poolSize = ranked.length > 5 ? 2 : 1;
  return randomItem(ranked.slice(0, poolSize), random).move;
}

function boardKey(board, player, depth) {
  let key = player === BLACK ? 'b' : 'w';
  for (const cell of board) key += cell === BLACK ? '1' : cell === WHITE ? '2' : '0';
  return `${key}:${depth}`;
}

function orderedMoves(board, player, legalMoves) {
  return legalMoves.slice().sort((a, b) => {
    const aBoard = applyMove(board, a, player);
    const bBoard = applyMove(board, b, player);
    return evaluateBoard(bBoard, player) - evaluateBoard(aBoard, player);
  });
}

function alphaBeta(board, currentPlayer, rootPlayer, depth, alpha, beta, deadline, table) {
  if (performance.now() > deadline) throw new Error('AI_TIMEOUT');

  const legal = getLegalMoves(board, currentPlayer);
  const enemyLegal = getLegalMoves(board, opponent(currentPlayer));
  if (depth === 0 || (legal.length === 0 && enemyLegal.length === 0)) {
    return evaluateBoard(board, rootPlayer);
  }

  const key = boardKey(board, currentPlayer, depth);
  if (table.has(key)) return table.get(key);

  if (legal.length === 0) {
    const value = alphaBeta(board, opponent(currentPlayer), rootPlayer, depth - 1, alpha, beta, deadline, table);
    table.set(key, value);
    return value;
  }

  const maximizing = currentPlayer === rootPlayer;
  let value = maximizing ? -Infinity : Infinity;

  for (const move of orderedMoves(board, currentPlayer, legal)) {
    const nextBoard = applyMove(board, move, currentPlayer);
    const turn = getNextTurn(nextBoard, currentPlayer);
    const nextPlayer = turn.gameOver ? opponent(currentPlayer) : turn.currentPlayer;
    const candidate = alphaBeta(nextBoard, nextPlayer, rootPlayer, depth - 1, alpha, beta, deadline, table);

    if (maximizing) {
      value = Math.max(value, candidate);
      alpha = Math.max(alpha, value);
    } else {
      value = Math.min(value, candidate);
      beta = Math.min(beta, value);
    }
    if (beta <= alpha) break;
  }

  table.set(key, value);
  return value;
}

export function chooseHardMove(board, player, options = {}) {
  const legal = getLegalMoves(board, player);
  if (legal.length === 0) return null;
  if (legal.length === 1) return legal[0];

  const timeLimitMs = Math.max(80, options.timeLimitMs ?? 650);
  const deadline = performance.now() + timeLimitMs;
  const empty = countPieces(board).empty;
  const maxDepth = options.maxDepth ?? (empty <= 12 ? empty + 2 : empty <= 22 ? 7 : 6);
  let bestMove = chooseNormalMove(board, player, options.random ?? Math.random);
  let bestScore = -Infinity;

  for (let depth = 2; depth <= maxDepth; depth += 1) {
    const table = new Map();
    let completed = true;
    let iterationBest = bestMove;
    let iterationScore = -Infinity;

    try {
      for (const move of orderedMoves(board, player, legal)) {
        const next = applyMove(board, move, player);
        const turn = getNextTurn(next, player);
        const nextPlayer = turn.gameOver ? opponent(player) : turn.currentPlayer;
        const score = alphaBeta(next, nextPlayer, player, depth - 1, -Infinity, Infinity, deadline, table);
        if (score > iterationScore) {
          iterationScore = score;
          iterationBest = move;
        }
      }
    } catch (error) {
      if (error.message !== 'AI_TIMEOUT') throw error;
      completed = false;
    }

    if (!completed) break;
    bestMove = iterationBest;
    bestScore = iterationScore;
    if (Math.abs(bestScore) >= 100000) break;
  }

  return bestMove;
}

export function chooseCpuMove(board, player, difficulty, options = {}) {
  if (difficulty === 'easy') return chooseEasyMove(board, player, options.random);
  if (difficulty === 'hard') return chooseHardMove(board, player, options);
  return chooseNormalMove(board, player, options.random);
}
