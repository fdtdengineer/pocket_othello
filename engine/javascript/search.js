import {
  BLACK,
  applyMove,
  countPieces,
  getLegalMoves,
  getNextTurn,
  opponent,
} from './rules.js';
import { CORNERS, evaluateBoard } from './evaluation.js';

export const TURN_RESPONSE_LIMIT_MS = 150;
export const SEARCH_SAFETY_MARGIN_MS = 30;
export const MAX_SEARCH_TIME_MS = TURN_RESPONSE_LIMIT_MS - SEARCH_SAFETY_MARGIN_MS;
export const DEFAULT_SEARCH_TIME_MS = 100;

function defaultClock() {
  if (globalThis.performance && typeof globalThis.performance.now === 'function') {
    return globalThis.performance.now();
  }
  return Date.now();
}

export function clampSearchTimeMs(value = DEFAULT_SEARCH_TIME_MS) {
  const requested = Number.isFinite(value) ? value : DEFAULT_SEARCH_TIME_MS;
  return Math.max(20, Math.min(MAX_SEARCH_TIME_MS, requested));
}

function randomItem(items, random = Math.random) {
  return items[Math.floor(random() * items.length)];
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
  for (const cell of board) key += cell === BLACK ? '1' : cell === -1 ? '2' : '0';
  return `${key}:${depth}`;
}

function orderedMoves(board, player, legalMoves) {
  return legalMoves.slice().sort((a, b) => {
    const aBoard = applyMove(board, a, player);
    const bBoard = applyMove(board, b, player);
    return evaluateBoard(bBoard, player) - evaluateBoard(aBoard, player);
  });
}

function assertWithinDeadline(clock, deadline) {
  if (clock() >= deadline) throw new Error('AI_TIMEOUT');
}

function alphaBeta(board, currentPlayer, rootPlayer, depth, alpha, beta, deadline, table, clock) {
  assertWithinDeadline(clock, deadline);

  const legal = getLegalMoves(board, currentPlayer);
  const enemyLegal = getLegalMoves(board, opponent(currentPlayer));
  if (depth === 0 || (legal.length === 0 && enemyLegal.length === 0)) {
    return evaluateBoard(board, rootPlayer);
  }

  const key = boardKey(board, currentPlayer, depth);
  if (table.has(key)) return table.get(key);

  if (legal.length === 0) {
    const value = alphaBeta(
      board,
      opponent(currentPlayer),
      rootPlayer,
      depth - 1,
      alpha,
      beta,
      deadline,
      table,
      clock,
    );
    table.set(key, value);
    return value;
  }

  const maximizing = currentPlayer === rootPlayer;
  let value = maximizing ? -Infinity : Infinity;

  for (const move of orderedMoves(board, currentPlayer, legal)) {
    assertWithinDeadline(clock, deadline);
    const nextBoard = applyMove(board, move, currentPlayer);
    const turn = getNextTurn(nextBoard, currentPlayer);
    const nextPlayer = turn.gameOver ? opponent(currentPlayer) : turn.currentPlayer;
    const candidate = alphaBeta(
      nextBoard,
      nextPlayer,
      rootPlayer,
      depth - 1,
      alpha,
      beta,
      deadline,
      table,
      clock,
    );

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

  const clock = options.clock ?? defaultClock;
  const timeLimitMs = clampSearchTimeMs(options.timeLimitMs);
  const deadline = clock() + timeLimitMs;
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
        assertWithinDeadline(clock, deadline);
        const next = applyMove(board, move, player);
        const turn = getNextTurn(next, player);
        const nextPlayer = turn.gameOver ? opponent(player) : turn.currentPlayer;
        const score = alphaBeta(
          next,
          nextPlayer,
          player,
          depth - 1,
          -Infinity,
          Infinity,
          deadline,
          table,
          clock,
        );
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
