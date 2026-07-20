// Pocket Othello - CodinGame standalone submission
// Paste this entire file into the JavaScript editor for:
// https://www.codingame.com/multiplayer/bot-programming/othello-1
//
// CodinGame allows 150 ms per normal turn. The search itself is capped at
// 100 ms, leaving time for parsing, move validation, and output.

const EMPTY = 0;
const BLACK = 1;
const WHITE = -1;
const BOARD_SIZE = 8;
const TURN_RESPONSE_LIMIT_MS = 150;
const SEARCH_TIME_LIMIT_MS = 100;

const DIRECTIONS = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1],           [0, 1],
  [1, -1],  [1, 0],  [1, 1],
];

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

function opponent(player) {
  return -player;
}

function indexOf(row, col) {
  return row * BOARD_SIZE + col;
}

function rowCol(index) {
  return [Math.floor(index / BOARD_SIZE), index % BOARD_SIZE];
}

function isInside(row, col) {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

function getFlips(board, index, player) {
  if (board[index] !== EMPTY) return [];
  const start = rowCol(index);
  const allFlips = [];

  for (const direction of DIRECTIONS) {
    const dr = direction[0];
    const dc = direction[1];
    let row = start[0] + dr;
    let col = start[1] + dc;
    const line = [];

    while (isInside(row, col)) {
      const value = board[indexOf(row, col)];
      if (value === opponent(player)) {
        line.push(indexOf(row, col));
        row += dr;
        col += dc;
        continue;
      }
      if (value === player && line.length > 0) allFlips.push(...line);
      break;
    }
  }

  return allFlips;
}

function getLegalMoves(board, player) {
  const moves = [];
  for (let index = 0; index < 64; index += 1) {
    if (board[index] !== EMPTY) continue;
    const flips = getFlips(board, index, player);
    if (flips.length > 0) moves.push({ index, flips });
  }
  return moves;
}

function applyMove(board, move, player) {
  const next = board.slice();
  next[move.index] = player;
  for (const flipIndex of move.flips) next[flipIndex] = player;
  return next;
}

function countPieces(board) {
  let black = 0;
  let white = 0;
  for (const cell of board) {
    if (cell === BLACK) black += 1;
    if (cell === WHITE) white += 1;
  }
  return { black, white, empty: 64 - black - white };
}

function getNextPlayer(board, playerWhoMoved) {
  const preferred = opponent(playerWhoMoved);
  if (getLegalMoves(board, preferred).length > 0) return preferred;
  if (getLegalMoves(board, playerWhoMoved).length > 0) return playerWhoMoved;
  return EMPTY;
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
        if (isInside(r, c) && board[indexOf(r, c)] === EMPTY) {
          touchesEmpty = true;
          break;
        }
      }
    }
    if (touchesEmpty) frontier += 1;
  }
  return frontier;
}

function cornerScore(board, player) {
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

function evaluateBoard(board, player) {
  const pieces = countPieces(board);
  const myDiscs = player === BLACK ? pieces.black : pieces.white;
  const theirDiscs = player === BLACK ? pieces.white : pieces.black;
  const myMoves = getLegalMoves(board, player).length;
  const theirMoves = getLegalMoves(board, opponent(player)).length;
  const discDiff = myDiscs - theirDiscs;
  const mobility = myMoves - theirMoves;
  const frontier = frontierCount(board, opponent(player)) - frontierCount(board, player);
  const corners = cornerScore(board, player);
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

function boardKey(board, player, depth) {
  let key = player === BLACK ? 'b' : 'w';
  for (const cell of board) key += cell === BLACK ? '1' : cell === WHITE ? '2' : '0';
  return key + ':' + depth;
}

function orderedMoves(board, player, legalMoves) {
  return legalMoves.slice().sort((a, b) => {
    const aBoard = applyMove(board, a, player);
    const bBoard = applyMove(board, b, player);
    return evaluateBoard(bBoard, player) - evaluateBoard(aBoard, player);
  });
}

function checkDeadline(deadline) {
  if (Date.now() >= deadline) throw new Error('AI_TIMEOUT');
}

function alphaBeta(board, currentPlayer, rootPlayer, depth, alpha, beta, deadline, table) {
  checkDeadline(deadline);
  const legal = getLegalMoves(board, currentPlayer);
  const enemyLegal = getLegalMoves(board, opponent(currentPlayer));

  if (depth === 0 || (legal.length === 0 && enemyLegal.length === 0)) {
    return evaluateBoard(board, rootPlayer);
  }

  const key = boardKey(board, currentPlayer, depth);
  if (table.has(key)) return table.get(key);

  if (legal.length === 0) {
    const passedValue = alphaBeta(
      board,
      opponent(currentPlayer),
      rootPlayer,
      depth - 1,
      alpha,
      beta,
      deadline,
      table,
    );
    table.set(key, passedValue);
    return passedValue;
  }

  const maximizing = currentPlayer === rootPlayer;
  let value = maximizing ? -Infinity : Infinity;

  for (const move of orderedMoves(board, currentPlayer, legal)) {
    checkDeadline(deadline);
    const nextBoard = applyMove(board, move, currentPlayer);
    const nextPlayer = getNextPlayer(nextBoard, currentPlayer);
    const candidate = alphaBeta(
      nextBoard,
      nextPlayer === EMPTY ? opponent(currentPlayer) : nextPlayer,
      rootPlayer,
      depth - 1,
      alpha,
      beta,
      deadline,
      table,
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

function chooseFallbackMove(board, player, legal) {
  let bestMove = legal[0];
  let bestScore = -Infinity;
  for (const move of legal) {
    const score = evaluateBoard(applyMove(board, move, player), player);
    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
  }
  return bestMove;
}

function chooseMove(board, player) {
  const legal = getLegalMoves(board, player);
  if (legal.length === 0) return null;
  if (legal.length === 1) return legal[0];

  const deadline = Date.now() + SEARCH_TIME_LIMIT_MS;
  const empty = countPieces(board).empty;
  const maxDepth = empty <= 12 ? empty + 2 : empty <= 22 ? 7 : 6;
  let bestMove = chooseFallbackMove(board, player, legal);
  let bestScore = -Infinity;

  for (let depth = 2; depth <= maxDepth; depth += 1) {
    const table = new Map();
    let completed = true;
    let iterationBest = bestMove;
    let iterationScore = -Infinity;

    try {
      for (const move of orderedMoves(board, player, legal)) {
        checkDeadline(deadline);
        const nextBoard = applyMove(board, move, player);
        const nextPlayer = getNextPlayer(nextBoard, player);
        const score = alphaBeta(
          nextBoard,
          nextPlayer === EMPTY ? opponent(player) : nextPlayer,
          player,
          depth - 1,
          -Infinity,
          Infinity,
          deadline,
          table,
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

function parseBoard(rows) {
  const board = [];
  for (const row of rows) {
    for (const cell of row) {
      board.push(cell === '0' ? BLACK : cell === '1' ? WHITE : EMPTY);
    }
  }
  return board;
}

function indexToCoordinate(index) {
  const row = Math.floor(index / 8);
  const col = index % 8;
  return String.fromCharCode(97 + col) + String(row + 1);
}

const myId = parseInt(readline(), 10);
const boardSize = parseInt(readline(), 10);
const myPlayer = myId === 0 ? BLACK : WHITE;

while (true) {
  const rows = [];
  for (let i = 0; i < boardSize; i += 1) rows.push(readline());

  const actionCount = parseInt(readline(), 10);
  const refereeActions = [];
  for (let i = 0; i < actionCount; i += 1) refereeActions.push(readline());

  const board = parseBoard(rows);
  const selected = chooseMove(board, myPlayer);
  let output = selected ? indexToCoordinate(selected.index) : 'PASS';

  // The referee's legal-action list is the final guard against a rules or
  // coordinate-conversion regression.
  if (refereeActions.length > 0 && !refereeActions.includes(output)) {
    output = refereeActions[0];
  }

  console.log(output);
}
