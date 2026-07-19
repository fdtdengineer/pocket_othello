export const EMPTY = 0;
export const BLACK = 1;
export const WHITE = -1;
export const BOARD_SIZE = 8;

const DIRECTIONS = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1],           [0, 1],
  [1, -1],  [1, 0],  [1, 1],
];

export function opponent(player) {
  return -player;
}

export function indexOf(row, col) {
  return row * BOARD_SIZE + col;
}

export function rowCol(index) {
  return [Math.floor(index / BOARD_SIZE), index % BOARD_SIZE];
}

export function isInside(row, col) {
  return row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE;
}

export function createInitialBoard() {
  const board = Array(BOARD_SIZE * BOARD_SIZE).fill(EMPTY);
  board[indexOf(3, 3)] = WHITE;
  board[indexOf(3, 4)] = BLACK;
  board[indexOf(4, 3)] = BLACK;
  board[indexOf(4, 4)] = WHITE;
  return board;
}

export function getFlips(board, index, player) {
  if (!Array.isArray(board) || board.length !== 64 || board[index] !== EMPTY) {
    return [];
  }

  const [startRow, startCol] = rowCol(index);
  const allFlips = [];

  for (const [dr, dc] of DIRECTIONS) {
    let row = startRow + dr;
    let col = startCol + dc;
    const line = [];

    while (isInside(row, col)) {
      const value = board[indexOf(row, col)];
      if (value === opponent(player)) {
        line.push(indexOf(row, col));
        row += dr;
        col += dc;
        continue;
      }
      if (value === player && line.length > 0) {
        allFlips.push(...line);
      }
      break;
    }
  }

  return allFlips;
}

export function getLegalMoves(board, player) {
  const moves = [];
  for (let index = 0; index < board.length; index += 1) {
    if (board[index] !== EMPTY) continue;
    const flips = getFlips(board, index, player);
    if (flips.length > 0) moves.push({ index, flips });
  }
  return moves;
}

export function applyMove(board, moveOrIndex, player) {
  const index = typeof moveOrIndex === 'number' ? moveOrIndex : moveOrIndex.index;
  const flips = typeof moveOrIndex === 'number'
    ? getFlips(board, index, player)
    : moveOrIndex.flips;

  if (!flips || flips.length === 0) {
    throw new Error('Illegal Othello move.');
  }

  const next = board.slice();
  next[index] = player;
  for (const flipIndex of flips) next[flipIndex] = player;
  return next;
}

export function countPieces(board) {
  let black = 0;
  let white = 0;
  for (const value of board) {
    if (value === BLACK) black += 1;
    if (value === WHITE) white += 1;
  }
  return { black, white, empty: 64 - black - white };
}

export function getNextTurn(board, playerWhoMoved) {
  const preferred = opponent(playerWhoMoved);
  if (getLegalMoves(board, preferred).length > 0) {
    return { currentPlayer: preferred, passedPlayer: null, gameOver: false };
  }
  if (getLegalMoves(board, playerWhoMoved).length > 0) {
    return { currentPlayer: playerWhoMoved, passedPlayer: preferred, gameOver: false };
  }
  return { currentPlayer: EMPTY, passedPlayer: null, gameOver: true };
}

export function isGameOver(board) {
  return getLegalMoves(board, BLACK).length === 0 && getLegalMoves(board, WHITE).length === 0;
}

export function winner(board) {
  const { black, white } = countPieces(board);
  if (black > white) return BLACK;
  if (white > black) return WHITE;
  return EMPTY;
}

export function playerName(player) {
  if (player === BLACK) return 'Black';
  if (player === WHITE) return 'White';
  return 'None';
}
