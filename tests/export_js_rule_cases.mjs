import {
  BLACK,
  WHITE,
  applyMove,
  countPieces,
  createInitialBoard,
  getLegalMoves,
  getNextTurn,
  isGameOver,
  winner,
} from '../engine/javascript/rules.js';

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function snapshot(board, player) {
  const legalMoves = getLegalMoves(board, player);
  return {
    board: board.slice(),
    player,
    countPieces: countPieces(board),
    legalMoves: legalMoves.map((move) => ({
      index: move.index,
      flips: move.flips.slice(),
    })),
    legalActionMask: Array.from(
      { length: 64 },
      (_, index) => legalMoves.some((move) => move.index === index),
    ),
    gameOver: isGameOver(board),
    winner: winner(board),
    transitions: legalMoves.map((move) => {
      const nextBoard = applyMove(board, move, player);
      const turn = getNextTurn(nextBoard, player);
      return {
        index: move.index,
        board: nextBoard,
        turn,
      };
    }),
  };
}

const cases = [];
let passTransitions = 0;
let terminalTransitions = 0;

for (let game = 0; game < 8; game += 1) {
  const random = mulberry32(0xC0FFEE + game * 7919);
  let board = createInitialBoard();
  let player = BLACK;

  for (let ply = 0; ply < 60; ply += 1) {
    const state = snapshot(board, player);
    cases.push(state);

    if (state.legalMoves.length === 0) break;
    const choice = game === 0
      ? 0
      : Math.floor(random() * state.legalMoves.length);
    const move = state.legalMoves[choice];
    board = applyMove(board, move, player);
    const turn = getNextTurn(board, player);
    if (turn.passedPlayer !== null) passTransitions += 1;
    if (turn.gameOver) terminalTransitions += 1;
    if (turn.gameOver) {
      cases.push(snapshot(board, BLACK));
      cases.push(snapshot(board, WHITE));
      break;
    }
    player = turn.currentPlayer;
  }
}

const fullBlack = Array(64).fill(BLACK);
const fullWhite = Array(64).fill(WHITE);
cases.push(snapshot(fullBlack, BLACK));
cases.push(snapshot(fullBlack, WHITE));
cases.push(snapshot(fullWhite, BLACK));
cases.push(snapshot(fullWhite, WHITE));

process.stdout.write(JSON.stringify({
  cases,
  metadata: {
    passTransitions,
    terminalTransitions,
  },
}));
