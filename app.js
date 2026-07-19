import {
  BLACK,
  WHITE,
  EMPTY,
  applyMove,
  countPieces,
  createInitialBoard,
  getLegalMoves,
  getNextTurn,
  playerName,
  winner,
} from './engine.js';
import { chooseCpuMove } from './ai.js';
import { OnlineSession, normalizeRoomCode } from './online.js';

const elements = {
  board: document.querySelector('#board'),
  blackScore: document.querySelector('#blackScore'),
  whiteScore: document.querySelector('#whiteScore'),
  blackLabel: document.querySelector('#blackLabel'),
  whiteLabel: document.querySelector('#whiteLabel'),
  blackCard: document.querySelector('#blackCard'),
  whiteCard: document.querySelector('#whiteCard'),
  statusText: document.querySelector('#statusText'),
  statusDot: document.querySelector('#connectionDot'),
  modeText: document.querySelector('#modeText'),
  thinkingBadge: document.querySelector('#thinkingBadge'),
  undoButton: document.querySelector('#undoButton'),
  newGameButton: document.querySelector('#newGameButton'),
  menuButton: document.querySelector('#menuButton'),
  onlineButton: document.querySelector('#onlineButton'),
  modalBackdrop: document.querySelector('#modalBackdrop'),
  closeModalButton: document.querySelector('#closeModalButton'),
  modeChooser: document.querySelector('#modeChooser'),
  onlinePanel: document.querySelector('#onlinePanel'),
  waitingPanel: document.querySelector('#waitingPanel'),
  cpuModeButton: document.querySelector('#cpuModeButton'),
  localModeButton: document.querySelector('#localModeButton'),
  onlineModeButton: document.querySelector('#onlineModeButton'),
  backToModesButton: document.querySelector('#backToModesButton'),
  createRoomButton: document.querySelector('#createRoomButton'),
  joinRoomButton: document.querySelector('#joinRoomButton'),
  roomCodeInput: document.querySelector('#roomCodeInput'),
  roomCodeText: document.querySelector('#roomCodeText'),
  roomCodeBox: document.querySelector('#roomCodeBox'),
  shareRoomButton: document.querySelector('#shareRoomButton'),
  cancelOnlineButton: document.querySelector('#cancelOnlineButton'),
  waitingTitle: document.querySelector('#waitingTitle'),
  onlineStatusText: document.querySelector('#onlineStatusText'),
  toast: document.querySelector('#toast'),
};

const cells = [];
for (let index = 0; index < 64; index += 1) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'cell';
  button.dataset.index = String(index);
  button.setAttribute('role', 'gridcell');
  button.setAttribute('aria-label', `Row ${Math.floor(index / 8) + 1}, column ${(index % 8) + 1}`);
  button.addEventListener('click', () => onCellClick(index));
  cells.push(button);
  elements.board.append(button);
}

let state = freshState();
let cpuDifficulty = localStorage.getItem('pocketOthelloDifficulty') || 'normal';
let history = [];
let cpuToken = 0;
let online = null;
let toastTimer = null;

function freshState(overrides = {}) {
  return {
    board: createInitialBoard(),
    currentPlayer: BLACK,
    gameOver: false,
    lastMove: null,
    passedPlayer: null,
    mode: 'cpu',
    humanPlayer: BLACK,
    onlineRole: null,
    connected: false,
    version: 0,
    ...overrides,
  };
}

function cloneState(value) {
  return { ...value, board: value.board.slice() };
}

function showToast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add('is-visible');
  toastTimer = setTimeout(() => elements.toast.classList.remove('is-visible'), 2600);
}

function setStatus(message, kind = 'ok') {
  elements.statusText.textContent = message;
  elements.statusDot.classList.toggle('is-waiting', kind === 'waiting');
  elements.statusDot.classList.toggle('is-error', kind === 'error');
}

function canLocalPlayerMove() {
  if (state.gameOver) return false;
  if (state.mode === 'cpu') return state.currentPlayer === state.humanPlayer;
  if (state.mode === 'local') return true;
  if (state.mode === 'online') {
    if (!state.connected) return false;
    const myPlayer = state.onlineRole === 'host' ? BLACK : WHITE;
    return state.currentPlayer === myPlayer;
  }
  return false;
}

function render() {
  const counts = countPieces(state.board);
  elements.blackScore.textContent = counts.black;
  elements.whiteScore.textContent = counts.white;
  elements.blackCard.classList.toggle('is-active', state.currentPlayer === BLACK && !state.gameOver);
  elements.whiteCard.classList.toggle('is-active', state.currentPlayer === WHITE && !state.gameOver);

  const legal = canLocalPlayerMove() ? new Set(getLegalMoves(state.board, state.currentPlayer).map((move) => move.index)) : new Set();
  for (let index = 0; index < 64; index += 1) {
    const cell = cells[index];
    cell.replaceChildren();
    cell.classList.toggle('is-legal', legal.has(index));
    cell.disabled = !legal.has(index);
    const value = state.board[index];
    if (value !== EMPTY) {
      const disc = document.createElement('span');
      disc.className = `disc ${value === BLACK ? 'black' : 'white'}`;
      if (state.lastMove === index) disc.classList.add('last-move');
      cell.append(disc);
    }
  }

  if (state.mode === 'cpu') {
    elements.blackLabel.textContent = state.humanPlayer === BLACK ? 'You · Black' : 'CPU · Black';
    elements.whiteLabel.textContent = state.humanPlayer === WHITE ? 'You · White' : 'CPU · White';
    elements.modeText.textContent = `CPU · ${capitalize(cpuDifficulty)}`;
  } else if (state.mode === 'local') {
    elements.blackLabel.textContent = 'Player 1 · Black';
    elements.whiteLabel.textContent = 'Player 2 · White';
    elements.modeText.textContent = 'Two Players';
  } else {
    elements.blackLabel.textContent = state.onlineRole === 'host' ? 'You · Black' : 'Opponent · Black';
    elements.whiteLabel.textContent = state.onlineRole === 'guest' ? 'You · White' : 'Opponent · White';
    elements.modeText.textContent = `Online · ${state.onlineRole === 'host' ? 'Host' : 'Guest'}`;
  }

  elements.undoButton.disabled = history.length === 0 || state.mode === 'online' || state.gameOver;
  updateStatusFromState();
}

function updateStatusFromState() {
  if (state.gameOver) {
    const gameWinner = winner(state.board);
    const counts = countPieces(state.board);
    if (gameWinner === EMPTY) setStatus(`Draw · ${counts.black}–${counts.white}`);
    else setStatus(`${playerName(gameWinner)} wins · ${counts.black}–${counts.white}`);
    return;
  }

  if (state.mode === 'online' && !state.connected) {
    setStatus('Waiting for opponent', 'waiting');
    return;
  }

  if (state.passedPlayer) {
    const passed = playerName(state.passedPlayer);
    setStatus(`${passed} has no legal move · ${playerName(state.currentPlayer)} plays again`);
    return;
  }

  if (state.mode === 'cpu') {
    setStatus(state.currentPlayer === state.humanPlayer ? 'Your turn' : 'CPU turn', state.currentPlayer === state.humanPlayer ? 'ok' : 'waiting');
  } else if (state.mode === 'online') {
    const myPlayer = state.onlineRole === 'host' ? BLACK : WHITE;
    setStatus(state.currentPlayer === myPlayer ? 'Your turn' : "Opponent's turn", state.currentPlayer === myPlayer ? 'ok' : 'waiting');
  } else {
    setStatus(`${playerName(state.currentPlayer)} to move`);
  }
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function onCellClick(index) {
  if (!canLocalPlayerMove()) return;
  const legalMove = getLegalMoves(state.board, state.currentPlayer).find((move) => move.index === index);
  if (!legalMove) return;

  if (state.mode === 'online' && state.onlineRole === 'guest') {
    const sent = online?.send({ type: 'move', index, version: state.version });
    if (sent) setStatus('Move sent…', 'waiting');
    return;
  }

  commitMove(legalMove, state.currentPlayer, { saveHistory: state.mode !== 'online' });
  if (state.mode === 'online' && state.onlineRole === 'host') broadcastState();
  maybeRunCpu();
}

function commitMove(move, player, options = {}) {
  if (options.saveHistory) history.push(cloneState(state));
  const nextBoard = applyMove(state.board, move, player);
  const turn = getNextTurn(nextBoard, player);
  state = {
    ...state,
    board: nextBoard,
    currentPlayer: turn.currentPlayer,
    gameOver: turn.gameOver,
    passedPlayer: turn.passedPlayer,
    lastMove: move.index,
    version: state.version + 1,
  };
  render();
}

function maybeRunCpu() {
  cpuToken += 1;
  const token = cpuToken;
  if (state.mode !== 'cpu' || state.gameOver || state.currentPlayer === state.humanPlayer) {
    elements.thinkingBadge.hidden = true;
    return;
  }

  elements.thinkingBadge.hidden = false;
  setTimeout(() => {
    if (token !== cpuToken || state.mode !== 'cpu' || state.gameOver || state.currentPlayer === state.humanPlayer) return;
    const move = chooseCpuMove(state.board, state.currentPlayer, cpuDifficulty, { timeLimitMs: cpuDifficulty === 'hard' ? 720 : 250 });
    elements.thinkingBadge.hidden = true;
    if (move) commitMove(move, state.currentPlayer, { saveHistory: true });
    if (!state.gameOver && state.currentPlayer !== state.humanPlayer) maybeRunCpu();
  }, cpuDifficulty === 'hard' ? 180 : 320);
}

function resetGame(mode = state.mode) {
  cpuToken += 1;
  history = [];
  const onlineFields = mode === 'online' ? {
    onlineRole: state.onlineRole,
    connected: state.connected,
  } : {};
  state = freshState({ mode, ...onlineFields });
  render();
  if (mode === 'online' && state.onlineRole === 'host') broadcastState();
  maybeRunCpu();
}

function startCpuMode() {
  stopOnline();
  state = freshState({ mode: 'cpu', humanPlayer: BLACK });
  history = [];
  closeModal();
  render();
}

function startLocalMode() {
  stopOnline();
  state = freshState({ mode: 'local' });
  history = [];
  closeModal();
  render();
}

function openModal(view = 'modes') {
  elements.modalBackdrop.classList.add('is-visible');
  showModalView(view);
}

function closeModal() {
  elements.modalBackdrop.classList.remove('is-visible');
}

function showModalView(view) {
  elements.modeChooser.hidden = view !== 'modes';
  elements.onlinePanel.hidden = view !== 'online';
  elements.waitingPanel.hidden = view !== 'waiting';
}

function setupOnlineSession() {
  stopOnline();
  online = new OnlineSession();
  online.addEventListener('ready', (event) => {
    const { roomCode } = event.detail;
    elements.roomCodeText.textContent = roomCode;
    elements.onlineStatusText.textContent = online.role === 'host' ? 'Keep this page open while your opponent joins.' : 'Connecting to the host…';
  });
  online.addEventListener('connected', () => {
    state.connected = true;
    elements.waitingTitle.textContent = 'Opponent connected!';
    elements.onlineStatusText.textContent = 'Starting the match…';
    if (online.role === 'host') {
      state = freshState({ mode: 'online', onlineRole: 'host', connected: true });
      broadcastState({ includeAssignment: true });
      setTimeout(closeModal, 500);
      render();
    }
  });
  online.addEventListener('message', (event) => handleOnlineMessage(event.detail.data));
  online.addEventListener('disconnected', (event) => {
    state.connected = false;
    render();
    showToast(event.detail.message || 'Opponent disconnected.');
  });
  online.addEventListener('error', (event) => {
    elements.onlineStatusText.textContent = event.detail.message;
    setStatus(event.detail.message, 'error');
    showToast(event.detail.message);
  });
}

function hostOnlineGame() {
  setupOnlineSession();
  showModalView('waiting');
  elements.waitingTitle.textContent = 'Waiting for an opponent…';
  elements.roomCodeBox.hidden = false;
  elements.shareRoomButton.hidden = false;
  elements.onlineStatusText.textContent = 'Creating room…';
  try {
    const code = online.host();
    state = freshState({ mode: 'online', onlineRole: 'host', connected: false });
    elements.roomCodeText.textContent = code;
    render();
  } catch (error) {
    elements.onlineStatusText.textContent = error.message;
  }
}

function joinOnlineGame() {
  const code = normalizeRoomCode(elements.roomCodeInput.value);
  elements.roomCodeInput.value = code;
  if (code.length !== 6) {
    showToast('Enter a valid 6-character room code.');
    return;
  }
  setupOnlineSession();
  showModalView('waiting');
  elements.waitingTitle.textContent = 'Connecting to room…';
  elements.roomCodeBox.hidden = false;
  elements.shareRoomButton.hidden = true;
  elements.roomCodeText.textContent = code;
  elements.onlineStatusText.textContent = 'Looking for the host…';
  state = freshState({ mode: 'online', onlineRole: 'guest', connected: false });
  render();
  try {
    online.join(code);
  } catch (error) {
    elements.onlineStatusText.textContent = error.message;
  }
}

function handleOnlineMessage(message) {
  if (!message || typeof message !== 'object') return;

  if (message.type === 'state' && online.role === 'guest') {
    const incoming = message.state;
    if (!incoming || !Array.isArray(incoming.board) || incoming.board.length !== 64) return;
    state = {
      ...freshState(),
      ...incoming,
      board: incoming.board.slice(),
      mode: 'online',
      onlineRole: 'guest',
      connected: true,
    };
    closeModal();
    render();
    return;
  }

  if (message.type === 'move' && online.role === 'host') {
    if (message.version !== state.version || state.gameOver || state.currentPlayer !== WHITE) {
      broadcastState();
      return;
    }
    const legalMove = getLegalMoves(state.board, WHITE).find((move) => move.index === message.index);
    if (!legalMove) {
      broadcastState();
      return;
    }
    commitMove(legalMove, WHITE);
    broadcastState();
    return;
  }

  if (message.type === 'new-game' && online.role === 'host') resetGame('online');
}

function serializableState() {
  return {
    board: state.board.slice(),
    currentPlayer: state.currentPlayer,
    gameOver: state.gameOver,
    lastMove: state.lastMove,
    passedPlayer: state.passedPlayer,
    version: state.version,
  };
}

function broadcastState(options = {}) {
  online?.send({ type: 'state', state: serializableState(), assignment: options.includeAssignment ? WHITE : undefined });
}

function stopOnline() {
  if (online) online.destroy();
  online = null;
}

async function shareRoom() {
  const code = online?.roomCode || elements.roomCodeText.textContent;
  const url = new URL(window.location.href);
  url.searchParams.set('room', code);
  const text = `Join my Pocket Othello room: ${code}`;
  try {
    if (navigator.share) {
      await navigator.share({ title: 'Pocket Othello', text, url: url.toString() });
    } else {
      await navigator.clipboard.writeText(`${text}\n${url}`);
      showToast('Invite copied to clipboard.');
    }
  } catch (error) {
    if (error.name !== 'AbortError') showToast('Could not share the invite.');
  }
}

for (const button of document.querySelectorAll('[data-difficulty]')) {
  button.classList.toggle('is-selected', button.dataset.difficulty === cpuDifficulty);
  button.addEventListener('click', () => {
    cpuDifficulty = button.dataset.difficulty;
    localStorage.setItem('pocketOthelloDifficulty', cpuDifficulty);
    for (const other of document.querySelectorAll('[data-difficulty]')) {
      other.classList.toggle('is-selected', other === button);
    }
    if (state.mode === 'cpu') {
      elements.modeText.textContent = `CPU · ${capitalize(cpuDifficulty)}`;
      resetGame('cpu');
    }
  });
}

elements.undoButton.addEventListener('click', () => {
  if (!history.length || state.mode === 'online') return;
  cpuToken += 1;
  state = history.pop();
  if (state.mode === 'cpu' && state.currentPlayer !== state.humanPlayer && history.length) state = history.pop();
  elements.thinkingBadge.hidden = true;
  render();
});
elements.newGameButton.addEventListener('click', () => {
  if (state.mode === 'online' && state.onlineRole === 'guest') {
    online?.send({ type: 'new-game' });
    showToast('New game requested.');
  } else {
    resetGame(state.mode);
  }
});
elements.menuButton.addEventListener('click', () => openModal('modes'));
elements.onlineButton.addEventListener('click', () => openModal('online'));
elements.closeModalButton.addEventListener('click', closeModal);
elements.modalBackdrop.addEventListener('click', (event) => { if (event.target === elements.modalBackdrop) closeModal(); });
elements.cpuModeButton.addEventListener('click', startCpuMode);
elements.localModeButton.addEventListener('click', startLocalMode);
elements.onlineModeButton.addEventListener('click', () => showModalView('online'));
elements.backToModesButton.addEventListener('click', () => showModalView('modes'));
elements.createRoomButton.addEventListener('click', hostOnlineGame);
elements.joinRoomButton.addEventListener('click', joinOnlineGame);
elements.roomCodeInput.addEventListener('input', () => { elements.roomCodeInput.value = normalizeRoomCode(elements.roomCodeInput.value); });
elements.roomCodeInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') joinOnlineGame(); });
elements.shareRoomButton.addEventListener('click', shareRoom);
elements.cancelOnlineButton.addEventListener('click', () => { stopOnline(); showModalView('online'); startCpuMode(); openModal('online'); });

const invitedRoom = normalizeRoomCode(new URLSearchParams(window.location.search).get('room'));
if (invitedRoom.length === 6) {
  elements.roomCodeInput.value = invitedRoom;
  showModalView('online');
}

render();

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('./service-worker.js').catch(() => {}));
}
