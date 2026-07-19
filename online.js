const ROOM_PREFIX = 'pocket-othello-';
const CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

export function normalizeRoomCode(value) {
  return String(value || '')
    .toUpperCase()
    .replace(/[^A-Z2-9]/g, '')
    .replace(/[IO01]/g, '')
    .slice(0, 6);
}

export function generateRoomCode(random = Math.random) {
  let code = '';
  for (let i = 0; i < 6; i += 1) {
    code += CODE_ALPHABET[Math.floor(random() * CODE_ALPHABET.length)];
  }
  return code;
}

export class OnlineSession extends EventTarget {
  constructor() {
    super();
    this.peer = null;
    this.connection = null;
    this.role = null;
    this.roomCode = null;
  }

  emit(type, detail = {}) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  ensurePeerJs() {
    if (!window.Peer) throw new Error('Online library could not be loaded.');
  }

  destroy() {
    if (this.connection) this.connection.close();
    if (this.peer && !this.peer.destroyed) this.peer.destroy();
    this.connection = null;
    this.peer = null;
    this.role = null;
    this.roomCode = null;
  }

  host(code = generateRoomCode()) {
    this.destroy();
    this.ensurePeerJs();
    this.role = 'host';
    this.roomCode = normalizeRoomCode(code);
    const peerId = ROOM_PREFIX + this.roomCode.toLowerCase();
    this.peer = new window.Peer(peerId, { debug: 1 });

    this.peer.on('open', () => this.emit('ready', { role: this.role, roomCode: this.roomCode }));
    this.peer.on('connection', (connection) => {
      if (this.connection?.open) {
        connection.on('open', () => connection.close());
        return;
      }
      this.attachConnection(connection);
    });
    this.attachPeerErrors();
    return this.roomCode;
  }

  join(code) {
    this.destroy();
    this.ensurePeerJs();
    this.role = 'guest';
    this.roomCode = normalizeRoomCode(code);
    if (this.roomCode.length !== 6) throw new Error('Enter a valid 6-character room code.');

    this.peer = new window.Peer(undefined, { debug: 1 });
    this.peer.on('open', () => {
      const connection = this.peer.connect(ROOM_PREFIX + this.roomCode.toLowerCase(), {
        serialization: 'json',
        reliable: true,
        metadata: { game: 'pocket-othello', version: 1 },
      });
      this.attachConnection(connection);
    });
    this.attachPeerErrors();
  }

  attachPeerErrors() {
    this.peer.on('error', (error) => {
      let message = error.message || 'Online connection failed.';
      if (error.type === 'unavailable-id') message = 'That room code is already in use. Try another one.';
      if (error.type === 'peer-unavailable') message = 'Room not found. Check the code and make sure the host is waiting.';
      this.emit('error', { message, error });
    });
  }

  attachConnection(connection) {
    this.connection = connection;
    connection.on('open', () => this.emit('connected', { role: this.role, roomCode: this.roomCode }));
    connection.on('data', (data) => this.emit('message', { data }));
    connection.on('close', () => this.emit('disconnected', { message: 'Opponent disconnected.' }));
    connection.on('error', (error) => this.emit('error', { message: error.message || 'Connection error.', error }));
  }

  send(data) {
    if (!this.connection?.open) return false;
    this.connection.send(data);
    return true;
  }
}
