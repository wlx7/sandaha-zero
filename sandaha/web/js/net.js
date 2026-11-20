// WebSocket connection with automatic reconnect and outbound message queue.
import { store, toast } from './util.js';

const listeners = {};
const outbox = [];
let ws = null;
let reconnectTimer = null;
let retry = 0;
let lastMsgAt = Date.now();

export const me = {
  id: store.get('sdh.client_id', null),
  name: store.get('sdh.name', '玩家' + Math.floor(Math.random() * 900 + 100)),
};

export function on(op, fn) {
  (listeners[op] ||= new Set()).add(fn);
  return () => listeners[op].delete(fn);
}

function emit(op, msg) {
  [...(listeners[op] || [])].forEach((fn) => fn(msg));
  [...(listeners['*'] || [])].forEach((fn) => fn(msg));
}

export function setName(n) {
  me.name = (n || '').trim().slice(0, 16) || me.name;
  store.set('sdh.name', me.name);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ op: 'hello', client_id: me.id, name: me.name }));
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  retry = Math.min(retry + 1, 6);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 300 * retry);
}

export function connect() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
    return;
  }
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const sock = new WebSocket(`${proto}://${location.host}/ws`);
  ws = sock;

  sock.onopen = () => {
    if (ws !== sock) return;
    retry = 0;
    lastMsgAt = Date.now();
    sock.send(JSON.stringify({ op: 'hello', client_id: me.id, name: me.name }));
    while (outbox.length && sock.readyState === WebSocket.OPEN) {
      sock.send(outbox.shift());
    }
    emit('open', {});
  };

  sock.onmessage = (e) => {
    lastMsgAt = Date.now();
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }
    if (msg.op === 'welcome') {
      me.id = msg.client_id;
      store.set('sdh.client_id', me.id);
    }
    if (msg.op === 'error') toast(msg.msg);
    emit(msg.op, msg);
  };

  sock.onerror = () => {
    // onclose follows and schedules reconnect
  };

  sock.onclose = () => {
    if (ws === sock) ws = null;
    emit('close', {});
    scheduleReconnect();
  };
}

export function send(obj) {
  const payload = JSON.stringify(obj);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(payload);
    return;
  }
  if (outbox.length < 64) outbox.push(payload);
  connect();
}

// Drop a socket that looks open but no longer delivers anything, and reconnect right away.
function restartSocket() {
  const old = ws;
  ws = null;
  if (old) {
    old.onopen = old.onmessage = old.onerror = old.onclose = null;
    try { old.close(); } catch { /* ignore */ }
  }
  emit('close', {});
  retry = 0;
  connect();
}

// Heartbeat. A phone that slept, switched Wi-Fi or lost a packet can leave the socket "open" while
// nothing arrives any more: the table then looks frozen. Ping every few seconds; the server answers
// with the revision of the room state, so a missed update is noticed too (see game.js 'pong').
const PING_MS = 4000;
setInterval(() => {
  if (document.hidden) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (Date.now() - lastMsgAt > PING_MS * 2.5) {
    restartSocket();
    return;
  }
  try { ws.send('{"op":"ping"}'); } catch { /* onclose will follow */ }
}, PING_MS);

document.addEventListener('visibilitychange', () => {
  if (document.hidden) return;
  lastMsgAt = Date.now();   // timers were throttled while hidden: do not count that as silence
  if (!ws || ws.readyState > WebSocket.OPEN) {
    retry = 0;
    connect();
  } else if (ws.readyState === WebSocket.OPEN) {
    try { ws.send('{"op":"ping"}'); } catch { /* ignore */ }
  }
});

window.addEventListener('online', () => {
  if (!ws || ws.readyState > WebSocket.OPEN) {
    retry = 0;
    connect();
  }
});
