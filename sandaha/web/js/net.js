// WebSocket connection with automatic reconnect.
import { store, toast } from './util.js';

const listeners = {};
let ws = null;
let openPromise = null;
let retry = 0;

export const me = {
  id: store.get('sdh.client_id', null),
  name: store.get('sdh.name', '玩家' + Math.floor(Math.random() * 900 + 100)),
};

export function on(op, fn) { (listeners[op] ||= new Set()).add(fn); return () => listeners[op].delete(fn); }
function emit(op, msg) { (listeners[op] || []).forEach((fn) => fn(msg)); (listeners['*'] || []).forEach((fn) => fn(msg)); }

export function setName(n) {
  me.name = (n || '').trim().slice(0, 16) || me.name;
  store.set('sdh.name', me.name);
  if (ws && ws.readyState === 1) send({ op: 'hello', client_id: me.id, name: me.name });
}

export function connect() {
  if (ws && (ws.readyState === 0 || ws.readyState === 1)) return openPromise;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  openPromise = new Promise((resolve) => {
    ws.onopen = () => {
      retry = 0;
      ws.send(JSON.stringify({ op: 'hello', client_id: me.id, name: me.name }));
      emit('open', {});
      resolve();
    };
  });
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.op === 'welcome') { me.id = msg.client_id; store.set('sdh.client_id', me.id); }
    if (msg.op === 'error') toast(msg.msg);
    emit(msg.op, msg);
  };
  ws.onclose = () => {
    emit('close', {});
    retry = Math.min(retry + 1, 6);
    setTimeout(connect, 400 * retry);
  };
  return openPromise;
}

export async function send(obj) {
  if (!ws || ws.readyState !== 1) await connect();
  ws.send(JSON.stringify(obj));
}
