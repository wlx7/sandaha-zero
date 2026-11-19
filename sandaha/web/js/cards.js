// Card rendering and hand layout.
import { h } from './util.js';

export const SUIT_SYM = { S: '♠', H: '♥', C: '♣', D: '♦' };
export const SUIT_IDX = { S: 0, H: 1, C: 2, D: 3 };
const RED = new Set(['H', 'D']);

export function parse(code) {
  if (code === 'BJ') return { joker: 'big', suit: null, rank: '大王' };
  if (code === 'LJ') return { joker: 'small', suit: null, rank: '小王' };
  return { joker: null, suit: code[0], rank: code.slice(1) };
}

export function isTrump(code, trump) {
  const c = parse(code);
  if (c.joker) return true;
  if (c.rank === '7' || c.rank === '2') return true;
  return trump !== null && trump !== undefined && trump < 4 && SUIT_IDX[c.suit] === trump;
}

export function label(code) {
  const c = parse(code);
  if (c.joker) return c.rank;
  return SUIT_SYM[c.suit] + c.rank;
}

export function colorOf(code) {
  const c = parse(code);
  if (c.joker) return c.joker === 'big' ? 'red' : 'black';
  return RED.has(c.suit) ? 'red' : 'black';
}

export function cardEl(code, { trump = null, mini = false } = {}) {
  const c = parse(code);
  const tr = isTrump(code, trump);
  const cls = ['card', colorOf(code)];
  if (mini) cls.push('mini');
  if (c.joker) cls.push('joker');
  if (tr) cls.push('trump');
  const el = h('div', { class: cls.join(' '), 'data-code': code, title: label(code) });
  if (c.joker) {
    el.append(h('div', { class: 'idx' }, c.joker === 'big' ? '大王' : '小王'));
    el.append(h('div', { class: 'pip' }, '★'));
  } else {
    // Keep "主" inside top-left .idx so adjacent fanned cards or stacked rows never cover it!
    el.append(h('div', { class: 'idx' },
      h('span', { class: 'rk' }, c.rank),
      h('span', { class: 's' }, SUIT_SYM[c.suit]),
      tr ? h('span', { class: 'tmark' }, '主') : null));
    el.append(h('div', { class: 'pip' }, SUIT_SYM[c.suit]));
  }
  return el;
}

export function chipEl(code, trump) {
  const cls = ['chip', colorOf(code)];
  if (isTrump(code, trump)) cls.push('trump');
  return h('span', { class: cls.join(' ') }, label(code));
}

export function stackEl(codes, trump, extraClass = '') {
  const el = h('div', { class: 'stack ' + extraClass });
  for (const c of codes) el.append(cardEl(c, { trump, mini: true }));
  return el;
}

// group key so that the hand shows trump first then each side suit
function groupOf(code, trump) {
  if (isTrump(code, trump)) return 'T';
  return parse(code).suit;
}

/**
 * Lay out a hand into rows inside `container` (absolute positioning).
 * In landscape mobile viewports, keeps up to 29 cards in a single row so the
 * table fits 100% inside 100dvh with zero vertical scrolling or clipping.
 */
export function layoutHand(container, codes, { trump = null, selected = new Set(), onToggle = null, disabled = false } = {}) {
  container.innerHTML = '';
  const n = codes.length;
  if (!n) { container.style.height = '0px'; return; }

  const style = getComputedStyle(document.documentElement);
  const cw = parseFloat(style.getPropertyValue('--cw')) || 70;
  const ch = parseFloat(style.getPropertyValue('--ch')) || 98;
  const W = Math.max(cw * 3, container.clientWidth || container.parentElement.clientWidth || 360);

  const vw = window.innerWidth || W;
  const vh = window.innerHeight || 700;
  const shortLandscape = vw > vh && vh <= 540;

  const groupGap = shortLandscape ? Math.max(4, Math.round(cw * 0.10)) : Math.round(cw * 0.18);
  const minStep = shortLandscape ? Math.max(17, Math.round(cw * 0.37)) : Math.max(23, Math.round(cw * 0.48));
  const maxStep = Math.round(cw * 0.66);
  const maxRows = shortLandscape ? 1 : 4;

  let rows = 1;
  const groups = codes.map((c) => groupOf(c, trump));
  const nGroups = (row) => { let g = 0; for (let i = 1; i < row.length; i++) if (row[i] !== row[i - 1]) g++; return g; };
  let step;
  for (;;) {
    const per = Math.ceil(n / rows);
    const gaps = nGroups(groups.slice(0, per));
    step = (W - cw - gaps * groupGap) / Math.max(1, per - 1);
    if (step >= minStep || rows >= maxRows) break;
    rows++;
  }
  const absFloor = shortLandscape ? 15 : 19;
  step = Math.min(maxStep, Math.max(absFloor, step));
  const per = Math.ceil(n / rows);
  const rowGap = rows > 1 ? Math.round(ch * (rows > 2 ? 0.56 : 0.66)) : 0;
  const lift = shortLandscape ? 9 : 14;

  for (let r = 0; r < rows; r++) {
    const slice = codes.slice(r * per, (r + 1) * per);
    const g = groups.slice(r * per, (r + 1) * per);
    const width = cw + (slice.length - 1) * step + nGroups(g) * groupGap;
    let x = Math.max(0, (W - width) / 2);
    slice.forEach((code, i) => {
      if (i > 0 && g[i] !== g[i - 1]) x += groupGap;
      const el = cardEl(code, { trump });
      el.style.left = x + 'px';
      el.style.top = (lift + r * rowGap) + 'px';
      el.style.zIndex = String(r * 100 + i);
      const key = `${code}#${r * per + i}`;
      el.dataset.key = key;
      if (selected.has(key)) el.classList.add('sel');
      if (onToggle && !disabled) el.addEventListener('click', () => onToggle(key, code, el));
      container.append(el);
      x += step;
    });
  }
  container.style.height = (lift + (rows - 1) * rowGap + ch + 2) + 'px';
}
