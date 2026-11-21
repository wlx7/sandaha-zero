// Save slots: the "谁在玩？" picker (create / rename / delete / choose a user) and the stats page.
import { $, h, modal, toast } from './util.js';
import { me, on, send } from './net.js';

const MODE_NAMES = { battle: '对战', teach: '教学' };

const signed = (n) => (n > 0 ? `+${n}` : `${n}`);
const pctOf = (a, b) => (b ? `${Math.round((a / b) * 100)}%` : '—');

function when(t) {
  if (!t) return '还没玩过';
  const d = new Date(t * 1000);
  const p = (x) => String(x).padStart(2, '0');
  return `${d.getMonth() + 1}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function initial(name) {
  return [...(name || '?')][0].toUpperCase();
}

export function profileLine(p) {
  const s = (p && p.stats && p.stats.battle) || { hands: 0, score: 0 };
  return s.hands ? `对战 ${s.hands} 局 · 累计 ${signed(s.score)}` : '还没有对战记录';
}

// Make a modal ignore the backdrop and the Android back button (no way out but choosing a user)
function lock(m) {
  const back = m.box.parentElement;
  if (back) back.dataset.locked = '1';
}

// One-line name prompt. On a landscape phone the keyboard leaves ~120px of height, so the input and
// both buttons share one row (and the note sits next to the title) to keep 确定 visible.
export function askName(title, value, onOk, note) {
  const input = h('input', { value: value || '', maxlength: 12, placeholder: '最多 12 个字', enterkeyhint: 'done', autocomplete: 'off' });
  const ok = () => {
    const v = input.value.trim();
    if (!v) { toast('请输入用户名'); input.focus(); return; }
    m.close();
    onOk(v);
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') ok(); });
  const m = modal(h('div', { class: 'name-ask' },
    h('h2', {}, title, note ? h('span', { class: 'hint note' }, note) : null),
    h('div', { class: 'name-row' },
      input,
      h('button', { class: 'btn ghost', onclick: () => m.close() }, '取消'),
      h('button', { class: 'btn primary', onclick: ok }, '确定'))));
  setTimeout(() => { input.focus(); input.select(); }, 30);
}

let picker = null;   // the open picker, if any

export function pickerOpen() {
  return Boolean(picker && picker.m.box.isConnected);
}

/** Show the user picker. ``force``: no profile is active, so it cannot be dismissed without choosing. */
export function openProfilePicker({ force = false, onDone } = {}) {
  if (pickerOpen()) {
    if (force) picker.requireChoice();
    return;
  }
  if (force && me.guest) return;   // a guest has deliberately chosen not to have a user
  const st = { list: null, inUse: {}, sel: me.profileId, pending: false, force };
  const body = h('div', { class: 'pp-body' });
  const offs = [];
  const m = modal(h('div', { class: 'profile-picker' }, h('h2', {}, '谁在玩？'), body),
    { closable: !force, onClose: () => { offs.forEach((f) => f()); picker = null; } });
  // No active user any more: hide 取消 and lock the popup until someone is chosen
  function requireChoice() {
    lock(m);
    if (!st.force) { st.force = true; render(); }
  }
  picker = { m, requireChoice };
  if (force) lock(m);

  function close() {
    m.close();
  }

  function playAsGuest() {
    askName('游客模式', me.guest || !me.profile ? me.name : '', (name) => {
      st.pending = true;
      send({ op: 'profile_guest', name });
    }, '游客不保存战绩，牌局结束后分数就没了。');
  }

  function choose(id) {
    if (!id) return;
    if (st.inUse[id]) { toast('这个用户正在另一台设备上使用'); return; }
    st.pending = true;
    send({ op: 'profile_select', id });
  }

  function render() {
    body.innerHTML = '';
    if (st.list === null) {
      body.append(h('div', { class: 'hint', style: { padding: '18px 0', textAlign: 'center' } }, '正在读取用户列表…'));
      return;
    }
    if (!st.list.length) {
      body.append(
        h('div', { class: 'pp-empty' }, '欢迎！先给自己起个名字吧，战绩都会记在这个名字下。'),
        h('div', { class: 'row', style: { justifyContent: 'center', marginTop: '12px' } },
          h('button', { class: 'btn ghost', onclick: playAsGuest }, '游客模式'),
          h('button', { class: 'btn primary', onclick: create }, '新建用户')),
        h('div', { class: 'hint', style: { textAlign: 'center', marginTop: '8px' } }, '游客模式不保存战绩，牌局结束后分数就没了。'));
      return;
    }
    const list = h('div', { class: 'pp-list', role: 'listbox' });
    for (const p of st.list) {
      const busy = Boolean(st.inUse[p.id]);
      const row = h('button', {
        class: 'pp-row' + (p.id === st.sel ? ' sel' : '') + (busy ? ' busy' : ''),
        role: 'option', 'aria-selected': p.id === st.sel ? 'true' : 'false',
        onclick: () => { st.sel = p.id; render(); },
        ondblclick: () => choose(p.id),
      },
      h('span', { class: 'pp-av' }, initial(p.name)),
      h('span', { class: 'pp-main' },
        h('b', {}, p.name, p.id === me.profileId ? h('span', { class: 'tag' }, '当前') : null,
          busy ? h('span', { class: 'tag warn' }, '使用中') : null),
        h('span', { class: 'hint' }, profileLine(p))),
      h('span', { class: 'pp-when hint' }, when(p.last_played)));
      list.append(row);
    }
    list.append(h('button', { class: 'pp-row pp-new', onclick: create }, h('span', { class: 'pp-av' }, '+'),
      h('span', { class: 'pp-main' }, h('b', {}, '新建用户'))));
    const selected = st.list.find((p) => p.id === st.sel);
    const canAct = Boolean(selected);
    body.append(list,
      h('div', { class: 'pp-actions' },
        h('div', { class: 'row' },
          h('button', { class: 'btn sm', disabled: !canAct, onclick: () => selected && showProfileStats(selected.id) }, '战绩'),
          h('button', { class: 'btn sm', disabled: !canAct, onclick: () => selected && rename(selected) }, '改名'),
          h('button', { class: 'btn sm danger', disabled: !canAct || Boolean(selected && st.inUse[selected.id]), onclick: () => selected && remove(selected) }, '删除')),
        h('div', { class: 'row' },
          h('button', { class: 'btn ghost', onclick: playAsGuest, title: '不保存战绩' }, me.guest ? '继续当游客' : '游客模式'),
          st.force ? null : h('button', { class: 'btn ghost', onclick: close }, '取消'),
          h('button', { class: 'btn primary', disabled: !canAct || st.pending, onclick: () => choose(st.sel) },
            selected && selected.id === me.profileId ? '继续' : '就是我'))));
  }

  function create() {
    askName('新建用户', st.list && st.list.length ? '' : (me.name || ''), (name) => {
      st.pending = true;
      send({ op: 'profile_create', name });
    });
  }

  function rename(p) {
    askName(`给「${p.name}」改名`, p.name, (name) => send({ op: 'profile_rename', id: p.id, name }));
  }

  function remove(p) {
    const s = p.stats.battle;
    const c = modal(h('div', {},
      h('h2', {}, `删除「${p.name}」？`),
      h('div', { style: { margin: '8px 0 14px' } },
        s.hands ? `这个用户的 ${s.hands} 局对战记录（累计 ${signed(s.score)}）会一起删除，无法恢复。` : '删除后无法恢复。'),
      h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
        h('button', { class: 'btn ghost', onclick: () => c.close() }, '取消'),
        h('button', { class: 'btn danger', onclick: () => { c.close(); send({ op: 'profile_delete', id: p.id }); } }, '删除'))));
  }

  offs.push(on('profiles', (msg) => {
    st.list = msg.profiles || [];
    st.inUse = msg.in_use || {};
    if (!st.list.some((p) => p.id === st.sel)) {
      const free = st.list.find((p) => !st.inUse[p.id]);
      st.sel = me.profileId || (free ? free.id : null);
    }
    render();
  }));
  offs.push(on('profile', (msg) => {
    if (msg.guest && st.pending) {
      st.pending = false;
      toast(`游客 ${msg.name}，祝你好运！`);
      close();
      return;
    }
    if (msg.profile && st.pending) {
      st.pending = false;
      toast(`你好，${msg.profile.name}！`);
      close();
      if (onDone) onDone(msg.profile);
      return;
    }
    if (!msg.profile) requireChoice();
    render();
  }));
  offs.push(on('error', () => { if (st.pending) { st.pending = false; render(); } }));
  render();
  send({ op: 'profile_list' });
}

/** Open the picker whenever the server says this device has no active user. */
export function ensureProfile(onDone) {
  if (me.welcomed && !me.profile && !me.guest) openProfilePicker({ force: true, onDone });
}

export function showProfileStats(id) {
  const body = h('div', { class: 'hint', style: { padding: '18px 0', textAlign: 'center' } }, '正在读取战绩…');
  const m = modal(h('div', { class: 'profile-stats' }, body,
    h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } },
      h('button', { class: 'btn primary', onclick: () => m.close() }, '关闭'))));
  const offErr = on('error', () => { offErr(); off(); m.close(); });
  const off = on('profile_detail', (msg) => {
    off();
    offErr();
    const p = msg.profile;
    const box = h('div', {});
    box.append(h('h2', {}, h('span', { class: 'pp-av' }, initial(p.name)), ` ${p.name} 的战绩`));
    for (const mode of ['battle', 'teach']) {
      const s = p.stats[mode];
      if (mode === 'teach' && !s.hands) continue;
      const tiles = [
        ['局数', s.hands],
        ['胜率', pctOf(s.wins, s.hands)],
        ['累计得分', signed(s.score)],
        ['平均每局', s.hands ? signed(+(s.score / s.hands).toFixed(2)) : '—'],
        ['做庄', s.banker_hands ? `${s.banker_hands} 次 · 胜 ${pctOf(s.banker_wins, s.banker_hands)}` : '—'],
        ['单局最高', s.best === null ? '—' : signed(s.best)],
        ['单局最低', s.worst === null ? '—' : signed(s.worst)],
        ['最长连胜', s.best_streak || 0],
      ];
      box.append(h('div', { class: 'ps-mode' }, `${MODE_NAMES[mode]}模式`),
        h('div', { class: 'ps-grid' }, tiles.map(([k, v]) => h('div', { class: 'ps-tile' }, h('span', { class: 'hint' }, k), h('b', {}, String(v))))));
    }
    if (!p.stats.battle.hands && !p.stats.teach.hands) {
      box.append(h('div', { class: 'pp-empty' }, '还没有记录，打完一局就会出现在这里。'));
    }
    if (p.recent && p.recent.length) {
      box.append(h('div', { class: 'ps-mode' }, '最近对局'),
        h('div', { class: 'ps-recent' },
          h('div', { class: 'ps-rrow hdr' }, h('span', {}, '时间'), h('span', {}, '身份'), h('span', {}, '结果'), h('span', {}, '得分')),
          p.recent.map((r) => h('div', { class: 'ps-rrow' },
            h('span', {}, when(r.t), r.mode === 'teach' ? h('span', { class: 'tag' }, '教学') : null),
            h('span', {}, r.banker ? `庄 · ${r.contract}分` : `闲 · 庄${r.contract}分`),
            h('span', {}, `${r.tier} · 闲${r.xian_points}`),
            h('b', { class: r.score >= 0 ? 'pos' : 'neg' }, signed(r.score))))));
    }
    box.append(h('div', { class: 'hint', style: { marginTop: '8px' } }, `建档于 ${when(p.created)}。分数单位为底注。`));
    body.replaceWith(box);
  });
  send({ op: 'profile_get', id });
  return m;
}

export function profileModalLocked() {
  const top = $('#modal-root .modal-back:last-child');
  return Boolean(top && top.dataset.locked);
}
