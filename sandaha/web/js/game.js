// Lobby + game table.
import { $, api, fmt, h, modal, pct, store, toast } from './util.js';
import { cardEl, chipEl, label, layoutHand, SUIT_SYM } from './cards.js';
import { me, on, send } from './net.js';
import { goHome, showRules } from './main.js';

const TRUMP_NAMES = ['♠ 黑桃', '♥ 红桃', '♣ 梅花', '♦ 方块', '无主', '投降'];
const PHASE_NAMES = { bid: '叫分', trump: '定主', bury: '埋底', play: '出牌', over: '结束' };
const KIND_NAMES = ['单张', '对子', '拖拉机', '甩牌'];
const REL_CLASS = ['me', 'right', 'top', 'left'];

export function mountPlay(app) {
  const st = {
    view: null, rooms: [], models: [], defaultModel: null, timeOffset: 0,
    sel: new Set(), handSig: '', panel: store.get('sdh.panel', null), reveal: store.get('sdh.reveal', true),
    shownResult: null, seenPlays: new Set(), lastHint: null,
    lobby: store.get('sdh.lobby', { mode: 'battle', seats: ['me', 'ai', 'ai', 'ai'], models: [null, null, null, null], ai_delay: 0.9, rules: {} }),
  };
  const offs = [];
  let alive = true;
  let timerInt = null;
  let resizeT = null;
  let resultT = null;

  offs.push(on('rooms', (m) => {
    if (!alive) return;
    st.rooms = m.rooms;
    if (!st.view || !st.view.room) render();
  }));
  offs.push(on('room_closed', (m) => {
    if (!alive) return;
    if (!m.room || (st.view && st.view.room && st.view.room.id === m.room)) {
      st.view = null;
      st.sel.clear();
      const mr = $('#modal-root');
      if (mr) mr.innerHTML = '';
      toast(m.msg || '牌局已结束');
      render();
    }
  }));
  offs.push(on('state', (m) => {
    if (!alive) return;
    st.timeOffset = m.now - Date.now() / 1000;
    const prevPhase = st.view && st.view.phase;
    st.view = m;
    const sig = (m.hand || []).join(',') + '|' + m.phase + '|' + (m.hand_no || 0);
    if (sig !== st.handSig) { st.sel.clear(); st.handSig = sig; }
    render();
    if (m.phase === 'over' && m.result && st.shownResult !== m.hand_no) {
      st.shownResult = m.hand_no;
      clearTimeout(resultT);
      resultT = setTimeout(() => { if (alive) showResult(); }, prevPhase === 'over' ? 0 : 900);
    }
  }));
  offs.push(on('hint', (m) => { if (alive) showHint(m.comment); }));
  offs.push(on('welcome', () => { if (alive) send({ op: 'sync' }); }));
  offs.push(on('close', () => { /* reconnect handled by net */ }));

  api('/api/models').then((r) => {
    if (!alive) return;
    st.models = r.models || [];
    st.defaultModel = r.default;
    if (!st.view || !st.view.room) render();
  }).catch(() => {});
  send({ op: 'sync' });

  const onResize = () => {
    if (!alive) return;
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { if (alive) render(); }, 100);
  };
  window.addEventListener('resize', onResize);
  timerInt = setInterval(tick, 250);

  function canManageRoomSummary(r) {
    if (r.host_id && r.host_id === me.id) return true;
    const isMine = r.seats.some((s) => s.kind === 'human' && s.client_id === me.id);
    const otherOnline = r.seats.some((s) => s.kind === 'human' && s.client_id !== me.id && s.connected);
    if (isMine && !otherOnline) return true;
    const anyHumanOnline = r.seats.some((s) => s.kind === 'human' && s.connected);
    return !anyHumanOnline;
  }

  function promptLeaveRoom() {
    if (!alive) return;
    const v = st.view;
    if (!v || !v.room) return;
    const canManage = Boolean(v.can_manage || v.is_host);
    if (canManage) {
      const m = modal(h('div', {},
        h('h2', {}, '退出或结束牌局'),
        h('div', { class: 'hint', style: { margin: '6px 0 14px' } },
          '选择「结束牌局」将直接解散并清理当前房间；选择「暂时离开」会保留房间，稍后可在大厅点击「回到牌局」继续。'),
        h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
          h('button', { class: 'btn ghost', onclick: () => m.close() }, '继续对局'),
          h('button', { class: 'btn', onclick: () => {
            m.close();
            send({ op: 'leave_room' });
            st.view = null;
            render();
          } }, '暂时离开'),
          h('button', { class: 'btn danger', onclick: () => {
            m.close();
            send({ op: 'close_room', room: v.room.id });
            st.view = null;
            render();
          } }, '结束牌局'))));
      return;
    }
    if (v.phase && v.phase !== 'over' && v.my_seat !== null && v.my_seat !== undefined) {
      const m = modal(h('div', {},
        h('h2', {}, '离开房间'),
        h('div', { style: { margin: '8px 0 14px' } }, '对局正在进行中，确定离开当前房间？'),
        h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
          h('button', { class: 'btn ghost', onclick: () => m.close() }, '取消'),
          h('button', { class: 'btn danger', onclick: () => {
            m.close();
            send({ op: 'leave_room' });
            st.view = null;
            render();
          } }, '确定离开'))));
      return;
    }
    send({ op: 'leave_room' });
    st.view = null;
    render();
  }

  function showManageModal() {
    if (!alive) return;
    const v = st.view;
    if (!v || !v.room) return;
    const curDelay = v.ai_delay ?? st.lobby.ai_delay ?? 0.9;
    const delaySlider = h('input', { type: 'range', min: 0, max: 3, step: 0.1, value: curDelay, style: { flex: '1' } });
    const delayTxt = h('span', { class: 'hint', style: { minWidth: '34px', textAlign: 'right' } }, `${fmt(curDelay, 1)}s`);
    delaySlider.addEventListener('input', () => { delayTxt.textContent = `${fmt(+delaySlider.value, 1)}s`; });
    delaySlider.addEventListener('change', () => {
      st.lobby.ai_delay = +delaySlider.value;
      store.set('sdh.lobby', st.lobby);
      send({ op: 'settings', ai_delay: +delaySlider.value });
    });

    const btnGrid = h('div', { class: 'manage-grid' });
    let mRef = null;
    if (v.phase && v.phase !== 'over') {
      btnGrid.append(h('button', { class: 'btn primary', onclick: () => { mRef.close(); send({ op: 'restart_hand' }); } }, '重开本局（重新发牌）'));
    }
    if (v.phase) {
      btnGrid.append(h('button', { class: 'btn', onclick: () => { mRef.close(); send({ op: 'abort_hand' }); } }, '结束本局（调整座位 / AI）'));
    }
    const hasScore = (v.hand_no > 0) || (v.totals && v.totals.some((x) => x !== 0));
    if (hasScore) {
      btnGrid.append(h('button', { class: 'btn', onclick: () => { mRef.close(); send({ op: 'reset_scores' }); toast('累计总分已清零'); } }, '清零累计总分'));
    }
    btnGrid.append(h('button', { class: 'btn danger', onclick: () => {
      mRef.close();
      send({ op: 'close_room', room: v.room.id });
      st.view = null;
      render();
    } }, '结束并解散牌局'));

    mRef = modal(h('div', {},
      h('h2', {}, '牌局管理'),
      h('div', { class: 'row', style: { margin: '8px 0 12px', flexWrap: 'nowrap' } },
        h('span', { class: 'hint' }, 'AI 出牌间隔'), delaySlider, delayTxt),
      btnGrid,
      h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } },
        h('button', { class: 'btn ghost', onclick: () => mRef.close() }, '关闭'))));
  }

  function showSeatConfigModal(seatIdx) {
    const v = st.view;
    if (!v || !v.room) return;
    const cur = v.room.seats[seatIdx];
    const msel = modelOptions(cur && cur.model ? cur.model : st.defaultModel);
    let mRef = null;
    mRef = modal(h('div', {},
      h('h2', {}, `设置 ${seatIdx + 1} 号位`),
      h('div', { class: 'manage-grid', style: { marginTop: '10px' } },
        h('button', { class: 'btn primary', onclick: () => { mRef.close(); send({ op: 'set_seat', seat: seatIdx, kind: 'me' }); } }, '我坐这里（本机）'),
        h('div', { class: 'row', style: { flexWrap: 'nowrap' } },
          msel,
          h('button', { class: 'btn', onclick: () => { mRef.close(); send({ op: 'set_seat', seat: seatIdx, kind: 'ai', model: msel.value }); } }, '设为该 AI')),
        v.mode === 'battle' ? h('button', { class: 'btn', onclick: () => { mRef.close(); send({ op: 'set_seat', seat: seatIdx, kind: 'open' }); } }, '设为空位（等待玩家加入）') : null),
      h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } },
        h('button', { class: 'btn ghost', onclick: () => mRef.close() }, '取消'))));
  }

  window.__sdhPlayBack = () => {
    if (!alive) return false;
    if (st.panel) {
      st.panel = null;
      store.set('sdh.panel', null);
      render();
      return true;
    }
    if (st.view && st.view.room) {
      promptLeaveRoom();
      return true;
    }
    return false;
  };

  function render() {
    if (!alive) return;
    app.innerHTML = '';
    if (st.view && st.view.room) renderTable(); else renderLobby();
  }

  // ======================================================== lobby =====
  function resolveModelSpec(spec) {
    if (!spec) return st.defaultModel || null;
    if (spec === 'rule' || spec === 'random') return spec;
    if (!st.models.length) return spec;
    const raw = spec.endsWith('?search') ? spec.slice(0, -7) : spec;
    if (st.models.some((m) => m.path === raw)) return spec;
    return st.defaultModel || 'rule';
  }

  function modelOptions(selected) {
    const fmtLabel = (m) => {
      if (m.label === 'sandaha-best') return 'sandaha-best（最强主力）';
      if (m.label === 'sandaha-lite') return 'sandaha-lite（轻量快速）';
      return m.label;
    };
    const opts = [];
    for (const m of st.models.filter((x) => x.kind !== 'checkpoint')) {
      opts.push(h('option', { value: m.path }, fmtLabel(m)));
      opts.push(h('option', { value: m.path + '?search' }, `${m.label} + 搜索增强`));
    }
    opts.push(h('option', { value: 'rule' }, '规则AI (基准)'), h('option', { value: 'random' }, '随机AI'));
    const ck = st.models.filter((x) => x.kind === 'checkpoint').slice(-12);
    if (ck.length) opts.push(h('optgroup', { label: '训练检查点' }, ck.map((m) => h('option', { value: m.path }, m.label))));
    const sel = h('select', {}, opts);
    sel.value = resolveModelSpec(selected) || st.defaultModel || 'rule';
    if (!sel.value) sel.value = (sel.options[0] && sel.options[0].value) || 'rule';
    return sel;
  }

  function renderLobby() {
    const L = st.lobby;
    const save = () => store.set('sdh.lobby', L);
    const modeSeg = h('div', { class: 'seg' },
      h('button', { class: L.mode === 'battle' ? 'on' : '', onclick: () => { L.mode = 'battle'; save(); render(); } }, '对战模式'),
      h('button', { class: L.mode === 'teach' ? 'on' : '', onclick: () => { L.mode = 'teach'; save(); render(); } }, '教学模式'));
    const seatRows = [];
    const dirNames = ['1号位', '2号位', '3号位', '4号位'];
    if (L.mode === 'battle') {
      for (let i = 0; i < 4; i++) {
        const kind = h('select', {},
          h('option', { value: 'me' }, '我（本机）'),
          h('option', { value: 'human' }, '等待玩家加入'),
          h('option', { value: 'ai' }, 'AI'));
        kind.value = L.seats[i];
        kind.addEventListener('change', () => {
          if (kind.value === 'me') L.seats = L.seats.map((x, j) => (j !== i && x === 'me' ? 'human' : x));
          L.seats[i] = kind.value; save(); render();
        });
        if (L.seats[i] === 'ai') {
          const msel = modelOptions(L.models[i]);
          msel.addEventListener('change', () => { L.models[i] = msel.value; save(); });
          seatRows.push(h('div', { class: 'seat-cfg' }, h('span', {}, dirNames[i]), kind, msel));
        } else {
          seatRows.push(h('div', { class: 'seat-cfg human-seat' }, h('span', {}, dirNames[i]), kind));
        }
      }
      seatRows.push(h('div', { class: 'hint' }, '出牌顺序为 1→2→3→4（逆时针）。每步限时 30 秒，超时系统随机出牌。座位都设为 AI 即可观战。'));
    } else {
      const msel = modelOptions(L.models[1]);
      msel.addEventListener('change', () => { L.models = [null, msel.value, msel.value, msel.value]; save(); });
      seatRows.push(h('div', { class: 'form' }, h('label', {}, '对手 / 教练 AI'), msel));
      seatRows.push(h('div', { class: 'hint', style: { marginTop: '6px' } }, '教学模式：你与三个 AI 对局，不限时；可查看所有 AI 手牌，每步均有期望得分与胜率点评，并可随时请求提示。'));
    }
    const delay = h('input', { type: 'range', min: 0, max: 3, step: 0.1, value: L.ai_delay, style: { flex: '1', minWidth: '100px' } });
    const delayLbl = h('span', { class: 'hint' }, `${L.ai_delay}s`);
    delay.addEventListener('input', () => { L.ai_delay = +delay.value; delayLbl.textContent = `${L.ai_delay}s`; save(); });
    const R = L.rules;
    const rnum = (k, d, min, max, step = 5) => {
      const i = h('input', { type: 'number', value: R[k] ?? d, min, max, step, style: { width: '90px' } });
      i.addEventListener('change', () => { R[k] = +i.value; save(); });
      return i;
    };
    const rchk = (k, d, text) => {
      const i = h('input', { type: 'checkbox' });
      i.checked = R[k] ?? d;
      i.addEventListener('change', () => { R[k] = i.checked; save(); });
      return h('label', { class: 'row', style: { gap: '6px', color: 'var(--text)' } }, i, text);
    };
    const adv = h('details', {}, h('summary', {}, '高级规则'),
      h('div', { class: 'form' },
        h('label', {}, '起叫分'), rnum('max_bid', 80, 50, 100),
        h('label', {}, '常主强制叫分'), rchk('force_enabled', true, '持 7 张以上常主须叫到 70 分'),
        h('label', {}, '庄家选项'), h('div', { class: 'row' }, rchk('allow_no_trump', true, '允许无主'), rchk('allow_surrender', true, '允许投降')),
        h('label', {}, '主牌甩牌'), rchk('allow_trump_throw', true, '允许甩主牌（副牌永远不可甩）'),
        h('label', {}, '单张扣底倍数'), rnum('koudi_single', 1, 1, 4, 1),
        h('label', {}, '对子扣底倍数'), rnum('koudi_pair', 2, 1, 8, 1),
        h('label', {}, '双进单出'), rchk('shuang_jin_dan_chu', true, '庄家获胜倍数翻倍（默认开启）'),
        h('div', { class: 'full hint' }, '注：AI 模型以默认规则训练，修改规则后 AI 仍可正常出牌，但强度可能下降。')));
    const createBtn = h('button', { class: 'btn primary', onclick: () => {
      const seats = [];
      for (let i = 0; i < 4; i++) {
        if (L.mode === 'teach') seats.push(i === 0 ? { kind: 'me' } : { kind: 'ai', model: resolveModelSpec(L.models[1]) });
        else if (L.seats[i] === 'ai') seats.push({ kind: 'ai', model: resolveModelSpec(L.models[i]) });
        else seats.push({ kind: L.seats[i] });
      }
      send({ op: 'create_room', mode: L.mode, seats, ai_delay: L.ai_delay, rules: L.rules,
        coach: resolveModelSpec(L.models[1]), name: `${me.name} 的房间` });
    } }, L.mode === 'teach' ? '开始教学局' : '创建房间');

    const roomList = h('div', { class: 'list' });
    if (!st.rooms.length) roomList.append(h('div', { class: 'empty' }, '暂无进行中的牌局。可在左侧新建对局，或让局域网设备扫码加入。'));
    for (const r of st.rooms) {
      const open = r.seats.filter((s) => s.kind === 'open').length;
      const isMine = r.seats.some((s) => s.kind === 'human' && s.client_id === me.id);
      const canManage = canManageRoomSummary(r);
      const joinLabel = isMine ? '回到牌局' : (open ? '加入' : '观战');
      const joinCls = (isMine || open) ? 'btn sm primary' : 'btn sm';
      roomList.append(h('div', { class: 'room-item' },
        h('div', { class: 'meta' }, h('b', {}, r.name), ' ',
          isMine ? h('span', { class: 'tag gold' }, '我的牌局') : null, isMine ? ' ' : null,
          h('span', { class: 'tag ' + (r.mode === 'teach' ? 'gold' : 'green') }, r.mode === 'teach' ? '教学' : '对战'), ' ',
          h('span', { class: 'tag' }, r.status === 'waiting' ? `等待中 · 空位 ${open}` : `第 ${r.hand_no} 局`),
          h('div', { class: 'hint' }, r.seats.map((s) => s.kind === 'open' ? '空位' : s.name).join(' · '))),
        h('div', { class: 'row', style: { flexWrap: 'nowrap', gap: '6px' } },
          h('button', { class: joinCls, onclick: () => send({ op: 'join_room', room: r.id }) }, joinLabel),
          canManage ? h('button', { class: 'btn sm danger', onclick: () => send({ op: 'close_room', room: r.id }) }, '结束') : null)));
    }
    const closableCount = st.rooms.filter(canManageRoomSummary).length;
    app.append(h('div', { class: 'page lobby-page' },
      h('div', { class: 'pagebar' }, h('button', { class: 'btn sm ghost', onclick: () => goHome() }, '← 首页'),
        h('h2', {}, '对弈大厅'), h('span', { class: 'hint' }, `昵称：${me.name}`),
        h('button', { class: 'btn sm ghost', onclick: showRules }, '规则')),
      h('div', { class: 'grid2 lobby-grid' },
        h('div', { class: 'card-box' }, h('h3', {}, '新建对局'), h('div', { class: 'row lobby-top-row', style: { marginBottom: '10px' } }, modeSeg, createBtn),
          ...seatRows,
          h('div', { class: 'form lobby-delay-row', style: { marginTop: '8px' } }, h('label', {}, 'AI 出牌间隔'), h('div', { class: 'row', style: { flexWrap: 'nowrap' } }, delay, delayLbl)),
          adv),
        h('div', { class: 'card-box' },
          h('h3', {},
            h('span', { style: { marginRight: 'auto' } }, '牌局列表'),
            h('button', { class: 'btn sm ghost', onclick: () => send({ op: 'list_rooms' }) }, '刷新'),
            closableCount > 1 ? h('button', { class: 'btn sm danger', onclick: () => {
              const m = modal(h('div', {},
                h('h2', {}, '全部结束'),
                h('div', { style: { margin: '8px 0 14px' } }, `确定结束并清理当前 ${closableCount} 个牌局房间？`),
                h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
                  h('button', { class: 'btn ghost', onclick: () => m.close() }, '取消'),
                  h('button', { class: 'btn danger', onclick: () => { m.close(); send({ op: 'close_all_rooms' }); } }, '全部结束'))));
            } }, '全部结束') : null),
          roomList))));
  }

  // ======================================================== table =====
  function seatName(i) {
    const s = st.view.room.seats[i];
    if (!s) return '';
    if (s.kind === 'open') return '空位';
    return s.name || (s.kind === 'ai' ? 'AI' : '玩家');
  }
  function viewer() { return st.view.my_seat ?? 0; }
  function rel(i) { return (i - viewer() + 4) % 4; }

  function lastBid(i) {
    const bids = st.view.bids || [];
    for (let k = bids.length - 1; k >= 0; k--) if (bids[k].seat === i) return bids[k].value;
    return undefined;
  }

  function seatBox(i, cls) {
    const v = st.view;
    const s = v.room.seats[i];
    const isAI = s.kind === 'ai';
    const active = v.phase && v.phase !== 'over' && v.to_act === i;
    const av = h('div', { class: 'avatar' + (isAI ? ' ai' : '') + (active ? ' active' : '') },
      s.kind === 'open' ? '?' : (isAI ? 'AI' : (seatName(i)[0] || '玩')),
      v.banker === i ? h('span', { class: 'badge' }, '庄') : null);
    const sub = h('div', { class: 'sub' });
    if (v.phase) sub.append(h('span', {}, `剩${v.hand_counts[i]}张`));
    if (s.kind === 'human' && s.connected === false) sub.append(h('span', { class: 'tag red' }, '离线'));
    if (v.totals) sub.append(h('span', { class: v.totals[i] >= 0 ? 'pos' : 'neg' }, `${v.totals[i] >= 0 ? '+' : ''}${v.totals[i]}`));
    if (v.phase === 'bid') {
      const b = lastBid(i);
      if (b !== undefined) sub.append(h('span', { class: 'bubble' + (b === 0 ? ' pass' : '') }, b === 0 ? '不叫' : `${b}分`));
    }
    const voidSuits = [];
    if (v.void && v.phase === 'play' && i !== v.my_seat) {
      v.void[i].forEach((x, c) => { if (x) voidSuits.push(c === 4 ? '主' : SUIT_SYM['SHCD'[c]]); });
    }
    if (voidSuits.length) sub.append(h('span', { class: 'tag', title: '已知缺门' }, '缺' + voidSuits.join('')));
    const timer = active && v.deadline ? h('div', { class: 'timer', 'data-deadline': v.deadline }, h('i', {})) : null;
    const box = h('div', { class: `seat ${cls}` }, av, h('div', { class: 'name', title: seatName(i) }, seatName(i)), sub, timer);
    if (!v.phase) {
      if (v.can_manage || v.is_host) {
        box.append(h('button', { class: 'btn sm', onclick: () => showSeatConfigModal(i) }, s.kind === 'open' ? '安排座位' : '调整'));
      } else if (s.kind === 'open') {
        box.append(h('button', { class: 'btn sm', onclick: () => send({ op: 'take_seat', seat: i }) }, '坐这里'));
      }
    }
    if (v.all_hands && st.reveal && i !== v.my_seat) {
      const mh = h('div', { class: 'mini-hand chips' }, v.all_hands[i].map((c) => chipEl(c, v.trump)));
      box.append(mh);
    }
    return box;
  }

  function fitStack(slot, codes, trump, cls, animate) {
    const el = h('div', { class: 'stack ' + (cls || '') });
    slot.append(el);
    if (!codes || !codes.length) return el;
    // Measure real rendered mini-card size (CSS --tcw may use clamp() which parseFloat cannot parse)
    const probe = cardEl(codes[0], { trump, mini: true });
    probe.style.position = 'absolute';
    probe.style.visibility = 'hidden';
    el.append(probe);
    const pr = probe.getBoundingClientRect();
    const cw = pr.width || 40;
    const ch = pr.height || 56;
    probe.remove();

    const avail = Math.max(cw * 2, slot.clientWidth - 4);
    const n = codes.length;
    // .card.mini .idx occupies left ~16px; keep >= 19px step so rank, suit & 主 are 100% uncovered
    const minS = Math.max(19, Math.ceil(cw * 0.54));
    const maxS = Math.max(minS + 3, Math.round(cw * 0.70));
    let step = n > 1 ? Math.min(maxS, (avail - cw) / (n - 1)) : 0;
    let perRow = n;
    if (n > 1 && step < minS) {
      step = minS;
      perRow = Math.max(2, Math.floor((avail - cw) / step) + 1);
    }
    el.style.flexDirection = 'column';
    el.style.alignItems = 'center';
    const rowOverlap = Math.max(0, Math.min(Math.round(ch * 0.36), Math.round(ch - 32)));
    for (let r = 0; r * perRow < n; r++) {
      const row = h('div', { style: { display: 'flex', marginTop: r ? `-${rowOverlap}px` : '0' } });
      codes.slice(r * perRow, (r + 1) * perRow).forEach((c, k) => {
        const ce = cardEl(c, { trump, mini: true });
        if (k) ce.style.marginLeft = `${Math.round(step - cw)}px`;
        row.append(ce);
      });
      el.append(row);
    }
    if (animate) el.classList.add('fly-' + animate);
    return el;
  }

  function showBuriedModal() {
    const v = st.view;
    if (!v || !v.buried) return;
    const pts = v.buried.reduce((a, c) => a + cardPts(c), 0);
    const m = modal(h('div', {},
      h('h2', {}, '已埋底牌'),
      h('div', { class: 'row', style: { justifyContent: 'center', gap: '6px', margin: '10px 0' } },
        v.buried.map((c) => cardEl(c, { trump: v.trump, mini: true }))),
      h('div', { class: 'hint', style: { textAlign: 'center' } }, `共 8 张 · 底牌分 ${pts} 分`),
      h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } },
        h('button', { class: 'btn primary', onclick: () => m.close() }, '关闭'))));
  }

  function renderTable() {
    const v = st.view;
    const teach = v.mode === 'teach';
    const compactScreen = (window.innerWidth || 1024) <= 900 || (window.innerHeight || 768) <= 540;
    const wrap = h('div', { class: 'table-wrap' + (teach && st.reveal ? ' reveal' : '') });

    // ---- top bar
    const info = h('div', { class: 'info' });
    info.append(h('span', { class: 'pill' }, teach ? '教学' : '对战', v.hand_no ? `·${v.hand_no}局` : ''));
    if (v.phase) info.append(h('span', { class: 'pill' }, PHASE_NAMES[v.phase] || v.phase));
    if (v.banker !== null && v.banker !== undefined) info.append(h('span', { class: 'pill' }, '庄 ', h('b', {}, seatName(v.banker))));
    if (v.contract) info.append(h('span', { class: 'pill' }, '庄分 ', h('b', {}, v.contract)));
    else if (v.best_bid) info.append(h('span', { class: 'pill' }, '当前 ', h('b', {}, v.best_bid)));
    if (v.trump_label) info.append(h('span', { class: 'pill' }, '主 ', h('b', {}, v.trump_label)));
    if (v.phase === 'play' || v.phase === 'over') info.append(h('span', { class: 'pill' }, '闲 ', h('b', {}, `${v.xian_points}`), v.contract ? `/${v.contract}` : ''));

    const tbtn = (key, text) => h('button', { class: 'btn sm' + (st.panel === key ? ' on' : ''), onclick: () => togglePanel(key) }, text);
    const tools = h('div', { class: 'tools' });
    if (v.buried && (v.phase === 'play' || v.phase === 'over')) {
      tools.append(h('button', { class: 'btn sm', onclick: showBuriedModal }, '底牌'));
    }
    tools.append(tbtn('counter', '记牌器'), tbtn('history', '历史'));
    if (teach) {
      tools.append(tbtn('comments', '点评'));
      tools.append(h('button', {
        class: 'btn sm' + ((compactScreen ? st.panel === 'hands' : st.reveal) ? ' on' : ''),
        onclick: () => {
          if (compactScreen) {
            togglePanel('hands');
          } else {
            st.reveal = !st.reveal;
            store.set('sdh.reveal', st.reveal);
            render();
          }
        },
      }, 'AI手牌'));
    }
    tools.append(h('button', { class: 'btn sm ghost', onclick: showRules }, '规则'));
    if (v.can_manage || v.is_host) {
      tools.append(h('button', { class: 'btn sm', onclick: showManageModal }, '管理'));
    }
    tools.append(h('button', { class: 'btn sm ghost', onclick: promptLeaveRoom }, '离开'));
    wrap.append(h('div', { class: 'topbar' }, info, tools));

    // ---- felt
    const felt = h('div', { class: 'felt' });
    const seatsByRel = [0, 1, 2, 3].map((r) => (viewer() + r) % 4);
    felt.append(seatBox(seatsByRel[2], 'top'), seatBox(seatsByRel[3], 'left'), seatBox(seatsByRel[1], 'right'));

    const center = h('div', { class: 'center' });
    const slots = {};
    for (const [r, c] of [[2, 'p-top'], [3, 'p-left'], [1, 'p-right'], [0, 'p-me']]) {
      slots[r] = h('div', { class: `pslot ${c}` });
      center.append(slots[r]);
    }
    center.append(midInfo());
    felt.append(center);

    // ---- unified bottom control bar + hand
    const meStatus = h('div', { class: 'me-status' });
    const ms = v.my_seat;
    if (ms !== null && ms !== undefined) {
      const active = v.phase && v.phase !== 'over' && v.to_act === ms;
      meStatus.append(h('div', { class: 'avatar me-av' + (active ? ' active' : '') },
        seatName(ms)[0] || '我', v.banker === ms ? h('span', { class: 'badge' }, '庄') : null));
      meStatus.append(h('b', { class: 'me-name' }, seatName(ms)));
      if (v.totals) meStatus.append(h('span', { class: (v.totals[ms] >= 0 ? 'pos' : 'neg') + ' me-score' }, `${v.totals[ms] >= 0 ? '+' : ''}${v.totals[ms]}`));
      if (v.phase === 'bid') {
        const b = lastBid(ms);
        if (b !== undefined) meStatus.append(h('span', { class: 'bubble' + (b === 0 ? ' pass' : '') }, b === 0 ? '不叫' : `${b}分`));
      }
      if (active && v.deadline) {
        meStatus.append(
          h('div', { class: 'timer me-timer', 'data-deadline': v.deadline }, h('i', {})),
          h('span', { class: 'hint me-cd', 'data-countdown': v.deadline }, '')
        );
      }
      if (v.buried && v.phase === 'play') {
        meStatus.append(h('span', { class: 'buried-inline' },
          h('span', { class: 'hint' }, '底牌：'),
          h('span', { class: 'chips' }, v.buried.map((c) => chipEl(c, v.trump)))));
      }
    } else {
      meStatus.append(h('span', { class: 'hint' }, '观战中'));
    }

    const actions = h('div', { class: 'actions' });
    const controlBar = h('div', { class: 'control-bar' }, meStatus, actions);
    const hand = h('div', { class: 'hand' });
    const handArea = h('div', { class: 'hand-area' }, controlBar, hand);
    felt.append(handArea);

    const main = h('div', { class: 'main-area' }, felt);
    if (st.panel) {
      main.append(h('div', { class: 'side-backdrop', onclick: () => togglePanel(st.panel) }));
      main.append(sidePanel());
    }
    wrap.append(main);
    app.append(wrap);

    // fill trick slots & hand (after DOM mount so clientWidth is exact)
    fillTrick(slots);
    renderActions(actions);
    if (v.hand) {
      const canSelect = v.my_seat === v.to_act && (v.phase === 'play' || v.phase === 'bury');
      layoutHand(hand, v.hand, { trump: v.trump, selected: st.sel, disabled: !canSelect,
        onToggle: (key, code, el) => {
          if (st.sel.has(key)) { st.sel.delete(key); el.classList.remove('sel'); } else { st.sel.add(key); el.classList.add('sel'); }
          updateSelInfo();
        } });
    }
    tick();
  }

  function midInfo() {
    const v = st.view;
    const mid = h('div', { class: 'mid' });
    if (!v.phase) {
      const open = v.room.seats.filter((s) => s.kind === 'open').length;
      mid.append(h('div', { class: 'big' }, open ? `等待玩家 (${open} 空位)` : '准备开始'));
      if ((v.can_manage || v.is_host) && !open) {
        mid.append(h('button', { class: 'btn primary', onclick: () => send({ op: 'start' }) }, '开始发牌'));
      } else if (v.can_manage || v.is_host) {
        mid.append(h('div', { class: 'mid-sub' }, '点击各座位「安排座位」可设为 AI'));
      }
      return mid;
    }
    if (v.phase === 'bid') {
      mid.append(h('div', { class: 'big' }, v.best_bid ? `当前 ${v.best_bid} 分` : '叫分'));
      mid.append(h('div', { class: 'mid-sub' }, `${seatName(v.to_act)} 叫分中`));
    } else if (v.phase === 'trump') {
      mid.append(h('div', { class: 'big' }, `庄分 ${v.contract}`), h('div', { class: 'mid-sub' }, `${seatName(v.banker)} 定主中`));
    } else if (v.phase === 'bury') {
      mid.append(h('div', { class: 'big' }, v.trump_label), h('div', { class: 'mid-sub' }, `${seatName(v.banker)} 埋底中`));
    } else if (v.phase === 'play') {
      const n = (v.tricks || []).length + (v.cur.length ? 1 : 0);
      const pts = v.cur.reduce((a, p) => a + p.cards.reduce((b, c) => b + cardPts(c), 0), 0);
      mid.append(h('div', { class: 'big' }, `第 ${Math.max(1, n)} 轮`));
      if (v.cur.length) mid.append(h('div', { class: 'mid-sub' }, `本轮 ${pts} 分`));
      else if (v.tricks.length) {
        const t = v.tricks[v.tricks.length - 1];
        mid.append(h('div', { class: 'mid-sub' }, `${seatName(t.winner)} 赢 ${t.points} 分`));
      }
    } else if (v.phase === 'over' && v.result) {
      mid.append(h('div', { class: 'big' }, v.result.tier), h('div', { class: 'mid-sub' }, `闲家 ${v.result.xian_points} / ${v.result.contract}`));
    }
    return mid;
  }

  function cardPts(c) { const r = c.slice(1); return r === '5' ? 5 : (r === '10' || r === 'K') ? 10 : 0; }

  function fillTrick(slots) {
    const v = st.view;
    if (v.phase !== 'play' && v.phase !== 'over') return;
    let plays = v.cur;
    let done = false;
    let winner = null;
    if (!plays.length && v.tricks.length) {
      const t = v.tricks[v.tricks.length - 1];
      plays = t.plays; done = true; winner = t.winner;
    }
    const tIdx = v.tricks.length + (done ? -1 : 0);
    for (const p of plays) {
      const key = `${v.hand_no}:${tIdx}:${p.seat}`;
      const r = rel(p.seat);
      const anim = st.seenPlays.has(key) ? null : REL_CLASS[r];
      st.seenPlays.add(key);
      fitStack(slots[r], p.cards, v.trump, (done ? 'done-trick ' : '') + (winner === p.seat ? 'win' : ''), anim);
    }
  }

  function selectedCodes() {
    return [...st.sel].map((k) => k.split('#')[0]);
  }

  function updateSelInfo() {
    const el = $('[data-selinfo]');
    if (!el) return;
    const v = st.view;
    if (v && v.phase === 'bury') el.textContent = `已选 ${st.sel.size}/${v.bury_need}`;
    else el.textContent = `已选 ${st.sel.size} 张`;
  }

  function renderActions(box) {
    const v = st.view;
    const mine = v.my_seat !== null && v.my_seat !== undefined && v.to_act === v.my_seat;
    const compact = (window.innerHeight || 768) <= 540 && (window.innerWidth || 1024) > (window.innerHeight || 768);
    if (!v.phase) return;
    if (v.phase === 'over') {
      box.append(h('span', { class: 'msg' }, `${v.result.tier} · 闲家 ${v.result.xian_points} 分`),
        h('button', { class: 'btn', onclick: showResult }, '结算详情'));
      if ((v.my_seat !== null && v.my_seat !== undefined) || v.is_host) {
        box.append(h('button', { class: 'btn primary', onclick: () => send({ op: 'next_hand' }) }, '下一局'));
      }
      return;
    }
    if (!mine) {
      box.append(h('span', { class: 'msg' }, `等待 ${seatName(v.to_act)} ${PHASE_NAMES[v.phase]}…`));
      return;
    }
    if (v.phase === 'bid') {
      const grid = h('div', { class: 'bid-grid' });
      for (const b of v.legal_bids || []) {
        grid.append(h('button', { class: 'btn' + (b === 0 ? ' ghost' : ''), onclick: () => send({ op: 'action', kind: 'bid', value: b }) }, b === 0 ? '不叫' : `${b}`));
      }
      box.append(h('span', { class: 'msg' }, compact ? '叫分:' : '请叫分（越低越难）：'), grid);
      if (v.mode === 'teach') box.append(h('button', { class: 'btn sm', onclick: () => send({ op: 'hint' }) }, '提示'));
    } else if (v.phase === 'trump') {
      box.append(h('span', { class: 'msg' }, '底牌:'), h('span', { class: 'chips' }, (v.bottom || []).map((c) => chipEl(c, null))));
      for (const t of v.legal_trumps || []) {
        box.append(h('button', { class: 'btn' + (t === 5 ? ' danger' : ''), onclick: () => {
          if (t === 5) {
            const m = modal(h('div', {},
              h('h2', {}, '确定投降？'),
              h('div', { style: { margin: '8px 0 14px' } }, `当前庄分 ${v.contract} 分，投降将按「${v.contract >= 55 ? '下庄' : '小倒'}」直接结算本局。`),
              h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
                h('button', { class: 'btn ghost', onclick: () => m.close() }, '取消'),
                h('button', { class: 'btn danger', onclick: () => { m.close(); send({ op: 'action', kind: 'trump', value: t }); } }, '确定投降'))));
            return;
          }
          send({ op: 'action', kind: 'trump', value: t });
        } }, TRUMP_NAMES[t]));
      }
      if (v.mode === 'teach') box.append(h('button', { class: 'btn sm', onclick: () => send({ op: 'hint' }) }, '提示'));
    } else if (v.phase === 'bury') {
      box.append(
        h('span', { class: 'msg' }, compact ? '选牌埋底' : `请选择 ${v.bury_need} 张牌埋底`),
        h('span', { class: 'hint', 'data-selinfo': 1 }, `已选 ${st.sel.size}/${v.bury_need}`),
        h('button', { class: 'btn primary', onclick: () => {
          const cards = selectedCodes();
          if (cards.length !== v.bury_need) { toast(`请选择 ${v.bury_need} 张牌`); return; }
          send({ op: 'action', kind: 'bury', cards });
        } }, '埋底'),
        h('button', { class: 'btn', onclick: () => { st.sel.clear(); render(); } }, '重选')
      );
      if (v.mode === 'teach') box.append(h('button', { class: 'btn sm', onclick: () => send({ op: 'hint' }) }, '提示'));
    } else if (v.phase === 'play') {
      let need = '';
      if (v.lead) need = compact ? `跟${v.lead.n}张(${KIND_NAMES[v.lead.kind]})` : `需跟 ${v.lead.n} 张（${KIND_NAMES[v.lead.kind]}）`;
      else need = compact ? '你首出（主牌可甩）' : '你首出：单张 / 对子 / 拖拉机（主牌可甩）';
      box.append(
        h('span', { class: 'msg' }, need),
        h('span', { class: 'hint', 'data-selinfo': 1 }, `已选 ${st.sel.size} 张`),
        h('button', { class: 'btn primary', onclick: () => {
          const cards = selectedCodes();
          if (!cards.length) { toast('请先选择要出的牌'); return; }
          send({ op: 'action', kind: 'play', cards });
        } }, '出牌'),
        h('button', { class: 'btn', onclick: () => { st.sel.clear(); render(); } }, '重选')
      );
      if (v.mode === 'teach') box.append(h('button', { class: 'btn sm', onclick: () => send({ op: 'hint' }) }, '提示'));
    }
  }

  function selectCodes(codes) {
    st.sel.clear();
    const hand = st.view.hand || [];
    const used = new Set();
    for (const c of codes) {
      const idx = hand.findIndex((x, i) => x === c && !used.has(i));
      if (idx >= 0) { used.add(idx); st.sel.add(`${c}#${idx}`); }
    }
    render();
  }

  function showHint(c) {
    if (!alive) return;
    if (!c) { toast('暂无可用的 AI 模型'); return; }
    if (c.codes) { selectCodes(c.codes); toast(`AI 建议：${c.suggest}`, 4000); return; }
    const top = c.top || [];
    if (!top.length) return;
    if (top[0].codes) selectCodes(top[0].codes);
    const m = modal(h('div', {}, h('h2', {}, 'AI 提示'), commentBody(c, true),
      h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '10px' } }, h('button', { class: 'btn primary', onclick: () => m.close() }, '好的'))));
  }

  // ===================================================== side panel =====
  function togglePanel(key) {
    st.panel = st.panel === key ? null : key;
    store.set('sdh.panel', st.panel);
    render();
  }

  function sidePanel() {
    const v = st.view;
    const tabs = [['counter', '记牌器'], ['history', '历史']];
    if (v.mode === 'teach') tabs.push(['comments', '点评'], ['hands', '手牌']);
    const body = h('div', { class: 'body' });
    if (st.panel === 'counter') body.append(counterView());
    else if (st.panel === 'history') body.append(historyView());
    else if (st.panel === 'comments') body.append(commentsView());
    else if (st.panel === 'hands') body.append(handsView());
    return h('aside', { class: 'side' },
      h('div', { class: 'tabs' }, tabs.map(([k, t]) => h('button', { class: st.panel === k ? 'on' : '', onclick: () => { st.panel = k; store.set('sdh.panel', k); render(); } }, t)),
        h('button', { class: 'closebtn', 'aria-label': '关闭', onclick: () => togglePanel(st.panel) }, '×')),
      body);
  }

  function counterView() {
    const v = st.view;
    if (!v.counter) return h('div', { class: 'empty' }, '对局开始后显示');
    const ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '2'];
    const cell = (code) => {
      const n = v.counter[code] ?? 0;
      return h('td', { class: n === 0 ? 'z' : n === 1 ? 'one' : 'two' }, n);
    };
    const rows = [];
    for (const s of 'SHCD') {
      const isT = v.trump !== null && v.trump !== undefined && 'SHCD'[v.trump] === s;
      rows.push(h('tr', {}, h('th', { class: 'suit ' + (s === 'H' || s === 'D' ? 'red-s' : '') }, SUIT_SYM[s] + (isT ? '主' : '')), ranks.map((r) => cell(s + r))));
    }
    const out = h('div', {},
      h('div', { class: 'hint', style: { marginBottom: '6px' } }, '外面还剩的牌（已扣除你的手牌和已出的牌）'),
      h('table', { class: 'counter-table' },
        h('tr', {}, h('th', {}, ''), ranks.map((r) => h('th', {}, r))), rows),
      h('table', { class: 'counter-table', style: { marginTop: '8px', width: 'auto' } },
        h('tr', {}, h('th', { style: { paddingRight: '10px' } }, '大王'), cell('BJ'), h('th', { style: { padding: '0 10px' } }, '小王'), cell('LJ'))));
    const pts = Object.entries(v.counter).reduce((a, [c, n]) => a + cardPts(c) * n, 0);
    out.append(h('div', { class: 'hint', style: { marginTop: '8px' } }, `外面剩余分数：${pts} 分`));
    return out;
  }

  function trickRow(t, idx, trump) {
    return h('div', { class: 'trick-row' },
      h('div', { class: 'hdr' }, h('span', {}, `第 ${idx} 轮 · ${seatName(t.leader)} 首出`), h('span', {}, `${seatName(t.winner)} 赢 · ${t.points} 分`)),
      t.plays.map((p) => h('div', { class: 'pl' + (p.seat === t.winner ? ' win' : '') }, h('span', { class: 'nm' }, seatName(p.seat)),
        h('span', { class: 'chips' }, p.cards.map((c) => chipEl(c, trump))))));
  }

  function historyView() {
    const v = st.view;
    const out = h('div', {});
    if (v.bids && v.bids.length) {
      out.append(h('div', { class: 'trick-row' }, h('div', { class: 'hdr' }, h('span', {}, '叫分记录')),
        h('div', { class: 'chips' }, v.bids.map((b) => h('span', { class: 'tag' }, `${seatName(b.seat)}: ${b.value ? b.value : '不叫'}`)))));
    }
    const tricks = (v.tricks || []).map((t, i) => trickRow(t, i + 1, v.trump)).reverse();
    if (!tricks.length) out.append(h('div', { class: 'empty' }, '本局还没有完成的出牌轮次'));
    out.append(...tricks);
    if (v.past_hands && v.past_hands.length) {
      out.append(h('h3', { style: { margin: '12px 0 6px' } }, '往局记录'));
      const tb = h('table', { class: 'evals' }, h('tr', {}, h('th', {}, '局'), h('th', {}, '庄'), h('th', {}, '庄分'), h('th', {}, '闲得分'), h('th', {}, '结果'), h('th', {}, '我')));
      for (const p of [...v.past_hands].reverse()) {
        const my = v.my_seat ?? 0;
        tb.append(h('tr', {}, h('td', {}, p.hand_no), h('td', {}, seatName(p.banker)), h('td', {}, p.contract), h('td', {}, p.xian_points),
          h('td', {}, p.tier), h('td', { class: p.scores[my] >= 0 ? 'pos' : 'neg' }, p.scores[my])));
      }
      out.append(tb);
    }
    return out;
  }

  function verdict(c) {
    if (c.regret === undefined || c.regret === null) return null;
    if (c.chosen_rank === 1 || c.regret < 0.05) return h('div', { class: 'verdict good' }, '✓ 最优或接近最优');
    if (c.regret < 0.3) return h('div', { class: 'verdict ok' }, `△ 可接受（期望少 ${fmt(c.regret, 2)} 底注）`);
    return h('div', { class: 'verdict bad' }, `✗ 失误（期望少 ${fmt(c.regret, 2)} 底注）`);
  }

  function commentBody(c, isHint) {
    const box = h('div', {});
    if (c.error) { box.append(h('div', { class: 'hint' }, c.error)); return box; }
    if (c.suggest) box.append(h('div', {}, `AI 建议：${c.suggest}`));
    if (c.chosen && !isHint) box.append(h('div', {}, '实际：', h('b', {}, c.chosen), c.chosen_rank ? `（AI 排名 ${c.chosen_rank}/${c.n_options}）` : ''));
    const top = c.top || [];
    if (top.length) {
      const maxW = Math.max(...top.map((t) => t.win_prob));
      box.append(h('div', { class: 'opt hint' }, h('span', {}, '候选'), h('span', {}, '期望得分'), h('span', {}, '胜率')));
      top.forEach((t, i) => box.append(h('div', { class: 'opt' + (i === 0 ? ' best' : '') },
        h('span', {}, `${i + 1}. ${t.label}`), h('span', {}, `${t.exp_score >= 0 ? '+' : ''}${fmt(t.exp_score, 2)}`),
        h('span', { title: `预测闲家得分 ${fmt(t.xian_points, 0)}` }, pct(t.win_prob, 0)))));
      box.append(h('div', { class: 'bar', title: '最佳选项胜率' }, h('i', { style: { width: `${Math.round(maxW * 100)}%` } })));
      box.append(h('div', { class: 'hint' }, `预测闲家最终得分 ≈ ${fmt(top[0].xian_points, 0)} 分`));
    }
    if (c.chosen_eval && c.chosen_rank > 3) {
      box.append(h('div', { class: 'opt' }, h('span', {}, `${c.chosen_rank}. ${c.chosen_eval.label}`),
        h('span', {}, fmt(c.chosen_eval.exp_score, 2)), h('span', {}, pct(c.chosen_eval.win_prob, 0))));
    }
    const vd = isHint ? null : verdict(c);
    if (vd) box.append(vd);
    return box;
  }

  function commentsView() {
    const v = st.view;
    const list = (v.comments || []).filter((c) => c.hand === v.hand_no || c.hand === undefined).slice().reverse();
    if (!list.length) return h('div', { class: 'empty' }, '每一步出牌后，这里会显示 AI 的点评。');
    return h('div', {}, list.map((c) => h('div', { class: 'cmt' },
      h('div', { class: 'hdr' }, h('b', {}, `${seatName(c.seat)}${c.seat === v.my_seat ? '（你）' : ''}`),
        h('span', { class: 'hint' }, `${PHASE_NAMES[c.phase] || ''}${c.phase === 'play' ? ` · 第 ${c.trick} 轮` : ''}`)),
      commentBody(c, false))));
  }

  function handsView() {
    const v = st.view;
    if (!v.all_hands) return h('div', { class: 'empty' }, '对局开始后显示');
    return h('div', {}, [0, 1, 2, 3].filter((i) => i !== v.my_seat).map((i) => h('div', { class: 'trick-row' },
      h('div', { class: 'hdr' }, h('b', {}, seatName(i)), h('span', {}, `${v.all_hands[i].length} 张`)),
      h('div', { class: 'chips' }, v.all_hands[i].map((c) => chipEl(c, v.trump))))));
  }

  // ========================================================= result =====
  function showResult() {
    if (!alive) return;
    const v = st.view;
    if (!v || !v.result) return;
    const r = v.result;
    const my = v.my_seat;
    const myWin = my !== null && my !== undefined ? r.scores[my] > 0 : null;
    const head = r.xian_win ? '闲家胜' : '庄家胜';
    const k = r.koudi;
    const content = h('div', {},
      h('h2', {}, `第 ${v.hand_no} 局结算`),
      h('div', { class: 'result-big ' + (myWin === null ? '' : myWin ? 'win' : 'lose') }, `${head} · ${r.tier}`),
      h('div', { style: { textAlign: 'center' } }, `庄家 ${seatName(r.banker)} · 庄分 ${r.contract} · 闲家得分 ${r.xian_points}`),
      k && k.mult ? h('div', { style: { textAlign: 'center' }, class: 'hint' }, `扣底：底分 ${k.bottom_points} × ${k.mult} = ${k.gained}`) :
        (k ? h('div', { style: { textAlign: 'center' }, class: 'hint' }, `底牌分 ${k.bottom_points}（未扣底）`) : null),
      h('div', { class: 'score-grid' }, [0, 1, 2, 3].map((i) => h('div', {}, seatName(i) + (i === r.banker ? '（庄）' : ''),
        h('b', { class: r.scores[i] >= 0 ? 'pos' : 'neg' }, `${r.scores[i] >= 0 ? '+' : ''}${r.scores[i]}`),
        h('span', { class: 'hint' }, `累计 ${v.totals[i]}`)))),
      v.buried ? h('div', {}, h('div', { class: 'hint' }, '底牌'), h('div', { class: 'chips' }, v.buried.map((c) => chipEl(c, v.trump)))) : null);
    const m = modal(h('div', {}, content, h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '10px' } },
      (v.can_manage || v.is_host) ? h('button', { class: 'btn danger', onclick: () => {
        m.close();
        send({ op: 'close_room', room: v.room.id });
        st.view = null;
        render();
      } }, '结束牌局') : null,
      h('button', { class: 'btn', onclick: () => m.close() }, '关闭'),
      (my !== null && my !== undefined) || v.is_host || v.can_manage ? h('button', { class: 'btn primary', onclick: () => { m.close(); send({ op: 'next_hand' }); } }, '下一局') : null)));
  }

  // ========================================================== timer =====
  function tick() {
    if (!alive) return;
    const now = Date.now() / 1000 + st.timeOffset;
    const total = (st.view && st.view.turn_seconds) || 30;
    for (const el of document.querySelectorAll('.timer[data-deadline]')) {
      const left = +el.dataset.deadline - now;
      const i = el.querySelector('i');
      i.style.width = `${Math.max(0, Math.min(100, (left / total) * 100))}%`;
      el.classList.toggle('warn', left < 8);
    }
    for (const el of document.querySelectorAll('[data-countdown]')) {
      const left = Math.max(0, Math.ceil(+el.dataset.countdown - now));
      el.textContent = `${left}s`;
    }
  }

  render();
  return () => {
    alive = false;
    window.__sdhPlayBack = null;
    if (st.view && st.view.room) {
      send({ op: 'leave_room' });
    }
    offs.forEach((f) => f());
    clearInterval(timerInt);
    clearTimeout(resizeT);
    clearTimeout(resultT);
    window.removeEventListener('resize', onResize);
  };
}
