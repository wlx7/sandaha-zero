import { $, api, h, modal, store } from './util.js';
import { connect, me, setName } from './net.js';
import { mountPlay } from './game.js';

const ICON_PLAY = `<svg viewBox="0 0 64 64" fill="none"><rect x="8" y="14" width="26" height="36" rx="4" fill="#fffdf8" stroke="#e3b341" stroke-width="2" transform="rotate(-10 21 32)"/><text x="12" y="30" font-size="12" font-weight="800" fill="#c62828" transform="rotate(-10 21 32)">K</text><rect x="28" y="12" width="26" height="36" rx="4" fill="#fffdf8" stroke="#e3b341" stroke-width="2" transform="rotate(10 41 30)"/><text x="33" y="27" font-size="12" font-weight="800" fill="#1a1a1a" transform="rotate(10 41 30)">A</text><text x="38" y="42" font-size="14" fill="#1a1a1a" transform="rotate(10 41 30)">♠</text></svg>`;
const ICON_TEACH = `<svg viewBox="0 0 64 64" fill="none"><circle cx="32" cy="26" r="14" fill="#13342a" stroke="#e3b341" stroke-width="2.5"/><path d="M26 40h12v6a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4v-6z" fill="#234b3d" stroke="#e3b341" stroke-width="2"/><path d="M32 18v10M27 24h10" stroke="#f5d77a" stroke-width="2.5" stroke-linecap="round"/><path d="M28 54h8" stroke="#9fbcb0" stroke-width="2.5" stroke-linecap="round"/></svg>`;
const ICON_TRAIN = `<svg viewBox="0 0 64 64" fill="none"><rect x="6" y="8" width="52" height="44" rx="6" fill="#0b1f18" stroke="#3d7a64" stroke-width="2"/><path d="M12 44 L22 34 L30 38 L40 24 L52 16" stroke="#e3b341" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 46 L22 40 L30 42 L40 34 L52 30" stroke="#4a9eff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity=".8"/><circle cx="52" cy="16" r="3.5" fill="#e3b341"/><rect x="20" y="54" width="24" height="4" rx="2" fill="#3d7a64"/></svg>`;

const RULES_HTML = `
<h2>三打哈 规则速览（本程序默认规则）</h2>
<ul style="line-height:1.65;padding-left:18px;margin:8px 0">
<li>两副牌去掉 3、4，共 92 张；每人 21 张，底牌 8 张。分牌：5=5 分，10/K=10 分，共 200 分。</li>
<li><b>叫分</b>：从 80 分起叫，每次至少降 5 分，最低 5 分；不叫后不能再叫；叫分最低者坐庄（庄分）。四家都不叫则首叫者以 80 分坐庄。持有 ≥7 张常主（王/7/2）者在别人叫 75 或以上时不能不叫。</li>
<li><b>定主 / 投降 / 埋底</b>：庄家拿底牌后选主花色（或无主），也可投降（庄分≥55 按下庄、否则按小倒结算），然后埋 8 张底牌。</li>
<li><b>主牌大小</b>：大王 &gt; 小王 &gt; 主7 &gt; 副7 &gt; 主2 &gt; 副2 &gt; 主A K Q J 10 9 8 6 5。副牌：A K Q J 10 9 8 6 5。</li>
<li><b>出牌</b>：单张、对子（同花色同点）、拖拉机（相连对子，如 主7主7+副7副7、副2副2+主A主A、8866）。<b>不允许甩副牌</b>；主牌可以甩，每一部分都必须是最大的，否则甩牌失败，只能出被管住的那一部分。</li>
<li>跟牌须跟同门，有对跟对、有拖拉机跟拖拉机；无同门可毙（用主牌杀）或垫牌。庄家首出。</li>
<li><b>扣底</b>：闲家用主牌赢得最后一轮，底牌分 ×倍数（单张×1、对子×2、拖拉机×对数×2、甩牌×张数）计入闲家。</li>
<li><b>结算</b>（以底注为单位，庄家输赢为每个闲家的 3 倍）：闲家得分 ≥庄分+70 大倒 ×3；≥庄分+40 小倒 ×2；≥庄分 下庄 ×1；闲家 0 分 大光 ×3；≤30 分 小光 ×2；其余 保庄 ×1。</li>
</ul>
<p class="hint">各地规则差异较大，可在创建房间的“高级规则”中调整。</p>`;

export function showRules() {
  const m = modal(h('div', {}, h('div', { html: RULES_HTML }), h('div', { class: 'row', style: { justifyContent: 'flex-end' } },
    h('button', { class: 'btn primary', onclick: () => m.close() }, '知道了'))));
}

let cachedInfo = null;
async function loadInfo() {
  if (cachedInfo) return cachedInfo;
  try {
    cachedInfo = await api('/api/info');
  } catch {
    cachedInfo = { can_train: false, lan: [] };
  }
  return cachedInfo;
}

function enterPlayWithMode(nameVal, mode) {
  setName(nameVal);
  const lobby = store.get('sdh.lobby', { mode: 'battle', seats: ['me', 'ai', 'ai', 'ai'], models: [null, null, null, null], ai_delay: 0.9, rules: {} });
  lobby.mode = mode;
  store.set('sdh.lobby', lobby);
  location.hash = '#/play';
}

async function home(app) {
  const info = await loadInfo();
  const canTrain = info.can_train !== false;
  const nameInput = h('input', { value: me.name, maxlength: 16, 'aria-label': '昵称', style: { width: '148px' } });
  nameInput.addEventListener('change', () => setName(nameInput.value));
  const lan = h('div', { class: 'lan' });

  let choices;
  if (canTrain) {
    choices = h('div', { class: 'choices' },
      h('button', { class: 'choice', onclick: () => { setName(nameInput.value); location.hash = '#/play'; } },
        h('span', { html: ICON_PLAY }), h('b', {}, '对弈'), h('span', {}, '对战 / 教学模式，人机任意组合，支持局域网')),
      h('button', { class: 'choice', onclick: () => { location.hash = '#/train'; } },
        h('span', { html: ICON_TRAIN }), h('b', {}, '训练'), h('span', {}, '自对弈训练、调参、评测与模型管理')));
  } else {
    choices = h('div', { class: 'choices' },
      h('button', { class: 'choice', onclick: () => enterPlayWithMode(nameInput.value, 'battle') },
        h('span', { html: ICON_PLAY }), h('b', {}, '对战模式'), h('span', {}, '人机自由组队 · 局域网好友联机')),
      h('button', { class: 'choice', onclick: () => enterPlayWithMode(nameInput.value, 'teach') },
        h('span', { html: ICON_TEACH }), h('b', {}, '教学模式'), h('span', {}, 'AI 实时动作评分、胜率点评与提示')));
  }

  const brandBox = h('div', { class: 'home-brand' },
    h('img', { class: 'logo', src: '/static/icon.svg', alt: '' }),
    h('div', { class: 'home-titles' },
      h('h1', {}, '三打哈 AI'),
      h('div', { class: 'sub' }, '湖南三打哈 · 强化学习 AI · 局域网联机 · 教学点评')),
    h('div', { class: 'namebox' }, h('label', { class: 'hint' }, '昵称'), nameInput,
      h('button', { class: 'btn sm ghost', onclick: showRules }, '规则说明')));

  const mainBox = h('div', { class: 'home-main' }, choices, lan);

  app.append(h('div', { class: 'home' },
    h('div', { class: 'home-inner' }, brandBox, mainBox),
    h('div', { class: 'foot' }, 'SanDaHa-Zero · 非商业许可 (PolyForm Noncommercial 1.0.0)')));

  if (info.lan && info.lan.length) {
    const url = info.lan[0];
    const showQrModal = () => {
      const m = modal(h('div', { style: { textAlign: 'center' } },
        h('h2', {}, '局域网扫码联机'),
        h('img', { src: `/api/qr?url=${encodeURIComponent(url)}`, alt: '局域网二维码', style: { width: '180px', height: '180px', background: '#fff', borderRadius: '10px', padding: '6px' } }),
        h('div', { style: { marginTop: '8px' } }, ...info.lan.map((u) => h('div', {}, h('b', {}, u)))),
        h('div', { class: 'hint', style: { marginTop: '6px' } }, '同一 Wi-Fi 下的手机或电脑扫码即可加入房间'),
        h('div', { class: 'row', style: { justifyContent: 'center', marginTop: '12px' } },
          h('button', { class: 'btn primary', onclick: () => m.close() }, '关闭'))));
    };
    const shownLan = info.lan.slice(0, 2);
    lan.append(
      h('img', { class: 'lan-qr', src: `/api/qr?url=${encodeURIComponent(url)}`, alt: '局域网二维码', onclick: showQrModal }),
      h('div', { class: 'lan-text' },
        h('div', {}, '同局域网设备可访问：'),
        ...shownLan.map((u) => h('div', {}, h('b', {}, u))),
        info.lan.length > 2 ? h('div', { class: 'hint' }, `另有 ${info.lan.length - 2} 个网卡地址（点击查看）`) : null,
        h('button', { class: 'btn sm ghost lan-qr-btn', onclick: showQrModal }, '显示联机二维码'))
    );
  }
}

let cleanup = null;
async function route() {
  const app = $('#app');
  if (cleanup) { cleanup(); cleanup = null; }
  app.innerHTML = '';
  const r = location.hash || '#/';
  if (r.startsWith('#/play')) {
    cleanup = mountPlay(app);
  } else if (r.startsWith('#/train')) {
    const info = await loadInfo();
    if (info.can_train === false) {
      location.hash = '#/';
      return;
    }
    const { mountTrain } = await import('./train.js');
    cleanup = mountTrain(app);
  } else {
    await home(app);
  }
}

// Global back handler used by Android hardware/gesture back
window.__sdhBack = () => {
  const openModal = $('#modal-root .modal-back:last-child');
  if (openModal) {
    openModal.remove();
    return true;
  }
  if (typeof window.__sdhPlayBack === 'function' && window.__sdhPlayBack()) {
    return true;
  }
  const r = location.hash || '#/';
  if (r !== '#/' && r !== '#') {
    location.hash = '#/';
    return true;
  }
  return false;
};

window.addEventListener('hashchange', route);
connect();
route();
