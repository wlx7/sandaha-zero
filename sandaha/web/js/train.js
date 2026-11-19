// Training dashboard.
import { api, big, dur, fmt, h, modal, pct, store, toast } from './util.js';
import { lineChart } from './chart.js';
import { goHome } from './main.js';

const COLORS = { play: '#e3b341', bid: '#4a9eff', bury: '#3fbf7f', trump: '#c77dff', rule: '#e3b341', base: '#4a9eff', prev: '#3fbf7f' };
const ICONS = {
  games: '<svg viewBox="0 0 24 24" fill="none" stroke="#e3b341" stroke-width="2"><rect x="3" y="4" width="10" height="14" rx="2"/><rect x="11" y="6" width="10" height="14" rx="2" fill="#13342a"/></svg>',
  speed: '<svg viewBox="0 0 24 24" fill="none" stroke="#4a9eff" stroke-width="2"><path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-6"/></svg>',
  win: '<svg viewBox="0 0 24 24" fill="none" stroke="#3fbf7f" stroke-width="2"><path d="M7 4h10v4a5 5 0 0 1-10 0z"/><path d="M12 13v4M8 20h8"/><path d="M7 6H4a3 3 0 0 0 3 4M17 6h3a3 3 0 0 1-3 4"/></svg>',
  score: '<svg viewBox="0 0 24 24" fill="none" stroke="#c77dff" stroke-width="2"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>',
  time: '<svg viewBox="0 0 24 24" fill="none" stroke="#9fbcb0" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>',
  brain: '<svg viewBox="0 0 24 24" fill="none" stroke="#e3b341" stroke-width="2"><circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="12" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M8 6h2M8 18h2M14 6l2 5M14 18l2-5"/></svg>',
};

const kfmt = (v) => (v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + 'M' : v.toFixed(0) + 'k');
const STATE_ZH = { running: '训练中', starting: '启动中', evaluating: '评测中', finished: '已完成', stale: '已中断', error: '出错' };

export function mountTrain(app) {
  const st = { presets: {}, defaults: {}, torch: {}, cpus: 4, runs: [], selected: store.get('sdh.run', null), detail: null,
    models: [], form: null, installing: false };
  let poll = null;
  let alive = true;

  async function load() {
    const [p, m] = await Promise.all([api('/api/train/presets'), api('/api/models')]);
    st.presets = p.presets; st.defaults = p.defaults; st.torch = p.torch; st.cpus = p.cpus; st.models = m.models;
    st.form = store.get('sdh.trainform', null) || { ...p.defaults, name: '' };
    await refresh();
  }

  async function refresh() {
    try {
      const r = await api('/api/train/runs');
      st.runs = r.runs;
      if (!st.selected && st.runs.length) st.selected = st.runs[0].name;
      if (st.selected && st.runs.some((x) => x.name === st.selected)) st.detail = await api(`/api/train/run/${st.selected}`);
      else st.detail = null;
    } catch (e) { toast(e.message); }
    if (alive) render();
  }

  function schedule() {
    if (!alive) return;
    clearTimeout(poll);
    const running = st.runs.some((r) => ['running', 'starting', 'evaluating'].includes(r.status && r.status.state));
    poll = setTimeout(async () => { if (!alive) return; await refresh(); schedule(); }, running || st.installing ? 4000 : 15000);
  }

  function field(key, type = 'number', attrs = {}) {
    const F = st.form;
    const i = h('input', { type, value: F[key] ?? '', ...attrs });
    i.addEventListener('change', () => { F[key] = type === 'number' ? +i.value : i.value; store.set('sdh.trainform', F); });
    return i;
  }

  function modelSelect(key, emptyLabel) {
    const F = st.form;
    const sel = h('select', {}, h('option', { value: '' }, emptyLabel),
      st.models.map((m) => h('option', { value: m.path }, `${m.label}`)));
    sel.value = F[key] || '';
    sel.addEventListener('change', () => { F[key] = sel.value; store.set('sdh.trainform', F); });
    return sel;
  }

  function lrSched() {
    const F = st.form;
    const sel = h('select', { title: '学习率衰减方式（按最长时间计算进度）' },
      h('option', { value: 'cosine' }, '余弦衰减'), h('option', { value: 'linear' }, '线性衰减'), h('option', { value: 'constant' }, '不衰减'));
    sel.value = F.lr_schedule || 'cosine';
    sel.addEventListener('change', () => { F.lr_schedule = sel.value; store.set('sdh.trainform', F); });
    return sel;
  }

  function formBox() {
    const F = st.form;
    const presetSel = h('select', {}, Object.entries(st.presets).map(([k, v]) =>
      h('option', { value: k }, `${k} · ${big(v.params)} 参数 · [${v.arch.play.join(',')}]`)), h('option', { value: 'custom' }, '自定义'));
    presetSel.value = F.preset;
    const custom = h('div', { class: 'row', style: { display: F.preset === 'custom' ? 'flex' : 'none' } });
    const playH = h('input', { value: (F.play_hidden || []).join(','), placeholder: '出牌网络隐层, 如 1024,512,256' });
    const auxH = h('input', { value: (F.aux_hidden || []).join(','), placeholder: '叫分/定主/埋底网络隐层' });
    const parseList = (s) => s.split(/[,\s]+/).map(Number).filter((x) => x > 0);
    playH.addEventListener('change', () => { F.play_hidden = parseList(playH.value); store.set('sdh.trainform', F); });
    auxH.addEventListener('change', () => { F.aux_hidden = parseList(auxH.value); store.set('sdh.trainform', F); });
    custom.append(playH, auxH);
    presetSel.addEventListener('change', () => { F.preset = presetSel.value; custom.style.display = F.preset === 'custom' ? 'flex' : 'none'; store.set('sdh.trainform', F); });
    const dev = h('select', {}, h('option', { value: 'auto' }, '自动'), h('option', { value: 'cuda' }, 'GPU (CUDA)'), h('option', { value: 'cpu' }, 'CPU'));
    dev.value = F.device || 'auto';
    dev.addEventListener('change', () => { F.device = dev.value; store.set('sdh.trainform', F); });
    const t = st.torch || {};
    const torchLine = t.available
      ? h('span', { class: 'tag green' }, `PyTorch ${t.version} · ${t.cuda ? 'GPU: ' + t.gpu : '仅 CPU'}`)
      : h('span', {}, h('span', { class: 'tag red' }, '未安装 PyTorch'), ' ', h('button', { class: 'btn sm', onclick: installTorch }, '安装训练依赖'));
    const start = h('button', { class: 'btn primary', disabled: !t.available, onclick: async () => {
      try {
        const cfg = { ...F };
        if (!cfg.name) cfg.name = 'run-' + new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
        const r = await api('/api/train/start', cfg);
        st.selected = r.name; store.set('sdh.run', r.name);
        toast(`训练 ${r.name} 已启动`);
        setTimeout(refresh, 1500);
      } catch (e) { toast(e.message); }
    } }, '开始训练');
    return h('div', { class: 'card-box' },
      h('h3', { html: ICONS.brain + ' 新建训练' }),
      h('div', { class: 'row', style: { marginBottom: '10px' } }, torchLine),
      h('div', { class: 'form' },
        h('label', {}, '名称'), field('name', 'text', { placeholder: '留空自动命名' }),
        h('label', {}, '模型大小'), presetSel,
        h('div', { class: 'full' }, custom),
        h('label', {}, '设备'), dev,
        h('label', {}, '自对弈进程数'), h('div', { class: 'row' }, field('num_actors', 'number', { min: 0, max: 256, style: { width: '90px' } }), h('span', { class: 'hint' }, `0 = 自动 (本机 ${st.cpus} 线程)`)),
        h('label', {}, '学习率'), h('div', { class: 'row' }, field('lr', 'number', { step: 0.0001, style: { width: '100px' } }), '→', field('lr_end', 'number', { step: 0.00001, style: { width: '100px' } }), lrSched()),
        h('label', {}, '批大小'), field('batch_size', 'number', { step: 256 }),
        h('label', {}, '样本复用率'), field('replay_ratio', 'number', { step: 0.5 }),
        h('label', {}, '探索率 ε'), h('div', { class: 'row' }, field('epsilon_start', 'number', { step: 0.01, style: { width: '80px' } }), '→', field('epsilon_end', 'number', { step: 0.005, style: { width: '80px' } })),
        h('label', {}, 'ε 衰减局数'), field('epsilon_decay_games', 'number', { step: 10000 }),
        h('label', {}, '最多局数'), h('div', { class: 'row' }, field('max_games', 'number', { step: 100000, style: { width: '140px' } }), h('span', { class: 'hint' }, '0 = 不限')),
        h('label', {}, '最长时间(小时)'), h('div', { class: 'row' }, field('max_hours', 'number', { step: 0.5, min: 0, style: { width: '140px' } }), h('span', { class: 'hint' }, '0 = 不限，可随时手动停止')),
        h('label', {}, '评测间隔(分钟)'), field('eval_every_minutes', 'number', { step: 1 }),
        h('label', {}, '评测牌局数'), h('div', { class: 'row' }, field('eval_deals', 'number', { step: 50, style: { width: '90px' } }), h('span', { class: 'hint' }, '每副牌换位各打一次')),
        h('label', {}, '基准模型'), modelSelect('baseline', '无（只对比规则AI与上个检查点）'),
        h('label', {}, '继续训练自'), modelSelect('init_from', '从零开始'),
        h('label', {}, '投降课程'), h('div', { class: 'row' }, field('no_surrender_games', 'number', { step: 50000, style: { width: '110px' } }), h('span', { class: 'hint' }, '局前禁止投降')),
      ),
      h('div', { class: 'row', style: { marginTop: '12px' } }, start,
        h('button', { class: 'btn', onclick: () => { st.form = { ...st.defaults, name: '' }; store.set('sdh.trainform', st.form); render(); } }, '恢复默认')),
      st.installing || (st.install && st.install.state !== 'idle') ? h('pre', { class: 'log' }, (st.install && st.install.log) || '安装中…') : null);
  }

  async function installTorch() {
    st.installing = true;
    try { st.install = await api('/api/train/install', {}); } catch (e) { toast(e.message); }
    const pollInstall = async () => {
      st.install = await api('/api/train/install');
      if (st.install.state === 'running') { render(); setTimeout(pollInstall, 2000); return; }
      st.installing = false;
      st.torch = st.install.torch;
      toast(st.install.state === 'done' ? 'PyTorch 安装完成' : '安装失败，请查看日志');
      render();
    };
    pollInstall();
  }

  function runsBox() {
    const list = h('div', { class: 'runs' });
    if (!st.runs.length) list.append(h('div', { class: 'empty' }, '还没有训练记录'));
    for (const r of st.runs) {
      const s = (r.status && r.status.state) || 'unknown';
      const tag = { running: 'green', starting: 'green', evaluating: 'gold', finished: '', stale: 'red', error: 'red' }[s] ?? '';
      const ev = r.last_eval && r.last_eval.vs_rule;
      list.append(h('div', { class: 'run-item' + (r.name === st.selected ? ' on' : ''), onclick: async () => { st.selected = r.name; store.set('sdh.run', r.name); await refresh(); } },
        h('span', { class: 'nm' }, r.name), ev ? h('span', { class: 'hint' }, `vs规则 ${pct(ev.win_rate, 0)}`) : null,
        h('span', { class: 'tag ' + tag }, STATE_ZH[s] || s)));
    }
    return h('div', { class: 'card-box' }, h('h3', {}, '训练记录', h('button', { class: 'btn sm ghost', onclick: refresh }, '刷新')), list);
  }

  function statTile(icon, value, key) {
    return h('div', { class: 'stat' }, h('span', { html: ICONS[icon] }), h('div', {}, h('div', { class: 'v' }, value), h('div', { class: 'k' }, key)));
  }

  function detailBox() {
    const d = st.detail;
    if (!d) return h('div', { class: 'card-box empty' }, '选择或新建一个训练查看曲线');
    const ms = d.metrics || [];
    const ev = d.evals || [];
    const last = ms[ms.length - 1] || {};
    const lastE = ev[ev.length - 1];
    const s = d.status || {};
    const running = ['running', 'starting', 'evaluating'].includes(s.state);
    const elapsed = running ? (Date.now() / 1000 - (s.start || Date.now() / 1000)) : ((s.end || 0) - (s.start || 0));
    const tiles = h('div', { class: 'stats' },
      statTile('games', big(last.games || 0), '自对弈局数'),
      statTile('speed', fmt(last.games_per_s || 0, 0) + ' 局/秒', '生成速度'),
      statTile('win', lastE ? pct(lastE.vs_rule.win_rate) : '—', '对规则AI胜率'),
      statTile('score', lastE ? (lastE.vs_rule.avg_score >= 0 ? '+' : '') + fmt(lastE.vs_rule.avg_score, 2) : '—', '对规则AI每局净得分'),
      statTile('time', dur(elapsed > 0 ? elapsed : 0), running ? '已运行' : '总用时'),
      statTile('brain', big(d.params), `参数量 · ${d.config.preset}`));
    const xs = (m) => (m.games || 0) / 1000;
    const winChart = lineChart([
      { name: '对规则AI', color: COLORS.rule, points: ev.map((e) => [xs(e), e.vs_rule.win_rate]) },
      { name: '对基准模型', color: COLORS.base, points: ev.filter((e) => e.vs_baseline).map((e) => [xs(e), e.vs_baseline.win_rate]) },
      { name: '对上一检查点', color: COLORS.prev, points: ev.filter((e) => e.vs_prev).map((e) => [xs(e), e.vs_prev.win_rate]), dash: '5 4' },
    ], { yFmt: (v) => (v * 100).toFixed(0) + '%', xFmt: kfmt, xName: '局数(千)', refLine: 0.5, yMin: 0, yMax: 1 });
    const scoreChart = lineChart([
      { name: '对规则AI 净得分/局', color: COLORS.rule, points: ev.map((e) => [xs(e), e.vs_rule.avg_score]) },
      { name: '对基准模型', color: COLORS.base, points: ev.filter((e) => e.vs_baseline).map((e) => [xs(e), e.vs_baseline.avg_score]) },
      { name: '做庄胜率 (vs规则)', color: '#c77dff', points: ev.map((e) => [xs(e), e.vs_rule.banker_win_rate]), dash: '3 3' },
    ], { yFmt: (v) => fmt(v, 2), xFmt: kfmt, xName: '局数(千)', refLine: 0 });
    const lossChart = lineChart(['play', 'bid', 'bury', 'trump'].map((p) => ({
      name: { play: '出牌', bid: '叫分', bury: '埋底', trump: '定主' }[p], color: COLORS[p],
      points: ms.filter((m) => m[`loss_${p}`] !== undefined).map((m) => [xs(m), m[`loss_${p}`]]),
    })), { yFmt: (v) => fmt(v, 3), xFmt: kfmt, xName: '局数(千)' });
    const lrChart = lineChart([{ name: '学习率', color: '#c77dff', points: ms.filter((m) => m.lr !== undefined).map((m) => [xs(m), m.lr]) }],
      { yFmt: (v) => v.toExponential(1), xFmt: kfmt, xName: '局数(千)', yMin: 0 });
    const speedChart = lineChart([{ name: '局/秒', color: COLORS.base, points: ms.map((m) => [xs(m), m.games_per_s]) }],
      { yFmt: (v) => fmt(v, 0), xFmt: kfmt, xName: '局数(千)', yMin: 0 });
    const evalTable = h('table', { class: 'evals' }, h('tr', {}, h('th', {}, '检查点'), h('th', {}, 'vs规则 胜率'), h('th', {}, '净得分'), h('th', {}, '做庄率'), h('th', {}, '做庄胜率'), h('th', {}, 'vs基准'), h('th', {}, 'vs上一版')));
    for (const e of ev.slice().reverse().slice(0, 12)) {
      evalTable.append(h('tr', {}, h('td', {}, big(e.games)), h('td', {}, pct(e.vs_rule.win_rate)), h('td', {}, fmt(e.vs_rule.avg_score, 2)),
        h('td', {}, pct(e.vs_rule.banker_rate, 0)), h('td', {}, pct(e.vs_rule.banker_win_rate, 0)),
        h('td', {}, e.vs_baseline ? pct(e.vs_baseline.win_rate) : '—'), h('td', {}, e.vs_prev ? pct(e.vs_prev.win_rate) : '—')));
    }
    const actions = h('div', { class: 'row' });
    if (running) actions.append(h('button', { class: 'btn danger', onclick: async () => { await api(`/api/train/stop/${d.name}`, {}); toast('已发送停止信号，保存并评测后结束'); refresh(); } }, '停止训练'));
    else actions.append(h('button', { class: 'btn danger', onclick: async () => { if (!confirm(`删除训练 ${d.name} 及其所有模型？`)) return; try { await api(`/api/train/delete/${d.name}`, {}); st.selected = null; refresh(); } catch (e) { toast(e.message); } } }, '删除'));
    actions.append(h('button', { class: 'btn', onclick: () => publishDialog(d) }, '发布为对弈AI'));
    actions.append(h('button', { class: 'btn', onclick: () => evalDialog(d) }, '对比评测'));
    actions.append(h('button', { class: 'btn ghost', onclick: () => { st.form = { ...d.config, name: d.config.name + '-2', init_from: d.checkpoints.length ? d.checkpoints[d.checkpoints.length - 1] : '' }; store.set('sdh.trainform', st.form); render(); toast('已载入配置，可在左侧修改后继续训练'); } }, '载入配置/继续训练'));
    return h('div', {},
      h('div', { class: 'pagebar' }, h('h2', {}, d.name), h('span', { class: 'tag ' + (running ? 'green' : '') }, STATE_ZH[s.state] || s.state || ''), actions),
      tiles,
      h('div', { class: 'charts' },
        h('div', { class: 'card-box' }, h('h3', {}, '胜率曲线'), winChart),
        h('div', { class: 'card-box' }, h('h3', {}, '得分 / 做庄胜率'), scoreChart),
        h('div', { class: 'card-box' }, h('h3', {}, '训练损失 (Q 值 MSE)'), lossChart),
        h('div', { class: 'card-box' }, h('h3', {}, '自对弈速度'), speedChart),
        h('div', { class: 'card-box' }, h('h3', {}, '学习率'), lrChart)),
      h('div', { class: 'card-box', style: { marginTop: '14px', overflowX: 'auto' } }, h('h3', {}, '评测结果（换位复式赛）'), ev.length ? evalTable : h('div', { class: 'empty' }, '第一次评测将在评测间隔后进行')),
      h('details', { style: { marginTop: '10px' } }, h('summary', {}, '训练配置与日志'),
        h('pre', { class: 'log' }, JSON.stringify(d.config, null, 1)), h('pre', { class: 'log' }, d.log || '')));
  }

  function publishDialog(d) {
    const src = h('select', {}, h('option', { value: '' }, 'final（最终模型）'), d.checkpoints.slice().reverse().map((c) => h('option', { value: c }, c.split('/').pop())));
    const name = h('input', { value: d.name, maxlength: 40 });
    const m = modal(h('div', {}, h('h2', {}, '发布为对弈 AI'), h('div', { class: 'form' }, h('label', {}, '来源'), src, h('label', {}, '模型名称'), name),
      h('div', { class: 'hint', style: { marginTop: '8px' } }, '发布后会复制到 models/ 目录，可在对弈时选择。命名为 sandaha-best 将成为默认 AI。'),
      h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } }, h('button', { class: 'btn', onclick: () => m.close() }, '取消'),
        h('button', { class: 'btn primary', onclick: async () => {
          try { const r = await api(`/api/train/publish/${d.name}`, { source: src.value, target: name.value }); toast(`已发布：${r.path}`); m.close(); st.models = (await api('/api/models')).models; }
          catch (e) { toast(e.message); }
        } }, '发布'))));
  }

  function evalDialog(d) {
    const opts = () => [h('option', { value: 'rule' }, '规则AI'), h('option', { value: 'random' }, '随机AI'), ...st.models.map((m) => h('option', { value: m.path }, m.label))];
    const a = h('select', {}, opts());
    const b = h('select', {}, opts());
    const last = d.checkpoints[d.checkpoints.length - 1];
    if (last) a.value = last;
    const deals = h('input', { type: 'number', value: 100, min: 10, max: 500, step: 10 });
    const out = h('div', {});
    const m = modal(h('div', {}, h('h2', {}, '对比评测'), h('div', { class: 'form' }, h('label', {}, 'A'), a, h('label', {}, 'B'), b, h('label', {}, '牌局数'), deals), out,
      h('div', { class: 'row', style: { justifyContent: 'flex-end', marginTop: '12px' } }, h('button', { class: 'btn', onclick: () => m.close() }, '关闭'),
        h('button', { class: 'btn primary', onclick: async (e) => {
          e.target.disabled = true; out.innerHTML = '<div class="hint">评测中…</div>';
          try {
            const r = await api('/api/eval', { a: a.value, b: b.value, deals: +deals.value });
            out.innerHTML = '';
            out.append(h('div', { class: 'stats', style: { marginTop: '10px' } },
              statTile('win', pct(r.win_rate), 'A 的胜率'), statTile('score', fmt(r.avg_score, 2), 'A 每局净得分'),
              statTile('games', r.games, '局数'), statTile('brain', pct(r.banker_win_rate, 0), 'A 做庄胜率')));
          } catch (err) { out.textContent = err.message; }
          e.target.disabled = false;
        } }, '开始'))));
  }

  function render() {
    if (!alive) return;
    app.innerHTML = '';
    if (!st.form) { app.append(h('div', { class: 'page' }, h('div', { class: 'empty' }, '加载中…'))); return; }
    app.append(h('div', { class: 'page' },
      h('div', { class: 'pagebar' }, h('button', { class: 'btn sm ghost', onclick: () => goHome() }, '← 首页'), h('h2', {}, '训练中心'),
        h('span', { class: 'hint' }, '算法：Deep Monte-Carlo 自对弈 (DouZero 风格) · 叫分/定主/埋底/出牌四个 Q 网络')),
      h('div', { style: { display: 'grid', gridTemplateColumns: 'minmax(0, 380px) minmax(0, 1fr)', gap: '14px' }, class: 'train-grid' },
        h('div', { style: { display: 'flex', flexDirection: 'column', gap: '14px' } }, runsBox(), formBox()),
        detailBox())));
    if (window.innerWidth < 900) {
      const g = app.querySelector('.train-grid');
      if (g) g.style.gridTemplateColumns = '1fr';
    }
  }

  render();
  load().then(schedule).catch((e) => toast(e.message));
  return () => { alive = false; clearTimeout(poll); };
}
