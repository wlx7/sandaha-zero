export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

let toastTimer;
export function toast(msg, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

export async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `请求失败 (${r.status})`);
  return data;
}

export function modal(content, { onClose, closable = true } = {}) {
  const root = $('#modal-root');
  const back = h('div', { class: 'modal-back' });
  const box = h('div', { class: 'modal', role: 'dialog' }, content);
  back.append(box);
  const close = () => { back.remove(); onClose && onClose(); };
  // A locked popup (data-locked) must be answered: the backdrop does not close it
  if (closable) back.addEventListener('click', (e) => { if (e.target === back && !back.dataset.locked) close(); });
  root.append(back);
  return { close, box };
}

export const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
};

export function fmt(n, d = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toFixed(d);
}
export function pct(x, d = 1) { return x === null || x === undefined ? '—' : (x * 100).toFixed(d) + '%'; }
export function big(n) {
  if (n === null || n === undefined) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(Math.round(n));
}
export function dur(sec) {
  sec = Math.max(0, Math.round(sec));
  const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
  return hh ? `${hh}h${String(mm).padStart(2, '0')}m` : `${mm}m${String(ss).padStart(2, '0')}s`;
}

// ------------------------------------------------ unified landscape stage ---
export const stage = {
  width: 1024,
  height: 600,
  rotated: false,
  compact: false,
  narrow: false,
  isApk: false,
};

export function syncStage() {
  const vv = window.visualViewport;
  const useVV = vv && (!vv.scale || Math.abs(vv.scale - 1) < 0.05);
  const rawW = Math.round((useVV ? vv.width : window.innerWidth) || document.documentElement.clientWidth || 800);
  const rawH = Math.round((useVV ? vv.height : window.innerHeight) || document.documentElement.clientHeight || 450);
  const offX = useVV ? Math.round(vv.offsetLeft || 0) : 0;
  const offY = useVV ? Math.round(vv.offsetTop || 0) : 0;

  const minScreen = Math.min(window.screen?.width || rawW, window.screen?.height || rawH);
  const rotated = rawH > rawW && (rawW <= 860 || (navigator.maxTouchPoints > 0 && minScreen <= 900));
  const sw = rotated ? rawH : rawW;
  const sh = rotated ? rawW : rawH;
  const compact = sh <= 540;
  const narrow = sw <= 1000;

  const changed = stage.rotated !== rotated || stage.width !== sw || stage.height !== sh;
  stage.width = sw;
  stage.height = sh;
  stage.rotated = rotated;
  stage.compact = compact;
  stage.narrow = narrow;

  const root = document.documentElement;
  root.classList.toggle('rotated-landscape', rotated);
  root.classList.toggle('compact-landscape', compact);
  root.classList.toggle('narrow-stage', narrow);
  root.style.setProperty('--stage-w', `${sw}px`);
  root.style.setProperty('--stage-h', `${sh}px`);
  root.style.setProperty('--stage-left', `${offX}px`);
  root.style.setProperty('--stage-top', `${offY}px`);
  root.style.setProperty('--vw1', `${(sw / 100).toFixed(3)}px`);
  root.style.setProperty('--vh1', `${(sh / 100).toFixed(3)}px`);

  if (changed) {
    window.dispatchEvent(new CustomEvent('sdh-stage-resize'));
  }
}

let stageInited = false;
export function initStage() {
  if (stageInited) return;
  stageInited = true;
  syncStage();
  window.addEventListener('resize', syncStage);
  window.addEventListener('orientationchange', () => setTimeout(syncStage, 60));
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', syncStage);
    window.visualViewport.addEventListener('scroll', syncStage);
  }
  document.addEventListener('fullscreenchange', syncStage);
  document.addEventListener('webkitfullscreenchange', syncStage);

  // Smooth touch-scroll & range-slider drag mapping when CSS 90deg rotation is active
  let touchCtx = null;

  function findScrollable(target, root) {
    let el = target instanceof Element ? target : null;
    while (el && el !== document.body && el !== document.documentElement) {
      const cs = getComputedStyle(el);
      const canY = (cs.overflowY === 'auto' || cs.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 2;
      const canX = (cs.overflowX === 'auto' || cs.overflowX === 'scroll') && el.scrollWidth > el.clientWidth + 2;
      if (canY || canX) return el;
      if (el === root) break;
      el = el.parentElement;
    }
    return root && (root.scrollHeight > root.clientHeight + 2) ? root : null;
  }

  function applyRotatedRange(rangeEl, clientY) {
    const rect = rangeEl.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientY - rect.top) / Math.max(1, rect.height)));
    const min = rangeEl.min !== '' ? +rangeEl.min : 0;
    const max = rangeEl.max !== '' ? +rangeEl.max : 100;
    const step = rangeEl.step && rangeEl.step !== 'any' ? +rangeEl.step : 1;
    let val = min + ratio * (max - min);
    val = Math.round((val - min) / step) * step + min;
    val = Math.max(min, Math.min(max, +val.toFixed(6)));
    if (+rangeEl.value !== val) {
      rangeEl.value = String(val);
      rangeEl.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  document.addEventListener('touchstart', (e) => {
    if (!stage.rotated || e.touches.length !== 1) {
      touchCtx = null;
      return;
    }
    const t = e.touches[0];
    const rangeEl = e.target instanceof Element ? e.target.closest('input[type="range"]') : null;
    if (rangeEl) {
      touchCtx = { kind: 'range', rangeEl };
      applyRotatedRange(rangeEl, t.clientY);
      return;
    }
    if (e.target instanceof Element && e.target.closest('select')) {
      touchCtx = null;
      return;
    }
    const stg = $('#stage');
    const scroller = findScrollable(e.target, stg);
    if (!scroller) {
      touchCtx = null;
      return;
    }
    touchCtx = {
      kind: 'scroll',
      el: scroller,
      startX: t.clientX,
      startY: t.clientY,
      lastX: t.clientX,
      lastY: t.clientY,
      moved: false,
    };
  }, { passive: true });

  document.addEventListener('touchmove', (e) => {
    if (!stage.rotated || !touchCtx || e.touches.length !== 1) return;
    const t = e.touches[0];
    if (touchCtx.kind === 'range') {
      applyRotatedRange(touchCtx.rangeEl, t.clientY);
      if (e.cancelable) e.preventDefault();
      return;
    }
    const dx = t.clientX - touchCtx.lastX;
    const dy = t.clientY - touchCtx.lastY;
    if (!touchCtx.moved && Math.hypot(t.clientX - touchCtx.startX, t.clientY - touchCtx.startY) < 6) {
      return;
    }
    touchCtx.moved = true;
    touchCtx.lastX = t.clientX;
    touchCtx.lastY = t.clientY;
    // Under +90deg clockwise rotation: stageDown = -screenX, stageRight = +screenY
    if (touchCtx.el.scrollHeight > touchCtx.el.clientHeight) {
      touchCtx.el.scrollTop += dx;
    }
    if (touchCtx.el.scrollWidth > touchCtx.el.clientWidth) {
      touchCtx.el.scrollLeft -= dy;
    }
    if (e.cancelable) e.preventDefault();
  }, { passive: false });

  document.addEventListener('touchend', () => {
    if (touchCtx && touchCtx.kind === 'range') {
      touchCtx.rangeEl.dispatchEvent(new Event('change', { bubbles: true }));
    }
    touchCtx = null;
  }, { passive: true });
}

export function canBrowserFullscreen() {
  if (stage.isApk) return false;
  const el = document.documentElement;
  return Boolean(el.requestFullscreen || el.webkitRequestFullscreen);
}

export function isFullscreenActive() {
  return Boolean(document.fullscreenElement || document.webkitFullscreenElement);
}

export async function tryLandscapeFullscreen({ toggle = false, autoOnlyMobile = false } = {}) {
  if (stage.isApk) return;
  const doc = document;
  const el = doc.documentElement;
  const fsEl = doc.fullscreenElement || doc.webkitFullscreenElement;
  if (toggle && fsEl) {
    try {
      if (doc.exitFullscreen) await doc.exitFullscreen();
      else if (doc.webkitExitFullscreen) await doc.webkitExitFullscreen();
    } catch {}
    return;
  }
  if (autoOnlyMobile) {
    const isMobileTouch = navigator.maxTouchPoints > 0 && Math.min(stage.width, stage.height) <= 540;
    if (!isMobileTouch) return;
  }
  try {
    if (!fsEl) {
      if (el.requestFullscreen) await el.requestFullscreen({ navigationUI: 'hide' });
      else if (el.webkitRequestFullscreen) await el.webkitRequestFullscreen();
    }
    if (screen.orientation && screen.orientation.lock) {
      await screen.orientation.lock('landscape').catch(() => {});
    }
  } catch {
    // Browser denied fullscreen or orientation lock; CSS 90deg landscape remains active seamlessly.
  }
}
