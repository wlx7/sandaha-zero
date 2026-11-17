// Minimal responsive SVG line chart with hover tooltip.
import { h } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
const s = (tag, attrs = {}) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

function niceTicks(min, max, n = 5) {
  if (!(max > min)) { max = min + 1; }
  const span = max - min;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => x >= step0) || step0;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
  return out;
}

/**
 * series: [{name, color, points: [[x, y], ...], dash}]
 * opts: {title, yFmt, xFmt, yMin, yMax, refLine}
 */
export function lineChart(series, opts = {}) {
  const W = 440, H = 230, L = 46, R = 10, T = 12, B = 30;
  const wrap = h('div', { class: 'chart' });
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opts.title || 'chart' });
  wrap.append(svg);
  const pts = series.flatMap((x) => x.points);
  if (!pts.length) {
    wrap.append(h('div', { class: 'empty' }, '暂无数据'));
    svg.remove();
    return wrap;
  }
  let xmin = Math.min(...pts.map((p) => p[0])), xmax = Math.max(...pts.map((p) => p[0]));
  if (xmax === xmin) xmax = xmin + 1;
  let ymin = opts.yMin ?? Math.min(...pts.map((p) => p[1]));
  let ymax = opts.yMax ?? Math.max(...pts.map((p) => p[1]));
  if (opts.refLine !== undefined) { ymin = Math.min(ymin, opts.refLine); ymax = Math.max(ymax, opts.refLine); }
  const yt = niceTicks(ymin, ymax, 4);
  ymin = opts.yMin ?? yt[0]; ymax = opts.yMax ?? yt[yt.length - 1];
  const X = (x) => L + (x - xmin) / (xmax - xmin) * (W - L - R);
  const Y = (y) => T + (1 - (y - ymin) / ((ymax - ymin) || 1)) * (H - T - B);
  const yFmt = opts.yFmt || ((v) => String(+v.toFixed(3)));
  const xFmt = opts.xFmt || ((v) => String(Math.round(v)));
  for (const v of yt) {
    if (v < ymin - 1e-9 || v > ymax + 1e-9) continue;
    svg.append(s('line', { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: '#23493c', 'stroke-width': 1 }));
    const t = s('text', { x: L - 6, y: Y(v) + 4, 'text-anchor': 'end', fill: '#9fbcb0', 'font-size': 13 });
    t.textContent = yFmt(v);
    svg.append(t);
  }
  for (const v of niceTicks(xmin, xmax, 5)) {
    if (v < xmin || v > xmax) continue;
    const t = s('text', { x: X(v), y: H - 8, 'text-anchor': 'middle', fill: '#9fbcb0', 'font-size': 13 });
    t.textContent = xFmt(v);
    svg.append(t);
  }
  if (opts.refLine !== undefined) {
    svg.append(s('line', { x1: L, x2: W - R, y1: Y(opts.refLine), y2: Y(opts.refLine), stroke: '#e3b341',
      'stroke-dasharray': '4 4', 'stroke-width': 1, opacity: 0.7 }));
  }
  for (const se of series) {
    if (!se.points.length) continue;
    const d = se.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    svg.append(s('path', { d, fill: 'none', stroke: se.color, 'stroke-width': 2.2, 'stroke-linejoin': 'round',
      'stroke-dasharray': se.dash || '' }));
    if (se.points.length < 40) {
      for (const p of se.points) svg.append(s('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.8, fill: se.color }));
    }
  }
  const legend = h('div', { class: 'legend' },
    series.map((se) => h('span', {}, h('i', { style: { background: se.color } }), se.name)));
  wrap.append(legend);
  // hover
  const tip = h('div', { class: 'tip' });
  wrap.append(tip);
  const guide = s('line', { y1: T, y2: H - B, stroke: '#ffffff', opacity: 0, 'stroke-width': 1 });
  svg.append(guide);
  svg.addEventListener('mousemove', (e) => {
    const rect = svg.getBoundingClientRect();
    const x = xmin + ((e.clientX - rect.left) / rect.width * W - L) / (W - L - R) * (xmax - xmin);
    const rows = [];
    let gx = null;
    for (const se of series) {
      if (!se.points.length) continue;
      let best = se.points[0];
      for (const p of se.points) if (Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
      gx = gx ?? best[0];
      rows.push(`<span style="color:${se.color}">●</span> ${se.name}: <b>${yFmt(best[1])}</b>`);
    }
    if (gx === null) return;
    guide.setAttribute('x1', X(gx)); guide.setAttribute('x2', X(gx)); guide.setAttribute('opacity', 0.25);
    tip.innerHTML = `${opts.xName || 'x'}: ${xFmt(gx)}<br>` + rows.join('<br>');
    tip.style.display = 'block';
    const px = (X(gx) / W) * rect.width;
    tip.style.left = Math.min(rect.width - 170, Math.max(0, px + 10)) + 'px';
    tip.style.top = '8px';
  });
  svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; guide.setAttribute('opacity', 0); });
  return wrap;
}
