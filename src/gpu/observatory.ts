/**
 * The Observatory: a dedicated, full-screen "biologist" view over the living
 * ocean. Three areas, all fed from the periodic GPU read-backs the sim already
 * does (so it adds no per-frame cost):
 *
 *   1. The world — time series of population, food, lineage diversity and an
 *      energy band, plus a sampled strategy mix.
 *   2. Lineages over time — a table of clades with their rise-and-fall curves
 *      and evolved traits.
 *   3. Tracked creatures — a small watch-list (capped) of individuals you pin
 *      from the brain panel, each with its energy/speed history.
 *
 * The ocean keeps running behind the dimmed overlay. Rendering happens only
 * while the view is open; data keeps accumulating either way (in gpuSim).
 */
import { t, onLang } from './i18n.js';

export interface WorldSample {
  tick: number;
  alive: number;
  foodAlive: number;
  lineages: number;
  energyAvg: number;
  energyMin: number;
  energyMax: number;
  /** Predation kills in the last tick (Phase 6). */
  predKills: number;
  /** Predation gain (0 = predation disabled) — drives the "active/off" wording. */
  predGain: number;
  /** Population-weighted mean body size (Phase 6 morphology). */
  meanSize: number;
  /** Day/night light level 0..1 (noon=1, midnight=0); -1 when the cycle is off. */
  daylight: number;
  /** descKey -> count, sampled from the dominant pool (a strategy snapshot). */
  strategy: Record<string, number>;
}

export interface LineageHistory {
  lineage: number;
  hue: number;
  count: number;
  trend: number;
  descKey: string;
  fast: boolean;
  seek: number;
  forage: number;
  cruise: number;
  aggression: number;
  neurons: number;
  /** Population samples over time (oldest -> newest), for the rise/fall curve. */
  samples: number[];
}

export interface WatchSample {
  tick: number;
  energy: number;
  speed: number;
  turn: number;
  thrust: number;
  alive: boolean;
}

export interface Watched {
  id: number;
  lineage: number;
  hue: number;
  history: WatchSample[];
}

export interface ObservatoryData {
  world: WorldSample[];
  lineages: LineageHistory[];
  watched: Watched[];
}

export interface Observatory {
  panel: HTMLElement;
  toggle: HTMLButtonElement;
  isOpen(): boolean;
  update(data: ObservatoryData): void;
}

const STRAT_COLOR: Record<string, string> = {
  desc_chase: '#3ff0d8',
  desc_steer: '#5ad1ff',
  desc_straight: '#9b8cff',
  desc_ambush: '#ff9f43',
  desc_circler: '#ffd23f',
  desc_away: '#ff5aa6',
  desc_erratic: '#8aa0b4',
};

const ACCENT = '#3ff0d8';
const CARD =
  'background:rgba(6,18,41,0.82);border:1px solid rgba(63,240,216,0.18);' +
  'border-radius:12px;padding:14px 16px;';

function setupCanvas(c: HTMLCanvasElement, wCss: number, hCss: number): CanvasRenderingContext2D {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  c.width = Math.round(wCss * dpr);
  c.height = Math.round(hCss * dpr);
  c.style.width = `${wCss}px`;
  c.style.height = `${hCss}px`;
  const ctx = c.getContext('2d')!;
  ctx.scale(dpr, dpr);
  return ctx;
}

/** Draw one or more line series sharing a y-scale. Returns the max used. */
function lineChart(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  series: { values: number[]; color: string }[],
): number {
  ctx.clearRect(0, 0, w, h);
  let max = 1;
  for (const s of series) for (const v of s.values) if (v > max) max = v;
  ctx.strokeStyle = 'rgba(120,160,200,0.14)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - 0.5);
  ctx.lineTo(w, h - 0.5);
  ctx.stroke();
  for (const s of series) {
    if (s.values.length < 2) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    s.values.forEach((v, i) => {
      const x = (i / (s.values.length - 1)) * w;
      const y = h - (v / max) * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  return max;
}

/** Energy band: filled min..max area with a mean line on top. */
function bandChart(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  mins: number[],
  means: number[],
  maxs: number[],
): void {
  ctx.clearRect(0, 0, w, h);
  const n = means.length;
  if (n < 2) return;
  let max = 1;
  for (const v of maxs) if (v > max) max = v;
  const xAt = (i: number): number => (i / (n - 1)) * w;
  const yAt = (v: number): number => h - (v / max) * (h - 4) - 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) ctx.lineTo(xAt(i), yAt(maxs[i]!));
  for (let i = n - 1; i >= 0; i--) ctx.lineTo(xAt(i), yAt(mins[i]!));
  ctx.closePath();
  ctx.fillStyle = 'rgba(63,240,216,0.14)';
  ctx.fill();
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  means.forEach((v, i) => (i === 0 ? ctx.moveTo(xAt(i), yAt(v)) : ctx.lineTo(xAt(i), yAt(v))));
  ctx.stroke();
}

function mkCard(titleKey: string): { card: HTMLElement; title: HTMLElement; body: HTMLElement } {
  const card = document.createElement('div');
  card.style.cssText = CARD;
  const title = document.createElement('div');
  title.style.cssText = `font-weight:600;letter-spacing:.1em;color:${ACCENT};margin-bottom:10px;`;
  title.dataset.key = titleKey;
  const body = document.createElement('div');
  card.append(title, body);
  return { card, title, body };
}

export function buildObservatory(onRemoveWatch: (id: number) => void): Observatory {
  let open = false;
  let last: ObservatoryData = { world: [], lineages: [], watched: [] };

  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;inset:0;display:none;z-index:30;overflow:auto;' +
    'background:rgba(2,4,10,0.9);font:13px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;';

  const inner = document.createElement('div');
  inner.style.cssText = 'max-width:1100px;margin:0 auto;padding:22px 22px 60px;';
  panel.append(inner);

  const header = document.createElement('div');
  header.style.cssText =
    'display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;';
  const h1 = document.createElement('div');
  h1.style.cssText = `font-size:20px;font-weight:600;letter-spacing:.12em;color:${ACCENT};`;
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  closeBtn.onclick = () => setOpen(false);
  header.append(h1, closeBtn);
  inner.append(header);

  const worldCard = mkCard('obs_world');
  const lineageCard = mkCard('obs_lineages');
  const watchCard = mkCard('obs_watched');
  inner.append(worldCard.card, lineageCard.card, watchCard.card);
  lineageCard.card.style.marginTop = '16px';
  watchCard.card.style.marginTop = '16px';

  const toggle = document.createElement('button');
  toggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  toggle.onclick = () => setOpen(!open);

  function setOpen(v: boolean): void {
    open = v;
    panel.style.display = v ? 'block' : 'none';
    if (v) render();
  }
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape' && open) setOpen(false);
  });

  // --- Renderers (only run while open) ---
  function tile(label: string, value: string, color = '#cfe8ff'): string {
    return (
      `<div style="${CARD}padding:10px 12px;min-width:120px">` +
      `<div style="opacity:.55;font-size:11px">${label}</div>` +
      `<div style="font-size:22px;color:${color};margin-top:2px">${value}</div></div>`
    );
  }

  // A plain-language read of what's happening right now, from the latest sample
  // (population trend, predation, dominant strategy, mean brain complexity).
  function narrative(world: WorldSample[], lineages: LineageHistory[]): string {
    const cur = world[world.length - 1]!;
    const prev = world[Math.max(0, world.length - 8)]!;
    const d = cur.alive - prev.alive;
    const band = Math.max(20, prev.alive * 0.08);
    const trend = d > band ? t('nar_rising') : d < -band ? t('nar_falling') : t('nar_stable');
    let wn = 0;
    let wsum = 0;
    for (const l of lineages) {
      wn += l.neurons * l.count;
      wsum += l.count;
    }
    const meanN = wsum ? (wn / wsum).toFixed(1) : '—';
    const entries = Object.entries(cur.strategy).sort((a, b) => b[1] - a[1]);
    const strat = entries.length ? t(entries[0]![0]) : '—';
    const pred =
      cur.predGain > 0
        ? `${t('nar_predActive')}${cur.predKills > 0 ? ` (${cur.predKills}/tick)` : ''}`
        : t('nar_predOff');
    const phase =
      cur.daylight < 0
        ? ''
        : ` ${cur.daylight > 0.6 ? '☀ ' + t('nar_day') : cur.daylight < 0.4 ? '🌙 ' + t('nar_night') : '🌅 ' + t('nar_dusk')}.`;
    return (
      `${t('nar_pop')} ${cur.alive.toLocaleString()} · ${trend}.${phase} ${pred}. ` +
      `${t('nar_strategy')}: ${strat}. ${t('nar_complexity')} ${meanN} ${t('neurons')} · ` +
      `${t('nar_size')} ${cur.meanSize.toFixed(2)}×.`
    );
  }

  function renderWorld(world: WorldSample[]): void {
    const body = worldCard.body;
    body.innerHTML = '';
    if (world.length === 0) {
      body.innerHTML = `<div style="opacity:.5">${t('sampling')}</div>`;
      return;
    }
    const cur = world[world.length - 1]!;
    const summary = document.createElement('div');
    summary.style.cssText =
      `margin-bottom:12px;padding:9px 11px;border-radius:8px;line-height:1.5;` +
      `background:rgba(63,240,216,0.07);border:1px solid rgba(63,240,216,0.14);color:#cfe8ff;`;
    summary.textContent = narrative(world, last.lineages);
    body.append(summary);
    const tiles = document.createElement('div');
    tiles.style.cssText = 'display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;';
    tiles.innerHTML =
      tile(t('obs_alive'), cur.alive.toLocaleString(), ACCENT) +
      tile(t('obs_food'), cur.foodAlive.toLocaleString()) +
      tile(t('obs_diversity'), cur.lineages.toLocaleString()) +
      tile(t('obs_meanEnergy'), cur.energyAvg.toFixed(1)) +
      tile(t('obs_predation'), `${cur.predKills}/tick`, '#ff9f43');
    body.append(tiles);

    const charts = document.createElement('div');
    charts.style.cssText = 'display:flex;gap:18px;flex-wrap:wrap;';
    const popWrap = document.createElement('div');
    popWrap.innerHTML =
      `<div style="opacity:.6;font-size:11px;margin-bottom:4px">${t('obs_population')} ` +
      `<span style="color:${ACCENT}">▬</span> · ${t('obs_food')} <span style="color:#5ad1ff">▬</span></div>`;
    const popCanvas = document.createElement('canvas');
    popWrap.append(popCanvas);
    const enWrap = document.createElement('div');
    enWrap.innerHTML = `<div style="opacity:.6;font-size:11px;margin-bottom:4px">${t('obs_energyBand')}</div>`;
    const enCanvas = document.createElement('canvas');
    enWrap.append(enCanvas);
    charts.append(popWrap, enWrap);
    body.append(charts);

    const cw = 420;
    const ch = 120;
    lineChart(setupCanvas(popCanvas, cw, ch), cw, ch, [
      { values: world.map((s) => s.alive), color: ACCENT },
      { values: world.map((s) => s.foodAlive), color: '#5ad1ff' },
    ]);
    bandChart(
      setupCanvas(enCanvas, cw, ch),
      cw,
      ch,
      world.map((s) => s.energyMin),
      world.map((s) => s.energyAvg),
      world.map((s) => s.energyMax),
    );

    // Strategy mix (stacked bar + legend), from the latest sample.
    const total = Object.values(cur.strategy).reduce((a, b) => a + b, 0);
    if (total > 0) {
      const mix = document.createElement('div');
      mix.style.cssText = 'margin-top:14px';
      const entries = Object.entries(cur.strategy).sort((a, b) => b[1] - a[1]);
      const bar = entries
        .map(([k, v]) => {
          const pct = (v / total) * 100;
          const c = STRAT_COLOR[k] ?? '#8aa0b4';
          return `<div title="${t(k)}" style="width:${pct}%;background:${c}"></div>`;
        })
        .join('');
      const legend = entries
        .map(([k, v]) => {
          const c = STRAT_COLOR[k] ?? '#8aa0b4';
          return (
            `<span style="margin-right:12px;white-space:nowrap">` +
            `<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${c};margin-right:5px"></span>` +
            `${t(k)} ${Math.round((v / total) * 100)}%</span>`
          );
        })
        .join('');
      mix.innerHTML =
        `<div style="opacity:.6;font-size:11px;margin-bottom:5px">${t('obs_strategy')}</div>` +
        `<div style="display:flex;height:14px;border-radius:4px;overflow:hidden">${bar}</div>` +
        `<div style="margin-top:7px;font-size:11px;opacity:.8">${legend}</div>`;
      body.append(mix);
    }
  }

  function miniCurve(samples: number[], color: string): HTMLCanvasElement {
    const c = document.createElement('canvas');
    c.style.verticalAlign = 'middle';
    const w = 96;
    const h = 26;
    lineChart(setupCanvas(c, w, h), w, h, [{ values: samples, color }]);
    return c;
  }

  function renderLineages(lineages: LineageHistory[]): void {
    const body = lineageCard.body;
    body.innerHTML = '';
    if (lineages.length === 0) {
      body.innerHTML = `<div style="opacity:.5">${t('sampling')}</div>`;
      return;
    }
    const rows = [...lineages].sort((a, b) => b.count - a.count).slice(0, 12);
    const table = document.createElement('div');
    table.style.cssText = 'display:flex;flex-direction:column;gap:8px';
    for (const r of rows) {
      const c = `hsl(${Math.round(r.hue * 360)}, 90%, 62%)`;
      const arrow = r.trend > 1 ? '▲' : r.trend < -1 ? '▼' : '—';
      const ac = r.trend > 1 ? ACCENT : r.trend < -1 ? '#ff5aa6' : 'rgba(207,232,255,.5)';
      const row = document.createElement('div');
      row.style.cssText =
        'display:flex;align-items:center;gap:12px;border-bottom:1px solid rgba(120,160,200,0.08);padding-bottom:8px';
      const left = document.createElement('div');
      left.style.cssText = 'flex:1;min-width:0';
      left.innerHTML =
        `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>` +
        `<b>#${r.lineage}</b> · ${r.count} <span style="color:${ac}">${arrow}</span></div>` +
        `<div style="opacity:.7;margin-left:16px">${t(r.descKey)} · ${t(r.fast ? 'fast' : 'slow')}</div>` +
        `<div style="opacity:.5;margin-left:16px;font-size:11px">` +
        `${t('tr_seek')} ${r.seek.toFixed(2)} · ${t('tr_forage')} ${r.forage.toFixed(2)} · ` +
        `${t('tr_cruise')} ${r.cruise.toFixed(2)} · ${t('tr_aggr')} ${r.aggression.toFixed(2)} · ` +
        `${t('neurons')} ${r.neurons}/10</div>`;
      row.append(left, miniCurve(r.samples, c));
      table.append(row);
    }
    body.append(table);
  }

  function renderWatched(watched: Watched[]): void {
    const body = watchCard.body;
    body.innerHTML = '';
    if (watched.length === 0) {
      body.innerHTML = `<div style="opacity:.5">${t('obs_watchHint')}</div>`;
      return;
    }
    const grid = document.createElement('div');
    grid.style.cssText = 'display:flex;gap:14px;flex-wrap:wrap';
    for (const wch of watched) {
      const cur = wch.history[wch.history.length - 1];
      const c = `hsl(${Math.round(wch.hue * 360)}, 90%, 62%)`;
      const card = document.createElement('div');
      card.style.cssText = `${CARD}width:260px`;
      const alive = cur?.alive ?? false;
      const head = document.createElement('div');
      head.style.cssText = 'display:flex;justify-content:space-between;align-items:center';
      head.innerHTML =
        `<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>` +
        `<b>#${wch.id}</b> · ${t('lineageWord')} #${wch.lineage}` +
        `${alive ? '' : ` · <span style="color:#ff5aa6">${t('deceased')}</span>`}</div>`;
      const rm = document.createElement('button');
      rm.textContent = '✕';
      rm.style.cssText =
        'background:none;border:none;color:#cfe8ff;cursor:pointer;font-size:15px;line-height:1;opacity:.7';
      rm.onclick = () => onRemoveWatch(wch.id);
      head.append(rm);
      card.append(head);

      const stat = document.createElement('div');
      stat.style.cssText = 'margin:6px 0;opacity:.85';
      if (cur) {
        const turnTxt =
          cur.turn > 0.1 ? t('turnRight') : cur.turn < -0.1 ? t('turnLeft') : t('straight');
        stat.innerHTML =
          `${t('energyWord')} ${cur.energy.toFixed(1)} · ${t('speedWord')} ${cur.speed.toFixed(1)}<br>` +
          `${t('decision')}: ${turnTxt} · ${t('out_thrust')} ${(cur.thrust * 100).toFixed(0)}%<br>` +
          `<span style="opacity:.6;font-size:11px">${t('obs_age')} ${wch.history.length} ${t('obs_ticks')}</span>`;
      }
      card.append(stat);

      const lbl = document.createElement('div');
      lbl.style.cssText = 'font-size:11px;opacity:.6;margin-top:4px';
      lbl.innerHTML = `${t('obs_energyLine')} <span style="color:${ACCENT}">▬</span> · ${t('obs_speedLine')} <span style="color:#9b8cff">▬</span>`;
      const cv = document.createElement('canvas');
      const cw = 228;
      const chh = 64;
      lineChart(setupCanvas(cv, cw, chh), cw, chh, [
        { values: wch.history.map((s) => s.energy), color: ACCENT },
        { values: wch.history.map((s) => s.speed), color: '#9b8cff' },
      ]);
      card.append(lbl, cv);
      grid.append(card);
    }
    body.append(grid);
  }

  function render(): void {
    if (!open) return;
    h1.textContent = `PELAGIA · ${t('observatory')}`;
    worldCard.title.textContent = t('obs_world');
    lineageCard.title.textContent = t('obs_lineages');
    watchCard.title.textContent = t('obs_watched');
    renderWorld(last.world);
    renderLineages(last.lineages);
    renderWatched(last.watched);
  }

  function relabelToggle(): void {
    toggle.textContent = '📊 ' + t('observatory');
  }
  relabelToggle();
  onLang(() => {
    relabelToggle();
    render();
  });

  return {
    panel,
    toggle,
    isOpen: () => open,
    update(data) {
      last = data;
      render();
    },
  };
}

export interface HistoryPanel {
  panel: HTMLElement;
  toggle: HTMLButtonElement;
  isOpen(): boolean;
  update(data: ObservatoryData): void;
}

/**
 * Evolutionary history as a Muller plot: each clade is a coloured band stacked
 * over time, its thickness = its population. Clades visibly rise, take over and
 * go extinct — the evolutionary story drawn from the per-clade population curves
 * the sampler already records (all on one shared time axis).
 */
export function buildEvolutionHistory(): HistoryPanel {
  let open = false;
  let last: LineageHistory[] = [];

  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;inset:0;display:none;z-index:31;overflow:auto;' +
    'background:rgba(2,4,10,0.92);font:13px ui-monospace,SFMono-Regular,Menlo,monospace;color:#cfe8ff;';
  const inner = document.createElement('div');
  inner.style.cssText = 'max-width:1100px;margin:0 auto;padding:22px 22px 60px;';
  panel.append(inner);

  const header = document.createElement('div');
  header.style.cssText =
    'display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;';
  const h1 = document.createElement('div');
  h1.style.cssText = `font-size:20px;font-weight:600;letter-spacing:.12em;color:${ACCENT};`;
  const closeBtn = document.createElement('button');
  closeBtn.textContent = '✕';
  closeBtn.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  closeBtn.onclick = () => setOpen(false);
  header.append(h1, closeBtn);

  const note = document.createElement('div');
  note.style.cssText = 'opacity:.7;margin-bottom:10px;line-height:1.5;';
  const card = document.createElement('div');
  card.style.cssText = CARD;
  const cw = 1040;
  const ch = 340;
  const chart = document.createElement('canvas');
  chart.style.cssText = `display:block;width:100%;max-width:${cw}px;height:${ch}px;`;
  const timeLbl = document.createElement('div');
  timeLbl.style.cssText = 'opacity:.5;font-size:11px;margin-top:4px;';
  const legend = document.createElement('div');
  legend.style.cssText = 'margin-top:12px;display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;';
  card.append(chart, timeLbl, legend);
  inner.append(header, note, card);

  const toggle = document.createElement('button');
  toggle.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  toggle.onclick = () => setOpen(!open);

  function setOpen(v: boolean): void {
    open = v;
    panel.style.display = v ? 'block' : 'none';
    if (v) render();
  }
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape' && open) setOpen(false);
  });

  function valAt(l: LineageHistory, c: number, T: number): number {
    const idx = l.samples.length - (T - c); // align series to the newest sample
    return idx >= 0 ? l.samples[idx]! : 0;
  }

  function drawMuller(lineages: LineageHistory[]): void {
    const ctx = setupCanvas(chart, cw, ch);
    ctx.clearRect(0, 0, cw, ch);
    const lins = lineages.filter((l) => l.samples.length > 0).sort((a, b) => a.lineage - b.lineage);
    if (lins.length === 0) {
      ctx.fillStyle = 'rgba(207,232,255,0.5)';
      ctx.font = '13px ui-monospace, monospace';
      ctx.fillText(t('ph_empty'), 8, 20);
      return;
    }
    let T = 0;
    for (const l of lins) T = Math.max(T, l.samples.length);
    let maxTot = 1;
    for (let c = 0; c < T; c++) {
      let s = 0;
      for (const l of lins) s += valAt(l, c, T);
      if (s > maxTot) maxTot = s;
    }
    const colW = cw / T;
    for (let c = 0; c < T; c++) {
      let y = ch;
      for (const l of lins) {
        const v = valAt(l, c, T);
        if (v <= 0) continue;
        const h = (v / maxTot) * (ch - 2);
        ctx.fillStyle = `hsl(${Math.round(l.hue * 360)}, 85%, 58%)`;
        ctx.fillRect(c * colW, y - h, Math.ceil(colW) + 0.6, h + 0.6);
        y -= h;
      }
    }
  }

  function renderLegend(lineages: LineageHistory[]): void {
    const top = [...lineages]
      .filter((l) => l.count > 0)
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
    legend.innerHTML = top
      .map((l) => {
        const c = `hsl(${Math.round(l.hue * 360)}, 85%, 58%)`;
        return (
          `<span style="white-space:nowrap"><span style="display:inline-block;width:10px;height:10px;` +
          `border-radius:2px;background:${c};margin-right:5px"></span>#${l.lineage} · ${l.count} · ${t(l.descKey)}</span>`
        );
      })
      .join('');
  }

  function render(): void {
    if (!open) return;
    h1.textContent = `PELAGIA · ${t('ph_title')}`;
    note.textContent = t('ph_note');
    timeLbl.textContent = t('ph_time');
    drawMuller(last);
    renderLegend(last);
  }

  function relabelToggle(): void {
    toggle.textContent = '🌳 ' + t('ph_history');
  }
  relabelToggle();
  onLang(() => {
    relabelToggle();
    render();
  });

  return {
    panel,
    toggle,
    isOpen: () => open,
    update(data) {
      last = data.lineages;
      render();
    },
  };
}
