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
import { icon } from './icons.js';
import { pcgHash, floatFromU32 } from '../core/rng.js';
import { lineageLabelHtml, displayLineage, onLineageNamesChange } from './lineageNames.js';
import { openLineageRename } from './lineageRename.js';

/** Lineage colour from its id — matches the GPU's per-creature hue. */
const hueOf = (id: number): number => floatFromU32(pcgHash(id));

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
  /** Tick when this clade was first sampled (≈ its birth) — for the time axis. */
  birthTick: number;
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
  /** new lineage id -> parent lineage id (for the family tree / cladogram). */
  parents: Map<number, number>;
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
  let last: ObservatoryData = { world: [], lineages: [], watched: [], parents: new Map() };

  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;inset:0;display:none;z-index:1001;overflow:auto;' +
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
  closeBtn.className = 'pg-btn pg-iconbtn';
  closeBtn.innerHTML = icon('close');
  closeBtn.title = t('close');
  closeBtn.onclick = () => setOpen(false);
  header.append(h1, closeBtn);
  inner.append(header);

  const worldCard = mkCard('obs_world');
  const lineageCard = mkCard('obs_lineages');
  const watchCard = mkCard('obs_watched');
  inner.append(worldCard.card, lineageCard.card, watchCard.card);
  lineageCard.card.style.marginTop = '16px';
  watchCard.card.style.marginTop = '16px';

  // Delegated click-to-rename: a clade label/row in either card opens the popover.
  const renameOnClick = (e: MouseEvent): void => {
    const tgt = (e.target as HTMLElement).closest<HTMLElement>('[data-rename]');
    if (tgt?.dataset.rename) openLineageRename(+tgt.dataset.rename, e.clientX, e.clientY);
  };
  lineageCard.body.addEventListener('click', renameOnClick);
  watchCard.body.addEventListener('click', renameOnClick);

  const toggle = document.createElement('button');
  toggle.className = 'pg-btn';
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
        : ` ${cur.daylight > 0.6 ? t('nar_day') : cur.daylight < 0.4 ? t('nar_night') : t('nar_dusk')}.`;
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
        `<div data-rename="${r.lineage}" title="${t('nameLineage')}" style="cursor:pointer">` +
        `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${c};margin-right:6px"></span>` +
        `${lineageLabelHtml(r.lineage)} · ${r.count} <span style="color:${ac}">${arrow}</span>` +
        `<span style="opacity:.35;margin-left:6px;vertical-align:-1px;display:inline-block">${icon('pencil', 11)}</span></div>` +
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
        `<b>#${wch.id}</b> · ${t('lineageWord')} <span data-rename="${wch.lineage}" title="${t('nameLineage')}" style="cursor:pointer">${lineageLabelHtml(wch.lineage)}</span>` +
        `${alive ? '' : ` · <span style="color:#ff5aa6">${t('deceased')}</span>`}</div>`;
      const rm = document.createElement('button');
      rm.innerHTML = icon('close', 15);
      rm.title = t('close');
      rm.style.cssText =
        'background:none;border:none;color:var(--ink-dim);cursor:pointer;line-height:0;padding:2px;';
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
    toggle.innerHTML = icon('activity', 16) + `<span>${t('observatory')}</span>`;
    closeBtn.title = t('close');
  }
  relabelToggle();
  onLang(() => {
    relabelToggle();
    render();
  });
  onLineageNamesChange(render); // a rename re-labels the lineage rows/watch-list

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
  let lastParents = new Map<number, number>();
  let lastTick = 0;

  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;inset:0;display:none;z-index:1002;overflow:auto;' +
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
  closeBtn.className = 'pg-btn pg-iconbtn';
  closeBtn.innerHTML = icon('close');
  closeBtn.title = t('close');
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

  // Family tree (cladogram): who descends from whom, once speciation has branched.
  // Laid out left→right by birth order with a tick axis, in a horizontal scroller
  // so deep histories spread out instead of crowding.
  const treeCard = document.createElement('div');
  treeCard.style.cssText = `${CARD}margin-top:16px;`;
  const treeLbl = document.createElement('div');
  treeLbl.style.cssText = 'opacity:.7;font-size:12px;margin-bottom:8px;';
  const tch = 360;
  const treeScroll = document.createElement('div');
  treeScroll.style.cssText = 'overflow-x:auto;overflow-y:hidden;';
  const tree = document.createElement('canvas');
  tree.style.cssText = `display:block;height:${tch}px;cursor:pointer;`;
  treeScroll.append(tree);
  treeCard.append(treeLbl, treeScroll);
  inner.append(header, note, card, treeCard);

  // Hit-boxes (CSS px within the tree canvas) for click-to-rename.
  let treeHits: { id: number; x: number; y: number; r: number }[] = [];
  tree.addEventListener('click', (e) => {
    const rect = tree.getBoundingClientRect();
    const lx = e.clientX - rect.left;
    const ly = e.clientY - rect.top;
    let best = -1;
    let bestD = 16 * 16;
    for (const h of treeHits) {
      const d = (h.x - lx) ** 2 + (h.y - ly) ** 2;
      if (d < bestD && d < (h.r + 8) ** 2) {
        bestD = d;
        best = h.id;
      }
    }
    if (best >= 0) openLineageRename(best, e.clientX, e.clientY);
  });
  // Muller legend: click a clade to name it.
  legend.addEventListener('click', (e) => {
    const tgt = (e.target as HTMLElement).closest<HTMLElement>('[data-rename]');
    if (tgt?.dataset.rename) openLineageRename(+tgt.dataset.rename, e.clientX, e.clientY);
  });

  const toggle = document.createElement('button');
  toggle.className = 'pg-btn';
  toggle.onclick = () => setOpen(!open);

  function setOpen(v: boolean): void {
    open = v;
    panel.style.display = v ? 'block' : 'none';
    if (v) {
      render();
      treeScroll.scrollLeft = 0; // start at the roots; scroll right toward "now"
    }
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
          `<span data-rename="${l.lineage}" title="${t('nameLineage')}" style="white-space:nowrap;cursor:pointer">` +
          `<span style="display:inline-block;width:10px;height:10px;` +
          `border-radius:2px;background:${c};margin-right:5px"></span>${displayLineage(l.lineage)} · ${l.count} · ${t(l.descKey)}</span>`
        );
      })
      .join('');
  }

  interface TNode {
    id: number;
    hue: number;
    count: number;
    parent: number;
    children: TNode[];
    birth: number; // birth tick (or estimated for history-less ancestors)
    col: number; // column index by birth order
    y: number; // tidy y slot
  }

  const fmtTick = (tk: number): string =>
    tk >= 1000 ? `${(tk / 1000).toFixed(1)}k` : `${Math.round(tk)}`;

  // Cladogram: top clades + their ancestors (via parent pointers), laid out
  // left→right by BIRTH ORDER (one column per clade, evenly spaced so nothing
  // crowds) with the real birth tick under each column and a horizontal scroll.
  // y is a tidy layout (leaves on their own row, parents centred on their kids).
  function drawTree(
    lineages: LineageHistory[],
    parents: Map<number, number>,
    nowTick: number,
  ): void {
    const byId = new Map(lineages.map((l) => [l.lineage, l]));
    const nodes = new Map<number, TNode>();
    const add = (id: number): TNode => {
      let nd = nodes.get(id);
      if (!nd) {
        const l = byId.get(id);
        nd = {
          id,
          hue: l ? l.hue : hueOf(id),
          count: l ? l.count : 0,
          parent: parents.get(id) ?? -1,
          children: [],
          birth: l ? l.birthTick : -1,
          col: 0,
          y: 0,
        };
        nodes.set(id, nd);
      }
      return nd;
    };
    const top = [...lineages]
      .filter((l) => l.count > 0)
      .sort((a, b) => b.count - a.count)
      .slice(0, 16);
    for (const l of top) {
      add(l.lineage);
      let id = l.lineage;
      let guard = 0;
      while (parents.has(id) && guard++ < 24) {
        const p = parents.get(id)!;
        add(p);
        id = p;
      }
    }
    for (const nd of nodes.values()) {
      if (nd.parent >= 0 && nodes.has(nd.parent)) nodes.get(nd.parent)!.children.push(nd);
    }
    if (nodes.size === 0) {
      const ctx0 = setupCanvas(tree, cw, tch);
      ctx0.clearRect(0, 0, cw, tch);
      ctx0.fillStyle = 'rgba(207,232,255,0.5)';
      ctx0.font = '13px ui-monospace, monospace';
      ctx0.fillText(t('ph_empty'), 8, 20);
      treeHits = [];
      return;
    }
    const roots = [...nodes.values()].filter((n) => n.parent < 0 || !nodes.has(n.parent));
    // Fill in missing birth ticks (a history-less ancestor is older than its kids).
    const estimate = (nd: TNode): number => {
      if (nd.birth >= 0) return nd.birth;
      let m = nowTick;
      for (const c of nd.children) m = Math.min(m, estimate(c));
      nd.birth = Math.max(0, m - 1);
      return nd.birth;
    };
    roots.forEach(estimate);

    // x = column by birth order (even spacing); real ticks go on the axis.
    const ordered = [...nodes.values()].sort((a, b) => a.birth - b.birth || a.id - b.id);
    ordered.forEach((nd, i) => (nd.col = i));
    const cols = ordered.length;

    // y = tidy layout: leaves take sequential rows; parents centre on their kids.
    let leaf = 0;
    const layoutY = (nd: TNode): void => {
      if (nd.children.length === 0) {
        nd.y = leaf++;
      } else {
        nd.children.sort((a, b) => a.birth - b.birth || a.id - b.id);
        nd.children.forEach(layoutY);
        nd.y = nd.children.reduce((s, c) => s + c.y, 0) / nd.children.length;
      }
    };
    roots.sort((a, b) => a.birth - b.birth || a.id - b.id);
    roots.forEach(layoutY);
    const leaves = Math.max(1, leaf - 1);

    const COL_W = 108;
    const mxL = 60;
    const mxR = 78;
    const myT = 40;
    const myB = 34;
    const contentW = Math.max(cw, mxL + mxR + (cols - 1) * COL_W);
    const ctx = setupCanvas(tree, contentW, tch);
    ctx.clearRect(0, 0, contentW, tch);
    const px = (col: number): number =>
      cols <= 1 ? contentW / 2 : mxL + col * ((contentW - mxL - mxR) / (cols - 1));
    const py = (y: number): number => myT + (y / leaves) * (tch - myT - myB);

    // bottom tick axis: a faint vertical guide + the birth tick under each column
    ctx.font = '9px ui-monospace, monospace';
    ctx.textBaseline = 'alphabetic';
    let lastLabel = '';
    for (const nd of ordered) {
      const x = px(nd.col);
      ctx.strokeStyle = 'rgba(120,160,200,0.06)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, myT - 12);
      ctx.lineTo(x, tch - myB + 4);
      ctx.stroke();
      // only label when the tick changes, so a run of same-birth founders reads cleanly
      const lbl = fmtTick(nd.birth);
      if (lbl !== lastLabel) {
        ctx.fillStyle = 'rgba(207,232,255,0.4)';
        ctx.textAlign = 'center';
        ctx.fillText(lbl, x, tch - myB + 18);
        lastLabel = lbl;
      }
    }
    ctx.fillStyle = 'rgba(207,232,255,0.55)';
    ctx.textAlign = 'left';
    ctx.fillText(t('tickWord'), 6, tch - myB + 18);
    if (nowTick > 0) {
      ctx.textAlign = 'right';
      ctx.fillText(`${t('now')} · ${fmtTick(nowTick)}`, contentW - 6, tch - myB + 18);
    }

    // edges: smooth left→right curves tinted parent-hue → child-hue
    ctx.lineWidth = 1.7;
    for (const nd of nodes.values()) {
      for (const c of nd.children) {
        const x0 = px(nd.col);
        const y0 = py(nd.y);
        const x1 = px(c.col);
        const y1 = py(c.y);
        const mid = (x0 + x1) / 2;
        const grad = ctx.createLinearGradient(x0, 0, x1, 0);
        grad.addColorStop(0, `hsla(${Math.round(nd.hue * 360)},80%,60%,0.22)`);
        grad.addColorStop(1, `hsla(${Math.round(c.hue * 360)},80%,62%,0.6)`);
        ctx.strokeStyle = grad;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.bezierCurveTo(mid, y0, mid, y1, x1, y1);
        ctx.stroke();
      }
    }

    // nodes: glowing discs + name labels above; record hit-boxes for rename
    treeHits = [];
    ctx.textBaseline = 'middle';
    for (const nd of nodes.values()) {
      const x = px(nd.col);
      const y = py(nd.y);
      const r = 4 + Math.min(13, Math.sqrt(nd.count));
      const col = `hsl(${Math.round(nd.hue * 360)}, 85%, 60%)`;
      ctx.save();
      ctx.shadowColor = col;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.restore();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.2)';
      ctx.stroke();
      const name = displayLineage(nd.id);
      const label = name.length > 14 ? name.slice(0, 13) + '…' : name;
      ctx.font = '10.5px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = nd.count > 0 ? 'rgba(207,232,255,0.85)' : 'rgba(207,232,255,0.45)';
      ctx.fillText(label, x, y - r - 9);
      treeHits.push({ id: nd.id, x, y, r });
    }
  }

  function render(): void {
    if (!open) return;
    h1.textContent = `PELAGIA · ${t('ph_title')}`;
    note.textContent = t('ph_note');
    timeLbl.textContent = t('ph_time');
    treeLbl.textContent = `${t('ph_tree')} — ${t('hist_births')}`;
    drawMuller(last);
    renderLegend(last);
    drawTree(last, lastParents, lastTick);
  }

  function relabelToggle(): void {
    toggle.innerHTML = icon('branch', 16) + `<span>${t('ph_history')}</span>`;
    closeBtn.title = t('close');
  }
  relabelToggle();
  onLang(() => {
    relabelToggle();
    render();
  });
  onLineageNamesChange(render); // a rename re-labels the Muller legend

  return {
    panel,
    toggle,
    isOpen: () => open,
    update(data) {
      last = data.lineages;
      lastParents = data.parents;
      lastTick = data.world[data.world.length - 1]?.tick ?? lastTick;
      render();
    },
  };
}
