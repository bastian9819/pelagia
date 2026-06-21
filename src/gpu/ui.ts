/**
 * Live stats HUD (population + sparkline + fps) and the bottom transport bar
 * (play/step/speed/fit/colour + a menu of panels). Text is localised via i18n.
 */
import { t, onLang, toggleLang, getLang } from './i18n.js';
import { icon, type IconName } from './icons.js';
import { attachTooltip } from './tooltip.js';

const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16];
const DEFAULT_SPEED_IDX = 3; // 1x

export interface OceanUi {
  panel: HTMLElement;
  /** Bottom transport bar (play/step/speed/fit/colour + the menu button). */
  controls: HTMLElement;
  /** Pop-up menu that holds panel toggles and one-shot actions (decluttered). */
  menu: HTMLElement;
  /** Add a panel toggle / action button into the pop-up menu (styled as a row). */
  addTool(btn: HTMLButtonElement): void;
  update(alive: number, fps: number, frame: number): void;
  readonly paused: boolean;
  /** Simulation ticks per rendered frame (may be < 1 for slow motion). */
  readonly speed: number;
}

/** How the bottom bar's colour-by-trait button cycles + labels itself. */
export interface ColorControl {
  cycle(): void;
  label(): string;
}

/** A text (optionally icon-led) button styled with the design system. */
export function mkBtn(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.className = 'pg-btn';
  b.textContent = label;
  b.onclick = onClick;
  return b;
}

/** An icon-only button (square). `title` doubles as the accessible label. */
export function mkIconBtn(name: IconName, title: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.className = 'pg-btn pg-iconbtn';
  b.innerHTML = icon(name);
  b.title = title;
  b.setAttribute('aria-label', title);
  b.onclick = onClick;
  return b;
}

/** Set an icon + text label on a button (used by menu rows that re-localise). */
export function setBtnIcon(btn: HTMLButtonElement, name: IconName, label: string): void {
  btn.innerHTML = icon(name, 16) + `<span>${label}</span>`;
}

// Floating panels: each can be open at once and dragged anywhere, so they coexist
// instead of fighting for one slot. Clicking/dragging a panel raises it above the
// others (overlays like the menu/help/observatory sit far higher, in the thousands).
let panelZ = 10;
export function bringToFront(panel: HTMLElement): void {
  panelZ += 1;
  panel.style.zIndex = String(panelZ);
}

/** Make `panel` draggable by `handle` (its header), kept within the viewport. */
export function makeDraggable(panel: HTMLElement, handle: HTMLElement): void {
  handle.style.cursor = 'move';
  handle.style.touchAction = 'none';
  let dragging = false;
  let sx = 0;
  let sy = 0;
  let ox = 0;
  let oy = 0;
  panel.addEventListener('pointerdown', () => bringToFront(panel), true);
  handle.addEventListener('pointerdown', (e) => {
    if ((e.target as HTMLElement).closest('button')) return; // let header buttons work
    dragging = true;
    bringToFront(panel);
    const r = panel.getBoundingClientRect();
    panel.style.left = `${r.left}px`;
    panel.style.top = `${r.top}px`;
    panel.style.right = 'auto';
    panel.style.bottom = 'auto';
    sx = e.clientX;
    sy = e.clientY;
    ox = r.left;
    oy = r.top;
    try {
      handle.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    e.preventDefault();
  });
  handle.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const nx = Math.max(4, Math.min(window.innerWidth - 60, ox + (e.clientX - sx)));
    const ny = Math.max(4, Math.min(window.innerHeight - 40, oy + (e.clientY - sy)));
    panel.style.left = `${nx}px`;
    panel.style.top = `${ny}px`;
  });
  const end = (e: PointerEvent): void => {
    dragging = false;
    try {
      handle.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };
  handle.addEventListener('pointerup', end);
  handle.addEventListener('pointercancel', end);
}

/**
 * Make `panel` resizable via a grip in its bottom-right corner. The panel is a
 * flex column (fixed header + a `flex:1; overflow:auto` body), so setting an
 * explicit width/height just resizes the scrolling body. Clamped to [min, viewport].
 */
export function makeResizable(
  panel: HTMLElement,
  opts: { minW?: number; minH?: number } = {},
): void {
  const minW = opts.minW ?? 200;
  const minH = opts.minH ?? 140;
  const grip = document.createElement('div');
  grip.innerHTML = icon('grip', 16);
  grip.style.cssText =
    'position:absolute;right:2px;bottom:2px;width:16px;height:16px;cursor:nwse-resize;' +
    'color:var(--ink-faint);z-index:3;touch-action:none;line-height:0;';
  panel.appendChild(grip);

  let resizing = false;
  let sx = 0;
  let sy = 0;
  let sw = 0;
  let sh = 0;
  let left = 0;
  let top = 0;
  grip.addEventListener('pointerdown', (e) => {
    resizing = true;
    bringToFront(panel);
    const r = panel.getBoundingClientRect();
    // Anchor by left/top (clear right/bottom + the max-* caps) so it grows predictably.
    panel.style.left = `${r.left}px`;
    panel.style.top = `${r.top}px`;
    panel.style.right = 'auto';
    panel.style.bottom = 'auto';
    panel.style.maxHeight = 'none';
    panel.style.maxWidth = 'none';
    left = r.left;
    top = r.top;
    sx = e.clientX;
    sy = e.clientY;
    sw = r.width;
    sh = r.height;
    try {
      grip.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    e.preventDefault();
    e.stopPropagation();
  });
  grip.addEventListener('pointermove', (e) => {
    if (!resizing) return;
    const maxW = window.innerWidth - left - 8;
    const maxH = window.innerHeight - top - 8;
    panel.style.width = `${Math.max(minW, Math.min(maxW, sw + (e.clientX - sx)))}px`;
    panel.style.height = `${Math.max(minH, Math.min(maxH, sh + (e.clientY - sy)))}px`;
  });
  const end = (e: PointerEvent): void => {
    resizing = false;
    try {
      grip.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };
  grip.addEventListener('pointerup', end);
  grip.addEventListener('pointercancel', end);
}

/**
 * A draggable panel header: an eyebrow title on the left and a close (×) button on
 * the right. Returns the header (the drag handle) + the title element to fill in.
 */
export function mkPanelHeader(onClose: () => void): { header: HTMLElement; title: HTMLElement } {
  const header = document.createElement('div');
  header.style.cssText =
    'display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:11px;flex:none;';
  const title = document.createElement('div');
  title.className = 'pg-eyebrow';
  const x = mkIconBtn('close', t('close'), onClose);
  x.style.cssText += 'width:26px;height:26px;';
  header.append(title, x);
  return { header, title };
}

export function buildUi(onFit: () => void, onStep: () => void, color: ColorControl): OceanUi {
  let paused = false;
  let speedIdx = DEFAULT_SPEED_IDX;
  const aliveHistory: number[] = [];

  const panel = document.createElement('div');
  panel.className = 'pg-panel';
  panel.style.cssText =
    'position:fixed;top:14px;left:14px;padding:13px 15px;min-width:196px;user-select:none;';

  const title = document.createElement('div');
  title.textContent = 'PELAGIA';
  title.style.cssText =
    'font-weight:600;letter-spacing:.22em;font-size:12px;color:var(--glow-cyan);';
  const aliveEl = document.createElement('div');
  aliveEl.textContent = '—';
  aliveEl.style.cssText =
    'font:600 26px var(--font-mono);margin-top:8px;letter-spacing:.01em;line-height:1;';
  const aliveLabel = document.createElement('div');
  aliveLabel.style.cssText =
    'color:var(--ink-faint);font-size:11px;margin-top:3px;letter-spacing:.04em;';
  const spark = document.createElement('canvas');
  spark.width = 392;
  spark.height = 76;
  spark.style.cssText = 'display:block;margin-top:12px;width:196px;height:38px;';
  const fpsEl = document.createElement('div');
  fpsEl.textContent = '—';
  fpsEl.style.cssText = 'color:var(--ink-dim);font:11px var(--font-mono);margin-top:9px;';
  const hint = document.createElement('div');
  hint.style.cssText =
    'color:var(--ink-faint);font-size:11px;margin-top:11px;line-height:1.55;white-space:pre-line;';
  panel.append(title, aliveEl, aliveLabel, spark, fpsEl, hint);
  attachTooltip(aliveEl, 'alive');
  attachTooltip(aliveLabel, 'alive');
  attachTooltip(fpsEl, 'tick');
  const sctx = spark.getContext('2d')!;

  const controls = document.createElement('div');
  controls.className = 'pg-panel';
  controls.style.cssText =
    'position:fixed;bottom:18px;left:50%;transform:translateX(-50%);display:flex;gap:6px;' +
    'align-items:center;padding:7px;border-radius:14px;max-width:96vw;';

  const pauseBtn = mkIconBtn('pause', t('pause'), () => {
    paused = !paused;
    pauseBtn.innerHTML = icon(paused ? 'play' : 'pause');
  });
  const speedBtn = mkBtn(`${SPEEDS[speedIdx]!}×`, () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    speedBtn.textContent = `${SPEEDS[speedIdx]}×`;
  });
  speedBtn.style.minWidth = '46px';
  speedBtn.style.fontFamily = 'var(--font-mono)';
  // Step: pause, then advance exactly one tick — for studying a decision frame.
  const stepBtn = mkIconBtn('step', t('step'), () => {
    if (!paused) {
      paused = true;
      pauseBtn.innerHTML = icon('play');
    }
    onStep();
  });
  const fitBtn = mkIconBtn('fit', t('fit'), onFit);
  const colorBtn = mkBtn('', () => {
    color.cycle();
    setBtnIcon(colorBtn, 'palette', color.label());
  });

  // Pop-up menu: everything that isn't constant transport lives here, so the bar
  // stays a single tidy row no matter how many tools we add.
  const menu = document.createElement('div');
  menu.className = 'pg-panel';
  menu.style.cssText =
    'position:fixed;bottom:70px;left:50%;transform:translateX(-50%);display:none;' +
    'flex-direction:column;gap:4px;width:230px;max-height:72vh;overflow:auto;padding:8px;z-index:900;';
  const menuBtn = mkIconBtn('menu', t('menu'), () => {
    menu.style.display = menu.style.display === 'none' ? 'flex' : 'none';
  });
  menu.addEventListener('click', () => (menu.style.display = 'none')); // close after a pick
  window.addEventListener('pointerdown', (e) => {
    const target = e.target as Node;
    if (menu.style.display !== 'none' && !menu.contains(target) && !menuBtn.contains(target)) {
      menu.style.display = 'none';
    }
  });

  // Thin separators group the bar: [play step] | [speed] | [fit colour] | [menu].
  const sep = (): HTMLElement => {
    const s = document.createElement('span');
    s.style.cssText = 'width:1px;height:20px;background:var(--border-1);margin:0 2px;';
    return s;
  };
  controls.append(pauseBtn, stepBtn, sep(), speedBtn, sep(), fitBtn, colorBtn, sep(), menuBtn);
  attachTooltip(pauseBtn, 'pause');
  attachTooltip(stepBtn, 'step');
  attachTooltip(speedBtn, 'speed');
  attachTooltip(fitBtn, 'fit');
  attachTooltip(colorBtn, 'color');
  attachTooltip(menuBtn, 'menu');

  const langBtn = mkBtn(getLang().toUpperCase(), () => toggleLang());
  function addTool(btn: HTMLButtonElement): void {
    btn.classList.add('pg-row');
    menu.append(btn);
  }
  addTool(langBtn);

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      pauseBtn.click();
    }
  });

  function relabel(): void {
    aliveLabel.textContent = t('creaturesAlive');
    hint.textContent = t('hint');
    langBtn.textContent = getLang().toUpperCase();
    setBtnIcon(colorBtn, 'palette', color.label());
    // Custom tooltips handle hover text; keep aria-labels for a11y (no native title).
    pauseBtn.setAttribute('aria-label', t('pause'));
    stepBtn.setAttribute('aria-label', t('step'));
    fitBtn.setAttribute('aria-label', t('fit'));
    menuBtn.setAttribute('aria-label', t('menu'));
  }
  relabel();
  onLang(relabel);

  function drawSpark(): void {
    const w = spark.width;
    const h = spark.height;
    sctx.clearRect(0, 0, w, h);
    if (aliveHistory.length < 2) return;
    let max = 1;
    for (const v of aliveHistory) if (v > max) max = v;
    // Soft area fill under the line for a more finished look.
    sctx.beginPath();
    for (let i = 0; i < aliveHistory.length; i++) {
      const x = (i / (aliveHistory.length - 1)) * w;
      const y = h - (aliveHistory[i]! / max) * (h - 4) - 2;
      if (i === 0) sctx.moveTo(x, y);
      else sctx.lineTo(x, y);
    }
    sctx.lineTo(w, h);
    sctx.lineTo(0, h);
    sctx.closePath();
    sctx.fillStyle = 'rgba(63,240,216,0.10)';
    sctx.fill();
    sctx.strokeStyle = 'rgba(63,240,216,0.9)';
    sctx.lineWidth = 2;
    sctx.beginPath();
    for (let i = 0; i < aliveHistory.length; i++) {
      const x = (i / (aliveHistory.length - 1)) * w;
      const y = h - (aliveHistory[i]! / max) * (h - 4) - 2;
      if (i === 0) sctx.moveTo(x, y);
      else sctx.lineTo(x, y);
    }
    sctx.stroke();
  }

  return {
    panel,
    controls,
    menu,
    addTool,
    update(alive, fps, frame) {
      aliveEl.textContent = alive.toLocaleString();
      fpsEl.textContent = `${fps.toFixed(0)} fps · tick ${frame.toLocaleString()}`;
      aliveHistory.push(alive);
      if (aliveHistory.length > 196) aliveHistory.shift();
      drawSpark();
    },
    get paused() {
      return paused;
    },
    get speed() {
      return SPEEDS[speedIdx]!;
    },
  };
}
