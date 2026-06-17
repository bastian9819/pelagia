/**
 * Live stats panel (population + sparkline + fps) and control bar (pause, speed,
 * fit, language). Text is localised via i18n.
 */
import { t, onLang, toggleLang, getLang } from './i18n.js';

const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4];
const DEFAULT_SPEED_IDX = 3; // 1x

export interface OceanUi {
  panel: HTMLElement;
  controls: HTMLElement;
  update(alive: number, fps: number, frame: number): void;
  readonly paused: boolean;
  /** Simulation ticks per rendered frame (may be < 1 for slow motion). */
  readonly speed: number;
}

export function mkBtn(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.style.cssText =
    'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
    'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
  b.onclick = onClick;
  return b;
}

export function buildUi(onFit: () => void, onStep: () => void): OceanUi {
  let paused = false;
  let speedIdx = DEFAULT_SPEED_IDX;
  const aliveHistory: number[] = [];

  const panel = document.createElement('div');
  panel.style.cssText =
    'position:fixed;top:12px;left:12px;padding:12px 14px;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;' +
    'color:#cfe8ff;background:rgba(2,4,10,0.62);border:1px solid rgba(63,240,216,0.18);border-radius:10px;' +
    'min-width:190px;user-select:none;';

  const title = document.createElement('div');
  title.textContent = 'PELAGIA';
  title.style.cssText = 'font-weight:600;letter-spacing:.14em;color:#3ff0d8;';
  const aliveEl = document.createElement('div');
  aliveEl.textContent = '—';
  aliveEl.style.cssText = 'font-size:24px;margin-top:6px;';
  const aliveLabel = document.createElement('div');
  aliveLabel.style.cssText = 'opacity:.55;font-size:11px;';
  const spark = document.createElement('canvas');
  spark.width = 190;
  spark.height = 38;
  spark.style.cssText = 'display:block;margin-top:10px;width:190px;height:38px;';
  const fpsEl = document.createElement('div');
  fpsEl.textContent = '—';
  fpsEl.style.cssText = 'opacity:.65;margin-top:8px;';
  const hint = document.createElement('div');
  hint.style.cssText =
    'opacity:.45;font-size:11px;margin-top:10px;line-height:1.5;white-space:pre-line;';
  panel.append(title, aliveEl, aliveLabel, spark, fpsEl, hint);
  const sctx = spark.getContext('2d')!;

  const controls = document.createElement('div');
  controls.style.cssText =
    'position:fixed;bottom:18px;left:50%;transform:translateX(-50%);display:flex;gap:8px;flex-wrap:wrap;' +
    'justify-content:center;max-width:96vw;font:13px ui-monospace,monospace;';
  const pauseBtn = mkBtn('⏸', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? '▶' : '⏸';
  });
  const speedBtn = mkBtn(`${SPEEDS[speedIdx]!}×`, () => {
    speedIdx = (speedIdx + 1) % SPEEDS.length;
    speedBtn.textContent = `${SPEEDS[speedIdx]}×`;
  });
  // Step: pause, then advance exactly one tick — for studying a decision frame.
  const stepBtn = mkBtn('⏭', () => {
    if (!paused) {
      paused = true;
      pauseBtn.textContent = '▶';
    }
    onStep();
  });
  const langBtn = mkBtn(getLang().toUpperCase(), () => toggleLang());
  controls.append(pauseBtn, speedBtn, stepBtn, mkBtn('⤢ fit', onFit), langBtn);

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
    sctx.strokeStyle = 'rgba(63,240,216,0.85)';
    sctx.lineWidth = 1.5;
    sctx.beginPath();
    for (let i = 0; i < aliveHistory.length; i++) {
      const x = (i / (aliveHistory.length - 1)) * w;
      const y = h - (aliveHistory[i]! / max) * (h - 2) - 1;
      if (i === 0) sctx.moveTo(x, y);
      else sctx.lineTo(x, y);
    }
    sctx.stroke();
  }

  return {
    panel,
    controls,
    update(alive, fps, frame) {
      aliveEl.textContent = alive.toLocaleString();
      fpsEl.textContent = `${fps.toFixed(0)} fps · tick ${frame.toLocaleString()}`;
      aliveHistory.push(alive);
      if (aliveHistory.length > 190) aliveHistory.shift();
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
