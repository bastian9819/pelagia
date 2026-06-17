/**
 * Lightweight DOM UI for the ocean: a live stats panel (population + sparkline +
 * fps) and a control bar (pause, speed, fit-view). Kept out of the render code.
 */
export interface OceanUi {
  panel: HTMLElement;
  controls: HTMLElement;
  /** Push the latest stats (call ~2/s). */
  update(alive: number, fps: number, frame: number): void;
  readonly paused: boolean;
  /** Simulation ticks to run per rendered frame. */
  readonly speed: number;
}

export function buildUi(onFit: () => void): OceanUi {
  let paused = false;
  let speed = 1;
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
  aliveLabel.textContent = 'creatures alive';
  aliveLabel.style.cssText = 'opacity:.55;font-size:11px;';
  const spark = document.createElement('canvas');
  spark.width = 190;
  spark.height = 38;
  spark.style.cssText = 'display:block;margin-top:10px;width:190px;height:38px;';
  const fpsEl = document.createElement('div');
  fpsEl.textContent = '—';
  fpsEl.style.cssText = 'opacity:.65;margin-top:8px;';
  const hint = document.createElement('div');
  hint.style.cssText = 'opacity:.45;font-size:11px;margin-top:10px;line-height:1.5;';
  hint.innerHTML = 'arrastra para mover · rueda para zoom<br>clic en una criatura → su cerebro';
  panel.append(title, aliveEl, aliveLabel, spark, fpsEl, hint);
  const sctx = spark.getContext('2d')!;

  const controls = document.createElement('div');
  controls.style.cssText =
    'position:fixed;bottom:18px;left:50%;transform:translateX(-50%);display:flex;gap:8px;' +
    'font:13px ui-monospace,monospace;';
  const mkBtn = (label: string, onClick: () => void): HTMLButtonElement => {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText =
      'padding:8px 14px;background:rgba(11,31,58,0.85);color:#cfe8ff;' +
      'border:1px solid rgba(63,240,216,0.25);border-radius:8px;cursor:pointer;font:inherit;';
    b.onclick = onClick;
    return b;
  };
  const pauseBtn = mkBtn('⏸', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? '▶' : '⏸';
  });
  const speedBtn = mkBtn('1×', () => {
    speed = speed === 1 ? 2 : speed === 2 ? 4 : 1;
    speedBtn.textContent = `${speed}×`;
  });
  controls.append(pauseBtn, speedBtn, mkBtn('⤢ fit', onFit));

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      pauseBtn.click();
    }
  });

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
      return speed;
    },
  };
}
