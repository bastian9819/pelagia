import './style.css';
import { World } from './sim/world.js';
import { DEFAULT_CONFIG } from './core/config.js';
import { t } from './gpu/i18n.js';

/**
 * Entry point.
 *
 * Default (no params): the production WebGPU ocean — thousands of neural
 * creatures evolving, beautifully rendered. Falls back to a small CPU ocean if
 * WebGPU is unavailable.
 *
 * Dev/diagnostic routes via URL params:
 *   ?engine=cpu                         the Phase 0 CPU reference view
 *   ?engine=gpu&mode=move|brain|grid    Phase 1 performance benchmarks
 *   ?n=20000&warmup=500                 tune the ocean
 */

const params = new URLSearchParams(location.search);
const canvasEl = document.querySelector<HTMLCanvasElement>('#ocean');
if (!canvasEl) throw new Error('Canvas #ocean not found');
const canvas: HTMLCanvasElement = canvasEl;

const engine = params.get('engine');
const mode = params.get('mode');

if (engine === 'cpu') {
  runCpuView(canvas);
} else if (engine === 'gpu' && (mode === 'move' || mode === 'brain' || mode === 'grid')) {
  void runBenchmark(canvas, mode);
} else {
  void bootOcean(canvas);
}

// --- The production ocean, with loading + WebGPU detection + CPU fallback ---
async function bootOcean(target: HTMLCanvasElement): Promise<void> {
  if (!('gpu' in navigator)) {
    fallbackToCpu('Tu navegador no soporta WebGPU.');
    return;
  }
  const n = Math.max(1, Number(params.get('n') ?? 20000));
  const warmup = Math.max(0, Number(params.get('warmup') ?? 500));
  const loading = showLoading();
  try {
    const { runGpuSim } = await import('./gpu/gpuSim.js');
    await runGpuSim(target, { n, warmup, onReady: () => loading.dismiss() });
  } catch (err) {
    console.error(err);
    loading.remove();
    fallbackToCpu('No se pudo iniciar WebGPU en este equipo.');
  }
}

function fallbackToCpu(reason: string): void {
  const b = document.createElement('div');
  b.className = 'banner';
  b.innerHTML = `${reason} Mostrando una versión reducida en CPU. Para la experiencia completa, usa un navegador con <a href="https://caniuse.com/webgpu" target="_blank" rel="noopener">WebGPU</a> (Chrome, Edge, Safari 26+).`;
  document.body.appendChild(b);
  runCpuView(canvas);
}

function showLoading(): { dismiss: () => void; remove: () => void } {
  const el = document.createElement('div');
  el.className = 'loading';
  el.innerHTML =
    `<div><div class="loading-title">PELAGIA</div>` +
    `<div class="loading-sub">${t('loading')}</div></div>`;
  document.body.appendChild(el);
  return {
    dismiss: () => {
      el.classList.add('hide');
      setTimeout(() => el.remove(), 700);
    },
    remove: () => el.remove(),
  };
}

async function runBenchmark(
  target: HTMLCanvasElement,
  benchMode: 'move' | 'brain' | 'grid',
): Promise<void> {
  const n = Math.max(1, Number(params.get('n') ?? 100000));
  try {
    if (benchMode === 'grid') {
      const { runGpuGrid } = await import('./gpu/gridBench.js');
      await runGpuGrid(target, n);
    } else {
      const { runGpuBenchmark } = await import('./gpu/benchmark.js');
      await runGpuBenchmark(target, n, benchMode);
    }
  } catch (err) {
    fallbackToCpu(`El benchmark GPU falló: ${String(err)}.`);
  }
}

// --- CPU reference view (Phase 0): Canvas2D, works anywhere ---
function runCpuView(target: HTMLCanvasElement): void {
  const seed = Number(params.get('seed') ?? DEFAULT_CONFIG.seed);
  const warmup = Number(params.get('warmup') ?? 0);
  const ticksPerFrame = Math.max(1, Number(params.get('speed') ?? 2));

  const world = new World({ ...DEFAULT_CONFIG, seed });
  for (let i = 0; i < warmup; i++) world.step();
  (globalThis as unknown as { __pelagia: unknown }).__pelagia = { world };

  const ctx = target.getContext('2d');
  if (!ctx) throw new Error('2D context unavailable');

  let viewW = 0;
  let viewH = 0;
  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    viewW = window.innerWidth;
    viewH = window.innerHeight;
    target.width = Math.floor(viewW * dpr);
    target.height = Math.floor(viewH * dpr);
    ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx!.fillStyle = '#02040a';
    ctx!.fillRect(0, 0, viewW, viewH);
  }
  resize();
  window.addEventListener('resize', resize);

  function view(): { scale: number; ox: number; oy: number } {
    const scale = Math.min(viewW / world.config.width, viewH / world.config.height);
    return {
      scale,
      ox: (viewW - world.config.width * scale) / 2,
      oy: (viewH - world.config.height * scale) / 2,
    };
  }

  function draw(): void {
    const { scale, ox, oy } = view();
    ctx!.globalCompositeOperation = 'source-over';
    ctx!.fillStyle = 'rgba(2, 4, 10, 0.30)';
    ctx!.fillRect(0, 0, viewW, viewH);
    ctx!.globalCompositeOperation = 'lighter';

    const food = world.food;
    ctx!.fillStyle = 'rgba(90, 150, 140, 0.55)';
    for (let i = 0; i < food.count; i++) {
      ctx!.fillRect(ox + food.x[i]! * scale - 1, oy + food.y[i]! * scale - 1, 2, 2);
    }

    const pop = world.population;
    const r = 3.2;
    for (let i = 0; i < pop.count; i++) {
      const sx = ox + pop.x[i]! * scale;
      const sy = oy + pop.y[i]! * scale;
      const h = pop.heading[i]!;
      ctx!.fillStyle = `hsla(${Math.round(pop.hue[i]! * 360)}, 90%, 62%, 0.95)`;
      ctx!.beginPath();
      ctx!.moveTo(sx + Math.cos(h) * r, sy + Math.sin(h) * r);
      ctx!.lineTo(sx + Math.cos(h + 2.5) * r * 0.8, sy + Math.sin(h + 2.5) * r * 0.8);
      ctx!.lineTo(sx + Math.cos(h - 2.5) * r * 0.8, sy + Math.sin(h - 2.5) * r * 0.8);
      ctx!.closePath();
      ctx!.fill();
    }

    ctx!.globalCompositeOperation = 'source-over';
    drawHud();
  }

  function drawHud(): void {
    const pop = world.population;
    let sumGen = 0;
    let maxGen = 0;
    for (let i = 0; i < pop.count; i++) {
      sumGen += pop.generation[i]!;
      if (pop.generation[i]! > maxGen) maxGen = pop.generation[i]!;
    }
    const meanGen = pop.count ? (sumGen / pop.count).toFixed(1) : '0';
    const lines = [
      `PELAGIA · CPU · seed ${world.config.seed}`,
      `tick ${world.tick}`,
      `pop ${pop.count}`,
      `food ${world.food.count}`,
      `gen ~${meanGen} (max ${maxGen})`,
    ];
    ctx!.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx!.textBaseline = 'top';
    ctx!.fillStyle = 'rgba(2, 4, 10, 0.45)';
    ctx!.fillRect(8, 8, 230, lines.length * 18 + 10);
    ctx!.fillStyle = 'rgba(207, 232, 255, 0.9)';
    lines.forEach((l, i) => ctx!.fillText(l, 16, 16 + i * 18));
  }

  function frame(): void {
    for (let i = 0; i < ticksPerFrame; i++) world.step();
    draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
