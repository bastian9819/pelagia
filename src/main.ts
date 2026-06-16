import './style.css';
import { World } from './sim/world.js';
import { DEFAULT_CONFIG } from './core/config.js';

/**
 * Entry point. Two engines, chosen by URL param:
 *   - default (CPU): the validated Phase 0 reference simulation, drawn with
 *     Canvas2D. URL params: ?seed=1&warmup=3000&speed=4
 *   - GPU spike:     ?engine=gpu&n=100000  — the Phase 1 WebGPU benchmark.
 */

const params = new URLSearchParams(location.search);

const canvas = document.querySelector<HTMLCanvasElement>('#ocean');
if (!canvas) throw new Error('Canvas #ocean not found');

if (params.get('engine') === 'gpu') {
  const n = Math.max(1, Number(params.get('n') ?? 100000));
  const mode = params.get('mode') === 'brain' ? 'brain' : 'move';
  void runGpu(canvas, n, mode);
} else {
  runCpuView(canvas);
}

async function runGpu(canvas: HTMLCanvasElement, n: number, mode: 'move' | 'brain'): Promise<void> {
  try {
    const { runGpuBenchmark } = await import('./gpu/benchmark.js');
    await runGpuBenchmark(canvas, n, mode);
  } catch (err) {
    const msg = document.createElement('div');
    msg.style.cssText =
      'position:fixed;inset:0;display:grid;place-items:center;padding:2rem;text-align:center;' +
      'font:14px ui-monospace,monospace;color:#cfe8ff';
    msg.textContent = `WebGPU benchmark failed: ${String(err)}`;
    document.body.appendChild(msg);
    throw err;
  }
}

function runCpuView(canvas: HTMLCanvasElement): void {
  const seed = Number(params.get('seed') ?? DEFAULT_CONFIG.seed);
  const warmup = Number(params.get('warmup') ?? 0);
  const ticksPerFrame = Math.max(1, Number(params.get('speed') ?? 2));

  const world = new World({ ...DEFAULT_CONFIG, seed });
  for (let i = 0; i < warmup; i++) world.step();
  (globalThis as unknown as { __pelagia: unknown }).__pelagia = { world };

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D context unavailable');

  let viewW = 0;
  let viewH = 0;
  function resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    viewW = window.innerWidth;
    viewH = window.innerHeight;
    canvas.width = Math.floor(viewW * dpr);
    canvas.height = Math.floor(viewH * dpr);
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
      `PELAGIA · seed ${world.config.seed}`,
      `tick ${world.tick}`,
      `pop ${pop.count}`,
      `food ${world.food.count}`,
      `gen ~${meanGen} (max ${maxGen})`,
    ];
    ctx!.font = '13px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx!.textBaseline = 'top';
    ctx!.fillStyle = 'rgba(2, 4, 10, 0.45)';
    ctx!.fillRect(8, 8, 210, lines.length * 18 + 10);
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
