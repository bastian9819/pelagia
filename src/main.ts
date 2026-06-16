import './style.css';
import { World } from './sim/world.js';
import { DEFAULT_CONFIG } from './core/config.js';

/**
 * Phase 0.8 live visualisation.
 *
 * A deliberately simple Canvas2D view of the simulation so emergence can be
 * watched directly (the beautiful WebGPU renderer arrives in Phase 2). Creatures
 * are triangles oriented by heading and coloured by lineage hue; food is dim
 * cyan; a faint trail reveals trajectories. URL params (seed, warmup, speed)
 * make runs reproducible and are the first seed of the Phase 5 share-by-URL.
 */

const params = new URLSearchParams(location.search);
const seed = Number(params.get('seed') ?? DEFAULT_CONFIG.seed);
const warmup = Number(params.get('warmup') ?? 0);
const ticksPerFrame = Math.max(1, Number(params.get('speed') ?? 2));

const world = new World({ ...DEFAULT_CONFIG, seed });
for (let i = 0; i < warmup; i++) world.step();

// Expose for debugging / verification.
(globalThis as unknown as { __pelagia: unknown }).__pelagia = { world };

const canvas = document.querySelector<HTMLCanvasElement>('#ocean');
if (!canvas) throw new Error('Canvas #ocean not found');
const ctx = canvas.getContext('2d');
if (!ctx) throw new Error('2D context unavailable');

let viewW = 0;
let viewH = 0;
function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  viewW = window.innerWidth;
  viewH = window.innerHeight;
  canvas!.width = Math.floor(viewW * dpr);
  canvas!.height = Math.floor(viewH * dpr);
  ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
  // Repaint solid background after a resize so trails build on a clean base.
  ctx!.fillStyle = '#02040a';
  ctx!.fillRect(0, 0, viewW, viewH);
}
resize();
window.addEventListener('resize', resize);

function worldToScreen(): { scale: number; ox: number; oy: number } {
  const scale = Math.min(viewW / world.config.width, viewH / world.config.height);
  const ox = (viewW - world.config.width * scale) / 2;
  const oy = (viewH - world.config.height * scale) / 2;
  return { scale, ox, oy };
}

function draw(): void {
  const { scale, ox, oy } = worldToScreen();

  // Fade previous frame for motion trails (bioluminescent persistence).
  ctx!.globalCompositeOperation = 'source-over';
  ctx!.fillStyle = 'rgba(2, 4, 10, 0.30)';
  ctx!.fillRect(0, 0, viewW, viewH);

  ctx!.globalCompositeOperation = 'lighter';

  // Food.
  const food = world.food;
  ctx!.fillStyle = 'rgba(90, 150, 140, 0.55)';
  for (let i = 0; i < food.count; i++) {
    const sx = ox + food.x[i]! * scale;
    const sy = oy + food.y[i]! * scale;
    ctx!.fillRect(sx - 1, sy - 1, 2, 2);
  }

  // Creatures: triangles pointing along heading, coloured by lineage hue.
  const pop = world.population;
  const r = 3.2;
  for (let i = 0; i < pop.count; i++) {
    const sx = ox + pop.x[i]! * scale;
    const sy = oy + pop.y[i]! * scale;
    const h = pop.heading[i]!;
    const hue = Math.round(pop.hue[i]! * 360);
    ctx!.fillStyle = `hsla(${hue}, 90%, 62%, 0.95)`;
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
